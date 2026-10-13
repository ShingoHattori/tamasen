#!/usr/bin/env python3
"""reBotArm 全自動キャリブレーション（重力補償 + 摩擦）

やること:
  1. ゼロ姿勢付近にいることを確認
  2. 自動で測定姿勢（CALIB_POSE_DEG）へゆっくり移動
  3. J2 / J3 を2種類の速度で往復させ、トルクを連続記録
  4. 関節ごとに  実測トルク = SCALE*g(q) + OFFSET + FC*sign(v) + FV*v  を最小二乗で当てはめ
  5. スタート位置へ戻る
  6. 結果が妥当なら grav_params.json に自動保存（前の値は grav_params.bak.json に退避）

使い方（reBotArm_control_py/example/ に置く）:
  uv run python example/auto_calib.py
  uv run python example/auto_calib.py --from-raw calib_raw.npz   # 記録データから再フィットだけ

使う場面: 初回、組み立て直したとき、ケーブルや手先の道具を付け替えたとき。
"""
import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import load_dynamics_model, compute_generalized_gravity

HERE = Path(__file__).resolve().parent
PARAM_FILE = HERE / "grav_params.json"
BACKUP_FILE = HERE / "grav_params.bak.json"
RAW_FILE = HERE / "calib_raw.npz"

# ── 姿勢 ─────────────────────────────────────────────────────────────────────
#                               J1    J2    J3   J4   J5   J6
CALIB_POSE_DEG = np.array([   0.0, -60.0, -75.0, 0.0, 0.0, 0.0])  # 測定姿勢
ZERO_TOL_DEG = 15.0          # 開始時、全関節がゼロからこの範囲内にあること

# ── スイープ (動かす関節 index, 振幅[deg], 他関節のずらし量 {index: deg}) ─────
SWEEPS = [
    (1, 15, {2: -20}),
    (1, 15, {2: 0}),
    (1, 15, {2: 20}),
    (2, 25, {1: -10}),
    (2, 25, {1: 0}),
    (2, 25, {1: 10}),
]
SWEEP_SPEEDS = [0.10, 0.25]  # 往復ごとに速度を変える（粘性摩擦を分離するため）[rad/s]
SWEEP_ACC = 0.6              # [rad/s^2]
MOVE_VEL = 0.2               # 姿勢間の移動速度 [rad/s]
PAUSE_S = 0.3                # 折り返しの停止時間

# ── ゲイン ───────────────────────────────────────────────────────────────────
KP = np.array([30.0, 60.0, 60.0, 15.0, 10.0, 10.0])
KD = np.array([ 1.5,  2.0,  2.0,  0.5,  0.4,  0.4])

# ── 安全 ─────────────────────────────────────────────────────────────────────
TRACK_ERR_MAX = np.radians(12)
TORQUE_MAX = 15.0

# ── 自動保存の条件 ───────────────────────────────────────────────────────────
SCALE_RANGE = (0.7, 1.3)
FC_RANGE = (0.0, 3.0)
RMS_MAX = 0.8
FIT_JOINTS_NOTE = "J2, J3 のみ同定。他の関節は既存の値を引き継ぐ"
# ─────────────────────────────────────────────────────────────────────────────


def load_params():
    p = {"scale": [1.0] * 6, "offset": [0.0] * 6, "fc": [0.0] * 6, "fv": [0.0] * 6}
    if PARAM_FILE.exists():
        p.update(json.loads(PARAM_FILE.read_text()))
    return {k: np.array(v, dtype=float) for k, v in p.items()}


PARAMS = load_params()   # 測定中の重力フィードフォワードに使う（同定結果には影響しない）


def g_model(q):
    return np.asarray(compute_generalized_gravity(q=q))


def trapezoid(p0, p1, vmax, amax):
    d = p1 - p0
    D, s = abs(d), (1.0 if d >= 0 else -1.0)
    if D < 1e-9:
        return 0.0, (lambda t: p1), (lambda t: 0.0)
    if D < vmax * vmax / amax:
        ta = np.sqrt(D / amax); vp = amax * ta; tc = 0.0
    else:
        ta = vmax / amax; vp = vmax; tc = D / vmax - ta
    T = 2 * ta + tc

    def pos(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            x = 0.5 * amax * t * t
        elif t < ta + tc:
            x = 0.5 * amax * ta * ta + vp * (t - ta)
        else:
            x = D - 0.5 * amax * (T - t) ** 2
        return p0 + s * x

    def vel(t):
        if t <= 0 or t >= T:
            return 0.0
        if t < ta:
            return s * amax * t
        if t < ta + tc:
            return s * vp
        return s * amax * (T - t)

    return T, pos, vel


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "idle"        # idle / move / hold / sweep / float
        self.q_hold = None
        self.q_from = None
        self.q_to = None
        self.T = 1.0
        self.t0 = 0.0
        self.joint = None
        self.segs = []            # [(t_start, T, pos, vel, vmax)]
        self.log = []             # (joint, q, v, tau, vref, vmax)
        self.error = None


S = State()


def controller(r: RebotArm, dt: float) -> None:
    try:
        n = r.arm.num_joints
        q = np.asarray(r.arm.get_positions(request_feedback=False))
        v = np.asarray(r.arm.get_velocities(request_feedback=False))
        tau_ff = g_model(q) * PARAMS["scale"] + PARAMS["offset"]
        now = time.perf_counter()

        with S.lock:
            if S.mode == "float":
                r.arm.send_mit(pos=q, vel=np.zeros(n), kp=np.full(n, 2.0),
                               kd=np.full(n, 1.0), tau=tau_ff)
                return
            if S.mode == "idle":
                return

            ref = S.q_hold.copy()
            vref = np.zeros(n)
            if S.mode == "move":
                u = min(1.0, (now - S.t0) / S.T)
                s = u * u * (3 - 2 * u)
                ref = S.q_from + (S.q_to - S.q_from) * s
                if u >= 1.0:
                    S.q_hold = S.q_to.copy()
                    S.mode = "hold"
            elif S.mode == "sweep":
                t = now - S.t0
                last = S.segs[-1]
                if t >= last[0] + last[1]:
                    ref[S.joint] = last[2](last[1])
                    S.q_hold = ref.copy()
                    S.mode = "hold"
                else:
                    seg = S.segs[0]
                    for sg in S.segs:
                        if sg[0] <= t:
                            seg = sg
                        else:
                            break
                    ts = t - seg[0]
                    ref[S.joint] = seg[2](min(ts, seg[1]))
                    vref[S.joint] = seg[3](ts) if ts < seg[1] else 0.0
                    tau_meas = np.asarray(r.get_torques())[:n]
                    S.log.append((S.joint, q.copy(), v.copy(), tau_meas.copy(),
                                  vref[S.joint], seg[4]))

            err = np.abs(ref - q)
            if err.max() > TRACK_ERR_MAX:
                S.error = f"追従誤差が上限を超えました(deg): {np.degrees(err).round(1)}"
                S.mode = "float"
                return
            r.arm.send_mit(pos=ref, vel=vref, kp=KP, kd=KD, tau=tau_ff)

            tau_now = np.asarray(r.get_torques())[:n]
            if np.abs(tau_now).max() > TORQUE_MAX:
                S.error = f"トルクが上限を超えました: {tau_now.round(2)}"
                S.mode = "float"
    except Exception as e:
        S.error = f"制御ループで例外: {e!r}"
        S.mode = "float"


def wait_mode(target):
    while True:
        if S.error:
            raise RuntimeError(S.error)
        with S.lock:
            if S.mode == target:
                return
        time.sleep(0.02)


def move_to(q_target):
    with S.lock:
        q_now = S.q_hold.copy()
        dist = np.max(np.abs(q_target - q_now))
        S.q_from, S.q_to = q_now, q_target.copy()
        S.T = max(1.0, 1.5 * dist / MOVE_VEL)
        S.t0 = time.perf_counter()
        S.mode = "move"
    wait_mode("hold")


def run_sweep(joint, center, amp):
    lo, hi = center[joint] - amp, center[joint] + amp
    start = center.copy()
    start[joint] = lo
    move_to(start)
    time.sleep(PAUSE_S)
    segs, t = [], 0.0
    for vmax in SWEEP_SPEEDS:
        for a, b in ((lo, hi), (hi, lo)):
            T, pos, vel = trapezoid(a, b, vmax, SWEEP_ACC)
            segs.append((t, T, pos, vel, vmax))
            t += T + PAUSE_S
    with S.lock:
        S.joint, S.segs = joint, segs
        S.t0 = time.perf_counter()
        S.mode = "sweep"
    wait_mode("hold")


# ── 同定 ─────────────────────────────────────────────────────────────────────
def fit(log):
    joints = np.array([x[0] for x in log])
    Q = np.array([x[1] for x in log])
    V = np.array([x[2] for x in log])
    T = np.array([x[3] for x in log])
    VR = np.array([x[4] for x in log])
    VM = np.array([x[5] for x in log])

    result = {}
    print("\n関節   | サンプル | SCALE  OFFSET |  FC    FV   | 残差RMS | 旧設定の残差")
    for j in sorted(set(joints.tolist())):
        m = joints == j
        vj, vr, vm = V[m, j], VR[m], VM[m]
        use = (np.abs(vr) >= vm * 0.99) & (np.sign(vj) == np.sign(vr)) & (np.abs(vj) >= vm * 0.6)
        n_up, n_dn = int(np.sum(use & (vr > 0))), int(np.sum(use & (vr < 0)))
        if min(n_up, n_dn) < 50:
            print(f"joint{j+1} | 上り{n_up}・下り{n_dn}。両方向のデータ不足でスキップ")
            continue
        q, v, tau = Q[m][use], vj[use], T[m][use, j]
        g = np.array([g_model(x)[j] for x in q])

        speeds_used = np.unique(np.round(vm[use], 3))
        cols = [g, np.ones_like(g), np.sign(v)]
        if len(speeds_used) >= 2:
            cols.append(v)
        A = np.stack(cols, axis=1)
        with np.errstate(all="ignore"):
            coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
            rms = float(np.sqrt(np.mean((A @ coef - tau) ** 2)))
            old = float(np.sqrt(np.mean((PARAMS["scale"][j] * g + PARAMS["offset"][j]
                                         + PARAMS["fc"][j] * np.sign(v) + PARAMS["fv"][j] * v
                                         - tau) ** 2)))
        a, b, fc = coef[:3]
        fv = coef[3] if len(coef) > 3 else 0.0
        print(f"joint{j+1} | {len(tau):6d}  | {a:5.2f}  {b:+5.2f} | {fc:4.2f}  {fv:+5.2f} | "
              f"{rms:.3f}   | {old:.3f}")
        result[j] = dict(scale=a, offset=b, fc=fc, fv=fv, rms=rms)
    return result


def check(result):
    problems = []
    for j, r_ in result.items():
        if not SCALE_RANGE[0] <= r_["scale"] <= SCALE_RANGE[1]:
            problems.append(f"joint{j+1}: SCALE {r_['scale']:.2f} が範囲外")
        if not FC_RANGE[0] <= r_["fc"] <= FC_RANGE[1]:
            problems.append(f"joint{j+1}: FC {r_['fc']:.2f} が範囲外")
        if r_["fv"] < -0.2:
            problems.append(f"joint{j+1}: FV {r_['fv']:.2f} が負")
        if r_["rms"] > RMS_MAX:
            problems.append(f"joint{j+1}: 残差 {r_['rms']:.2f} Nm が大きい")
    if not result:
        problems.append("同定できた関節がありません")
    return problems


def save(result):
    new = {k: PARAMS[k].copy() for k in ("scale", "offset", "fc", "fv")}
    for j, r_ in result.items():
        new["scale"][j] = r_["scale"]
        new["offset"][j] = r_["offset"]
        new["fc"][j] = max(r_["fc"], 0.0)
        new["fv"][j] = max(r_["fv"], 0.0)
    print("\n結果（" + FIT_JOINTS_NOTE + "）:")
    for k in ("scale", "offset", "fc", "fv"):
        print(f"  {k:6s} = {np.round(new[k], 3).tolist()}")

    problems = check(result)
    if problems:
        print("\n[!] 結果に気になる点があるので自動保存しません:")
        for p in problems:
            print("   - " + p)
        if input(">>> それでも保存しますか？ [y/N] ").strip().lower() != "y":
            return
    if PARAM_FILE.exists():
        shutil.copy(PARAM_FILE, BACKUP_FILE)
    PARAM_FILE.write_text(json.dumps({k: np.round(v, 4).tolist() for k, v in new.items()},
                                     indent=2))
    print(f"{PARAM_FILE.name} に保存しました"
          + (f"（前の値は {BACKUP_FILE.name}）" if BACKUP_FILE.exists() else ""))


def save_raw(log):
    np.savez(RAW_FILE,
             joint=np.array([x[0] for x in log]), q=np.array([x[1] for x in log]),
             v=np.array([x[2] for x in log]), tau=np.array([x[3] for x in log]),
             vref=np.array([x[4] for x in log]), vmax=np.array([x[5] for x in log]))
    print(f"記録データを {RAW_FILE.name} に保存しました（{len(log)} サンプル）")


def load_raw(path):
    d = np.load(path)
    return list(zip(d["joint"].tolist(), d["q"], d["v"], d["tau"],
                    d["vref"].tolist(), d["vmax"].tolist()))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", help="記録データ（.npz）から再フィットだけ行う")
    args = ap.parse_args()

    load_dynamics_model()
    if args.from_raw:
        save(fit(load_raw(args.from_raw)))
        return

    r = RebotArm()
    r.connect()
    r.arm.mode_mit()
    if r.has_gripper:
        r.gripper.mode_mit()
    r.disable_all()
    time.sleep(0.1)

    q_start = np.asarray(r.arm.get_positions())
    off = np.degrees(np.abs(q_start))
    if off.max() > ZERO_TOL_DEG:
        print(f"現在の関節角(deg): {np.degrees(q_start).round(1)}")
        print(f"ゼロ姿勢から {off.max():.0f}° 離れた関節があります。ゼロ姿勢付近に戻してから実行してください。")
        r.disconnect()
        return

    calib_pose = np.radians(CALIB_POSE_DEG)
    print("=" * 64)
    print("  全自動キャリブレーション（約3分）")
    print(f"  測定姿勢(deg): {CALIB_POSE_DEG.tolist()}")
    print("  動く範囲: " + ", ".join(
        f"J{j+1}±{amp}°" for j, amp, _ in SWEEPS[::3]) + "（＋相手関節を ±10〜20°）")
    print("  異常時は Ctrl-C（重力補償のみのフロートに切り替わります）")
    print("=" * 64)
    try:
        input(">>> アームの周りに何もないことを確認して Enter")
    except KeyboardInterrupt:
        r.disconnect()
        return

    try:
        with S.lock:
            S.q_hold = q_start.copy()
            S.mode = "hold"
        r.enable_all()
        r.start_control_loop(controller, rate=r.rate)
        time.sleep(0.3)

        print("測定姿勢へ移動中...")
        move_to(calib_pose)
        time.sleep(0.5)

        for i, (j, amp, offs) in enumerate(SWEEPS):
            center = calib_pose.copy()
            for k, d in offs.items():
                center[k] += np.radians(d)
            print(f"[{i+1}/{len(SWEEPS)}] joint{j+1} を ±{amp}° スイープ中...")
            run_sweep(j, center, np.radians(amp))

        print("スイープ完了。スタート位置へ戻ります...")
        move_to(calib_pose)
        move_to(q_start)
    except KeyboardInterrupt:
        print("\n中止しました")
    except RuntimeError as e:
        print(f"\n[!] {e}")
    finally:
        with S.lock:
            if S.mode != "hold":
                S.mode = "float"
            log = list(S.log)
        if log:
            save_raw(log)
            try:
                save(fit(log))
            except Exception as e:
                print(f"[!] フィットに失敗しました: {e!r}")
        try:
            input("\n>>> アームを手で支えてから Enter で終了します（トルクが抜けます）")
        except (EOFError, KeyboardInterrupt):
            pass
        r.disconnect()
        print("切断しました")


if __name__ == "__main__":
    main()
