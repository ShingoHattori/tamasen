#!/usr/bin/env python3
"""reBotArm 重力・摩擦のスイープ同定（J2 / J3）

仕組み:
  J2 または J3 を一定のゆっくりした速度で往復させ、動いている間のトルクを連続で記録する。
  関節ごとに
      実測トルク = SCALE*g(q) + OFFSET + FC*sign(v) + FV*v
  を最小二乗で当てはめ、重力補償（SCALE, OFFSET）と摩擦（クーロン摩擦 FC, 粘性摩擦 FV）を同時に求める。
  静止摩擦のばらつきに左右されないので、止めて測る方式より精度が出る。

使い方（reBotArm_control_py/example/ に置く）:
  uv run python example/sweep_calib.py
  1. 重力補償で浮いた状態になるので、安全な基準姿勢に手で動かして Enter
     （目安: 腕を斜め前に持ち上げた中間姿勢。J2 ±15°, J3 ±25° 動かしても何にも当たらないこと）
  2. J2 と J3 のスイープを自動で実行（約2分）
  3. 結果を表示し、grav_params.json に保存（teach.py が読み込む）

  uv run python example/sweep_calib.py --from-raw sweep_raw.npz   # 記録データから再フィットだけ行う

注意:
  - 測定中はアームが自動で動く。異常時は Ctrl-C（重力補償のみのフロートに切り替わる）
"""
import argparse
import json
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
RAW_FILE = HERE / "sweep_raw.npz"

# ── 測定中の重力補償（手で合わせた値。同定結果には影響しない。追従を助けるだけ） ─────
#                        joint1 joint2 joint3 joint4 joint5 joint6
BASE_SCALE  = np.array([  1.0,   0.9,   1.0,   1.0,   1.0,   1.0])
BASE_OFFSET = np.array([  0.0,   0.4,   0.0,   0.0,   0.0,   0.0])

# ── スイープ設定 ─────────────────────────────────────────────────────────────
# (動かす関節の index, 振幅[deg], 他の関節のずらし量 {index: deg})
SWEEPS = [
    (1, 15, {2: -20}),
    (1, 15, {2: 0}),
    (1, 15, {2: 20}),
    (2, 25, {1: -10}),
    (2, 25, {1: 0}),
    (2, 25, {1: 10}),
]
CYCLES = 2                # 各スイープの往復回数
SWEEP_VEL = 0.15          # スイープ速度 [rad/s]
SWEEP_ACC = 0.5           # 加減速 [rad/s^2]
MOVE_VEL = 0.2            # スイープ開始位置への移動速度 [rad/s]

# ── ゲイン（追従を良くするため J2/J3 は高め） ────────────────────────────────
KP = np.array([30.0, 60.0, 60.0, 15.0, 10.0, 10.0])
KD = np.array([ 1.5,  2.0,  2.0,  0.5,  0.4,  0.4])
FLOAT_KP = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
FLOAT_KD = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])

# ── 安全 ────────────────────────────────────────────────────────────────────
TRACK_ERR_MAX = np.radians(12)
TORQUE_MAX = 15.0

# ── フィット ─────────────────────────────────────────────────────────────────
VEL_MIN_RATIO = 0.6       # 指令速度のこの割合以上で動いているサンプルだけ使う
# ─────────────────────────────────────────────────────────────────────────────


def g_model(q):
    return np.asarray(compute_generalized_gravity(q=q))


def trapezoid(p0, p1, vmax, amax):
    """p0→p1 の台形速度プロファイル。(所要時間, pos(t), vel(t)) を返す"""
    d = p1 - p0
    D, s = abs(d), (1.0 if d >= 0 else -1.0)
    if D < 1e-9:
        return 0.0, (lambda t: p1), (lambda t: 0.0)
    if D < vmax * vmax / amax:
        ta = np.sqrt(D / amax)
        vp = amax * ta
        T = 2 * ta
        tc = 0.0
    else:
        ta = vmax / amax
        vp = vmax
        tc = D / vmax - ta
        T = 2 * ta + tc

    def pos(t):
        t = min(max(t, 0.0), T)
        if t < ta:
            x = 0.5 * amax * t * t
        elif t < ta + tc:
            x = 0.5 * amax * ta * ta + vp * (t - ta)
        else:
            tr = T - t
            x = D - 0.5 * amax * tr * tr
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
        self.mode = "float"       # float / move / hold / sweep
        self.q_hold = None        # move/hold/sweep 中の基準姿勢
        # move
        self.q_from = None
        self.q_to = None
        self.T = 1.0
        self.t0 = 0.0
        # sweep
        self.joint = None
        self.segs = []            # [(t_start, T, pos, vel), ...]
        self.v_cmd = 0.0
        self.log = []             # (joint, q, v, tau)
        self.error = None


S = State()


def controller(r: RebotArm, dt: float) -> None:
    try:
        n = r.arm.num_joints
        q = np.asarray(r.arm.get_positions(request_feedback=False))
        v = np.asarray(r.arm.get_velocities(request_feedback=False))
        tau_ff = g_model(q) * BASE_SCALE + BASE_OFFSET
        now = time.perf_counter()

        with S.lock:
            if S.mode == "float":
                r.arm.send_mit(pos=q, vel=np.zeros(n), kp=FLOAT_KP, kd=FLOAT_KD, tau=tau_ff)
                return

            ref = S.q_hold.copy()
            vref = np.zeros(n)
            if S.mode == "move":
                u = min(1.0, (now - S.t0) / S.T)
                s = u * u * (3 - 2 * u)  # smoothstep
                ref = S.q_from + (S.q_to - S.q_from) * s
                if u >= 1.0:
                    S.q_hold = S.q_to.copy()
                    S.mode = "hold"
            elif S.mode == "sweep":
                t = now - S.t0
                last = S.segs[-1]
                if t >= last[0] + last[1]:
                    # 全区間終了
                    ref[S.joint] = last[2](last[1])
                    S.q_hold = ref.copy()
                    S.mode = "hold"
                else:
                    # t 以前に始まった最後の区間を使う（折り返しの停止中は、その区間の終点で止まる）
                    seg = S.segs[0]
                    for sg in S.segs:
                        if sg[0] <= t:
                            seg = sg
                        else:
                            break
                    ts = min(t - seg[0], seg[1])
                    ref[S.joint] = seg[2](ts)
                    vref[S.joint] = seg[3](ts) if t - seg[0] < seg[1] else 0.0
                    tau_meas = np.asarray(r.get_torques())[:n]
                    S.log.append((S.joint, q.copy(), v.copy(), tau_meas.copy(), vref[S.joint]))

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
        S.T = max(0.8, 1.5 * dist / MOVE_VEL)  # smoothstep のピーク速度は平均の1.5倍
        S.t0 = time.perf_counter()
        S.mode = "move"
    wait_mode("hold")


def run_sweep(joint, center_pose, amp):
    lo = center_pose[joint] - amp
    hi = center_pose[joint] + amp
    start = center_pose.copy()
    start[joint] = lo
    move_to(start)
    time.sleep(0.3)

    segs, t = [], 0.0
    for _ in range(CYCLES):
        for a, b in ((lo, hi), (hi, lo)):
            T, pos, vel = trapezoid(a, b, SWEEP_VEL, SWEEP_ACC)
            segs.append((t, T, pos, vel))
            t += T + 0.3  # 折り返しで少し止まる
    with S.lock:
        S.joint = joint
        S.segs = segs
        S.t0 = time.perf_counter()
        S.mode = "sweep"
    wait_mode("hold")


def fit(log, n):
    joints = np.array([x[0] for x in log])
    Q = np.array([x[1] for x in log])
    V = np.array([x[2] for x in log])
    T = np.array([x[3] for x in log])
    VR = np.array([x[4] for x in log])

    result = {}
    print("\n関節   | サンプル | SCALE  OFFSET |  FC    FV   | 残差RMS [Nm] | 参考: 旧設定での残差")
    for j in sorted(set(joints.tolist())):
        m = joints == j
        vj, vr = V[m, j], VR[m]
        use = (np.abs(vr) >= SWEEP_VEL * 0.99) & (np.sign(vj) == np.sign(vr)) \
              & (np.abs(vj) >= SWEEP_VEL * VEL_MIN_RATIO)
        n_up = int(np.sum(use & (vr > 0)))
        n_dn = int(np.sum(use & (vr < 0)))
        if min(n_up, n_dn) < 50:
            print(f"joint{j+1} | 上り{n_up}・下り{n_dn} サンプル。両方向のデータが足りないのでフィットしません")
            continue
        q, v, tau = Q[m][use], vj[use], T[m][use, j]
        g = np.array([g_model(x)[j] for x in q])
        A = np.stack([g, np.ones_like(g), np.sign(v), v], axis=1)
        coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
        a, b, fc, fv = coef
        rms = np.sqrt(np.mean((A @ coef - tau) ** 2))
        old = np.sqrt(np.mean((BASE_SCALE[j] * g + BASE_OFFSET[j] - tau) ** 2))
        warn = ""
        if a < 0:
            warn += "  ← SCALEが負。要確認"
        if fc < 0:
            warn += "  ← 摩擦が負。要確認"
        print(f"joint{j+1} | {use.sum():6d}  | {a:5.2f}  {b:+5.2f} | {fc:4.2f}  {fv:5.2f} | "
              f"{rms:.3f}        | {old:.3f}{warn}")
        result[j] = (a, b, fc, fv)
    return result


def save(result, n):
    scale, offset = BASE_SCALE.copy(), BASE_OFFSET.copy()
    fc, fv = np.zeros(n), np.zeros(n)
    for j, (a, b, c, d) in result.items():
        scale[j], offset[j], fc[j], fv[j] = a, b, max(c, 0.0), max(d, 0.0)
    print("\n結果:")
    print(f"  scale  = {np.round(scale, 3).tolist()}")
    print(f"  offset = {np.round(offset, 3).tolist()}")
    print(f"  fc     = {np.round(fc, 3).tolist()}   # クーロン摩擦 [N·m]")
    print(f"  fv     = {np.round(fv, 3).tolist()}   # 粘性摩擦 [N·m/(rad/s)]")
    if input(">>> grav_params.json に保存しますか？ [y/N] ").strip().lower() == "y":
        PARAM_FILE.write_text(json.dumps({
            "scale": np.round(scale, 4).tolist(), "offset": np.round(offset, 4).tolist(),
            "fc": np.round(fc, 4).tolist(), "fv": np.round(fv, 4).tolist()}, indent=2))
        print(f"{PARAM_FILE.name} に保存しました")


def save_raw(log):
    np.savez(RAW_FILE,
             joint=np.array([x[0] for x in log]), q=np.array([x[1] for x in log]),
             v=np.array([x[2] for x in log]), tau=np.array([x[3] for x in log]),
             vref=np.array([x[4] for x in log]))
    print(f"記録データを {RAW_FILE.name} に保存しました（{len(log)} サンプル）")


def load_raw(path):
    d = np.load(path)
    return list(zip(d["joint"].tolist(), d["q"], d["v"], d["tau"], d["vref"].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", help="記録データ（.npz）から再フィットだけ行う")
    args = ap.parse_args()

    load_dynamics_model()
    if args.from_raw:
        log = load_raw(args.from_raw)
        n = log[0][1].shape[0]
        save(fit(log, n), n)
        return

    r = RebotArm()
    r.connect()
    r.arm.mode_mit()
    if r.has_gripper:
        r.gripper.mode_mit()
    r.disable_all()
    time.sleep(0.1)
    n = r.arm.num_joints

    try:
        r.enable_all()
        r.start_control_loop(controller, rate=r.rate)
        print(f"[制御ループ] {r.rate} Hz")
        print("重力補償中です。安全な基準姿勢に手で動かしてください。")
        input(">>> 基準姿勢に置いたら Enter")
        q_base = np.asarray(r.arm.get_positions())
        print(f"基準姿勢(deg): {np.degrees(q_base).round(1)}")
        print("動く範囲: " + ", ".join(
            f"J{j+1}±{amp}°(" + ",".join(f"J{k+1}{d:+d}°" for k, d in off.items()) + ")"
            for j, amp, off in SWEEPS))
        input(">>> 可動範囲に何もないことを確認して Enter（中止は Ctrl-C）")

        with S.lock:
            S.q_hold = q_base.copy()
            S.mode = "hold"

        for i, (j, amp, off) in enumerate(SWEEPS):
            center = q_base.copy()
            for k, d in off.items():
                center[k] += np.radians(d)
            print(f"[{i+1}/{len(SWEEPS)}] joint{j+1} を ±{amp}° スイープ中...")
            run_sweep(j, center, np.radians(amp))

        print("スイープ完了。基準姿勢へ戻ります...")
        move_to(q_base)
    except KeyboardInterrupt:
        print("\n中止しました")
    except RuntimeError as e:
        print(f"\n[!] {e}")
    finally:
        with S.lock:
            S.mode = "float"
            log = list(S.log)
        if log:
            save_raw(log)
            try:
                save(fit(log, n), n)
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
