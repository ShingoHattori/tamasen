#!/usr/bin/env python3
"""reBotArm 重力補償の自動キャリブレーション

仕組み:
  アームを複数の姿勢に移動させ、静止したときにモーターが出しているトルク（get_torques）を測る。
  静止中のトルク ≒ 重力を支えるトルク なので、関節ごとに
      実測トルク = SCALE × モデルの重力トルク + OFFSET
  を最小二乗で当てはめ、GRAV_SCALE / GRAV_OFFSET を求める。
  摩擦を打ち消すため、各姿勢に「上から」と「下から」の2方向で近づいて平均する。

使い方（reBotArm_control_py/example/ に置く）:
  uv run python example/auto_gravity_calib.py
  1. 起動すると重力補償で浮いた状態になる
  2. 周りに何もない、安全な「基準姿勢」に手で動かして Enter
  3. 基準姿勢のまわりの姿勢を自動で巡回して測定（数分かかる）
  4. 結果を表示し、grav_params.json に保存（teach.py が自動で読み込む）

注意:
  - 測定中はアームが自動で動く。可動範囲（下の OFFSETS_DEG）に何もないことを確認すること
  - 異常時は Ctrl-C。重力補償のみ（フロート）に切り替わる
"""
import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import load_dynamics_model, compute_generalized_gravity

PARAM_FILE = Path(__file__).with_name("grav_params.json")

# ── 現在の重力補償パラメータ（移動中の補助に使う。grav_params.json があればそちらを優先） ──
#                        joint1 joint2 joint3 joint4 joint5 joint6
GRAV_SCALE  = np.array([  1.0,   0.9,   1.0,   1.0,   1.0,   1.0])
GRAV_OFFSET = np.array([  0.0,   0.4,   0.0,   0.0,   0.0,   0.0])

# ── 測定する姿勢（基準姿勢からのずらし量 [deg]） ─────────────────────────────
J2_OFFSETS = [-20, -10, 0, 10, 20]
J3_OFFSETS = [-30, -15, 0, 15, 30]
J5_OFFSETS = [-30, 0, 30]          # J2/J3 の各組み合わせで順番に使う
APPROACH_DELTA_DEG = 5.0           # 摩擦キャンセル用に、この量だけ手前から近づく

# ── 動作パラメータ ───────────────────────────────────────────────────────────
KP = np.array([25.0, 25.0, 25.0, 10.0, 10.0, 10.0])
KD = np.array([ 1.5,  1.5,  1.5,  0.4,  0.4,  0.4])
FLOAT_KP = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
FLOAT_KD = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
MOVE_VEL = 0.3            # 移動速度 [rad/s]
RETURN_VEL = 0.12         # 測定後に基準姿勢へ戻るときの速度 [rad/s]
SETTLE_S = 1.0            # 到着後の待ち時間
SAMPLE_S = 0.8            # トルクを平均する時間
TRACK_ERR_MAX = np.radians(15)
TORQUE_MAX = 12.0         # これを超えるトルクが出たら中止 [N·m]
MIN_G_RANGE = 0.3         # モデルの重力トルクの変化幅がこれ未満の関節はフィットしない
# ─────────────────────────────────────────────────────────────────────────────

if PARAM_FILE.exists():
    p = json.loads(PARAM_FILE.read_text())
    GRAV_SCALE, GRAV_OFFSET = np.array(p["scale"]), np.array(p["offset"])


def gravity_model(q):
    return np.asarray(compute_generalized_gravity(q=q))


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "float"      # float / move / hold
        self.q_from = None
        self.q_to = None
        self.t0 = 0.0
        self.T = 1.0
        self.sampling = False
        self.samples = []        # (q, tau_meas)
        self.error = None


S = State()


def controller(r: RebotArm, dt: float) -> None:
    try:
        n = r.arm.num_joints
        q = np.asarray(r.arm.get_positions(request_feedback=False))
        tau_ff = gravity_model(q) * GRAV_SCALE + GRAV_OFFSET
        now = time.perf_counter()
        with S.lock:
            if S.mode == "float":
                r.arm.send_mit(pos=q, vel=np.zeros(n), kp=FLOAT_KP, kd=FLOAT_KD, tau=tau_ff)
                return
            if S.mode == "move":
                a = min(1.0, (now - S.t0) / S.T)
                ref = S.q_from + (S.q_to - S.q_from) * a
                if a >= 1.0:
                    S.mode = "hold"
            else:
                ref = S.q_to

            if np.max(np.abs(ref - q)) > TRACK_ERR_MAX:
                S.error = f"追従誤差が上限を超えました: {np.degrees(np.abs(ref - q)).round(1)}"
                S.mode = "float"
                return
            r.arm.send_mit(pos=ref, vel=np.zeros(n), kp=KP, kd=KD, tau=tau_ff)

            tau_meas = np.asarray(r.get_torques())[:n]
            if np.max(np.abs(tau_meas)) > TORQUE_MAX:
                S.error = f"トルクが上限を超えました: {tau_meas.round(2)}"
                S.mode = "float"
                return
            if S.sampling:
                S.samples.append((q.copy(), tau_meas.copy()))
    except Exception as e:
        S.error = f"制御ループで例外: {e!r}"
        S.mode = "float"


def check_error():
    if S.error:
        raise RuntimeError(S.error)


def move_to(q_target, q_now):
    dist = np.max(np.abs(q_target - q_now))
    with S.lock:
        S.q_from, S.q_to = q_now.copy(), q_target.copy()
        S.T = max(0.5, dist / MOVE_VEL)
        S.t0 = time.perf_counter()
        S.mode = "move"
    while True:
        check_error()
        with S.lock:
            if S.mode == "hold":
                return
        time.sleep(0.02)


def measure():
    time.sleep(SETTLE_S)
    check_error()
    with S.lock:
        S.samples = []
        S.sampling = True
    time.sleep(SAMPLE_S)
    with S.lock:
        S.sampling = False
        data = list(S.samples)
    check_error()
    if not data:
        raise RuntimeError("トルクのサンプルが取れませんでした")
    qs = np.array([d[0] for d in data])
    ts = np.array([d[1] for d in data])
    return qs.mean(axis=0), ts.mean(axis=0)


def build_poses(q_base):
    poses = []
    k = 0
    for i, d2 in enumerate(J2_OFFSETS):
        row = J3_OFFSETS if i % 2 == 0 else list(reversed(J3_OFFSETS))  # 往復順で移動を短く
        for d3 in row:
            d5 = J5_OFFSETS[k % len(J5_OFFSETS)]
            k += 1
            q = q_base.copy()
            q[1] += np.radians(d2)
            q[2] += np.radians(d3)
            q[4] += np.radians(d5)
            poses.append(q)
    return poses


def fit(q_list, tau_list, n):
    Q = np.array(q_list)
    T = np.array(tau_list)
    G = np.array([gravity_model(q) for q in Q])
    scale, offset = GRAV_SCALE.copy(), GRAV_OFFSET.copy()
    print("\n関節  | モデル範囲[Nm] | SCALE  OFFSET | 残差RMS 旧→新 [Nm]")
    for j in range(n):
        g, t = G[:, j], T[:, j]
        old_rms = np.sqrt(np.mean((GRAV_SCALE[j] * g + GRAV_OFFSET[j] - t) ** 2))
        g_range = g.max() - g.min()
        if g_range < MIN_G_RANGE:
            print(f"joint{j+1} | {g_range:6.2f}         | （変化が小さいので据え置き） | {old_rms:.3f}")
            continue
        A = np.stack([g, np.ones_like(g)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, t, rcond=None)
        new_rms = np.sqrt(np.mean((a * g + b - t) ** 2))
        warn = "  ← 符号が逆？要確認" if a < 0 else ""
        warn += "  ← 倍率が極端。要確認" if not (0.5 <= a <= 1.5) and a >= 0 else ""
        print(f"joint{j+1} | {g_range:6.2f}         | {a:5.2f}  {b:+5.2f} | "
              f"{old_rms:.3f} → {new_rms:.3f}{warn}")
        scale[j], offset[j] = a, b
    return scale, offset, G, T


def parse_log(path):
    """このスクリプトの出力ログ（q(deg)=[...] tau=[...] の行）から測定値を読み出す"""
    pat = re.compile(r"q\(deg\)=\[([^\]]*)\]\s+tau=\[([^\]]*)\]")
    qs, ts = [], []
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            qs.append(np.radians(np.array(m.group(1).split(), dtype=float)))
            ts.append(np.array(m.group(2).split(), dtype=float))
    return qs, ts


def save_results(records, n):
    scale, offset, G, T = fit([r_[0] for r_ in records], [r_[1] for r_ in records], n)
    print("\n結果:")
    print(f"GRAV_SCALE  = np.array({np.round(scale, 3).tolist()})")
    print(f"GRAV_OFFSET = np.array({np.round(offset, 3).tolist()})")

    raw = Path(__file__).with_name("grav_calib_raw.csv")
    with open(raw, "w") as f:
        f.write(",".join([f"q{j+1}" for j in range(n)] + [f"tau{j+1}" for j in range(n)]
                         + [f"g{j+1}" for j in range(n)]) + "\n")
        for (q, t), g in zip(records, G):
            f.write(",".join(f"{x:.5f}" for x in [*q, *t, *g]) + "\n")
    print(f"測定データを {raw.name} に保存しました")

    if input(">>> この値を grav_params.json に保存しますか？ [y/N] ").strip().lower() == "y":
        PARAM_FILE.write_text(json.dumps(
            {"scale": np.round(scale, 4).tolist(), "offset": np.round(offset, 4).tolist()},
            indent=2))
        print(f"{PARAM_FILE.name} に保存しました（teach.py が自動で読み込みます）")


def fit_from_log(path):
    load_dynamics_model()
    qs, ts = parse_log(path)
    if not qs:
        sys.exit("ログから測定値が見つかりませんでした")
    n = len(qs[0])
    print(f"ログから {len(qs)} 件の測定値を読み込みました")
    save_results(list(zip(qs, [t[:n] for t in ts])), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-log", help="実機を使わず、以前の実行ログからフィットだけ行う")
    args = ap.parse_args()
    if args.from_log:
        fit_from_log(args.from_log)
        return

    load_dynamics_model()
    r = RebotArm()
    r.connect()
    r.arm.mode_mit()
    if r.has_gripper:
        r.gripper.mode_mit()
    r.disable_all()
    time.sleep(0.1)

    n = r.arm.num_joints
    print(f"関節: {r.arm.joint_names}")
    t_test = np.asarray(r.get_torques())
    print(f"get_torques() の要素数: {len(t_test)}（先頭 {n} 個をアームの関節として使います）")

    records = []
    try:
        r.enable_all()
        r.start_control_loop(controller, rate=r.rate)
        print("重力補償中です。周りに何もない安全な基準姿勢に、手で動かしてください。")
        input(">>> 基準姿勢に置いたら Enter")

        q_base = np.asarray(r.arm.get_positions())
        poses = build_poses(q_base)
        print(f"\n測定姿勢: {len(poses)} 個 × 2方向（約 {len(poses) * 2 * 3.5 / 60:.0f} 分）")
        print(f"  J2: {J2_OFFSETS}°, J3: {J3_OFFSETS}°, J5: {J5_OFFSETS}° の範囲で動きます")
        input(">>> 可動範囲に何もないことを確認して Enter（中止は Ctrl-C）")

        q_now = q_base.copy()
        delta = np.zeros(n)
        delta[[1, 2, 4]] = np.radians(APPROACH_DELTA_DEG)
        for i, q_target in enumerate(poses):
            for sgn in (+1, -1):
                move_to(q_target + sgn * delta, q_now)
                move_to(q_target, q_target + sgn * delta)
                q_now = q_target
                q_meas, tau_meas = measure()
                records.append((q_meas, tau_meas))
                print(f"[{i+1:2d}/{len(poses)}{'↓' if sgn > 0 else '↑'}] "
                      f"q(deg)={np.degrees(q_meas).round(1)}  tau={tau_meas.round(2)}")

        print("\n測定完了。基準姿勢へゆっくり戻ります...")
        global MOVE_VEL
        MOVE_VEL = RETURN_VEL
        try:
            move_to(q_base, q_now)
        except RuntimeError as e:
            print(f"[!] 戻り動作で停止しました（測定データは有効です）: {e}")
            S.error = None
        with S.lock:
            S.mode = "float"
    except KeyboardInterrupt:
        print("\n中止しました")
    except RuntimeError as e:
        print(f"\n[!] {e}")
    finally:
        with S.lock:
            S.mode = "float"
        if len(records) >= 6:
            try:
                save_results(records, n)
            except Exception as e:
                print(f"[!] フィットに失敗しました: {e!r}")
        elif records:
            print(f"測定値が {len(records)} 件しかないのでフィットしません")
        with S.lock:
            S.mode = "float"
        try:
            input("\n>>> アームを手で支えてから Enter で終了します（トルクが抜けます）")
        except (EOFError, KeyboardInterrupt):
            pass
        r.disconnect()
        print("切断しました")


if __name__ == "__main__":
    main()
