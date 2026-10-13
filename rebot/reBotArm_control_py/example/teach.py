#!/usr/bin/env python3
"""reBotArm ダイレクトティーチング（重力補償つき 記録・再生）

reBotArm_control_py/example/ に置いて実行:
  uv run python example/teach.py record -o demo.csv   # 手で動かして記録
  uv run python example/teach.py replay -i demo.csv   # 記録を再生

record:
  重力補償でアームが浮いた状態になる。Enter で記録開始、もう一度 Enter で終了。
replay:
  現在姿勢 → 開始姿勢へゆっくり移動 → Enter で再生 → 最後の姿勢で保持。
  追従誤差が大きくなったら（ぶつかった等）、自動で重力補償のみ（フロート）に切り替えて止める。

注意:
  - 終了時（disconnect）にトルクが抜けてアームが落ちる。必ず手で支えてから Enter
  - Studio の gateway など、同じシリアルポートを使うものは止めておくこと
"""
import argparse
import bisect
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import load_dynamics_model, compute_generalized_gravity

# ── 重力補償（9_gravity_compensation_tuned.py で合わせた値） ─────────────────
#                        joint1 joint2 joint3 joint4 joint5 joint6
GRAV_SCALE  = np.array([  1.0,   0.9,   1.0,   1.0,   1.0,   1.0])
GRAV_OFFSET = np.array([  0.0,   0.4,   0.0,   0.0,   0.0,   0.0])  # [N·m]

# ── 記録時（手で動かす）のゲイン ─────────────────────────────────────────────
REC_KP      = np.array([  2.0,   2.0,   2.0,   1.0,   1.0,   1.0])
REC_KD      = np.array([  1.0,   1.0,   1.0,   0.3,   0.3,   0.3])
GRIP_REC_KP = 0.0   # グリッパーも手で動かせるように
GRIP_REC_KD = 0.1

# ── 再生時のゲイン（重力補償があるので低めでよい） ───────────────────────────
PLAY_KP     = np.array([ 15.0,  15.0,  15.0,   8.0,   8.0,   8.0])
PLAY_KD     = np.array([  1.0,   1.0,   1.0,   0.3,   0.3,   0.3])
GRIP_PLAY_KP = 5.0
GRIP_PLAY_KD = 0.3

# ── 安全関連 ─────────────────────────────────────────────────────────────────
APPROACH_VEL = 0.3              # 開始姿勢への移動速度 [rad/s]
MAX_STEP = 0.15                 # 記録データの1サンプル間の最大変化 [rad]（超えたら再生しない）
TRACK_ERR_MAX = np.radians(15)  # 再生中の追従誤差の上限。超えたらフロートに切替
SMOOTH = 5                      # 記録データの移動平均の窓幅（1で無効）
# 関節リミット [deg]。None ならチェックしない。例: [(-150, 150), (-180, 0), ...]
LIMITS_DEG = None
# ─────────────────────────────────────────────────────────────────────────────


def gravity(q):
    return compute_generalized_gravity(q=q) * GRAV_SCALE + GRAV_OFFSET


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.mode = "float"        # float / record / approach / hold / play
        self.t0 = 0.0
        self.rows = []             # 記録: [t, j1..j6, grip...]
        self.times = None
        self.arm_traj = None       # (N, 6)
        self.grip_traj = None      # (N, ng) or None
        self.q_from = None
        self.q_to = None
        self.g_from = None
        self.g_to = None
        self.approach_T = 1.0
        self.hold_arm = None
        self.hold_grip = None
        self.error = None


S = State()


def interp(t):
    """軌道を時刻 t で線形補間し、位置・速度・グリッパー位置を返す"""
    times, A = S.times, S.arm_traj
    if t <= times[0]:
        k = 0
    elif t >= times[-1]:
        k = len(times) - 2
    else:
        k = bisect.bisect_right(times, t) - 1
    span = times[k + 1] - times[k]
    r = 0.0 if span <= 0 else min(1.0, max(0.0, (t - times[k]) / span))
    q = A[k] + (A[k + 1] - A[k]) * r
    v = np.zeros_like(q) if span <= 0 else (A[k + 1] - A[k]) / span
    g = None
    if S.grip_traj is not None:
        G = S.grip_traj
        g = G[k] + (G[k + 1] - G[k]) * r
    return q, v, g


def send_grip(r, pos, kp, kd):
    if not r.has_gripper or pos is None:
        return
    n = len(pos)
    r.gripper.send_mit(pos=pos, vel=np.zeros(n), kp=np.full(n, kp),
                       kd=np.full(n, kd), tau=np.zeros(n))


def controller(r: RebotArm, dt: float) -> None:
    try:
        q = np.asarray(r.arm.get_positions(request_feedback=False))
        qg = np.asarray(r.gripper.get_positions()) if r.has_gripper else None
        tau = gravity(q)
        now = time.perf_counter()
        n = r.arm.num_joints

        with S.lock:
            mode = S.mode

            if mode in ("float", "record"):
                r.arm.send_mit(pos=q, vel=np.zeros(n), kp=REC_KP, kd=REC_KD, tau=tau)
                send_grip(r, qg, GRIP_REC_KP, GRIP_REC_KD)
                if mode == "record":
                    row = [now - S.t0] + list(q) + (list(qg) if qg is not None else [])
                    S.rows.append(row)

            elif mode == "approach":
                a = min(1.0, (now - S.t0) / S.approach_T)
                ref = S.q_from + (S.q_to - S.q_from) * a
                r.arm.send_mit(pos=ref, vel=np.zeros(n), kp=PLAY_KP, kd=PLAY_KD, tau=tau)
                if S.g_to is not None:
                    send_grip(r, S.g_from + (S.g_to - S.g_from) * a, GRIP_PLAY_KP, GRIP_PLAY_KD)
                else:
                    send_grip(r, qg, GRIP_PLAY_KP, GRIP_PLAY_KD)
                if a >= 1.0:
                    S.hold_arm, S.hold_grip = S.q_to.copy(), S.g_to
                    S.mode = "hold"

            elif mode == "hold":
                r.arm.send_mit(pos=S.hold_arm, vel=np.zeros(n), kp=PLAY_KP, kd=PLAY_KD, tau=tau)
                send_grip(r, S.hold_grip if S.hold_grip is not None else qg,
                          GRIP_PLAY_KP, GRIP_PLAY_KD)

            elif mode == "play":
                t = now - S.t0
                ref, v, g = interp(t)
                err = np.max(np.abs(ref - q))
                if err > TRACK_ERR_MAX:
                    S.error = (f"追従誤差 {np.degrees(err):.1f}° が上限を超えたので停止しました"
                               f"（t={t:.2f}s）")
                    S.mode = "float"
                    r.arm.send_mit(pos=q, vel=np.zeros(n), kp=REC_KP, kd=REC_KD, tau=tau)
                    return
                r.arm.send_mit(pos=ref, vel=v, kp=PLAY_KP, kd=PLAY_KD, tau=tau)
                send_grip(r, g if g is not None else qg, GRIP_PLAY_KP, GRIP_PLAY_KD)
                if t >= S.times[-1]:
                    S.hold_arm = S.arm_traj[-1].copy()
                    S.hold_grip = S.grip_traj[-1].copy() if S.grip_traj is not None else None
                    S.mode = "hold"
    except Exception as e:  # 制御ループ内の例外は記録してフロートへ
        S.error = f"制御ループで例外: {e!r}"
        S.mode = "float"


def connect():
    load_dynamics_model()
    r = RebotArm()
    r.connect()
    r.arm.mode_mit()
    if r.has_gripper:
        r.gripper.mode_mit()
    r.disable_all()
    time.sleep(0.1)
    return r


def start(r):
    r.enable_all()
    r.start_control_loop(controller, rate=r.rate)
    print(f"[制御ループ] {r.rate} Hz で開始")


def shutdown(r):
    with S.lock:
        if S.mode not in ("hold", "float"):
            S.mode = "float"
    try:
        input("\n>>> アームを手で支えてから Enter で終了します（トルクが抜けます）")
    except (EOFError, KeyboardInterrupt):
        pass
    r.disconnect()
    print("切断しました")


def wait_enter(msg):
    try:
        input(msg)
    finally:
        if S.error:
            print(f"\n[!] {S.error}")


# ── record ───────────────────────────────────────────────────────────────────
def cmd_record(args):
    r = connect()
    try:
        start(r)
        print("重力補償中です。手でアームを動かせます。")
        wait_enter(">>> Enter で記録開始")
        with S.lock:
            S.rows = []
            S.t0 = time.perf_counter()
            S.mode = "record"
        print("記録中... 動かし終わったら Enter")
        wait_enter("")
        with S.lock:
            S.mode = "float"
            rows = list(S.rows)

        if not rows:
            print("記録が空でした")
            return
        ng = len(rows[0]) - 7
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"j{i+1}" for i in range(6)] + [f"g{i+1}" for i in range(ng)])
            w.writerows(rows)
        dur = rows[-1][0]
        print(f"{len(rows)} サンプル / {dur:.1f}s（{len(rows)/max(dur, 1e-6):.0f} Hz）を "
              f"{args.output} に保存しました")
    except KeyboardInterrupt:
        with S.lock:
            S.mode = "float"
        print("\n中断しました（記録は保存していません）")
    finally:
        shutdown(r)


# ── replay ───────────────────────────────────────────────────────────────────
def load_traj(path):
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        data = np.array([[float(x) for x in row] for row in rd])
    if len(data) < 2:
        sys.exit("軌道が短すぎます")
    ng = len(header) - 7
    times = data[:, 0] - data[0, 0]
    arm = data[:, 1:7]
    grip = data[:, 7:] if ng > 0 else None
    return times, arm, grip


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((pad, w - 1 - pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def check_traj(arm):
    step = np.max(np.abs(np.diff(arm, axis=0)))
    if step > MAX_STEP:
        sys.exit(f"記録データに急な変化（{np.degrees(step):.1f}°/サンプル）があります。再生しません")
    if LIMITS_DEG is not None:
        lim = np.radians(np.array(LIMITS_DEG, dtype=float))
        lo, hi = lim[:, 0], lim[:, 1]
        bad = np.any((arm < lo) | (arm > hi), axis=0)
        if bad.any():
            names = [f"joint{i+1}" for i in np.where(bad)[0]]
            sys.exit(f"関節リミット外の値があります: {names}")


def cmd_replay(args):
    times, arm, grip = load_traj(args.input)
    arm = smooth(arm, SMOOTH)
    if grip is not None:
        grip = smooth(grip, SMOOTH)
    check_traj(arm)
    print(f"軌道: {len(times)} サンプル / {times[-1]:.1f}s")

    r = connect()
    try:
        if not r.has_gripper:
            grip = None
        q0 = np.asarray(r.arm.get_positions())
        g0 = np.asarray(r.gripper.get_positions()) if grip is not None else None

        dist = np.max(np.abs(arm[0] - q0))
        with S.lock:
            S.times, S.arm_traj, S.grip_traj = list(times), arm, grip
            S.q_from, S.q_to = q0, arm[0].copy()
            S.g_from, S.g_to = g0, (grip[0].copy() if grip is not None else None)
            S.approach_T = max(1.0, dist / APPROACH_VEL)
            S.t0 = time.perf_counter()
            S.mode = "approach"
        print(f"開始姿勢へ移動中（{S.approach_T:.1f}s）...")
        start(r)

        while True:
            with S.lock:
                m = S.mode
            if m == "hold":
                break
            if S.error:
                print(f"[!] {S.error}")
                return
            time.sleep(0.05)

        wait_enter(">>> 開始姿勢に着きました。Enter で再生開始")
        with S.lock:
            S.t0 = time.perf_counter()
            S.mode = "play"
        print("再生中...")

        while True:
            with S.lock:
                m = S.mode
            if m != "play":
                break
            time.sleep(0.05)

        if S.error:
            print(f"[!] {S.error}")
        else:
            print("再生終了。最後の姿勢で保持しています。")
    except KeyboardInterrupt:
        with S.lock:
            S.mode = "float"
        print("\n中止しました（重力補償のみに切り替え）")
    finally:
        shutdown(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["record", "replay"])
    p.add_argument("-o", "--output", default="demo.csv")
    p.add_argument("-i", "--input", default="demo.csv")
    args = p.parse_args()
    {"record": cmd_record, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    main()
