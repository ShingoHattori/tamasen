#!/usr/bin/env python3
"""reBotArm 重力補償（調整用）

公式 example/9_gravity_compensation.py をベースに、以下を追加:
  - 関節ごとの KP / KD
  - 関節ごとの重力補償係数（GRAV_SCALE）とオフセット（GRAV_OFFSET）
  - 関節角（deg）と tau_g の表示

配置: reBotArm_control_py/example/ に置いて
  uv run python example/9_gravity_compensation_tuned.py

注意:
  - Ctrl+C で終了するとトルクが抜けてアームが落ちる。先に手で支えるか下ろしておくこと
  - 配列のインデックスは 0 始まり（index 0 = joint1, 1 = joint2, ...）
"""
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.dynamics import (
    load_dynamics_model,
    compute_generalized_gravity,
    get_default_gravity,
)

# ── 調整パラメータ ───────────────────────────────────────────────────────────
#                        joint1 joint2 joint3 joint4 joint5 joint6
KP          = np.array([  2.0,   2.0,   2.0,   1.0,   1.0,   1.0])
KD          = np.array([  1.0,   1.0,   1.0,   0.3,   0.3,   0.3])
GRAV_SCALE  = np.array([  1.0,   0.9,   1.0,   1.0,   1.0,   1.0])  # 重力補償の倍率
#GRAV_OFFSET = np.array([  0.0,   0.0,   0.0,   0.0,   0.0,   0.0])  # 追加トルク [N·m]
GRAV_OFFSET = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])

PRINT_EVERY = 20  # 何周期ごとに表示するか

# 一部の関節だけ有効化する場合（空なら全関節）
ENABLED_JOINTS: list[str] = []
# ENABLED_JOINTS = ["joint2", "joint3"]
# ─────────────────────────────────────────────────────────────────────────────

_running = True


def _sigint_handler(signum, frame):
    global _running
    print("\n[gravity_comp] Ctrl+C を受信、停止します...")
    _running = False


signal.signal(signal.SIGINT, _sigint_handler)

_counter = 0


def gravity_compensation_controller(r: RebotArm, dt: float) -> None:
    global _counter

    q = r.arm.get_positions(request_feedback=False)
    tau_g_raw = compute_generalized_gravity(q=q)
    tau_g = tau_g_raw * GRAV_SCALE + GRAV_OFFSET

    r.arm.send_mit(
        pos=q,
        vel=np.zeros(r.arm.num_joints),
        kp=KP,
        kd=KD,
        tau=tau_g,
    )
    if r.has_gripper:
        r.gripper.send_mit(r.gripper.get_positions())

    _counter += 1
    if _counter % PRINT_EVERY == 0:
        print(
            f"[{_counter:5d}] "
            "q(deg) = " + " ".join(f"{np.degrees(x):+6.1f}" for x in q)
            + " | tau_g = " + " ".join(f"{t:+.2f}" for t in tau_g)
        )


def main() -> None:
    print("=" * 70)
    print("  reBotArm 重力補償（調整用）")
    print(f"  KP          = {KP}")
    print(f"  KD          = {KD}")
    print(f"  GRAV_SCALE  = {GRAV_SCALE}")
    print(f"  GRAV_OFFSET = {GRAV_OFFSET}")
    print("  Ctrl+C で停止（先にアームを支えること）")
    print("=" * 70)

    model = load_dynamics_model()
    g_vec = get_default_gravity()
    print(f"\n[Model] nq={model.nq}, nv={model.nv}")
    print(f"[Gravity] {g_vec}  m/s²")

    rebotarm = RebotArm()
    rebotarm.connect()

    n = rebotarm.arm.num_joints
    for name, arr in [("KP", KP), ("KD", KD),
                      ("GRAV_SCALE", GRAV_SCALE), ("GRAV_OFFSET", GRAV_OFFSET)]:
        if len(arr) != n:
            rebotarm.disconnect()
            sys.exit(f"{name} の要素数 {len(arr)} が関節数 {n} と一致しません")

    rebotarm.arm.mode_mit()
    rebotarm.gripper.mode_mit()
    rebotarm.disable_all()
    time.sleep(0.1)

    if ENABLED_JOINTS:
        for name in ENABLED_JOINTS:
            if name in rebotarm._motor_map:
                rebotarm._motor_map[name].enable()
        print(f"[Safety mode] 有効化した関節: {ENABLED_JOINTS}")
    else:
        rebotarm.enable_all()
        print("[Enabled] 全関節を有効化")

    rebotarm.start_control_loop(gravity_compensation_controller, rate=rebotarm.rate)
    print(f"[Control loop] {rebotarm.rate} Hz で開始")
    print("-" * 70)

    try:
        while _running:
            time.sleep(0.01)
    finally:
        print("\n[Stopping] 制御ループを停止します...")
        rebotarm.disconnect()
        print("[Done] 切断しました")


if __name__ == "__main__":
    main()
