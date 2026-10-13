#!/usr/bin/env python3
"""Headless checks of the simulator (no camera, no arm, no web server needed).

    MUJOCO_GL=egl python verify_sim.py [--real-arm-url http://127.0.0.1:8091]

1. FK correspondence: MuJoCo end_link vs the SDK's Pinocchio FK at the joint angles the read-only
   arm service reports (skipped if it is not running).
2. Mode hand-over: the trial starts from exactly the state the sync mode produced.
3. Grasp from the mock layout: block lifted, held by both jaws, penetration and slip during hold.
4. Expected failures: jaw gap too small; a block placed so that the grasp needs the floor.
Writes output/verify_<time>.json and prints a summary.
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import mujoco
import numpy as np

import planner
import scene
import sim_server

HERE = Path(__file__).resolve().parent


def make_sim(settings=None, arm_url="http://127.0.0.1:1", vision_url="http://127.0.0.1:1"):
    args = types.SimpleNamespace(arm_url=arm_url, vision_url=vision_url)
    s = sim_server.Sim(args)
    if settings:
        s.settings = settings
    return s


def make_plan_wide():
    sim = make_sim()
    sim.placement_source = "mock"
    sim.sync_tick()
    sim.do_plan()
    return sim


def run_trial_blocking(sim, realtime=False):
    if not realtime:
        sim_server.time_sleep_backup = sim_server.time.sleep
    msg = sim.start_trial(None)
    if sim.trial_thread is None or not sim.trial_thread.is_alive():
        return msg, None
    stats = {"max_pen_mm": 0.0, "slip_mm": None}
    rel0 = None
    while sim.trial_thread.is_alive():
        time.sleep(0.05)
        with sim.lock:
            m, d, ids = sim.m, sim.d, sim.ids
            for c in d.contact[:d.ncon]:
                b = {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]}
                if ids.block_body in b and (b & (ids.finger_left | ids.finger_right)):
                    stats["max_pen_mm"] = max(stats["max_pen_mm"], -c.dist * 1000)
            if sim.trial.get("phase", "").startswith("保持"):
                Te = np.eye(4)
                Te[:3, :3], Te[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
                rel = np.linalg.inv(Te) @ sim.block_T()
                if rel0 is None:
                    rel0 = rel
                stats["slip_mm"] = float(np.linalg.norm(rel[:3, 3] - rel0[:3, 3]) * 1000)
    return msg, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-arm-url", default="http://127.0.0.1:8091")
    args = ap.parse_args()
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mujoco": mujoco.__version__}

    # 1. FK correspondence with the real (read-only) joint angles
    arm = sim_server.fetch(args.real_arm_url + "/state.json", timeout=1)
    if arm.get("ok") and arm.get("fk"):
        sim = make_sim(arm_url=args.real_arm_url)
        sim.sync_tick()
        p_mj = sim.d.xpos[sim.ids.end_body]
        R_mj = sim.d.xmat[sim.ids.end_body].reshape(3, 3)
        p_pin, R_pin = np.array(arm["fk"]["p"]), np.array(arm["fk"]["R"])
        dq = [j["pos"] for j in arm["joints"][:6]]
        report["fk"] = {"q_real": dq, "dp_mm": float(np.linalg.norm(p_mj - p_pin) * 1000),
                        "drot_deg": math.degrees(math.acos(np.clip((np.trace(R_mj.T @ R_pin) - 1) / 2, -1, 1))),
                        "q_sim_equals_real": bool(np.allclose(sim.d.qpos[sim.ids.arm_qadr], dq))}
    else:
        report["fk"] = {"skipped": arm.get("error", "arm_state not ok")}

    # 2 + 3. mock-layout grasp, starting from the sync state
    sim = make_sim()
    sim.placement_source = "mock"
    sim.sync_tick()
    q_sync = sim.d.qpos.copy()
    sim.set_mode("trial")
    report["handover"] = {"qpos_identical": bool(np.array_equal(q_sync, sim.d.qpos)),
                          "basis": sim.trial["basis"]}
    sim.sync_tick()   # must be ignored in trial mode
    report["handover"]["live_overwrite_blocked"] = bool(np.array_equal(q_sync, sim.d.qpos))
    msg, stats = run_trial_blocking(sim)
    report["mock_grasp"] = {"message": msg, "result": sim.trial["result"], "success": sim.trial["success"],
                            "lift_mm": sim.trial.get("lift_mm"), "gap_closed_mm": sim.trial.get("gap_closed_mm"),
                            **(stats or {})}
    sim.reset()
    report["reset_mode"] = sim.mode

    # 4a. jaw gap too small (the 57 mm figure from older docs)
    s2 = scene.load_settings()
    s2["jaw"]["max_gap_m"] = 0.057
    sim = make_sim(s2)
    sim.placement_source = "mock"
    sim.sync_tick()
    sim.do_plan()
    report["narrow_jaw"] = {"ok_candidates": [c.name for c in sim.plan["candidates"] if c.ok],
                            "per_candidate": [(c.name, round(c.width_m * 1000), c.reasons[0][:40]) for c in sim.plan["candidates"]]}

    # 4a'. physics with an opening that is too small: force the top-down 78 mm candidate with a 57 mm jaw
    cand = next(c for c in make_plan_wide().plan["candidates"] if c.ok and "Y 面対" in c.name)
    sim.set_mode("trial")
    sim.trial_thread = __import__("threading").Thread(target=sim._run_trial, args=(cand,), daemon=True)
    sim.trial_thread.start()
    sim.trial_thread.join()
    report["narrow_jaw_forced_trial"] = {"candidate": cand.name, "result": sim.trial["result"],
                                         "lift_mm": sim.trial.get("lift_mm")}
    sim.reset()

    # 4c. floor interference: put the grasp point at the block centre (tips 9.5 mm below the floor)
    keep = planner.TIP_MARGIN
    planner.TIP_MARGIN = -0.02
    try:
        sim = make_sim()
        sim.placement_source = "mock"
        sim.sync_tick()
        sim.do_plan()
        report["floor_check"] = {"per_candidate": [(c.name, c.ok, c.reasons) for c in sim.plan["candidates"] if "+Z 面から" in c.name]}
    finally:
        planner.TIP_MARGIN = keep

    # 4b. block standing on its 78 x 31 face — the 31 mm pair is the only width that fits a 57 mm jaw
    s3 = scene.load_settings()
    s3["mock_block"]["position_m"] = [0.30, 0.0, 0.0335]
    sim = make_sim(s3)
    sim.placement_source = "mock"
    sim.sync_tick()
    with sim.lock:      # stand the block on end: rotate 90 deg about x so its Y axis points up
        T = sim.block_T()
        T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        T[2, 3] = 0.039
        sim.set_block(T)
        mujoco.mj_forward(sim.m, sim.d)
        sim.settings["mock_block"]["yaw_rad"] = 0.0
    sim.mode = "trial"
    sim.trial = {"phase": "", "log": [], "running": False, "done": False, "success": False, "result": "",
                 "basis": "mock", "basis_label": sim_server.BASIS_LABEL["mock"]}
    sim.plan = None
    sim.block_visible, sim.block_basis = True, "mock"
    msg, stats = run_trial_blocking(sim)
    report["standing_block"] = {"message": msg, "result": sim.trial["result"], **(stats or {}),
                                "candidates": [(c.name, c.ok, c.reasons) for c in sim.plan["candidates"]][:6]}

    out = HERE / "output" / f"verify_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
