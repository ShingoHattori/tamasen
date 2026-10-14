#!/usr/bin/env python3
"""Whole-arm interference check of the ID 1 marker mount (scene.toml [marker_mount]).

    MUJOCO_GL=egl python check_mount.py [--n 20000] [--arm-url http://127.0.0.1:8091]

Compares the scene with and without the mount (the same model otherwise):
  1. consistency: marker1 site vs the CAD spec, visual mesh vs the collision boxes
  2. random joint configurations inside the limits, 25 % of the joints put on a limit (fingers random): contacts of the mount boxes with
     the floor, the other links (convex hulls, coarse) and the jaws, and the closest approach per link;
     "new" = the configuration was free of floor / self contacts without the mount
  3. the real arm's current joint angles (GET /state.json of the arm service, read only) and the joint
     angles stored with the hand-eye calibration samples
  4. grasp planning (planner.plan) over a grid of block placements: candidates that the mount turns
     from ok to not ok, and why
Nothing is sent to the arm. Writes output/mount_check_<time>.json.
"""
import argparse
import copy
import glob
import json
import math
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

import planner
import scene

HERE = Path(__file__).resolve().parent


def build(settings, enabled):
    s = copy.deepcopy(settings)
    s["marker_mount"]["enabled"] = enabled
    m = mujoco.MjModel.from_xml_string(scene.build_xml(s))
    return m, s, scene.Ids(m)


def name(m, obj, i):
    return mujoco.mj_id2name(m, obj, i) or str(i)


def contacts(m, d, ids, mount):
    """(mount contacts [(partner, depth)], other floor/self contacts beyond tolerance) at the current state."""
    mc, other = [], []
    for c in d.contact[:d.ncon]:
        g1, g2 = c.geom1, c.geom2
        if g1 in mount or g2 in mount:
            p = g2 if g1 in mount else g1
            partner = "floor" if p == ids.floor_geom else name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[p])
            mc.append((partner, -c.dist))
            continue
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        env = [g for g in (g1, g2) if g == ids.floor_geom or g in ids.obstacle_geoms]
        if env:
            if (b2 if g1 == env[0] else b1) in ids.arm_bodies:
                other.append((name(m, mujoco.mjtObj.mjOBJ_GEOM, env[0]), -c.dist))
        elif b1 in ids.arm_bodies and b2 in ids.arm_bodies and -c.dist > planner.SELF_PEN_TOL:
            other.append((f"{name(m, mujoco.mjtObj.mjOBJ_BODY, b1)}-{name(m, mujoco.mjtObj.mjOBJ_BODY, b2)}", -c.dist))
    return mc, other


def set_q(m, d, ids, q, finger):
    d.qpos[:] = m.qpos0
    d.qpos[ids.arm_qadr] = q
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = finger, -finger
    d.qpos[ids.block_q:ids.block_q + 7] = [10, 10, 1, 1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)      # geometry only: no constraint solve (deep random overlaps overflow its stack)
    mujoco.mj_collision(m, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--arm-url", default="http://127.0.0.1:8091")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    settings = scene.load_settings()
    spec = scene.marker_mount_spec(dict(settings, marker_mount=dict(settings["marker_mount"], enabled=True)))
    mA, sA, idsA = build(settings, True)
    mB, sB, idsB = build(settings, False)
    dA, dB = mujoco.MjData(mA), mujoco.MjData(mB)
    mount = {mujoco.mj_name2id(mA, mujoco.mjtObj.mjOBJ_GEOM, f"marker_mount_{b['name']}_hull") for b in spec["collision_boxes_mm"]}
    end = idsA.end_body
    link6 = mujoco.mj_name2id(mA, mujoco.mjtObj.mjOBJ_BODY, "link6")
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "spec_status": spec["status"],
              "marker_centre_mm": spec["marker_centre_mm"], "n_random": args.n}

    # 1. consistency ------------------------------------------------------------------------------------
    set_q(mA, dA, idsA, np.zeros(6), 0.0)
    site = mujoco.mj_name2id(mA, mujoco.mjtObj.mjOBJ_SITE, "marker1")
    Re, pe = dA.xmat[end].reshape(3, 3), dA.xpos[end]
    p_site = Re.T @ (dA.site_xpos[site] - pe) * 1e3
    R_site = Re.T @ dA.site_xmat[site].reshape(3, 3)
    T_spec = np.array(spec["T_end_link_marker"])
    vis = mujoco.mj_name2id(mA, mujoco.mjtObj.mjOBJ_GEOM, "marker_mount_visual")
    mid = mA.geom_dataid[vis]
    V = mA.mesh_vert[mA.mesh_vertadr[mid]:mA.mesh_vertadr[mid] + mA.mesh_vertnum[mid]]
    Vw = (dA.geom_xmat[vis].reshape(3, 3) @ V.T).T + dA.geom_xpos[vis]
    Ve = (Re.T @ (Vw - pe).T).T * 1e3
    lo = np.min([np.array(b["centre"]) - b["half"] for b in spec["collision_boxes_mm"]], axis=0)
    hi = np.max([np.array(b["centre"]) + b["half"] for b in spec["collision_boxes_mm"]], axis=0)
    report["consistency"] = {
        "site_pos_err_mm": float(np.linalg.norm(p_site - T_spec[:3, 3])),
        "site_rot_err_deg": float(np.degrees(math.acos(np.clip((np.trace(T_spec[:3, :3].T @ R_site) - 1) / 2, -1, 1)))),
        "mesh_bounds_mm": [Ve.min(0).round(2).tolist(), Ve.max(0).round(2).tolist()],
        "boxes_bounds_mm": [lo.round(2).tolist(), hi.round(2).tolist()]}

    # 2. random configurations ---------------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    lo_q, hi_q = idsA.arm_range[:, 0], idsA.arm_range[:, 1]
    others = [g for g in range(mA.ngeom) if mA.geom_bodyid[g] in idsA.arm_bodies and g not in mount
              and mA.geom_bodyid[g] not in (end, link6) and (mA.geom_contype[g] | mA.geom_conaffinity[g])]
    fingers = idsA.finger_left | idsA.finger_right
    others = [g for g in others if mA.geom_bodyid[g] not in fingers]   # fingers stay ~15 mm below the platform
    env = [idsA.floor_geom] + list(idsA.obstacle_geoms)
    new_hits, near = {}, {}
    n_valid = any_new = 0
    worst = []
    for k in range(args.n):
        q = rng.uniform(lo_q, hi_q)
        edge = rng.random(6) < 0.25                     # push some joints onto their limits
        q[edge] = np.where(rng.random(edge.sum()) < 0.5, lo_q[edge], hi_q[edge])
        f = rng.uniform(0, idsA.fl_range[1])
        set_q(mB, dB, idsB, q, f)
        _, otherB = contacts(mB, dB, idsB, set())
        if otherB:
            continue                  # already blocked without the mount
        n_valid += 1
        set_q(mA, dA, idsA, q, f)
        mc, _ = contacts(mA, dA, idsA, mount)
        if mc:
            any_new += 1
            for p, dep in mc:
                new_hits[p] = new_hits.get(p, 0) + 1
            worst.append((max(dep for _, dep in mc), q.tolist(), f, sorted({p for p, _ in mc})))
        # closest approach per partner (within 30 mm), valid configurations only
        for g in env:
            for mg in mount:
                dist = mujoco.mj_geomDistance(mA, dA, mg, g, 0.03, None)
                if dist < 0.03:
                    b = name(mA, mujoco.mjtObj.mjOBJ_GEOM, g)
                    if dist < near.get(b, (1.0,))[0]:
                        near[b] = (float(dist), q.tolist())
        for g in others:
            for mg in mount:
                dist = mujoco.mj_geomDistance(mA, dA, mg, g, 0.03, None)
                if dist < 0.03:
                    b = name(mA, mujoco.mjtObj.mjOBJ_BODY, mA.geom_bodyid[g])
                    if dist < near.get(b, (1.0,))[0]:
                        near[b] = (float(dist), q.tolist())
    worst.sort(key=lambda w: -w[0])
    report["random"] = {
        "valid_without_mount": n_valid, "configs_newly_blocked": any_new,
        "newly_blocked_partner_counts": new_hits,
        "closest_per_body_mm": {b: {"min_mm": v[0] * 1e3, "q": v[1]} for b, v in sorted(near.items(), key=lambda kv: kv[1][0])},
        "worst_new": [{"depth_mm": w[0] * 1e3, "q": w[1], "finger_m": w[2], "partners": w[3]} for w in worst[:8]]}

    # 3. real pose and calibration samples ---------------------------------------------------------------
    rows = [("全関節 0", [0.0] * 6), ("模擬姿勢（scene.toml mock_joints）", settings["mock_joints"]["q_rad"])]
    try:
        with urllib.request.urlopen(args.arm_url + "/state.json", timeout=1.0) as r:
            arm = json.loads(r.read())
        if arm.get("ok"):
            rows.append(("現在の実機", [j["pos"] for j in arm["joints"][:6]]))
        else:
            report["real_arm"] = f"関節角が新しくない（ok = false、最終受信 {arm['joints'][0].get('age_s', float('nan')):.0f} s 前）"
            rows.append(("実機の最終受信（古い）", [j["pos"] for j in arm["joints"][:6]]))
    except Exception as e:
        report["real_arm"] = f"読めない: {e}"
    for f in sorted(glob.glob(str(HERE.parent / "oak" / "calib" / "handeye" / "*" / "samples*.jsonl"))):
        for i, line in enumerate(open(f)):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = rec.get("q") or rec.get("joints") or rec.get("q_rad")
            if isinstance(q, list) and len(q) >= 6:
                rows.append((f"{Path(f).parent.name}/{Path(f).name}#{i}", q[:6]))
    poses = []
    for label, q in rows:
        q = np.array(q, float)
        set_q(mA, dA, idsA, q, 0.0)
        mc, otherA = contacts(mA, dA, idsA, mount)
        dmin = min((mujoco.mj_geomDistance(mA, dA, mg, g, 0.2, None),
                    name(mA, mujoco.mjtObj.mjOBJ_GEOM, g) if g in env else name(mA, mujoco.mjtObj.mjOBJ_BODY, mA.geom_bodyid[g]))
                   for g in others + env for mg in mount)
        poses.append({"label": label, "q": q.tolist(), "mount_contacts": mc, "min_clearance_mm": dmin[0] * 1e3,
                      "closest": dmin[1]})
    report["poses"] = poses

    # 4. grasp planning ----------------------------------------------------------------------------------
    q_start = np.array(settings["mock_joints"]["q_rad"], float)
    changed, total, okA_n, okB_n = [], 0, 0, 0
    for x in (0.20, 0.27, 0.34):
        for y in (-0.12, 0.0, 0.12):
            for yaw in (0.0, 0.5, 1.0, 1.57):
                T = np.eye(4)
                T[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
                T[:3, 3] = [x, y, sA["block"]["size_m"][2] / 2]
                cA = {c.name: c for c in planner.plan(mA, sA, T, q_start)}
                cB = {c.name: c for c in planner.plan(mB, sB, T, q_start)}
                for n_, cb in cB.items():
                    total += 1
                    okA_n += cA[n_].ok
                    okB_n += cb.ok
                    if cb.ok and not cA[n_].ok:
                        changed.append({"block_xy_yaw": [x, y, yaw], "candidate": n_, "reasons": cA[n_].reasons})
    report["grasp"] = {"placements": 36, "candidates": total, "ok_without_mount": okB_n, "ok_with_mount": okA_n,
                       "lost_by_mount": changed}

    out = HERE / "output" / f"mount_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    c = report["consistency"]
    print(f"consistency: site {c['site_pos_err_mm']:.3f} mm / {c['site_rot_err_deg']:.3f} deg; mesh {c['mesh_bounds_mm']} boxes {c['boxes_bounds_mm']}")
    r = report["random"]
    print(f"random {args.n}: valid without mount {r['valid_without_mount']}, newly blocked by the mount {r['configs_newly_blocked']}")
    print("  partners (new):", r["newly_blocked_partner_counts"])
    print("  closest per body:", {b: round(v["min_mm"], 1) for b, v in r["closest_per_body_mm"].items()})
    for w in r["worst_new"][:5]:
        print(f"  worst new: {w['depth_mm']:.1f} mm {w['partners']} q(deg)={np.round(np.degrees(w['q']), 0).tolist()}")
    if "real_arm" in report:
        print("  real arm:", report["real_arm"])
    for p in poses:
        print(f"  pose {p['label']}: contacts {p['mount_contacts']} clearance {p['min_clearance_mm']:.1f} mm ({p['closest']})")
    g = report["grasp"]
    print(f"grasp: ok {g['ok_without_mount']} -> {g['ok_with_mount']} of {g['candidates']}; lost {len(g['lost_by_mount'])}")
    for x in g["lost_by_mount"][:10]:
        print("  lost:", x["block_xy_yaw"], x["candidate"], x["reasons"])
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
