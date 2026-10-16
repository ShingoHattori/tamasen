"""Rod jig grasp check in MuJoCo — simulation only (no real arm, no HTTP POST anywhere).

Run with the analysis venv (mujoco 3.15.0 + trimesh/scipy/matplotlib):
    ~/tamasen-sim-jig/.venv/bin/python sim/check_rod_jig.py geom
    ... placement | plan | trial | mc | sweep | render
Outputs: sim/output/rod_jig/ (json, png).

Steps:
  geom       jig collision pieces vs rod_jig.stl; model finger hulls and CAD finger slices vs the CAD finger;
             static finger-jig clearance along the insertion (gap 43.2) and when closed (42.0)
  placement  table clearance of the arm at full insertion vs support height (and tilt), over jig positions / yaws
  plan       IK plans for sampled jig poses; ID 1 + jig marker visibility along the approach; ranking
  trial      dynamic trial of the best plan: approach, insert (straight, 1 mm IK steps), close, lift, hold, move
  mc         Monte Carlo over perception / tracking error (multiprocessing, niced)
  sweep      single-axis error sweeps (maximum tolerable error per axis)
  render     images from the OAK viewpoint and a side view
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner  # noqa: E402
import rod_jig_scene as rj  # noqa: E402
import scene  # noqa: E402

OUT = HERE / "output" / "rod_jig"
VIEW_CACHE = OUT / "view_intrinsics.json"
PLACEHOLDER_JIG_ID = 5          # the jig's ArUco ID is not assigned yet


def get_view() -> dict:
    """Adopted hand-eye (oak/calib/handeye_active.json) + intrinsics (read once from the running vision service
    with a GET of /state.json, cached)."""
    he = json.loads((HERE.parent / "oak" / "calib" / "handeye_active.json").read_text())
    if not VIEW_CACHE.exists():
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8090/state.json", timeout=2) as r:
            vs = json.loads(r.read())
        OUT.mkdir(parents=True, exist_ok=True)
        VIEW_CACHE.write_text(json.dumps({"intrinsics": vs["intrinsics"],
                                          "id1_size": vs["config"]["markers"]["1"]["size"]["value"]}, indent=1))
    c = json.loads(VIEW_CACHE.read_text())
    intr = c["intrinsics"]
    return {"X": np.array(he["T_cam_base"]), "Y": np.array(he["T_ee_m1"]), "K": np.array(intr["K"]),
            "dist": np.array(intr.get("dist_coeffs") or np.zeros(5)), "image": intr["image"], "size": float(c["id1_size"]),
            "T_base_cam": np.array(he["T_base_cam"])}


_slices_cache = {}


def make_model(jig_T, support_h=0.0, mass=0.137, friction=0.35, finger_mode="cad", finger_friction=0.5, view=None,
               extra=None):
    s = scene.load_settings()
    s["jig"] = dict(s.get("jig", {}), enabled=False)
    if "slices" not in _slices_cache:
        m0, _ = scene.load_model(s)  # jig disabled above
        _slices_cache["slices"] = rj.cad_finger_slices(m0, mujoco.MjData(m0), scene.Ids(m0))
    if "inertia" not in _slices_cache:
        _slices_cache["inertia"] = rj.jig_inertia(1.0)
    I1 = _slices_cache["inertia"]
    inertia = dict(I1, full=(np.array(I1["full"]) * mass).tolist())
    cam = None
    if view is not None:
        fy, H = view["K"][1, 1], view["image"][1]
        cam = {"T_base_cam": view["T_base_cam"], "fovy": math.degrees(2 * math.atan(H / 2 / fy))}
    opts = dict(jig_T=jig_T, support=rj.support_for(jig_T, support_h), mass_kg=mass, friction=friction,
                finger_mode=finger_mode, finger_friction=finger_friction, finger_slices=_slices_cache["slices"],
                inertia=inertia, cam=cam)
    opts.update(extra or {})
    m = mujoco.MjModel.from_xml_string(rj.build_xml(s, opts))
    d = mujoco.MjData(m)
    ids = rj.JigIds(m)
    mujoco.mj_forward(m, d)
    return m, d, ids, s


_gap0 = {}


def gap0() -> float:
    """Jaw gap at finger q = 0, from the unmodified model (scene._gap_model needs its front/mid hulls)."""
    if "v" not in _gap0:
        m0, _ = scene.load_model(dict(scene.load_settings(), jig={"enabled": False}))
        _gap0["v"] = scene._gap_model(m0).gap0
    return _gap0["v"]


def gap_of(q: float) -> float:
    return gap0() + 2 * q


def set_fingers(m, d, ids, gap_m):
    q = float(np.clip((gap_m - gap0()) / 2, *ids.fl_range))
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = q, -q
    return q


def finger_geoms(m, ids, which=("front", "mid", "rear", "cad")):
    out = []
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if m.geom_bodyid[g] in ids.finger_left | ids.finger_right and m.geom_contype[g] and any(f"_{w}" in n for w in which):
            out.append(g)
    return out


# ------------------------------------------------------------------------------------------------ geom
def _chunked(f, P, n=1500):
    return np.concatenate([np.atleast_1d(f(P[i:i + n])) for i in range(0, len(P), n)]) if len(P) else np.zeros(0)


def cmd_geom(a):
    import trimesh
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(0)
    res = {}
    stl = trimesh.load(rj.JIG_DIR / "rod_jig.stl")
    pieces = rj.jig_pieces()
    hulls = [ConvexHull(v) for _, v in pieces]

    def in_pieces(P, tol=0.0):
        ins = np.zeros(len(P), bool)
        for h in hulls:
            ins |= np.all(P @ h.equations[:, :3].T + h.equations[:, 3] <= tol, axis=1)
        return ins
    screw = lambda P: np.any([(np.hypot(P[:, 0] - x, P[:, 1]) < 3.0) & (P[:, 2] > 9) for x in (-7.3333, 7.3333)], axis=0)
    # (1) volume
    vol = sum(h.volume for h in hulls)
    res["jig_pieces"] = {"n": len(pieces), "sum_volume_cm3": round(vol / 1000, 2), "stl_volume_cm3": round(stl.volume / 1000, 2),
                         "note": "difference = screw pilot holes + countersinks (left solid in the collision model)"}
    # (2) STL surface samples: offset 0.3 / 0.05 mm along the outward normal must be outside the pieces, inward inside
    P, fi = trimesh.sample.sample_surface_even(stl, 60000, seed=1)
    N = stl.face_normals[fi]
    keep = ~screw(P)
    P, N = P[keep], N[keep]
    pocket = (np.abs(P[:, 1]) > 15.0 - 1e-6) & (np.abs(P[:, 1]) < 37.0 + 2.6) & (np.abs(P[:, 2]) < 19.0 + 1) \
        & (np.abs(P[:, 1]) < rj.Y - 0.01) & (P[:, 0] > -rj.D / 2 + 0.01) & (P[:, 0] < rj.D / 2 - 0.01)
    for tol in (0.3, 0.1, 0.05):
        out_ok = ~in_pieces(P + N * tol)
        in_ok = in_pieces(P - N * tol)
        res[f"surface_offset_{tol}mm"] = {
            "all": {"n": int(len(P)), "outside_ok_frac": round(float(out_ok.mean()), 5), "inside_ok_frac": round(float(in_ok.mean()), 5)},
            "pocket_walls": {"n": int(pocket.sum()), "outside_ok_frac": round(float(out_ok[pocket].mean()), 5),
                             "inside_ok_frac": round(float(in_ok[pocket].mean()), 5)}}
    # (3) random volume points: disagreement STL vs pieces, distance of disagreeing points to the STL surface
    lo, hi = stl.bounds[0] - 1, stl.bounds[1] + 1
    Q = rng.uniform(lo, hi, size=(120000, 3))
    Q = Q[~screw(Q)]
    a_in = _chunked(stl.contains, Q, 5000).astype(bool)
    b_in = in_pieces(Q)
    bad = a_in != b_in
    dist = np.abs(_chunked(lambda X: trimesh.proximity.signed_distance(stl, X), Q[bad])) if bad.any() else np.zeros(0)
    res["volume_points"] = {"n": int(len(Q)), "disagree": int(bad.sum()),
                            "disagree_max_dist_to_stl_mm": round(float(dist.max()), 4) if len(dist) else 0.0,
                            "disagree_p99_dist_mm": round(float(np.percentile(dist, 99)), 4) if len(dist) else 0.0}

    # (4) fingers: model hulls and CAD slices vs the CAD finger, in the finger body frames
    s = scene.load_settings()
    m0, _ = scene.load_model(s)
    d0, ids0 = mujoco.MjData(m0), scene.Ids(m0)
    sl = rj.cad_finger_slices(m0, d0, ids0)
    mujoco.mj_kinematics(m0, d0)
    Re, pe = d0.xmat[ids0.end_body].reshape(3, 3), d0.xpos[ids0.end_body]
    fingers = {}
    for side in ("left", "right"):
        body = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, f"finger_{side}_link")
        Rb, pb = d0.xmat[body].reshape(3, 3), d0.xpos[body]
        cad = sl[side]["mesh_body"]
        to_end = lambda Pb: (Re.T @ ((Rb @ Pb.T).T + pb - pe).T).T * 1000      # body m -> end_link mm
        model = []
        for g in range(m0.ngeom):
            n = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if m0.geom_bodyid[g] == body and any(k in n for k in ("_front_collision", "_mid_collision", "_rear_collision")):
                mid = m0.geom_dataid[g]
                V = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]].astype(float)
                Rg = np.zeros(9)
                mujoco.mju_quat2Mat(Rg, m0.geom_quat[g])
                V = (Rg.reshape(3, 3) @ V.T).T + m0.geom_pos[g]
                model.append((n, V))
        cadp = [(f"cad{i}", v) for i, v in enumerate(sl[side]["pieces"])]
        rows = {}
        cad_end = trimesh.Trimesh(to_end(cad.vertices), cad.faces)
        for label, parts in (("model_hulls", model), ("cad_slices", cadp)):
            fat, thin_pts = [], []
            hull_eqs = []
            for n, V in parts:
                h = trimesh.Trimesh(to_end(V)).convex_hull
                Ps, _ = trimesh.sample.sample_surface_even(h, 3000, seed=2)
                sd = _chunked(lambda X: trimesh.proximity.signed_distance(cad_end, X), Ps)      # >0 inside the CAD finger
                fat.append(np.c_[Ps, -sd])
                hull_eqs.append(ConvexHull(h.vertices).equations)
            F = np.vstack(fat)
            # CAD surface not covered by the collision pieces (pieces thinner than the finger)
            Pc, _ = trimesh.sample.sample_surface_even(cad_end, 15000, seed=3)
            inside_any = np.zeros(len(Pc), bool)
            for E in hull_eqs:
                inside_any |= np.all(Pc @ E[:, :3].T + E[:, 3] <= 0.05, axis=1)
            unc = Pc[~inside_any]
            thin = np.zeros(0)
            if len(unc):
                # distance from uncovered CAD points to the nearest collision piece surface
                allp = trimesh.util.concatenate([trimesh.Trimesh(to_end(V)).convex_hull for _, V in parts])
                thin = np.abs(_chunked(lambda X: trimesh.proximity.closest_point(allp, X)[1], unc))
            zone = (F[:, 0] > -45.5) & (F[:, 0] < 0.5)                     # the part that enters the pocket
            bins = {}
            for x0 in range(-46, 0, 2):
                sel = zone & (F[:, 0] >= x0) & (F[:, 0] < x0 + 2)
                if sel.any():
                    bins[f"{x0}..{x0 + 2}"] = round(float(F[sel, 3].max()), 2)
            rows[label] = {"pieces": len(parts),
                           "excess_outside_cad_mm_in_pocket_zone": {"max": round(float(F[zone, 3].max()), 3),
                                                                    "p99": round(float(np.percentile(F[zone, 3], 99)), 3)},
                           "excess_by_end_x_mm": bins,
                           "cad_surface_uncovered_frac": round(float((~inside_any).mean()), 4),
                           "cad_uncovered_max_mm": round(float(thin.max()), 3) if len(thin) else 0.0}
        fingers[side] = rows
    res["fingers"] = fingers

    # (5) static clearance along the insertion: fingers at 43.2 mm, depth 0..45; closed on the walls at 42.0
    view = None
    Tj = rj.T_world_jig(0.30, 0.0, math.pi, 0.05)
    for mode in ("model", "cad"):
        m, d, ids, _ = make_model(Tj, 0.05, finger_mode=mode)
        fg = finger_geoms(m, ids)
        rows = []
        for depth in list(range(0, 45, 3)) + [45]:
            T = Tj @ np.linalg.inv(rj.T_end_jig(depth))
            put_end(m, d, ids, T)
            set_fingers(m, d, ids, rj.ENTRY_GAP * 1e-3)
            mujoco.mj_kinematics(m, d)
            dmin = min(mujoco.mj_geomDistance(m, d, g, j, 0.02, None) for g in fg for j in ids.jig_geoms)
            rows.append({"depth_mm": depth, "min_dist_mm": round(dmin * 1000, 3)})
        T = Tj @ np.linalg.inv(rj.T_end_jig(45))
        put_end(m, d, ids, T)
        closed = {}
        for gap in (42.0, 42.4, 43.2, 44.0, 46.0):
            set_fingers(m, d, ids, gap * 1e-3)
            mujoco.mj_kinematics(m, d)
            fr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jig_p0_inner")
            closed[f"{gap}"] = round(min(mujoco.mj_geomDistance(m, d, g, j, 0.02, None) for g in fg for j in ids.jig_geoms) * 1000, 3)
        res[f"insertion_static_{mode}_fingers"] = {"gap_mm": rj.ENTRY_GAP, "path": rows, "full_depth_min_dist_by_gap_mm": closed}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geom_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def put_end(m, d, ids, T_world_end, q_seed=None):
    """Kinematic only: solve IK for end_link = T (via the grip site) and set the arm joints."""
    q, pe, re = ik_end(m, d, ids, T_world_end, q_seed)
    d.qpos[ids.arm_qadr] = q
    return q, pe, re


GRIP_OFF = np.array([-0.025, 0, 0])


def ik_end(m, d, ids, T_world_end, q_seed=None, iters=300):
    p_grip = T_world_end[:3, 3] + T_world_end[:3, :3] @ GRIP_OFF
    best = None
    seeds = [q_seed] if q_seed is not None else [np.array([0, -0.9, -1.4, 0, 0, 0]), np.array([0, -1.6, -1.2, 0.5, 0, 0]),
                                                 np.array([0.0, -1.2, -1.0, 0, 0.8, 0]), np.zeros(6)]
    for sd in seeds:
        q, pe, re = planner.solve_ik(m, d, ids, p_grip, T_world_end[:3, :3], np.asarray(sd, float), iters=iters)
        if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
            best = (q, pe, re)
    return best



# ------------------------------------------------------------------------------------------------ views
def jig_marker_view(m, d, ids, view, size=0.030) -> dict:
    """Jig marker (placeholder ID) as seen by the OAK: facing angle, image position / margin / side, and rays
    from its centre and 4 inner corner points to the camera against the floor / support, the arm's visual meshes
    (including the gripper and ID 1 mount) and the obstacle boxes (the jig itself excluded). Same thresholds as
    the ID 1 check in planner.marker_view. Needs mj_kinematics done."""
    import cv2
    Rm = d.site_xmat[ids.jig_marker].reshape(3, 3)
    pm = d.site_xpos[ids.jig_marker].copy()
    X = view["X"]
    Pc = X[:3, :3] @ pm + X[:3, 3]
    Rc = X[:3, :3] @ Rm
    to_cam = -Pc / np.linalg.norm(Pc)
    facing = float(np.degrees(np.arccos(np.clip(Rc[:, 2] @ to_cam, -1, 1))))
    h = size / 2
    C = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    uv, _ = cv2.projectPoints((Rc @ C.T).T + Pc, np.zeros(3), np.zeros(3), view["K"], view["dist"])
    uv = uv.reshape(-1, 2)
    side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
    W, H = view["image"]
    M = planner.VIS_MARGIN_PX
    inside = bool(uv[:, 0].min() > M and uv[:, 1].min() > M and uv[:, 0].max() < W - M and uv[:, 1].max() < H - M)
    cam = view["T_base_cam"][:3, 3]
    blocked, hits = 0, set()
    for p_ in [np.zeros(3)] + [c * 0.7 for c in C]:
        h_ = ray_block(m, d, pm + Rm @ p_ + Rm[:, 2] * 0.003, cam, ids.jig_body)
        if h_:
            blocked += 1
            hits.add(h_)
    ok = facing <= planner.VIS_MAX_DEG and inside and side >= planner.VIS_MIN_SIDE_PX and Pc[2] > 0 and blocked == 0
    return {"ok": bool(ok), "facing_deg": round(facing, 1), "side_px": round(side, 1), "uv": np.round(uv.mean(0)).tolist(),
            "blocked": blocked, "hit": sorted(hits)}


_CAMBOX = {}


def ray_block(m, d, a, cam, exclude_body):
    """Name of the first geom between a and the camera, or None. Hits on the camera_head obstacle box are
    ignored: the box contains the camera, so every ray to the camera ends in it (planner.marker_view skips only
    the last 0.18 m of the ray, which oblique rays to the table exceed: an artifact of the inflated box)."""
    if id(m) not in _CAMBOX:
        _CAMBOX[id(m)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obst_camera_head")
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, a, cam - a, planner.RAY_GROUPS, 1, exclude_body, gid)
    if 0 <= dist < 1 - 0.02 / np.linalg.norm(cam - a) and int(gid[0]) != _CAMBOX[id(m)]:
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or f"geom{int(gid[0])}"
    return None


def id1_view(m, d, ids, q, view) -> dict:
    """ID 1 as in planner.marker_view (facing <= 55 deg, 40 px inside the image, side >= 20 px, 5 rays), with the
    occlusion rays done by ray_block."""
    import cv2
    d.qpos[ids.arm_qadr] = q
    mujoco.mj_kinematics(m, d)
    F = np.eye(4)
    F[:3, :3], F[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
    P = view["X"] @ F @ view["Y"]
    to_cam = -P[:3, 3] / np.linalg.norm(P[:3, 3])
    facing = float(np.degrees(np.arccos(np.clip(P[:3, 2] @ to_cam, -1, 1))))
    h = view["size"] / 2
    C = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    uv, _ = cv2.projectPoints((P[:3, :3] @ C.T).T + P[:3, 3], np.zeros(3), np.zeros(3), view["K"], view["dist"])
    uv = uv.reshape(-1, 2)
    side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
    W, H = view["image"]
    M = planner.VIS_MARGIN_PX
    inside = bool(uv[:, 0].min() > M and uv[:, 1].min() > M and uv[:, 0].max() < W - M and uv[:, 1].max() < H - M)
    B = F @ view["Y"]
    cam = view["T_base_cam"][:3, 3]
    blocked, hits = 0, set()
    for p_ in [np.zeros(3)] + [c * 0.7 for c in C]:
        h_ = ray_block(m, d, (B @ np.r_[p_, 1.0])[:3] + B[:3, 2] * 0.003, cam, ids.end_body)
        if h_:
            blocked += 1
            hits.add(h_)
    ok = facing <= planner.VIS_MAX_DEG and inside and side >= planner.VIS_MIN_SIDE_PX and P[2, 3] > 0 and blocked == 0
    return {"ok": bool(ok), "facing_deg": round(facing, 1), "side_px": round(side, 1), "uv": np.round(uv.mean(0)).tolist(),
            "blocked": blocked, "hit": sorted(hits)}


def both_views(m, d, ids, q, view, gap_m):
    d.qpos[ids.arm_qadr] = q
    set_fingers(m, d, ids, gap_m)
    v1 = id1_view(m, d, ids, q, view)
    vj = jig_marker_view(m, d, ids, view)
    return v1, vj


# ------------------------------------------------------------------------------------------------ clearance
def arm_geoms(m, ids):
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids.arm_bodies and m.geom_bodyid[g] != base
            and (m.geom_contype[g] | m.geom_conaffinity[g])]


def clearances(m, d, ids, q, gap_m, ag=None) -> dict:
    """Smallest distance of arm / gripper / finger geoms to the table, the support, the obstacle boxes and
    (non-finger geoms only) the jig. Fingers vs the jig are left out (they go into it)."""
    ag = ag or arm_geoms(m, ids)
    d.qpos[ids.arm_qadr] = q
    set_fingers(m, d, ids, gap_m)
    mujoco.mj_kinematics(m, d)
    fing = ids.finger_left | ids.finger_right
    out = {"floor": math.inf, "support": math.inf, "obstacle": math.inf, "jig_nonfinger": math.inf}
    who = {}
    for g in ag:
        for key, others in (("floor", [ids.floor_geom]), ("support", list(ids.support_geoms)), ("obstacle", ids.obstacle_geoms),
                            ("jig_nonfinger", [] if m.geom_bodyid[g] in fing else list(ids.jig_geoms))):
            for o in others:
                dd = mujoco.mj_geomDistance(m, d, g, o, 0.1, None)
                if dd < out[key]:
                    out[key], who[key] = dd, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
    return {k: (round(v * 1000, 1) if math.isfinite(v) else None) for k, v in out.items()} | {"worst": who}


# ------------------------------------------------------------------------------------------------ placement
def jig_pose_for(px, py, delta_deg, support_h, tilt_deg):
    """Jig on a support at (px, py): the approach (end_link x, palm -> tips, horizontal part) is the radial
    direction from the arm base rotated by delta; the jig's entry face (+x) faces back along it."""
    u = np.array([px, py]) / math.hypot(px, py)
    dl = math.radians(delta_deg)
    a = np.array([u[0] * math.cos(dl) - u[1] * math.sin(dl), u[0] * math.sin(dl) + u[1] * math.cos(dl)])
    yaw = math.atan2(-a[1], -a[0])
    return rj.T_world_jig(px, py, yaw, support_h, math.radians(tilt_deg))


_SEEDS = None


def seeds(ids, n=12):
    global _SEEDS
    if _SEEDS is None:
        rng = np.random.default_rng(7)
        _SEEDS = [np.array([0, -0.9, -1.4, 0, 0, 0]), np.array([0, -1.6, -1.2, 0.5, 0, 0]), np.array([0, -1.2, -1.0, 0, 0.8, 0])] + \
            [rng.uniform(ids.arm_range[:, 0], ids.arm_range[:, 1]) for _ in range(n)]
    return _SEEDS


def ik_multi(m, d, ids, T, extra_seeds=()):
    """All distinct IK solutions found from the seeds (list of (q, pe, re)), best first."""
    sols = []
    for sd in list(extra_seeds) + seeds(ids):
        q, pe, re = ik_end(m, d, ids, T, sd)
        if pe < planner.IK_POS_TOL and re < planner.IK_ROT_TOL and not any(np.max(np.abs(q - s_[0])) < 0.05 for s_ in sols):
            sols.append((q, pe, re))
    return sols


PRE_OUT_MM = 30.0          # fingertips this far outside the entry face at the pre-insertion pose


def _placement_job(args):
    H, tilt, grid, view_d = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mem_cap(2.0)
    view = {k: np.array(v) if isinstance(v, list) else v for k, v in view_d.items()}
    rows = []
    m = d = ids = None
    for px, py, dl in grid:
        Tj = jig_pose_for(px, py, dl, H, tilt)
        m, d, ids, _ = make_model(Tj, H, view=view)
        Tg = Tj @ np.linalg.inv(rj.T_end_jig(rj.INSERT))
        Tp = Tj @ np.linalg.inv(rj.T_end_jig(-PRE_OUT_MM))
        sols = ik_multi(m, d, ids, Tg)
        r = {"px": px, "py": py, "delta": dl, "H_mm": H * 1000, "tilt": tilt, "n_ik": len(sols), "sols": []}
        for q, _, _ in sols[:4]:
            qp, pe, re = ik_end(m, d, ids, Tp, q)
            c_g = clearances(m, d, ids, q, rj.ENTRY_GAP * 1e-3)
            c_p = clearances(m, d, ids, qp, rj.ENTRY_GAP * 1e-3)
            v1, vj = both_views(m, d, ids, q, view, rj.ENTRY_GAP * 1e-3)
            v1p, vjp = both_views(m, d, ids, qp, view, rj.ENTRY_GAP * 1e-3)
            r["sols"].append({"q": q.round(4).tolist(), "pre_ik_ok": bool(pe < planner.IK_POS_TOL and re < planner.IK_ROT_TOL),
                              "clear_grasp": c_g, "clear_pre": c_p, "id1_grasp": v1, "jig_grasp": vj, "id1_pre": v1p, "jig_pre": vjp})
        rows.append(r)
    return rows


def view_json(view):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in view.items()}


def cmd_placement(a):
    from multiprocessing import Pool
    view = get_view()
    Hs = [float(x) for x in a.heights.split(",")]
    tilts = [float(x) for x in a.tilts.split(",")]
    pxs = [float(x) for x in a.px.split(",")]
    pys = [float(x) for x in a.py.split(",")]
    dls = [float(x) for x in a.deltas.split(",")]
    grid = [(px, py, dl) for px in pxs for py in pys for dl in dls]
    jobs = []
    chunk = max(1, len(grid) // max(1, a.split))
    for H in Hs:
        for t in tilts:
            if H * 1000 < (rj.D / 2 + 1) * math.sin(math.radians(t)):     # the tilted support's low edge would be below the table
                continue
            for i in range(0, len(grid), chunk):
                jobs.append((H, t, grid[i:i + chunk], view_json(view)))
    make_model(rj.T_world_jig(0.4, 0, 0))      # build caches before forking
    gap0()
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = [r for rows in pool.map(_placement_job, jobs, chunksize=1) for r in rows]
    (OUT / f"placement_{a.tag}.json").write_text(json.dumps(res, indent=0))
    print(f"{len(res)} poses in {time.time() - t0:.0f} s")
    summarize_placement(res, a.clear)


def summarize_placement(res, clear_mm):
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in res:
        k = (r["H_mm"], r["tilt"])
        agg[k][0] += 1
        ok_ik = [s_ for s_ in r["sols"] if s_["pre_ik_ok"]]
        if ok_ik:
            agg[k][1] += 1
        ok_cl = [s_ for s_ in ok_ik if min(s_["clear_grasp"]["floor"], s_["clear_pre"]["floor"]) >= clear_mm
                 and (s_["clear_grasp"]["support"] is None or s_["clear_grasp"]["support"] >= 3)
                 and (s_["clear_grasp"]["obstacle"] is None or s_["clear_grasp"]["obstacle"] >= 0)
                 and s_["clear_grasp"]["jig_nonfinger"] >= 3]
        if ok_cl:
            agg[k][2] += 1
        if any(s_["id1_grasp"]["ok"] and s_["jig_grasp"]["ok"] for s_ in ok_cl):
            agg[k][3] += 1
        if any(s_["id1_grasp"]["ok"] and s_["jig_grasp"]["ok"] and s_["id1_pre"]["ok"] and s_["jig_pre"]["ok"] for s_ in ok_cl):
            agg[k][4] += 1
    print(f"support_mm tilt_deg  poses  IK(pre+grasp)  clear>={clear_mm}mm  +both_visible@grasp  +both@pre")
    for k in sorted(agg):
        print(f"{k[0]:8.0f} {k[1]:6.0f} {agg[k][0]:6d} {agg[k][1]:10d} {agg[k][2]:14d} {agg[k][3]:16d} {agg[k][4]:12d}")



# ------------------------------------------------------------------------------------------------ plan
LIFT_M, SIDE_M = 0.05, 0.10
INSERT_SPEED = 0.010            # m/s along the insertion (1 mm IK steps)
STEP_JUMP_MAX = 0.05            # rad: largest joint change between 1 mm steps (smoothness / branch flip)


def T_offset(dx=0, dy=0, dz=0, rx=0, ry=0, rz=0):
    """4x4 from a translation (m) and small rotations about x, y, z (rad), applied as Rz Ry Rx."""
    T = np.eye(4)
    T[:3, :3] = rj._rot([0, 0, 1], rz) @ rj._rot([0, 1, 0], ry) @ rj._rot([1, 0, 0], rx)
    T[:3, 3] = [dx, dy, dz]
    return T


def make_plan(m, d, ids, T_jig_plan, q_branch, track=np.zeros(3), side_dir=None):
    """Joint waypoints for the jig believed at T_jig_plan, the hand displaced by `track` (world, m: the arm's
    systematic error). q_branch picks the IK branch (solution at full insertion). Returns dict or None."""
    def T_at(depth_mm):
        T = T_jig_plan @ np.linalg.inv(rj.T_end_jig(depth_mm))
        T = T.copy()
        T[:3, 3] += track
        return T
    q_full, pe, re = ik_end(m, d, ids, T_at(rj.INSERT), q_branch)
    if pe > planner.IK_POS_TOL or re > planner.IK_ROT_TOL:
        return None
    ins, q, worst = [], q_full, 0.0
    depths = list(np.arange(rj.INSERT, -PRE_OUT_MM - 0.5, -1.0))      # solve backwards from the full insertion
    for dep in depths:
        q2, pe, re = ik_end(m, d, ids, T_at(dep), q, iters=60)
        if pe > planner.IK_POS_TOL or re > planner.IK_ROT_TOL:
            return None
        worst = max(worst, float(np.max(np.abs(q2 - q))))
        ins.append(q2)
        q = q2
    ins = ins[::-1]                                        # depth -30 .. 45
    Tf = T_at(rj.INSERT)
    if side_dir is None:
        a = -Tf[:3, 0]
        side_dir = np.cross([0, 0, 1], a)
        side_dir /= np.linalg.norm(side_dir)
    lift, q = [], ins[-1]
    for k in range(1, 11):
        T = Tf.copy()
        T[2, 3] += LIFT_M * k / 10
        q2, pe, re = ik_end(m, d, ids, T, q, iters=80)
        if pe > planner.IK_POS_TOL or re > planner.IK_ROT_TOL:
            return None
        worst = max(worst, float(np.max(np.abs(q2 - q)) / 1.0))
        lift.append(q2)
        q = q2
    side = []
    for k in range(1, 21):
        T = Tf.copy()
        T[2, 3] += LIFT_M
        T[:3, 3] += side_dir * SIDE_M * k / 20
        q2, pe, re = ik_end(m, d, ids, T, q, iters=80)
        if pe > planner.IK_POS_TOL or re > planner.IK_ROT_TOL:
            side = None
            break
        side.append(q2)
        q = q2
    T = T_at(-PRE_OUT_MM).copy()
    T[2, 3] += 0.06
    q_high, pe, re = ik_end(m, d, ids, T, ins[0], iters=200)
    return {"ins": np.array(ins), "depths": np.arange(-PRE_OUT_MM, rj.INSERT + 0.5, 1.0), "lift": np.array(lift),
            "side": None if side is None else np.array(side), "q_high": q_high, "high_ok": bool(pe < planner.IK_POS_TOL and re < planner.IK_ROT_TOL),
            "max_step_rad": worst, "side_dir": side_dir.tolist()}


def plan_report(m, d, ids, plan, view, start_q=None):
    """Static checks along the plan + visibility fractions (every 10 mm of the insertion, closed, lifted)."""
    gap_in, gap_cl = rj.ENTRY_GAP * 1e-3, rj.GRIP_W * 1e-3
    ag = arm_geoms(m, ids)
    cl = {"floor": math.inf, "support": math.inf, "obstacle": math.inf, "jig_nonfinger": math.inf}
    seq = [(q, gap_in, "ins") for q in plan["ins"][::5]] + [(plan["ins"][-1], gap_cl, "closed")] + [(q, gap_cl, "lift") for q in plan["lift"][::3]]
    for q, g, _ in seq:
        c = clearances(m, d, ids, q, g, ag)
        for k in cl:
            if c[k] is not None:
                cl[k] = min(cl[k], c[k])
    # transit high -> pre (joint interpolation) clearance
    tr = math.inf
    for t in np.linspace(0, 1, 15):
        c = clearances(m, d, ids, plan["q_high"] + (plan["ins"][0] - plan["q_high"]) * t, gap_in, ag)
        tr = min(tr, c["floor"], c["obstacle"] if c["obstacle"] is not None else math.inf)
    vis = []
    for i in list(range(0, len(plan["ins"]), 10)) + [len(plan["ins"]) - 1]:
        v1, vj = both_views(m, d, ids, plan["ins"][i], view, gap_in)
        vis.append({"depth_mm": float(plan["depths"][i]), "id1": v1, "jig": vj})
    v1c, vjc = both_views(m, d, ids, plan["ins"][-1], view, gap_cl)
    v1l, vjl = both_views(m, d, ids, plan["lift"][-1], view, gap_cl)
    n = len(vis)
    rep = {"clear_mm": cl, "transit_min_mm": round(tr, 1),
           "vis_path": vis, "closed": {"id1": v1c, "jig": vjc}, "lifted": {"id1": v1l, "jig": vjl},
           "frac_id1": sum(v["id1"]["ok"] for v in vis) / n, "frac_jig": sum(v["jig"]["ok"] for v in vis) / n,
           "frac_both": sum(v["id1"]["ok"] and v["jig"]["ok"] for v in vis) / n,
           "id1_facing_mean": float(np.mean([v["id1"]["facing_deg"] for v in vis])),
           "max_step_rad": plan["max_step_rad"], "high_ok": plan["high_ok"], "side_ok": plan["side"] is not None}
    if start_q is not None:
        rep["joint_travel_rad"] = float(np.abs(plan["q_high"] - start_q).sum())
    return rep


def plan_ok(rep, clear_mm):
    return (rep["clear_mm"]["floor"] >= clear_mm and rep["clear_mm"]["support"] >= 2.0 and rep["clear_mm"]["obstacle"] >= 0
            and rep["clear_mm"]["jig_nonfinger"] >= 3.0 and rep["max_step_rad"] < STEP_JUMP_MAX and rep["high_ok"]
            and rep["transit_min_mm"] >= 15.0)


def plan_score(rep):
    """Lower is better: both markers visible along the approach first, then ID 1 facing, then travel."""
    return (2.0 * (1 - rep["frac_both"]) + 1.0 * (1 - rep["frac_id1"]) + 1.0 * (1 - rep["frac_jig"])
            + 0.5 * (not rep["closed"]["jig"]["ok"]) + 0.5 * (not rep["closed"]["id1"]["ok"])
            + 0.01 * rep["id1_facing_mean"] + 0.05 * rep.get("joint_travel_rad", 0))


# ------------------------------------------------------------------------------------------------ trial
def _contact_info(m, d, ids):
    fing = ids.finger_left | ids.finger_right
    f6 = np.zeros(6)
    out = {"finger_jig_N": 0.0, "finger_jig_n": 0, "pen_mm": 0.0, "arm_floor": 0, "arm_support": 0, "arm_obst": 0,
           "nonfinger_jig": 0, "jig_floor_support": 0, "left": 0, "right": 0}
    obst = set(ids.obstacle_geoms)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        jig1, jig2 = g1 in ids.jig_geoms, g2 in ids.jig_geoms
        if jig1 or jig2:
            other_g, other_b = (g2, b2) if jig1 else (g1, b1)
            if other_b in fing:
                mujoco.mj_contactForce(m, d, i, f6)
                out["finger_jig_N"] += abs(f6[0])
                out["finger_jig_n"] += 1
                out["pen_mm"] = max(out["pen_mm"], -c.dist * 1000)
                out["left" if other_b in ids.finger_left else "right"] += 1
            elif other_b in ids.arm_bodies:
                out["nonfinger_jig"] += 1
            elif other_g == ids.floor_geom or other_g in ids.support_geoms:
                out["jig_floor_support"] += 1
        else:
            arm1, arm2 = b1 in ids.arm_bodies, b2 in ids.arm_bodies
            if arm1 != arm2:
                o = g2 if arm1 else g1
                ab = b1 if arm1 else b2
                if ab == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link"):
                    continue
                if o == ids.floor_geom:
                    out["arm_floor"] += 1
                elif o in ids.support_geoms:
                    out["arm_support"] += 1
                elif o in obst:
                    out["arm_obst"] += 1
    return out


def rel_pose(d, ids):
    """Jig pose in end_link (4x4)."""
    Te, Tj = np.eye(4), np.eye(4)
    Te[:3, :3], Te[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
    Tj[:3, :3], Tj[:3, 3] = d.xmat[ids.jig_body].reshape(3, 3), d.xpos[ids.jig_body]
    return np.linalg.inv(Te) @ Tj


def pose_diff(A, B):
    dp = (A[:3, 3] - B[:3, 3]) * 1000
    ang = math.degrees(math.acos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))
    return dp, ang


def run_trial(m, d, ids, plan, record=False, snap=None, close_target_gap=0.0):
    """Dynamic trial. The arm starts at q_high (already there), the jig rests at its true pose (settled first).
    snap(name) is called at pre / mid-insertion / closed / lifted (for rendering)."""
    dt = m.opt.timestep
    log, trace = [], []
    mujoco.mj_resetData(m, d)
    d.qpos[ids.arm_qadr] = plan["q_high"]
    q_in = set_fingers(m, d, ids, rj.ENTRY_GAP * 1e-3)
    d.ctrl[ids.act_arm] = plan["q_high"]
    d.ctrl[ids.act_grip] = q_in
    mujoco.mj_forward(m, d)
    for _ in range(int(0.5 / dt)):                         # the jig settles on its support
        mujoco.mj_step(m, d)
    Tj0 = np.eye(4)
    Tj0[:3, :3], Tj0[:3, 3] = d.xmat[ids.jig_body].reshape(3, 3), d.xpos[ids.jig_body].copy()
    st = {"max_force_N": 0.0, "max_force_depth": None, "max_pen_mm": 0.0, "arm_floor": 0, "arm_support": 0, "arm_obst": 0,
          "nonfinger_jig": 0, "jig_push_mm": 0.0, "jig_tilt_deg": 0.0, "track_mm": 0.0, "force_by_depth": {}}
    g_fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def drive(q_from, q_to, T, phase, depth=None):
        n = max(1, int(round(T / dt)))
        for k in range(n):
            s_ = (k + 1) / n
            d.ctrl[ids.act_arm] = q_from + (q_to - q_from) * s_
            mujoco.mj_step(m, d)
            if k % 10 == 0:
                c = _contact_info(m, d, ids)
                for key in ("arm_floor", "arm_support", "arm_obst", "nonfinger_jig"):
                    st[key] += c[key] > 0
                if phase in ("pre", "insert"):
                    if c["finger_jig_N"] > st["max_force_N"]:
                        st["max_force_N"], st["max_force_depth"] = c["finger_jig_N"], depth
                    st["max_pen_mm"] = max(st["max_pen_mm"], c["pen_mm"])
                    if depth is not None:
                        key = int(depth // 5) * 5
                        st["force_by_depth"][key] = max(st["force_by_depth"].get(key, 0.0), round(c["finger_jig_N"], 2))
                    dp, ang = pose_diff(np.c_[np.r_[d.xmat[ids.jig_body].reshape(3, 3), [[0, 0, 0]]], np.r_[d.xpos[ids.jig_body], 1]], Tj0)
                    st["jig_push_mm"] = max(st["jig_push_mm"], float(np.linalg.norm(dp)))
                    st["jig_tilt_deg"] = max(st["jig_tilt_deg"], ang)
                if record:
                    trace.append({"t": round(d.time, 3), "phase": phase, "depth": depth, "F": round(c["finger_jig_N"], 2),
                                  "jig": d.xpos[ids.jig_body].round(5).tolist()})

    q = plan["ins"][0]
    drive(plan["q_high"], q, 1.5, "pre")
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    if snap:
        snap("pre")
    step_T = 0.001 / INSERT_SPEED
    for i in range(1, len(plan["ins"])):
        drive(plan["ins"][i - 1], plan["ins"][i], step_T, "insert", float(plan["depths"][i]))
        if snap and abs(plan["depths"][i] - 22) < 0.5:
            snap("mid")
    drive(plan["ins"][-1], plan["ins"][-1], 0.3, "insert", rj.INSERT)
    # commanded vs actual end_link at the full insertion (arm compliance / jamming)
    T_ins = rel_pose(d, ids)
    depth_short = (T_ins[0, 3] - rj.T_end_jig(rj.INSERT)[0, 3]) * 1000        # >0: jig further out than planned
    # close
    f0 = float(d.ctrl[ids.act_grip])
    qc = float(np.clip((close_target_gap - gap0()) / 2, *ids.fl_range))
    n = int(1.0 / dt)
    for k in range(n):
        d.ctrl[ids.act_grip] = f0 + (qc - f0) * (k + 1) / n
        mujoco.mj_step(m, d)
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(m, d)
    gap_closed = (gap0() + float(d.qpos[ids.fl_q]) - float(d.qpos[ids.fr_q])) * 1000     # both fingers (soft coupling)
    finger_q = [round(float(d.qpos[ids.fl_q]) * 1000, 2), round(float(d.qpos[ids.fr_q]) * 1000, 2)]
    pen_closed = _contact_info(m, d, ids)["pen_mm"]
    T_closed = rel_pose(d, ids)
    if snap:
        snap("closed")
    z_before = float(d.xpos[ids.jig_body][2])
    lq = np.vstack([plan["ins"][-1], plan["lift"]])
    for i in range(1, len(lq)):
        drive(lq[i - 1], lq[i], 0.2, "lift")
    for _ in range(int(2.0 / dt)):
        mujoco.mj_step(m, d)
    if snap:
        snap("lifted")
    c_hold = _contact_info(m, d, ids)
    T_hold = rel_pose(d, ids)
    lift_mm = (float(d.xpos[ids.jig_body][2]) - z_before) * 1000
    T_side = None
    if plan["side"] is not None:
        sq = np.vstack([plan["lift"][-1], plan["side"]])
        for i in range(1, len(sq)):
            drive(sq[i - 1], sq[i], 0.125, "side")
        for _ in range(int(1.0 / dt)):
            mujoco.mj_step(m, d)
        T_side = rel_pose(d, ids)
        c_side = _contact_info(m, d, ids)
    slip_hold, rot_hold = pose_diff(T_hold, T_closed)
    res = {"depth_short_mm": round(float(depth_short), 2), "gap_closed_mm": round(gap_closed, 2), "finger_q_mm": finger_q,
           "pen_closed_mm": round(pen_closed, 2), "lift_mm": round(lift_mm, 1),
           "slip_hold_mm": round(float(np.linalg.norm(slip_hold)), 2), "rot_hold_deg": round(rot_hold, 2),
           "held_contacts": [c_hold["left"], c_hold["right"]], "jig_on_support_after_lift": c_hold["jig_floor_support"] > 0,
           "max_force_N": round(st["max_force_N"], 2), "max_force_depth_mm": st["max_force_depth"], "max_pen_mm": round(st["max_pen_mm"], 2),
           "force_by_depth": {str(k): v for k, v in sorted(st["force_by_depth"].items())},
           "jig_push_mm": round(st["jig_push_mm"], 2), "jig_tilt_deg": round(st["jig_tilt_deg"], 2),
           "arm_floor_steps": st["arm_floor"], "arm_support_steps": st["arm_support"], "arm_obst_steps": st["arm_obst"],
           "nonfinger_jig_steps": st["nonfinger_jig"]}
    if T_side is not None:
        ds, rs = pose_diff(T_side, T_closed)
        res.update(slip_side_mm=round(float(np.linalg.norm(ds)), 2), rot_side_deg=round(rs, 2), held_after_side=[c_side["left"], c_side["right"]])
    # classification
    fails = []
    if depth_short > 3.0:
        fails.append("not_fully_inserted" if st["max_force_depth"] is None or st["max_force_depth"] > rj.RECT else "stopped_at_entry")
    if st["max_force_N"] > 30.0:
        fails.append("jam_high_force" if (st["max_force_depth"] or 0) > rj.RECT - 1 else "tip_hits_entry")
    if st["jig_push_mm"] > 6.0 or st["jig_tilt_deg"] > 8.0:
        fails.append("jig_pushed_or_tipped")
    if lift_mm < 0.8 * LIFT_M * 1000 or c_hold["jig_floor_support"]:
        fails.append("not_lifted")
    if res["slip_hold_mm"] > 3.0 or res["rot_hold_deg"] > 3.0 or (T_side is not None and (res["slip_side_mm"] > 3.0 or res["rot_side_deg"] > 3.0)):
        fails.append("slip")
    if st["arm_floor"] or st["arm_obst"] or st["arm_support"]:
        fails.append("arm_collision")
    if st["nonfinger_jig"]:
        fails.append("palm_or_arm_touches_jig")
    res["fails"] = fails
    res["success"] = not fails
    if record:
        res["trace"] = trace
    return res



# ------------------------------------------------------------------------------------------------ commands
START_Q = np.array([0.0, -0.9, -1.4, 0.0, 0.0, 0.0])      # scene.toml mock_joints (a neutral start)


def placement_rows(tag):
    return json.loads((OUT / f"placement_{tag}.json").read_text())


def _plan_job(args):
    r, s_, clear, view_d = args
    view = {k: np.array(v) if isinstance(v, list) else v for k, v in view_d.items()}
    Tj = jig_pose_for(r["px"], r["py"], r["delta"], r["H_mm"] / 1000, r["tilt"])
    m, d, ids, _ = make_model(Tj, r["H_mm"] / 1000, view=view)
    pl = make_plan(m, d, ids, Tj, np.array(s_["q"]))
    if pl is None:
        return None
    rep = plan_report(m, d, ids, pl, view, START_Q)
    return {"placement": {k: r[k] for k in ("px", "py", "delta", "H_mm", "tilt")}, "q_branch": s_["q"], "ok": plan_ok(rep, clear),
            "score": plan_score(rep), **{k: rep[k] for k in ("clear_mm", "transit_min_mm", "frac_id1", "frac_jig", "frac_both",
                                                              "id1_facing_mean", "max_step_rad", "side_ok", "joint_travel_rad")},
            "closed_vis": [rep["closed"]["id1"]["ok"], rep["closed"]["jig"]["ok"]],
            "lifted_vis": [rep["lifted"]["id1"]["ok"], rep["lifted"]["jig"]["ok"]],
            "vis_path": [[v["depth_mm"], v["id1"]["ok"], v["jig"]["ok"], v["id1"]["facing_deg"], v["jig"]["facing_deg"],
                          v["id1"]["hit"], v["jig"]["hit"]] for v in rep["vis_path"]]}


def cmd_plan(a):
    """Full plans (every IK branch) for the placements that passed the survey; rank by visibility."""
    from multiprocessing import Pool
    view = get_view()
    rows = placement_rows(a.src or a.tag)
    cands = []
    for r in rows:
        if math.hypot(r["px"], r["py"]) < 0.1:
            continue
        for s_ in r["sols"]:
            cg, cp = s_["clear_grasp"], s_["clear_pre"]
            if not s_["pre_ik_ok"] or min(cg["floor"], cp["floor"]) < a.clear or (cg["support"] is not None and cg["support"] < 2) \
                    or cg["jig_nonfinger"] < 3:
                continue
            if not (s_["id1_grasp"]["ok"] or s_["jig_grasp"]["ok"]):
                continue
            cands.append((r, s_, a.clear, view_json(view)))
    print(f"{len(cands)} placement/branch pairs pass the survey (clear >= {a.clear} mm, a marker visible at the grasp)", flush=True)
    make_model(rj.T_world_jig(0.4, 0, 0))
    gap0()
    with Pool(a.workers) as pool:
        out = [o for o in pool.map(_plan_job, cands, chunksize=4) if o is not None]
    out.sort(key=lambda o: (not o["ok"], o["score"]))
    (OUT / f"plans_{a.tag}.json").write_text(json.dumps(out, indent=0))
    print(f"{sum(o['ok'] for o in out)} of {len(out)} full plans pass the path checks")
    print_plans(out)


def print_plans(out, n=25):
    print("rank ok  px    py   delta H_mm tilt | both  id1  jig | closed lifted | floor supp obst jigNF | step  ID1facing")
    for i, o in enumerate(out[:n]):
        p_ = o["placement"]
        c = o["clear_mm"]
        print(f"{i:3d} {int(o['ok'])}  {p_['px']:+.2f} {p_['py']:+.2f} {p_['delta']:+5.0f} {p_['H_mm']:4.0f} {p_['tilt']:4.0f} | "
              f"{o['frac_both']:.2f} {o['frac_id1']:.2f} {o['frac_jig']:.2f} | {o['closed_vis']} {o['lifted_vis']} | "
              f"{c['floor']:5.1f} {c['support']:5.1f} {c['obstacle']:6.1f} {c['jig_nonfinger']:5.1f} | {o['max_step_rad']:.3f} {o['id1_facing_mean']:5.1f}")


def load_choice(a):
    plans = json.loads((OUT / f"plans_{a.tag}.json").read_text())
    ch = plans[a.rank]
    p_ = ch["placement"]
    Tj = jig_pose_for(p_["px"], p_["py"], p_["delta"], p_["H_mm"] / 1000, p_["tilt"])
    return ch, Tj, p_["H_mm"] / 1000


def physics_opts(a):
    return dict(mass=a.mass, friction=a.friction, finger_friction=a.finger_friction)


class Snapper:
    """Renders the OAK view and a side view (EGL) at named moments of a trial."""

    def __init__(self, m, d, ids, prefix, approach_world):
        self.m, self.d, self.ids, self.prefix = m, d, ids, prefix
        self.r = mujoco.Renderer(m, 720, 960)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in (0, 1, 2, 5):
            self.opt.geomgroup[g] = 1
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        a = np.asarray(approach_world, float)
        self.side_az = math.degrees(math.atan2(a[1], a[0])) + 90
        self.files = []

    def __call__(self, name):
        import cv2
        m, d, ids = self.m, self.d, self.ids
        for view in ("oak", "side", "close"):
            if view == "oak":
                cam = ids.cam
            else:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = d.xpos[ids.jig_body]
                cam.distance = 0.55 if view == "side" else 0.22
                cam.azimuth = self.side_az if view == "side" else self.side_az - 35
                cam.elevation = -8 if view == "side" else -30
            self.r.update_scene(d, cam, self.opt)
            img = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
            label = f"{name} | {view} | simulation only | jig marker ID {PLACEHOLDER_JIG_ID} = placeholder"
            cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            f = OUT / "img" / f"{self.prefix}_{name}_{view}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(f), img)
            self.files.append(str(f))


def cmd_trial(a):
    view = get_view()
    ch, Tj, H = load_choice(a)
    m, d, ids, _ = make_model(Tj, H, view=view, finger_mode=a.fingers, **physics_opts(a))
    pl = make_plan(m, d, ids, Tj, np.array(ch["q_branch"]))
    snap = Snapper(m, d, ids, f"trial_{a.tag}_r{a.rank}", -Tj[:3, 0]) if a.render else None
    t0 = time.time()
    res = run_trial(m, d, ids, pl, record=True, snap=snap)
    res["wall_s"] = round(time.time() - t0, 1)
    res["placement"] = ch["placement"]
    res["physics"] = physics_opts(a) | {"fingers": a.fingers}
    (OUT / f"trial_{a.tag}_r{a.rank}_{a.fingers}.json").write_text(json.dumps(res, indent=0))
    print(json.dumps({k: v for k, v in res.items() if k != "trace"}, indent=1))
    if snap:
        print("\n".join(snap.files))


# ---- Monte Carlo / sweeps (worker processes share one model of the true scene)
_W = {}


def _winit(spec):
    os.nice(5)
    mem_cap(2.0)
    view = {k: np.array(v) if isinstance(v, list) else v for k, v in spec["view"].items()}
    Tj = np.array(spec["Tj"])
    m, d, ids, _ = make_model(Tj, spec["H"], view=view, finger_mode=spec["fingers"], **spec["phys"])
    _W.update(m=m, d=d, ids=ids, Tj=Tj, q_branch=np.array(spec["q_branch"]))


def _wtrial(job):
    m, d, ids, Tj = _W["m"], _W["d"], _W["ids"], _W["Tj"]
    e = job["err"]
    # perceived jig = world shift (ex, ey, ez) o true o small rotation in the jig frame (roll, pitch, yaw)
    Tp = T_offset(e["ex"], e["ey"], e.get("ez", 0.0)) @ Tj @ T_offset(rx=e["roll"], ry=e["pitch"], rz=e["yaw"])
    track = np.array([e["tx"], e["ty"], e["tz"]])
    if job.get("jig_frame"):     # sweep: offsets given in the end_link frame at the grasp (insertion x, closing y, width z)
        Tg = Tj @ np.linalg.inv(rj.T_end_jig(rj.INSERT))
        track = Tg[:3, :3] @ np.array([e["gx"], e["gy"], e["gz"]])
    pl = make_plan(m, d, ids, Tp, _W["q_branch"], track)
    if pl is None:
        return {"job": job, "res": {"success": False, "fails": ["plan_failed"]}}
    # effective error at the grasp in end_link axes (planned target vs ideal): x insertion, y closing, z width (up)
    ideal = Tj @ np.linalg.inv(rj.T_end_jig(rj.INSERT))
    tgt = Tp @ np.linalg.inv(rj.T_end_jig(rj.INSERT))
    tgt[:3, 3] += track
    E = np.linalg.inv(ideal) @ tgt
    ang = math.degrees(math.acos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
    yaw_e = math.degrees(math.atan2(E[1, 0], E[0, 0]))
    res = run_trial(m, d, ids, pl)
    return {"job": job, "eff": {"x": E[0, 3] * 1000, "y": E[1, 3] * 1000, "z": E[2, 3] * 1000, "ang": ang, "yaw_about_z": yaw_e}, "res": res}


def _pool_run(a, jobs, tag):
    from multiprocessing import Pool
    view = get_view()
    ch, Tj, H = load_choice(a)
    spec = {"view": view_json(view), "Tj": Tj.tolist(), "H": H, "fingers": a.fingers, "phys": physics_opts(a), "q_branch": ch["q_branch"]}
    make_model(Tj, H)
    gap0()
    t0 = time.time()
    out = []
    with Pool(a.workers, initializer=_winit, initargs=(spec,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_wtrial, jobs, chunksize=1)):
            out.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f} s", flush=True)
    meta = {"variant": a.variant, "placement": ch["placement"], "physics": spec["phys"], "fingers": a.fingers, "n": len(out), "wall_s": round(time.time() - t0)}
    (OUT / f"{tag}.json").write_text(json.dumps({"meta": meta, "trials": out}, indent=0, default=float))
    return out, meta


# nominal error model (the request): perception σ 3 mm (horizontal, per axis), yaw σ 2°, tilt σ 1°; tracking σ 1.5 mm per axis
NOMINAL = {"pos": 0.003, "yaw": math.radians(2.0), "tilt": math.radians(1.0), "track": 0.0015}
VISCORR = {"pos": 0.0006, "yaw": math.radians(0.5), "tilt": math.radians(0.3), "track": 0.0003}   # ~1 mm residual in all


def sample_err(rng, sig, scale=1.0, cap_pos=None):
    e = {"ex": rng.normal(0, sig["pos"] * scale), "ey": rng.normal(0, sig["pos"] * scale), "ez": 0.0,
         "yaw": rng.normal(0, sig["yaw"] * scale), "roll": rng.normal(0, sig["tilt"] * scale), "pitch": rng.normal(0, sig["tilt"] * scale),
         "tx": rng.normal(0, sig["track"] * scale), "ty": rng.normal(0, sig["track"] * scale), "tz": rng.normal(0, sig["track"] * scale)}
    if cap_pos:
        n = math.hypot(e["ex"], e["ey"])
        if n > cap_pos:
            e["ex"], e["ey"] = e["ex"] * cap_pos / n, e["ey"] * cap_pos / n
    return e


def cmd_mc(a):
    rng = np.random.default_rng(a.seed)
    jobs = []
    for sc in [float(x) for x in a.scales.split(",")]:
        for _ in range(a.n):
            jobs.append({"cond": f"nominal_x{sc:g}", "scale": sc, "err": sample_err(rng, NOMINAL, sc)})
    for _ in range(a.n_vc):
        jobs.append({"cond": "visual_correction", "scale": 1.0, "err": sample_err(rng, VISCORR)})
    print(f"{len(jobs)} trials, {a.workers} workers")
    out, meta = _pool_run(a, jobs, f"mc_{a.tag}{a.variant_tag}_{a.fingers}")
    summarize_mc(out)


def summarize_mc(out):
    from collections import Counter, defaultdict
    by = defaultdict(list)
    for r in out:
        by[r["job"]["cond"]].append(r)
    print("condition              n  success  failure modes")
    for k in sorted(by, key=lambda k: (k != "visual_correction", k)):
        rs = by[k]
        ok = sum(r["res"]["success"] for r in rs)
        fc = Counter(f for r in rs for f in r["res"]["fails"])
        print(f"{k:20s} {len(rs):4d}  {ok / len(rs) * 100:5.1f}%  {dict(fc)}")


def cmd_sweep(a):
    jobs = []
    want = set(a.axes.split(",")) if a.axes else None
    z = {"ex": 0, "ey": 0, "yaw": 0, "roll": 0, "pitch": 0, "tx": 0, "ty": 0, "tz": 0, "gx": 0, "gy": 0, "gz": 0}
    for v in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        jobs.append({"cond": "y_closing_mm", "value": v, "jig_frame": True, "err": z | {"gy": v / 1000}})
    for v in (-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4):
        jobs.append({"cond": "z_vertical_mm(+up)", "value": v, "jig_frame": True, "err": z | {"gz": v / 1000}})
    for v in (-6, -4, -3, -2, 2, 3, 4, 5, 6, 8):
        jobs.append({"cond": "x_depth_mm(+deeper)", "value": v, "jig_frame": True, "err": z | {"gx": v / 1000}})
    for v in (1, 2, 3, 4, 6, 8):
        jobs.append({"cond": "yaw_deg", "value": v, "jig_frame": True, "err": z | {"yaw": math.radians(v)}})
    for v in (-3, -2, -1, 1, 2, 3):
        jobs.append({"cond": "pitch_deg", "value": v, "jig_frame": True, "err": z | {"pitch": math.radians(v)}})
    for v in (1, 2, 3, 5):
        jobs.append({"cond": "roll_deg", "value": v, "jig_frame": True, "err": z | {"roll": math.radians(v)}})
    if want:
        jobs = [j for j in jobs if j["cond"].split("_")[0] in want]
    print(f"{len(jobs)} sweep trials")
    out, meta = _pool_run(a, jobs, f"sweep_{a.tag}{a.variant_tag}_{a.fingers}")
    out.sort(key=lambda r: (r["job"]["cond"], r["job"]["value"]))
    print("axis                  value  ok  fails                      Fmax N @depth  short  push  tilt  slip")
    for r in out:
        x = r["res"]
        print(f"{r['job']['cond']:22s} {r['job']['value']:5g}  {int(x['success'])}  {','.join(x['fails']) or '-':26s} "
              f"{x.get('max_force_N', 0):6.1f} @{x.get('max_force_depth_mm')}  {x.get('depth_short_mm', 0):5.1f} {x.get('jig_push_mm', 0):5.1f} "
              f"{x.get('jig_tilt_deg', 0):5.1f} {x.get('slip_hold_mm', 0):5.1f}")



def cmd_report(a):
    """Plots of the Monte Carlo (success vs error scale and vs the effective error at the grasp) and the sweeps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    mc = json.loads((OUT / f"mc_{a.tag}{a.variant_tag}_{a.fingers}.json").read_text())["trials"]
    img = OUT / "img"
    img.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    by = defaultdict(list)
    for r in mc:
        by[r["job"]["cond"]].append(r)
    conds = sorted([k for k in by if k.startswith("nominal")], key=lambda k: by[k][0]["job"]["scale"])
    xs = [by[k][0]["job"]["scale"] for k in conds]
    ys = [100 * np.mean([r["res"]["success"] for r in by[k]]) for k in conds]
    ax[0].plot(xs, ys, "o-", color="#2a6fdb", label="no correction (scaled nominal errors)")
    if "visual_correction" in by:
        vc = 100 * np.mean([r["res"]["success"] for r in by["visual_correction"]])
        ax[0].axhline(vc, color="#1a9e5a", ls="--", label=f"visual correction (~1 mm residual): {vc:.0f}%")
    ax[0].set_xlabel("error scale (1.0 = pos σ 3 mm/axis, yaw σ 2°, tilt σ 1°, track σ 1.5 mm/axis)")
    ax[0].set_ylabel("success %")
    ax[0].set_ylim(-3, 103)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Monte Carlo: success vs error scale")
    ok = np.array([r["res"]["success"] for r in mc if "eff" in r])
    E = [r["eff"] for r in mc if "eff" in r]
    ey = np.array([abs(e["y"]) + 45 * math.tan(math.radians(abs(e["yaw_about_z"]))) for e in E])
    ez = np.array([e["z"] for e in E])
    for axi, (vals, lab, bins) in zip(ax[1:], ((ey, "lateral error at the fingertip, mm: |y| + 45 tan|yaw| (closing axis)", np.arange(0, 9, 1.0)),
                                               (np.abs(ez), "|vertical error| at the grasp, mm (end_link z)", np.arange(0, 7, 0.75)))):
        cen, rate, cnt = [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = (vals >= b0) & (vals < b1)
            if sel.sum() >= 3:
                cen.append((b0 + b1) / 2)
                rate.append(100 * ok[sel].mean())
                cnt.append(int(sel.sum()))
        axi.bar(cen, rate, width=(bins[1] - bins[0]) * 0.85, color="#2a6fdb", alpha=0.8)
        for c_, r_, n_ in zip(cen, rate, cnt):
            axi.text(c_, r_ + 2, f"n={n_}", ha="center", fontsize=7)
        axi.set_xlabel(lab)
        axi.set_ylabel("success %")
        axi.set_ylim(0, 110)
        axi.grid(alpha=0.3, axis="y")
    ax[1].set_title("success vs effective error (all MC trials)")
    ax[2].set_title("vertical (confounded by lateral; sweep: ±3 mm OK)", fontsize=9)
    fig.suptitle(f"rod jig grasp, MuJoCo, {len(mc)} trials (simulation only)", fontsize=11)
    fig.tight_layout()
    f1 = img / f"mc_{a.tag}{a.variant_tag}_{a.fingers}.png"
    fig.savefig(f1, dpi=110)
    print(f1)
    sw_f = OUT / f"sweep_{a.tag}{a.variant_tag}_{a.fingers}.json"
    if sw_f.exists():
        sw = json.loads(sw_f.read_text())["trials"]
        axes = sorted({r["job"]["cond"] for r in sw})
        fig, ax = plt.subplots(1, len(axes), figsize=(3.2 * len(axes), 3.4), squeeze=False)
        for axi, c in zip(ax[0], axes):
            rs = sorted([r for r in sw if r["job"]["cond"] == c], key=lambda r: r["job"]["value"])
            v = [r["job"]["value"] for r in rs]
            F = [r["res"].get("max_force_N", 0) for r in rs]
            col = ["#1a9e5a" if r["res"]["success"] else "#d43d3d" for r in rs]
            axi.scatter(v, F, c=col, s=30)
            axi.set_title(c, fontsize=9)
            axi.set_xlabel("offset")
            axi.set_ylabel("max finger-jig force N (insertion)")
            axi.grid(alpha=0.3)
        fig.suptitle("single-axis error sweeps (green = success, red = failure)", fontsize=10)
        fig.tight_layout()
        f2 = img / f"sweep_{a.tag}{a.variant_tag}_{a.fingers}.png"
        fig.savefig(f2, dpi=110)
        print(f2)



# ------------------------------------------------------------------------------------------------ live pose
def mem_cap(gb=2.5):
    """Hard address-space cap for this process (robot1 runs live services; an OOM froze vision once)."""
    import resource
    lim = int(gb * 1024 ** 3)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))


def T_marker_jig_from_toml():
    import tomllib
    mk = tomllib.loads((HERE.parent / "oak" / "markers.toml").read_text())["marker"]["13"]
    T = np.eye(4)
    T[:3, :3] = np.array(mk["jig_rotation_in_marker"], float)
    T[:3, 3] = np.array(mk["jig_origin_in_marker_mm"], float) * 1e-3
    return T


def live_jig_pose(n=10, alt=False):
    """T_base_jig from the running vision service (GET /state.json, read only), averaged over n reads."""
    import urllib.request
    he = json.loads((HERE.parent / "oak" / "calib" / "handeye_active.json").read_text())
    Tbc = np.array(he["T_base_cam"])
    Tmj = T_marker_jig_from_toml()
    Ts, st = [], []
    for _ in range(n):
        with urllib.request.urlopen("http://127.0.0.1:8090/state.json", timeout=2) as r:
            vs = json.loads(r.read())
        mk = next((x for x in vs["markers"] if x["id"] == 13), None)
        if mk and mk.get("T_cam_marker"):
            Ts.append(Tbc @ np.array(mk["T_cam_marker_alt" if alt else "T_cam_marker"]) @ Tmj)
            st.append(mk["status"])
        time.sleep(0.15)
    if not Ts:
        raise SystemExit("ID 13 not seen")
    T = Ts[len(Ts) // 2].copy()
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], axis=0)
    spread = float(np.max([np.linalg.norm(t[:3, 3] - T[:3, 3]) for t in Ts]) * 1000)
    return T, st, spread


def cmd_live(a):
    view = get_view()
    T, st, spread = live_jig_pose()
    R = T[:3, :3]
    tilt = math.degrees(math.acos(np.clip(-R[2, 2], -1, 1)))         # jig z should point down
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    # height of the resting (+z) face centre and of its lowest corner above the arm base plane (table)
    face_c = T[:3, :3] @ np.array([0, 0, rj.ZH * 1e-3]) + T[:3, 3]
    corners = [T[:3, :3] @ np.array([x, y, rj.ZH]) * 1e-3 + T[:3, 3] for x in (-22, 22) for y in (-rj.Y, rj.Y)]
    x_in = R[:, 0]
    rad = np.array([T[0, 3], T[1, 3]]) / math.hypot(T[0, 3], T[1, 3])
    appr = -x_in[:2] / np.linalg.norm(x_in[:2])
    delta = math.degrees(math.atan2(rad[0] * appr[1] - rad[1] * appr[0], rad @ appr))
    info = {"status": st, "pos_spread_mm": round(spread, 1), "jig_centre_base_mm": (T[:3, 3] * 1000).round(1).tolist(),
            "resting_face_centre_z_mm": round(face_c[2] * 1000, 1), "resting_face_corner_z_mm": [round(c[2] * 1000, 1) for c in corners],
            "tilt_from_flat_deg": round(tilt, 1), "entry_face_yaw_deg": round(yaw, 1),
            "approach_vs_radial_deg": round(delta, 1), "T_base_jig": T.round(5).tolist()}
    print(json.dumps(info, indent=1))
    px, py = float(T[0, 3]), float(T[1, 3])
    rows = []
    # (1) as placed (yaw measured), stand heights / tilts; (2) other yaws at the same spot
    dls = [float(x) for x in a.deltas.split(",")] if a.live_grid else [-120, -90, -60, -30, 0, 30, 60, 90, 120]
    Hs = [float(x) for x in a.heights.split(",")] if a.live_grid else [0.0, 0.015, 0.025, 0.035, 0.05, 0.065]
    Ts = [float(x) for x in a.tilts.split(",")] if a.live_grid else [0, 15, 25, 35]
    for dl in sorted({round(delta)} | set(dls)):
        for H in Hs:
            for t in Ts:
                if H * 1000 < (rj.D / 2 + 1) * math.sin(math.radians(t)):
                    continue
                Tj = jig_pose_for(px, py, dl, H, t)
                m, d, ids, _ = make_model(Tj, H, view=view)
                Tg = Tj @ np.linalg.inv(rj.T_end_jig(rj.INSERT))
                sols = ik_multi(m, d, ids, Tg)
                best = None
                for q, _, _ in sols[:4]:
                    pl = make_plan(m, d, ids, Tj, q)
                    if pl is None:
                        continue
                    rep = plan_report(m, d, ids, pl, view, START_Q)
                    sc = (plan_ok(rep, a.clear), rep["frac_both"], rep["clear_mm"]["floor"])
                    if best is None or sc > best[0]:
                        best = (sc, rep)
                r = {"delta": dl, "as_placed": dl == round(delta), "H_mm": H * 1000, "tilt": t, "ik": len(sols)}
                if best:
                    rep = best[1]
                    r.update(ok=best[0][0], floor=rep["clear_mm"]["floor"], support=rep["clear_mm"]["support"],
                             obst=rep["clear_mm"]["obstacle"], both=round(rep["frac_both"], 2), step=round(rep["max_step_rad"], 3),
                             transit=rep["transit_min_mm"])
                else:
                    # flat / no full plan: report the best static floor clearance at the full insertion
                    fl = [clearances(m, d, ids, q, rj.ENTRY_GAP * 1e-3)["floor"] for q, _, _ in sols[:4]]
                    r.update(ok=False, floor=max(fl) if fl else None)
                rows.append(r)
    (OUT / f"live_{a.tag}.json").write_text(json.dumps({"pose": info, "rows": rows}, indent=0))
    print("delta  placed  H_mm tilt  ik  ok  floor_mm  both  step")
    for r in rows:
        if r["as_placed"] or r["ok"]:
            print(f"{r['delta']:+5.0f}  {'*' if r['as_placed'] else ' '}   {r['H_mm']:4.0f} {r['tilt']:4.0f} {r['ik']:3d}  {int(r['ok'])}  "
                  f"{r.get('floor')!s:>7}  {r.get('both', '-')!s:>4}  {r.get('step', '-')}")


def cmd_export(a):
    """sim/models/rod_jig/sim_data.json for the production scene (sim/.venv has no trimesh): CAD finger slices in
    the finger body frames (m) and the jig's COM / inertia per kg from rod_jig.stl."""
    m0, _ = scene.load_model(dict(scene.load_settings(), jig={"enabled": False}))
    sl = rj.cad_finger_slices(m0, mujoco.MjData(m0), scene.Ids(m0))
    data = {"note": "generated by sim/check_rod_jig.py export from fingers_closed_in_jig_frame.stl and rod_jig.stl",
            "finger_slices": {k: [np.round(v, 7).tolist() for v in sl[k]["pieces"]] for k in ("left", "right")},
            "inertia_per_kg": {k: v for k, v in rj.jig_inertia(1.0).items() if k in ("com_mm", "full")},
            "volume_cm3": round(rj.jig_inertia(1.0)["volume_cm3"], 2)}
    rj.SIM_DATA.write_text(json.dumps(data))
    print(rj.SIM_DATA, {k: len(v) for k, v in data["finger_slices"].items()}, data["inertia_per_kg"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", type=int, default=2)
    ap.add_argument("--heights", default="0,0.02,0.03,0.05")
    ap.add_argument("--tilts", default="0,20,35")
    ap.add_argument("--px", default="0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--py", default="-0.20,-0.10,0,0.10,0.20")
    ap.add_argument("--deltas", default="-90,-60,-30,0,30,60,90")
    ap.add_argument("--clear", type=float, default=17.0)
    ap.add_argument("--tag", default="survey")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--src", default=None, help="plan: placement survey tag to read")
    ap.add_argument("--fingers", default="cad", choices=["cad", "model"])
    ap.add_argument("--mass", type=float, default=0.137)
    ap.add_argument("--friction", type=float, default=0.35)
    ap.add_argument("--finger-friction", type=float, default=0.5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-vc", type=int, default=60)
    ap.add_argument("--scales", default="0.25,0.5,0.75,1.0,1.5")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--live-grid", action="store_true", help="live: use --heights/--tilts/--deltas")
    ap.add_argument("--axes", default=None, help="sweep: comma list of y,z,x,yaw,pitch,roll")
    ap.add_argument("--slack", type=float, default=None)
    ap.add_argument("--chamfer", type=float, default=None)
    ap.add_argument("--entry-gap", type=float, default=None)
    ap.add_argument("--fit-z", type=float, default=None)
    a = ap.parse_args()
    a.variant = rj.set_variant(a.slack, a.chamfer, a.entry_gap, a.fit_z)
    a.variant_tag = "" if (a.slack, a.chamfer, a.entry_gap, a.fit_z) == (None, None, None, None) else \
        f"_s{rj.SLACK:g}c{rj.CHAMFER:g}g{rj.ENTRY_GAP:g}" + ("" if a.fit_z is None else f"f{rj.FIT:g}")
    print("jig variant:", a.variant, flush=True)
    os.nice(10)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd != "render" and not a.render:
        mem_cap(3.0)
    {"export": cmd_export, "live": cmd_live, "geom": cmd_geom, "placement": cmd_placement, "plan": cmd_plan, "trial": cmd_trial, "mc": cmd_mc,
     "sweep": cmd_sweep, "report": cmd_report}[a.cmd](a)
