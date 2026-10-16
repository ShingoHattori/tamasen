"""Grasp candidates for the box, inverse kinematics and static checks — simulation only.

A candidate puts the gripper's closing axis (end_link y, the finger slide axis) along one block
axis — the jaws then sit on that pair of opposite faces — and approaches along another block axis
(end_link x points from the palm toward the fingertips). The grip site (midway along the pads)
goes to the block centre. For each candidate:

  width      required gap = block size along the closing axis + clearance, vs the configured
             maximum jaw gap (which is itself unverified on the real gripper)
  ik         damped least squares on the MuJoCo model within joint limits; position and
             orientation residuals must be small
  floor      with the block removed, no arm / jaw geometry may touch or penetrate the floor at
             the pre-grasp and grasp poses
  self       no link-hull self-contact deeper than SELF_PEN_TOL (hulls are coarse; see scene.py)
  limits     the solution stays inside the joint ranges (IK clamps; flagged if it sits on a limit)

Nothing here is sent anywhere outside the simulator.
"""
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

import scene

PREGRASP_BACKOFF = 0.08     # m, along -approach
LIFT = 0.08                 # m, straight up
IK_POS_TOL = 2e-3
IK_ROT_TOL = math.radians(2.0)
SELF_PEN_TOL = 3e-3
# Fingers / arm vs the table at the grasp pose. The path check before execution (sim_server
# FLOOR_CLEARANCE) wants >= 15 mm along the simulated motion; the plan keeps 17 mm so a candidate it
# accepts also passes there (2026-10-14: a top grasp of the standing block put the fingertips 8.5 mm above
# the table and the path check refused the move). The simulated arm is gravity-compensated like the real one
# (scene.py gravcomp), so it barely sags; the calibration error (~4 mm vertical) still leaves >= 11 mm.
PLAN_FLOOR_CLEARANCE = 0.017
# Fingertips at least this far past the block face they enter through. The jaw faces are 13 mm wide and 63 mm
# long; a flat block (31 mm tall) grasped from above with 17 mm under the tips leaves 14 mm.
MIN_ENGAGE = 0.010
GRIP_TO_TIP = 0.025         # grip site sits this far behind the fingertip plane (scene.py)
NEAR_BACKOFF = 0.03         # m, "near" point before the grasp: last place to look at ID 1 and correct
# Approach directions tilted about the closing axis, so that ID 1 (on the gripper's top face) can face the
# overhead camera; a straight top-down approach shows it edge-on (2026-10-14: 80-107 deg, never detected).
TILTS_DEG = (0, 20, -20, 35, -35)
VIS_MAX_DEG, VIS_MARGIN_PX, VIS_MIN_SIDE_PX = 55.0, 40, 20
TIP_MARGIN = 0.003
AXIS_NAMES = ["X", "Y", "Z"]


@dataclass
class Candidate:
    name: str
    close_axis: int
    approach: np.ndarray          # unit, base frame
    R_end: np.ndarray             # target end_link orientation, base frame
    width_m: float
    ok: bool = False
    reasons: list[str] = field(default_factory=list)
    q_pre: np.ndarray | None = None
    q_grasp: np.ndarray | None = None
    q_lift: np.ndarray | None = None
    score: float = math.inf
    grasp_point: np.ndarray | None = None
    tilt_deg: float = 0.0
    q_near: np.ndarray | None = None
    targets: dict = field(default_factory=dict)      # base-frame grip-site targets (for visual correction)
    vis: dict = field(default_factory=dict)          # predicted ID 1 visibility per pose
    correction: dict | None = None
    open_gap_m: float = 0.0                          # jaw gap while approaching (scene.toml [jaw] approach_gap_m)
    kind: str = "block"                              # "block" | "jig" (plan_jig)
    step_mm: float | None = None                     # step of the automatic approach (jig: scene.toml [jig] step_mm)
    q_path: list | None = None                       # jig: IK every 1 mm from near to grasp (simulation trial only)

    def json(self) -> dict:
        return {"name": self.name, "width_m": self.width_m, "ok": self.ok, "reasons": self.reasons,
                "approach": self.approach.tolist(), "score": None if not math.isfinite(self.score) else self.score,
                "q_pre": None if self.q_pre is None else self.q_pre.tolist(),
                "q_grasp": None if self.q_grasp is None else self.q_grasp.tolist(),
                "q_lift": None if self.q_lift is None else self.q_lift.tolist(),
                "q_near": None if self.q_near is None else self.q_near.tolist(),
                "tilt_deg": self.tilt_deg, "vis": self.vis, "correction": self.correction, "open_gap_m": self.open_gap_m,
                "kind": self.kind, "step_mm": self.step_mm}


def _axis_rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def candidates(T_base_obj: np.ndarray, size: np.ndarray, clearance: float, tilts=(0,)) -> list[Candidate]:
    R = T_base_obj[:3, :3]
    out = []
    for c in range(3):
        for a in range(3):
            if a == c:
                continue
            for sa in (1, -1):
                for sc in (1, -1):
                    for tilt in tilts:
                        y = sc * R[:, c]
                        approach = _axis_rot(y, math.radians(tilt)) @ (-sa * R[:, a])   # tilted about the closing axis
                        x = approach
                        Rend = np.column_stack([x, y, np.cross(x, y)])
                        face = f"{'+' if sa > 0 else '-'}{AXIS_NAMES[a]}"
                        name = (f"挟む {AXIS_NAMES[c]} 面対 / {face} 面から接近" + ("" if sc > 0 else "（手首 180°）")
                                + (f"、傾き {tilt:+d}°" if tilt else ""))
                        cand = Candidate(name, c, approach, Rend, float(size[c] + clearance))
                        cand.tilt_deg = float(tilt)
                        out.append(cand)
    return out


RAY_GROUPS = np.array([1, 0, 1, 0, 0, 1], np.uint8)   # block / floor, visual meshes, registered obstacles
RAY_CAM_SKIP_M = 0.18
VIS_STEP_M = 0.01           # approach sampled at this spacing for the visibility check (= the 10 mm steps)


def place_block(d, ids: scene.Ids, T_base_obj) -> None:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(T_base_obj, float)[:3, :3].ravel())
    d.qpos[ids.block_q:ids.block_q + 3], d.qpos[ids.block_q + 3:ids.block_q + 7] = T_base_obj[:3, 3], quat


def marker_view(m, d, ids: scene.Ids, q, view: dict) -> dict:
    """Predicted ID 1 view at joint angles q: angle to the camera, image position, side length, and
    occlusion: rays from the centre and four inner corner points of the square (3 mm off its face) to the
    camera, against the block, floor, the arm's visual meshes (not end_link's) and registered obstacles."""
    import cv2
    d.qpos[ids.arm_qadr] = q
    mujoco.mj_kinematics(m, d)
    F = np.eye(4)
    F[:3, :3], F[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
    P = view["X"] @ F @ view["Y"]
    to_cam = -P[:3, 3] / np.linalg.norm(P[:3, 3])
    facing = float(np.degrees(np.arccos(np.clip(P[:3, 2] @ to_cam, -1, 1))))
    h = view["size"] / 2
    C = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]])
    uv, _ = cv2.projectPoints((P @ C.T).T[:, :3], np.zeros(3), np.zeros(3), view["K"], view["dist"])
    uv = uv.reshape(-1, 2)
    side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
    W, H = view["image"]
    inside = bool(uv[:, 0].min() > VIS_MARGIN_PX and uv[:, 1].min() > VIS_MARGIN_PX
                  and uv[:, 0].max() < W - VIS_MARGIN_PX and uv[:, 1].max() < H - VIS_MARGIN_PX)
    B = F @ view["Y"]                                       # marker in base
    cam = np.linalg.inv(view["X"])[:3, 3]
    blocked, gid, hits = 0, np.zeros(1, np.int32), set()
    pts = [np.zeros(3)] + [p_ * 0.7 for p_ in C[:, :3]]
    for p_ in pts:
        a = (B @ np.r_[p_, 1.0])[:3] + B[:3, 2] * 0.003
        # end_link's own geoms (mount, square) are skipped: the calibrated marker origin sits a few mm off the
        # modelled square. The ray stops RAY_CAM_SKIP_M short of the camera (inside the camera_head box).
        dist = mujoco.mj_ray(m, d, a, cam - a, RAY_GROUPS, 1, ids.end_body, gid)
        if 0 <= dist < 1 - RAY_CAM_SKIP_M / np.linalg.norm(cam - a):
            blocked += 1
            hits.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or f"geom{int(gid[0])}")
    ok = facing <= VIS_MAX_DEG and inside and side >= VIS_MIN_SIDE_PX and P[2, 3] > 0 and blocked == 0
    return {"ok": bool(ok), "facing_deg": round(facing, 1), "side_px": round(side, 1), "uv": np.round(uv.mean(0)).tolist(),
            "blocked": blocked, "hit": sorted(hits)}


def approach_view(m, d, ids: scene.Ids, qs: list, view: dict) -> list[dict]:
    """marker_view along pre -> near -> grasp, sampled every VIS_STEP_M (joint-space interpolation of the
    short legs); each entry tagged with its leg."""
    out = []
    for leg, qa, qb, dist in (("pre-near", qs[0], qs[1], PREGRASP_BACKOFF - NEAR_BACKOFF), ("near-grasp", qs[1], qs[2], NEAR_BACKOFF)):
        n = max(1, int(round(dist / VIS_STEP_M)))
        for i in range(n + 1 if leg == "near-grasp" else n):
            v = marker_view(m, d, ids, qa + (qb - qa) * (i / n), view)
            out.append(dict(v, leg=leg))
    return out


def solve_ik(m, d, ids: scene.Ids, p_target, R_target, q0, iters=300) -> tuple[np.ndarray, float, float]:
    """Grip site to (p, R). Returns q, position error (m), rotation error (rad)."""
    q = q0.copy()
    lo, hi = ids.arm_range[:, 0], ids.arm_range[:, 1]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[ids.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[ids.grip_site]
        Rs = d.site_xmat[ids.grip_site].reshape(3, 3)
        ep = p_target - p
        er_m = R_target @ Rs.T
        er = 0.5 * np.array([er_m[2, 1] - er_m[1, 2], er_m[0, 2] - er_m[2, 0], er_m[1, 0] - er_m[0, 1]])
        if np.linalg.norm(ep) < IK_POS_TOL / 4 and np.linalg.norm(er) < IK_ROT_TOL / 4:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, ids.grip_site)
        J = np.vstack([jacp[:, ids.arm_dadr], jacr[:, ids.arm_dadr] * 0.3])
        e = np.concatenate([ep, er * 0.3])
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
    d.qpos[ids.arm_qadr] = q
    mujoco.mj_kinematics(m, d)
    Rs = d.site_xmat[ids.grip_site].reshape(3, 3)
    rot_err = math.acos(np.clip((np.trace(R_target @ Rs.T) - 1) / 2, -1, 1))
    return q, float(np.linalg.norm(p_target - d.site_xpos[ids.grip_site])), rot_err


def static_contacts(m, d, ids: scene.Ids, q, gap: float, block_away: bool) -> dict:
    """Contacts at a static configuration. With block_away the block is moved out of the scene."""
    d.qpos[ids.arm_qadr] = q
    half = max(0.0, (gap - scene._gap_model(m).gap0) / 2)
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = half, -half
    if block_away:
        d.qpos[ids.block_q:ids.block_q + 7] = [10, 10, 1, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    floor, self_, block = [], [], []
    obst = set(ids.obstacle_geoms)
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if ids.floor_geom in (c.geom1, c.geom2) or c.geom1 in obst or c.geom2 in obst:   # floor or registered obstacle
            other = b2 if c.geom1 == ids.floor_geom else b1
            if other in ids.arm_bodies:
                floor.append(-c.dist)
        elif b1 in ids.arm_bodies and b2 in ids.arm_bodies:
            self_.append((-c.dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1),
                          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)))
        elif ids.block_body in (b1, b2):
            block.append(-c.dist)
    return {"floor": floor, "self": self_, "block": block}


def floor_clearance(m, d, ids: scene.Ids, q, gap: float) -> float:
    """Smallest distance from any arm / jaw collision geom (not the base) to the floor, jaws open to gap."""
    d.qpos[ids.arm_qadr] = q
    half = max(0.0, (gap - scene._gap_model(m).gap0) / 2)
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = half, -half
    mujoco.mj_kinematics(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return min(mujoco.mj_geomDistance(m, d, g, ids.floor_geom, 0.2, None) for g in range(m.ngeom)
               if m.geom_bodyid[g] in ids.arm_bodies and m.geom_bodyid[g] != base
               and (m.geom_contype[g] | m.geom_conaffinity[g]))


def open_gap(c: Candidate, jaw: dict, model_max: float) -> float:
    """Jaw gap for the approach: approach_gap_m (wide, so a few mm of position error still lands the block
    between the jaws; requester 2026-10-14) but at least the required width + 10 mm, within the maximum."""
    return float(min(max(c.width_m + 0.01, jaw.get("approach_gap_m", 0.0)), jaw["max_gap_m"], model_max))


def retarget(m, settings: dict, c: Candidate, offset: np.ndarray, q_seed: np.ndarray) -> list[str]:
    """Shift all of a candidate's grip-site targets by offset (base frame) and re-solve them; returns problems."""
    d = mujoco.MjData(m)
    ids = scene.Ids(m)
    jaw = settings["jaw"]
    gap = c.open_gap_m or open_gap(c, jaw, scene._gap_model(m)(ids.fl_range[1]))
    qs, probs, seed = {}, [], q_seed
    for k in ("pre", "near", "grasp", "lift"):
        q, pe, re = solve_ik(m, d, ids, np.array(c.targets[k]) + offset, c.R_end, seed)
        if pe > IK_POS_TOL or re > IK_ROT_TOL:
            probs.append(f"{k}: IK 残差 {pe * 1000:.1f} mm / {math.degrees(re):.1f}°")
        qs[k], seed = q, q
    if floor_clearance(m, d, ids, qs["grasp"], gap) < PLAN_FLOOR_CLEARANCE - 1e-3:
        probs.append("補正後の把持姿勢で机とのすき間が足りない")
    if not probs:
        c.q_pre, c.q_near, c.q_grasp, c.q_lift = qs["pre"], qs["near"], qs["grasp"], qs["lift"]
    return probs


def plan(m, settings: dict, T_base_obj: np.ndarray, q_start: np.ndarray, view: dict | None = None) -> list[Candidate]:
    """Grasp candidates. With view (adopted camera calibration and intrinsics), approaches tilted about the
    closing axis are added and a candidate needs ID 1 predicted visible at the pre-grasp and near poses."""
    d = mujoco.MjData(m)
    ids = scene.Ids(m)
    size = np.array(settings["block"]["size_m"])
    jaw = settings["jaw"]
    max_gap = jaw["max_gap_m"]
    model_max = scene._gap_model(m)(ids.fl_range[1])
    centre = T_base_obj[:3, 3]
    out = candidates(T_base_obj, size, jaw["clearance_m"], TILTS_DEG if view else (0,))
    for c in out:
        if c.width_m > max_gap:
            c.reasons.append(f"開口不足: 必要 {c.width_m * 1000:.0f} mm > 設定の最大 {max_gap * 1000:.0f} mm")
        if c.width_m > model_max:
            c.reasons.append(f"モデルの最大開口 {model_max * 1000:.0f} mm も超える")
        if c.approach[2] > 0.3:
            c.reasons.append("下から接近する向き（床を通る）")
        if c.reasons:
            continue
        c.open_gap_m = open_gap(c, jaw, model_max)
        # Fingertips are GRIP_TO_TIP beyond the grip site. Keep them TIP_MARGIN inside the block
        # along the approach, so a flat block grasped from above does not drive the tips into the floor.
        half_a = 0.5 * float(np.abs(T_base_obj[:3, :3].T @ c.approach) @ size)
        back = max(0.0, GRIP_TO_TIP - half_a + TIP_MARGIN)
        p_grasp = centre - c.approach * back
        c.grasp_point = p_grasp
        p_pre = p_grasp - c.approach * PREGRASP_BACKOFF
        best = None
        for seed in (q_start, np.array([0, -0.9, -1.4, 0, 0, 0]), np.array([0, -1.6, -1.2, 0.5, 0, 0])):
            q1, e1, r1 = solve_ik(m, d, ids, p_pre, c.R_end, seed)
            q2, e2, r2 = solve_ik(m, d, ids, p_grasp, c.R_end, q1)
            p_lift = p_grasp + np.array([0, 0, LIFT])
            q3, e3, r3 = solve_ik(m, d, ids, p_lift, c.R_end, q2)
            err = max(e1, e2, e3), max(r1, r2, r3)
            if best is None or err[0] + err[1] * 0.05 < best[3][0] + best[3][1] * 0.05:
                best = (q1, q2, q3, err)
        c.q_pre, c.q_grasp, c.q_lift, (pe, re) = best
        if pe > IK_POS_TOL or re > IK_ROT_TOL:
            c.reasons.append(f"IK 解なし（残差 {pe * 1000:.1f} mm / {math.degrees(re):.1f}°）")
            continue
        # Table clearance at the grasp pose (jaws open): raise the grasp point until the fingers / arm keep
        # PLAN_FLOOR_CLEARANCE, then require the jaws to still reach MIN_ENGAGE into the block.
        gap_open = c.open_gap_m
        raised = 0.0
        for _ in range(4):
            clr = floor_clearance(m, d, ids, c.q_grasp, gap_open)
            if clr >= PLAN_FLOOR_CLEARANCE:
                break
            dz = PLAN_FLOOR_CLEARANCE - clr + 0.001
            raised += dz
            p_grasp = p_grasp + np.array([0, 0, dz])
            p_pre = p_grasp - c.approach * PREGRASP_BACKOFF
            c.q_pre, e1, r1 = solve_ik(m, d, ids, p_pre, c.R_end, c.q_pre)
            c.q_grasp, e2, r2 = solve_ik(m, d, ids, p_grasp, c.R_end, c.q_pre)
            c.q_lift, e3, r3 = solve_ik(m, d, ids, p_grasp + np.array([0, 0, LIFT]), c.R_end, c.q_grasp)
            pe, re = max(e1, e2, e3), max(r1, r2, r3)
        c.grasp_point = p_grasp
        p_near = p_grasp - c.approach * NEAR_BACKOFF
        c.q_near, e4, r4 = solve_ik(m, d, ids, p_near, c.R_end, c.q_pre)
        pe, re = max(pe, e4), max(re, r4)
        c.targets = {"pre": p_pre.tolist(), "near": p_near.tolist(), "grasp": p_grasp.tolist(),
                     "lift": (p_grasp + np.array([0, 0, LIFT])).tolist()}
        clr = floor_clearance(m, d, ids, c.q_grasp, gap_open)
        engage = half_a + float((p_grasp + c.approach * GRIP_TO_TIP - centre) @ c.approach)   # tips past the entry face
        note = (f"机とのすき間を保つため把持点を {raised * 1000:.0f} mm 上げた（指先がブロックに {engage * 1000:.0f} mm 入る）"
                if raised > 0 else None)
        if pe > IK_POS_TOL or re > IK_ROT_TOL:
            c.reasons.append(f"把持点を上げたら IK 解なし（残差 {pe * 1000:.1f} mm / {math.degrees(re):.1f}°）")
            continue
        if clr < PLAN_FLOOR_CLEARANCE - 1e-3:
            c.reasons.append(f"把持姿勢で机とのすき間が {clr * 1000:.1f} mm（{PLAN_FLOOR_CLEARANCE * 1000:.0f} mm 未満）")
        if engage < MIN_ENGAGE:
            c.reasons.append(f"顎がブロックに十分かからない（指先の入り込み {engage * 1000:.0f} mm < {MIN_ENGAGE * 1000:.0f} mm）")
        lo, hi = ids.arm_range[:, 0], ids.arm_range[:, 1]
        if view:
            place_block(d, ids, T_base_obj)
            half = max(0.0, (c.open_gap_m - scene._gap_model(m).gap0) / 2)
            d.qpos[ids.fl_q], d.qpos[ids.fr_q] = half, -half
            c.vis = {k: marker_view(m, d, ids, q, view) for k, q in (("pre", c.q_pre), ("near", c.q_near), ("grasp", c.q_grasp))}
            path = approach_view(m, d, ids, [c.q_pre, c.q_near, c.q_grasp], view)
            c.vis["path"] = {"n": len(path), "visible": sum(v["ok"] for v in path),
                             "visible_near_grasp": sum(v["ok"] for v in path if v["leg"] == "near-grasp"),
                             "n_near_grasp": sum(v["leg"] == "near-grasp" for v in path),
                             "occluded": sum(v["blocked"] > 0 for v in path),
                             "facing_mean_deg": round(float(np.mean([v["facing_deg"] for v in path])), 1),
                             "facing_max_deg": max(v["facing_deg"] for v in path)}
            bad = [v for v in path if v["leg"] == "pre-near" and not v["ok"]] + [v for k in ("pre", "near") if not (v := c.vis[k])["ok"]]
            if bad:
                w = max(bad, key=lambda v: (v["blocked"], v["facing_deg"]))
                c.reasons.append(f"接近前→近接で ID 1 が見えない点がある（{len(bad)} 点、例: 向き {w['facing_deg']:.0f}°"
                                 + (f"、遮蔽 {w['blocked']}/5 本" if w["blocked"] else "") + f"、上限 {VIS_MAX_DEG:.0f}°）")
        for label, q in (("接近前", c.q_pre), ("近接", c.q_near), ("把持", c.q_grasp), ("持ち上げ", c.q_lift)):
            at_lim = [scene.ARM_JOINTS[i] for i in range(6) if q[i] - lo[i] < 1e-3 or hi[i] - q[i] < 1e-3]
            if at_lim:
                c.reasons.append(f"{label}: 関節制限に当たっている（{', '.join(at_lim)}）")
            gap_open = c.open_gap_m
            k = static_contacts(m, d, ids, q, gap_open, block_away=True)
            if k["floor"]:
                c.reasons.append(f"{label}: 床と干渉（最大貫入 {max(k['floor']) * 1000:.1f} mm）")
            deep = [s for s in k["self"] if s[0] > SELF_PEN_TOL]
            if deep:
                s = max(deep)
                c.reasons.append(f"{label}: 自己衝突 {s[1]}–{s[2]}（{s[0] * 1000:.1f} mm）")
        # does the open gripper at the grasp pose already collide with the block (jaw not around it)?
        k = static_contacts(m, d, ids, c.q_grasp, c.open_gap_m, block_away=False)
        if k["block"] and max(k["block"]) > 1e-3:
            c.reasons.append(f"開いた顎・手がブロックに食い込む（{max(k['block']) * 1000:.1f} mm）")
        c.ok = not c.reasons
        c.score = float(np.abs(c.q_grasp - q_start).sum() + 2.0 * (1 + c.approach[2])   # prefer top-down
                        + 0.5 * math.radians(abs(c.tilt_deg))
                        # and ID 1 in view along the whole approach: the steps re-measure it before each move
                        + (0.08 * c.vis["path"]["facing_mean_deg"]
                           + 1.0 * (c.vis["path"]["n_near_grasp"] - c.vis["path"]["visible_near_grasp"]) if c.vis else 0.0))
        if c.ok:
            c.reasons.append("条件を満たす")
        if note:
            c.reasons.append(note)          # information only (after the ok decision)
    out.sort(key=lambda c: (not c.ok, c.score))
    return out


# ============================================================================================ rod jig
# Grasp of the rod jig (cad/tools/rod_jig.py): the fingers go into its two pockets straight along -x_jig, the jaws
# close onto the inner walls. Unlike the block there is no choice of faces: end_link x = -x_jig, y = y_jig,
# z = -z_jig (ID 1, on end_link +z, faces up when the jig's marker does). Candidates differ by IK branch only.
#   pre   fingertips pre_out_m before the entry face        near  near_out_m before it (the insertion starts)
#   grasp fingers insert_depth_m deep (tips 1 mm out of the far face)   lift  lift_m straight up
# Checks: IK and joint limits; the table (PLAN_FLOOR_CLEARANCE) and the support / obstacles at every pose and every
# 5 mm of the insertion; no joint jump > JIG_STEP_JUMP between 1 mm insertion steps; fingers vs the jig walls
# (the model's finger hulls are up to 2.5 mm fatter than the real finger: JIG_PEN_TOL); the palm / links >= 3 mm
# from the jig; ID 1 and the jig marker (ID 13) visible at pre and near (both are measured before the insertion).
JIG_STEP_JUMP = 0.05            # rad between 1 mm insertion steps
JIG_SUPPORT_CLEAR = 0.003       # m, arm / jaw geoms from the support (the fingers pass ~10 mm over it)
JIG_PEN_TOL = {"model": 0.0026, "cad": 0.0003}


def _ray_block(m, d, a, cam, exclude_body):
    """First geom between a and the camera (None if clear). The camera_head obstacle box is ignored: it contains the
    camera, so every ray to it ends there (marker_view skips only the last RAY_CAM_SKIP_M, which oblique rays to
    the table exceed)."""
    cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obst_camera_head")
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, a, cam - a, RAY_GROUPS, 1, exclude_body, gid)
    if 0 <= dist < 1 - 0.02 / np.linalg.norm(cam - a) and int(gid[0]) != cb:
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or f"geom{int(gid[0])}"
    return None


def _square_view(m, d, P_cam, B_base, size, view, exclude_body):
    """A square marker (pose P in the camera, B in base): facing angle, image margin, side, 5 occlusion rays."""
    import cv2
    to_cam = -P_cam[:3, 3] / np.linalg.norm(P_cam[:3, 3])
    facing = float(np.degrees(np.arccos(np.clip(P_cam[:3, 2] @ to_cam, -1, 1))))
    h = size / 2
    C = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    uv, _ = cv2.projectPoints((P_cam[:3, :3] @ C.T).T + P_cam[:3, 3], np.zeros(3), np.zeros(3), view["K"], view["dist"])
    uv = uv.reshape(-1, 2)
    side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
    W, H = view["image"]
    inside = bool(uv[:, 0].min() > VIS_MARGIN_PX and uv[:, 1].min() > VIS_MARGIN_PX
                  and uv[:, 0].max() < W - VIS_MARGIN_PX and uv[:, 1].max() < H - VIS_MARGIN_PX)
    cam = np.linalg.inv(view["X"])[:3, 3]
    blocked, hits = 0, set()
    for p_ in [np.zeros(3)] + [c * 0.7 for c in C]:
        hit = _ray_block(m, d, (B_base @ np.r_[p_, 1.0])[:3] + B_base[:3, 2] * 0.003, cam, exclude_body)
        if hit:
            blocked += 1
            hits.add(hit)
    ok = facing <= VIS_MAX_DEG and inside and side >= VIS_MIN_SIDE_PX and P_cam[2, 3] > 0 and blocked == 0
    return {"ok": bool(ok), "facing_deg": round(facing, 1), "side_px": round(side, 1), "uv": np.round(uv.mean(0)).tolist(),
            "blocked": blocked, "hit": sorted(hits)}


def jig_views(m, d, ids, q, view: dict) -> tuple[dict, dict]:
    """(ID 1, jig marker) predicted views at joint angles q (jig placed in d)."""
    d.qpos[ids.arm_qadr] = q
    mujoco.mj_kinematics(m, d)
    F = np.eye(4)
    F[:3, :3], F[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
    B1 = F @ view["Y"]
    v1 = _square_view(m, d, view["X"] @ B1, B1, view["size"], view, ids.end_body)
    Bj = np.eye(4)
    Bj[:3, :3], Bj[:3, 3] = d.site_xmat[ids.jig_marker].reshape(3, 3), d.site_xpos[ids.jig_marker]
    vj = _square_view(m, d, view["X"] @ Bj, Bj, view.get("jig_size", 0.03), view, ids.jig_body)
    return v1, vj


def place_jig(d, ids, T) -> None:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(T, float)[:3, :3].ravel())
    d.qpos[ids.jig_q:ids.jig_q + 3], d.qpos[ids.jig_q + 3:ids.jig_q + 7] = T[:3, 3], quat


def _arm_geoms(m, ids):
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids.arm_bodies and m.geom_bodyid[g] != base
            and (m.geom_contype[g] | m.geom_conaffinity[g])]


def jig_pose_checks(m, d, ids, q, gap, ag, pen_tol) -> dict:
    """Clearances at one pose (jaws at gap, jig placed in d): table, obstacles (incl. the support), self contact,
    fingers into the jig walls, palm / links to the jig."""
    d.qpos[ids.arm_qadr] = q
    half = float(np.clip((gap - scene._gap_model(m).gap0) / 2, *ids.fl_range))
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = half, -half
    mujoco.mj_forward(m, d)
    fing = ids.finger_left | ids.finger_right
    floor = min(mujoco.mj_geomDistance(m, d, g, ids.floor_geom, 0.2, None) for g in ag)
    obst = min([mujoco.mj_geomDistance(m, d, g, o, 0.2, None) for g in ag for o in ids.obstacle_geoms] or [math.inf])
    pen, palm, self_ = 0.0, math.inf, 0.0
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if ids.jig_body in (b1, b2):
            other = b2 if b1 == ids.jig_body else b1
            if other in fing:
                pen = max(pen, -c.dist)
        elif b1 in ids.arm_bodies and b2 in ids.arm_bodies:
            self_ = max(self_, -c.dist)
    jg = [g for g in ag if m.geom_bodyid[g] not in fing]
    palm = min(mujoco.mj_geomDistance(m, d, g, j, 0.05, None) for g in jg for j in ids.jig_geoms)
    return {"floor": floor, "obst": obst, "pen": pen, "palm": palm, "self": self_}


def plan_jig(m, settings: dict, T_base_jig: np.ndarray, q_start: np.ndarray, view: dict | None = None) -> list[Candidate]:
    import rod_jig_scene as rj
    d = mujoco.MjData(m)
    ids = scene.Ids(m)
    jg = settings["jig"]
    place_jig(d, ids, T_base_jig)
    insert = float(jg["insert_depth_m"]) * 1000
    gap_in, grip = float(jg["entry_gap_m"]), float(jg["grip_width_m"])
    pen_tol = JIG_PEN_TOL.get(jg.get("finger_collision", "model"), 0.0026)
    ag = _arm_geoms(m, ids)

    def T_end(depth_mm):                                     # end_link at insertion depth (mm; < 0: tips outside)
        T = np.eye(4)
        T[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        T[:3, 3] = [(rj.D / 2 - rj.INSERT + (insert - depth_mm)) * 1e-3, 0, 0]
        return T_base_jig @ np.linalg.inv(T)
    site = m.site_pos[ids.grip_site].copy()
    grip_pt = lambda T: T[:3, 3] + T[:3, :3] @ site
    R_end = T_end(insert)[:3, :3]
    approach = R_end[:, 0].copy()
    pre_d, near_d = -float(jg["pre_out_m"]) * 1000, -float(jg["near_out_m"]) * 1000
    tg = {"pre": grip_pt(T_end(pre_d)), "near": grip_pt(T_end(near_d)), "grasp": grip_pt(T_end(insert))}
    tg["lift"] = tg["grasp"] + np.array([0, 0, float(jg["lift_m"])])
    # IK branches at the full insertion
    rng = np.random.default_rng(7)
    seeds = [q_start, np.array([0, -0.9, -1.4, 0, 0, 0]), np.array([0, -1.6, -1.2, 0.5, 0, 0]), np.array([0, -1.2, -1.0, 0, 0.8, 0])] + \
        [rng.uniform(ids.arm_range[:, 0], ids.arm_range[:, 1]) for _ in range(12)]
    sols = []
    for sd in seeds:
        q, pe, re = solve_ik(m, d, ids, tg["grasp"], R_end, np.asarray(sd, float))
        if pe < IK_POS_TOL and re < IK_ROT_TOL and not any(np.max(np.abs(q - s_)) < 0.05 for s_ in sols):
            sols.append(q)
    out = []
    if not sols:
        c = Candidate("治具: 入口面から差し込み", 1, approach, R_end, grip + 0.01, kind="jig", step_mm=float(jg["step_mm"]))
        c.reasons.append("差し込み姿勢の IK 解なし（届かない）")
        return [c]
    lo, hi = ids.arm_range[:, 0], ids.arm_range[:, 1]
    for i, qg in enumerate(sols[:4]):
        c = Candidate(f"治具: 入口面から差し込み（IK 解 {i + 1}）", 1, approach, R_end, grip + 0.01, kind="jig",
                      step_mm=float(jg["step_mm"]), open_gap_m=gap_in)
        c.grasp_point = tg["grasp"]
        c.targets = {k: v.tolist() for k, v in tg.items()}
        # insertion path every 1 mm, solved backwards from the full insertion (stays on this branch)
        path, q, jump, bad = [], qg, 0.0, None
        for dep in np.arange(insert, near_d - 0.5, -1.0):
            q2, pe, re = solve_ik(m, d, ids, grip_pt(T_end(dep)), R_end, q, iters=80)
            if pe > IK_POS_TOL or re > IK_ROT_TOL:
                bad = f"差し込みの途中（深さ {dep:.0f} mm）で IK 解なし"
                break
            jump = max(jump, float(np.max(np.abs(q2 - q))))
            path.append(q2)
            q = q2
        if bad:
            c.reasons.append(bad)
            out.append(c)
            continue
        path = path[::-1]                                   # near .. grasp
        c.q_path = [p_.tolist() for p_ in path]
        c.q_grasp, c.q_near = path[-1], path[0]
        c.q_pre, pe1, re1 = solve_ik(m, d, ids, tg["pre"], R_end, c.q_near)
        c.q_lift, pe2, re2 = solve_ik(m, d, ids, tg["lift"], R_end, c.q_grasp)
        if max(pe1, pe2) > IK_POS_TOL or max(re1, re2) > IK_ROT_TOL:
            c.reasons.append(f"接近前 / 持ち上げの IK 解なし（残差 {max(pe1, pe2) * 1000:.1f} mm）")
        if jump > JIG_STEP_JUMP:
            c.reasons.append(f"差し込み中に関節が跳ぶ（1 mm で {jump:.3f} rad）")
        for label, q in (("接近前", c.q_pre), ("近接", c.q_near), ("把持", c.q_grasp), ("持ち上げ", c.q_lift)):
            at_lim = [scene.ARM_JOINTS[k] for k in range(6) if q[k] - lo[k] < 1e-3 or hi[k] - q[k] < 1e-3]
            if at_lim:
                c.reasons.append(f"{label}: 関節制限に当たっている（{', '.join(at_lim)}）")
        # clearances: poses + every 5 mm of the insertion (jaws at the entry gap; closed on the walls at the grasp)
        worst = {"floor": math.inf, "obst": math.inf, "pen": 0.0, "palm": math.inf, "self": 0.0}
        seq = [(c.q_pre, gap_in), (c.q_near, gap_in)] + [(p_, gap_in) for p_ in path[::5]] + [(c.q_grasp, gap_in), (c.q_grasp, grip + 0.0004)]
        for q, g in seq:
            r = jig_pose_checks(m, d, ids, q, g, ag, pen_tol)
            for k in ("floor", "obst", "palm"):
                worst[k] = min(worst[k], r[k])
            for k in ("pen", "self"):
                worst[k] = max(worst[k], r[k])
        # lift: the jig comes along (held), so only the table / obstacles for the arm
        r = jig_pose_checks(m, d, ids, c.q_lift, grip, ag, pen_tol)
        worst["floor"] = min(worst["floor"], r["floor"])
        worst["obst"] = min(worst["obst"], r["obst"])
        c.vis["clear_mm"] = {k: round(v * 1000, 1) for k, v in worst.items()}
        if worst["floor"] < PLAN_FLOOR_CLEARANCE - 1e-3:
            c.reasons.append(f"机とのすき間 {worst['floor'] * 1000:.1f} mm（{PLAN_FLOOR_CLEARANCE * 1000:.0f} mm 未満）。"
                             "治具を台に載せる / 台を傾ける（scene.toml [jig] support_height_m / support_tilt_deg）")
        if worst["obst"] < JIG_SUPPORT_CLEAR:
            c.reasons.append(f"台・障害物とのすき間 {worst['obst'] * 1000:.1f} mm（{JIG_SUPPORT_CLEAR * 1000:.0f} mm 未満）")
        if worst["pen"] > pen_tol:
            c.reasons.append(f"指が治具の壁に食い込む（{worst['pen'] * 1000:.1f} mm）")
        if worst["palm"] < 0.003:
            c.reasons.append(f"掌・リンクが治具に近い（{worst['palm'] * 1000:.1f} mm）")
        if worst["self"] > SELF_PEN_TOL:
            c.reasons.append(f"自己衝突（{worst['self'] * 1000:.1f} mm）")
        if view:
            path_v = []
            for k, q in [("pre", c.q_pre), ("near", c.q_near)] + [(f"d{i}", p_) for i, p_ in enumerate(path) if i % 10 == 0] + [("grasp", c.q_grasp)]:
                v1, vj = jig_views(m, d, ids, q, view)
                path_v.append({"at": k, "id1": v1, "jig": vj})
            n = len(path_v)
            c.vis.update({"pre": path_v[0]["id1"], "near": path_v[1]["id1"], "grasp": path_v[-1]["id1"],
                          "jig_pre": path_v[0]["jig"], "jig_near": path_v[1]["jig"],
                          "path": {"n": n, "visible": sum(v["id1"]["ok"] for v in path_v),
                                   "jig_visible": sum(v["jig"]["ok"] for v in path_v),
                                   "both_visible": sum(v["id1"]["ok"] and v["jig"]["ok"] for v in path_v),
                                   "facing_mean_deg": round(float(np.mean([v["id1"]["facing_deg"] for v in path_v])), 1)}})
            miss = [k for k in ("pre", "near") if not c.vis[k]["ok"]] + [f"治具マーカー@{k}" for k in ("pre", "near") if not c.vis["jig_" + k]["ok"]]
            if miss:
                c.reasons.append("接近前・近接で見えないマーカーがある（" + "、".join(miss) + "）")
        c.ok = not c.reasons
        pv = c.vis.get("path")
        c.score = float(np.abs(c.q_grasp - q_start).sum() + (3.0 * (pv["n"] - pv["both_visible"]) / pv["n"] + 0.01 * pv["facing_mean_deg"] if pv else 0.0))
        if c.ok:
            c.reasons.append("条件を満たす")
            if pv:
                c.reasons.append(f"ID 1 と治具マーカーが両方見える点 {pv['both_visible']}/{pv['n']}、"
                                 f"机とのすき間 最小 {c.vis['clear_mm']['floor']:.0f} mm")
        out.append(c)
    out.sort(key=lambda c: (not c.ok, c.score))
    return out
