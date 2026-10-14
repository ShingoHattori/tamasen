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

    def json(self) -> dict:
        return {"name": self.name, "width_m": self.width_m, "ok": self.ok, "reasons": self.reasons,
                "approach": self.approach.tolist(), "score": None if not math.isfinite(self.score) else self.score,
                "q_pre": None if self.q_pre is None else self.q_pre.tolist(),
                "q_grasp": None if self.q_grasp is None else self.q_grasp.tolist(),
                "q_lift": None if self.q_lift is None else self.q_lift.tolist()}


def candidates(T_base_obj: np.ndarray, size: np.ndarray, clearance: float) -> list[Candidate]:
    R = T_base_obj[:3, :3]
    out = []
    for c in range(3):
        for a in range(3):
            if a == c:
                continue
            for sa in (1, -1):
                for sc in (1, -1):
                    approach = -sa * R[:, a]          # moving toward the block along -(face normal)
                    y = sc * R[:, c]
                    x = approach
                    Rend = np.column_stack([x, y, np.cross(x, y)])
                    face = f"{'+' if sa > 0 else '-'}{AXIS_NAMES[a]}"
                    name = f"挟む {AXIS_NAMES[c]} 面対 / {face} 面から接近" + ("" if sc > 0 else "（手首 180°）")
                    out.append(Candidate(name, c, approach, Rend, float(size[c] + clearance)))
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


def plan(m, settings: dict, T_base_obj: np.ndarray, q_start: np.ndarray) -> list[Candidate]:
    d = mujoco.MjData(m)
    ids = scene.Ids(m)
    size = np.array(settings["block"]["size_m"])
    jaw = settings["jaw"]
    max_gap = jaw["max_gap_m"]
    model_max = scene._gap_model(m)(ids.fl_range[1])
    centre = T_base_obj[:3, 3]
    out = candidates(T_base_obj, size, jaw["clearance_m"])
    for c in out:
        if c.width_m > max_gap:
            c.reasons.append(f"開口不足: 必要 {c.width_m * 1000:.0f} mm > 設定の最大 {max_gap * 1000:.0f} mm")
        if c.width_m > model_max:
            c.reasons.append(f"モデルの最大開口 {model_max * 1000:.0f} mm も超える")
        if c.approach[2] > 0.3:
            c.reasons.append("下から接近する向き（床を通る）")
        if c.reasons:
            continue
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
        open_gap = min(c.width_m + 0.01, max_gap, model_max)
        raised = 0.0
        for _ in range(4):
            clr = floor_clearance(m, d, ids, c.q_grasp, open_gap)
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
        clr = floor_clearance(m, d, ids, c.q_grasp, open_gap)
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
        for label, q in (("接近前", c.q_pre), ("把持", c.q_grasp), ("持ち上げ", c.q_lift)):
            at_lim = [scene.ARM_JOINTS[i] for i in range(6) if q[i] - lo[i] < 1e-3 or hi[i] - q[i] < 1e-3]
            if at_lim:
                c.reasons.append(f"{label}: 関節制限に当たっている（{', '.join(at_lim)}）")
            open_gap = min(c.width_m + 0.01, max_gap, model_max)
            k = static_contacts(m, d, ids, q, open_gap, block_away=True)
            if k["floor"]:
                c.reasons.append(f"{label}: 床と干渉（最大貫入 {max(k['floor']) * 1000:.1f} mm）")
            deep = [s for s in k["self"] if s[0] > SELF_PEN_TOL]
            if deep:
                s = max(deep)
                c.reasons.append(f"{label}: 自己衝突 {s[1]}–{s[2]}（{s[0] * 1000:.1f} mm）")
        # does the open gripper at the grasp pose already collide with the block (jaw not around it)?
        k = static_contacts(m, d, ids, c.q_grasp, min(c.width_m + 0.01, max_gap, model_max), block_away=False)
        if k["block"] and max(k["block"]) > 1e-3:
            c.reasons.append(f"開いた顎・手がブロックに食い込む（{max(k['block']) * 1000:.1f} mm）")
        c.ok = not c.reasons
        c.score = float(np.abs(c.q_grasp - q_start).sum() + 2.0 * (1 + c.approach[2]))  # prefer top-down
        if c.ok:
            c.reasons.append("条件を満たす")
        if note:
            c.reasons.append(note)          # information only (after the ok decision)
    out.sort(key=lambda c: (not c.ok, c.score))
    return out
