#!/usr/bin/env python3
"""Fixed-camera (eye-to-hand) calibration: camera <-> arm base and end_link <-> ID 1 marker.

Model, for every static pose i:
    T_cam_m1(i) = X @ F_i @ Y
    X = T_cam_base   (unknown, 6 dof)      F_i = T_base_ee(i)  (measured joint angles -> FK of end_link)
    Y = T_ee_m1      (unknown, 6 dof)      the ID 1 mount on the gripper body — NOT assumed to be identity
The ID 1 marker pose is never equated with the end_link pose: Y is estimated.

Fit: Levenberg-Marquardt on the four ID 1 corner reprojection errors (px) of all calibration samples,
initialised with OpenCV's eye-to-hand calibrateHandEye. Using corners directly avoids the poor depth
and the IPPE two-solution ambiguity of a single small marker pose.

Not modelled: link / gearbox deflection and joint zero offsets (they show up as residual), the
grip centre (a separate calibration: T_m1_grip in markers.toml).

    python handeye.py solve calib/handeye/<session>        # writes result.json in that directory
    python handeye.py synth                                 # synthetic self-test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import pose

MIN_SAMPLES = 6
MIN_AXIS_SPREAD = 0.25          # 2nd singular value of the unit rotation axes of relative motions
MIN_REL_ROT_DEG = 15.0
HOLDOUT_MAX_PX = 2.0


def rt_to_T(r, t):
    R, _ = cv2.Rodrigues(np.asarray(r, float))
    return pose.rigid(R, t)


def T_to_rt(T):
    r, _ = cv2.Rodrigues(T[:3, :3])
    return r.ravel(), T[:3, 3].copy()


def project(T_cam_m, size, K, dist):
    rvec, _ = cv2.Rodrigues(T_cam_m[:3, :3])
    uv, _ = cv2.projectPoints(pose.marker_object_points(size), rvec, T_cam_m[:3, 3], K, dist)
    return uv.reshape(4, 2)


def diversity(F_list) -> dict:
    """How well the end_link rotations excite the problem (relative motions between samples)."""
    axes, angles = [], []
    for i in range(len(F_list)):
        for j in range(i + 1, len(F_list)):
            Rij = F_list[i][:3, :3].T @ F_list[j][:3, :3]
            a = pose.rotation_angle(Rij)
            if math.degrees(a) < MIN_REL_ROT_DEG:
                continue
            r, _ = cv2.Rodrigues(Rij)
            axes.append(r.ravel() / np.linalg.norm(r))
            angles.append(math.degrees(a))
    if len(axes) < 2:
        return {"ok": False, "pairs": len(axes), "axis_spread": 0.0, "max_rel_rot_deg": max(angles, default=0.0),
                "message": f"{MIN_REL_ROT_DEG:.0f}° 以上の相対回転を持つ組が足りない（並進だけ・小さい回転だけでは解けない）"}
    A = np.array(axes)
    A = np.vstack([A, -A])                      # axis sign does not matter
    s = np.linalg.svd(A / math.sqrt(len(A)), compute_uv=False)
    ok = s[1] >= MIN_AXIS_SPREAD
    return {"ok": bool(ok), "pairs": len(axes), "axis_spread": float(s[1]), "axis_spread_3rd": float(s[2]),
            "max_rel_rot_deg": float(max(angles)),
            "message": "回転軸が 2 方向以上に分散している" if ok else
                       f"回転軸の分散が小さい（{s[1]:.2f} < {MIN_AXIS_SPREAD}）: 異なる軸まわりの回転を加える"}


def init_guess(samples):
    """Closed-form start (Park & Martin): for pairs i, j,  A X = X B  with
    A = O_i O_j^-1 (marker motion seen by the camera), B = F_i F_j^-1 (end_link motion), X = T_cam_base.
    (This OpenCV 5 build has no calibrateHandEye, only its constants.)"""
    alphas, betas, rows_A, rows_b = [], [], [], []
    pairs = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    for i, j in pairs:
        A = samples[i]["O"] @ pose.invert(samples[j]["O"])
        B = samples[i]["F"] @ pose.invert(samples[j]["F"])
        if math.degrees(pose.rotation_angle(B[:3, :3])) < 5:
            continue
        a, _ = cv2.Rodrigues(A[:3, :3]); b, _ = cv2.Rodrigues(B[:3, :3])
        alphas.append(a.ravel()); betas.append(b.ravel())
        rows_A.append((A, B))
    M = sum(np.outer(a, b) for a, b in zip(alphas, betas))          # R_X b = a
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in rows_A])
    d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in rows_A])
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    X = pose.rigid(R, t)
    Y = _mean_T([pose.invert(s["F"]) @ pose.invert(X) @ s["O"] for s in samples])
    err = float(np.mean([np.linalg.norm(project(X @ s["F"] @ Y, s["size"], s["K"], s["dist"]) - s["corners"], axis=1).mean()
                         for s in samples]))
    return X, Y, err, "PARK_MARTIN"


def _mean_T(Ts):
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    return pose.rigid(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt, t)


# Prior on joint zero offsets: a few degrees for joint2..5; joint1 and joint6 offsets are not observable
# (absorbed by T_cam_base and T_ee_m1), so they are pinned to 0.
OFFSET_PRIOR = np.radians([0.01, 3.0, 3.0, 3.0, 3.0, 0.01])
_chain = None


def F_of(s, dq):
    """end_link pose of a sample, re-evaluated with joint zero offsets dq (URDF FK)."""
    global _chain
    if dq is None or s.get("q") is None:
        return s["F"]
    if _chain is None:
        import urdf_fk
        _chain = urdf_fk.Chain()
    return _chain(np.asarray(s["q"]) + dq)


def residuals(p, samples):
    X = rt_to_T(p[0:3], p[3:6])
    Y = rt_to_T(p[6:9], p[9:12])
    dq = p[12:18] if len(p) > 12 else None
    r = [(project(X @ F_of(s, dq) @ Y, s["size"], s["K"], s["dist"]) - s["corners"]).ravel() / s["sigma"] for s in samples]
    if dq is not None:
        r.append(dq / OFFSET_PRIOR)
    return np.concatenate(r)


def refine(X, Y, samples, iters=60, offsets=False, dq0=None):
    p = np.concatenate([*T_to_rt(X), *T_to_rt(Y)] + ([np.zeros(6) if dq0 is None else dq0] if offsets else []))
    npar = len(p)
    lam = 1e-3
    r = residuals(p, samples)
    cost = r @ r
    for _ in range(iters):
        J = np.empty((r.size, npar))
        for k in range(npar):
            h = 1e-6 if (k % 6 < 3 or k >= 12) else 1e-7
            dp = np.zeros(npar); dp[k] = h
            J[:, k] = (residuals(p + dp, samples) - r) / h
        A, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            r2 = residuals(p + step, samples)
            if r2 @ r2 < cost:
                p, r, cost, lam = p + step, r2, r2 @ r2, lam * 0.3
                break
            lam *= 10
            if lam > 1e8:
                break
        if np.linalg.norm(step) < 1e-10 or lam > 1e8:
            break
    J = np.empty((r.size, npar))
    for k in range(npar):
        h = 1e-6 if (k % 6 < 3 or k >= 12) else 1e-7
        dp = np.zeros(npar); dp[k] = h
        J[:, k] = (residuals(p + dp, samples) - r) / h
    dof = max(1, r.size - npar)
    s2 = (r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(npar, np.nan)
    dq = p[12:18] if offsets else None
    return rt_to_T(p[0:3], p[3:6]), rt_to_T(p[6:9], p[9:12]), sd, float(np.sqrt(s2)), dq


def evaluate(X, Y, samples, dq=None) -> dict:
    """Prediction X F Y vs observation, for samples (normally the held-out ones)."""
    rows = []
    for s in samples:
        P = X @ F_of(s, dq) @ Y
        px = float(np.linalg.norm(project(P, s["size"], s["K"], s["dist"]) - s["corners"], axis=1).mean())
        O = s["O"]
        rows.append({"id": s["id"], "reproj_px": px, "pos_err_mm": float(np.linalg.norm(P[:3, 3] - O[:3, 3]) * 1000),
                     # lateral (image-plane) part is well determined, depth along the ray is not
                     "lateral_err_mm": float(np.linalg.norm((P[:3, 3] - O[:3, 3])[:2]) * 1000),
                     "rot_err_deg": math.degrees(pose.rotation_angle(P[:3, :3].T @ O[:3, :3])),
                     "obs_status": s.get("status")})
    if not rows:
        return {"n": 0}
    a = lambda k: [r[k] for r in rows]
    return {"n": len(rows), "rows": rows,
            "reproj_px_mean": float(np.mean(a("reproj_px"))), "reproj_px_max": float(np.max(a("reproj_px"))),
            "pos_err_mm_median": float(np.median(a("pos_err_mm"))), "pos_err_mm_max": float(np.max(a("pos_err_mm"))),
            "lateral_err_mm_median": float(np.median(a("lateral_err_mm"))),
            "rot_err_deg_median": float(np.median(a("rot_err_deg"))), "rot_err_deg_max": float(np.max(a("rot_err_deg")))}


def load_samples(d: Path):
    out = []
    for line in (d / "samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({"id": r["id"], "F": np.array(r["T_base_ee"]), "O": np.array(r["T_cam_m1"]),
                    "corners": np.array(r["corners_px"]), "sigma": max(r.get("corner_std_px", 0.1), 0.05),
                    "size": r["marker_size_m"], "K": np.array(r["K"]), "dist": np.array(r["dist"]),
                    "role": r.get("role", "calib"), "status": r.get("status"), "q": r.get("q")})
    return out


def solve(samples) -> dict:
    cal = [s for s in samples if s["role"] == "calib"]
    ev = [s for s in samples if s["role"] == "eval"]
    res = {"n_calib": len(cal), "n_eval": len(ev), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "frames": "X = T_cam_base (base -> cam), Y = T_ee_m1 (ID 1 marker -> end_link), metres",
           "not_modelled": "link/gearbox deflection, joint zero offsets, grip centre (separate)"}
    res["diversity"] = diversity([s["F"] for s in cal])
    if len(cal) < MIN_SAMPLES:
        res.update(ok=False, message=f"校正用サンプルが {len(cal)} 個（{MIN_SAMPLES} 個以上必要）")
        return res
    if not res["diversity"]["ok"]:
        res.update(ok=False, message="姿勢の回転が不足: " + res["diversity"]["message"])
        return res
    X0, Y0, e0, method = init_guess(cal)
    X, Y, sd, rms_w, _ = refine(X0, Y0, cal)
    # Joint zero offsets: the real motor zeros need not match the URDF zero. Estimated together with
    # X and Y (weak prior of a few degrees); reported, never written to the motors.
    plain = evaluate(X, Y, cal)
    Xo, Yo, sdo, rmso, dq = refine(X, Y, cal, offsets=True)
    with_off = evaluate(Xo, Yo, cal, dq)
    use_off = with_off["reproj_px_mean"] < 0.7 * plain["reproj_px_mean"]
    res["joint_offsets"] = {"used": bool(use_off), "deg": np.degrees(dq).tolist(),
                            "sd_deg": np.degrees(sdo[12:18]).tolist(),
                            "fit_px_without": plain["reproj_px_mean"], "fit_px_with": with_off["reproj_px_mean"],
                            "note": "関節零点ずれ（URDF の 0 rad に対する実機の 0 rad）。推定値の報告のみで、モーターには書き込まない"}
    if use_off:
        X, Y, sd, rms_w = Xo, Yo, sdo, rmso
    dqu = dq if use_off else None
    res.update(init_method=method, init_reproj_px=e0,
               T_cam_base=X.tolist(), T_base_cam=pose.invert(X).tolist(), T_ee_m1=Y.tolist(),
               sd={"cam_base_rot_deg": np.degrees(sd[0:3]).tolist(), "cam_base_t_mm": (sd[3:6] * 1000).tolist(),
                   "ee_m1_rot_deg": np.degrees(sd[6:9]).tolist(), "ee_m1_t_mm": (sd[9:12] * 1000).tolist()},
               weighted_rms=rms_w, fit=evaluate(X, Y, cal, dqu), holdout=evaluate(X, Y, ev, dqu))
    h = res["holdout"]
    if h["n"] == 0:
        res.update(ok=False, message="評価用（校正に使わない）サンプルがない。評価なしでは採用しない")
    else:
        # Acceptance on held-out corner reprojection (what the camera actually measures). The rotation
        # error against a single small marker's own pose is reported but is dominated by that pose's
        # noise (~1-2 deg at 0.4 px corner noise), so it is not a pass/fail criterion.
        ok = h["reproj_px_max"] < HOLDOUT_MAX_PX
        res.update(ok=bool(ok), message=(f"評価用サンプルの再投影誤差が {HOLDOUT_MAX_PX} px 未満" if ok else
                                         f"評価用サンプルの再投影誤差が大きい（最大 {h['reproj_px_max']:.2f} px ≥ {HOLDOUT_MAX_PX} px）"),
                   rot_note="回転誤差は単一マーカー姿勢のノイズを含む参考値（合否判定に使わない）")
    return res


# --------------------------------------------------------------------------- synthetic self-test

def synth(n_cal=10, n_eval=4, corner_px=0.15, joint_rad=0.0, axes="xyz", seed=0) -> dict:
    """Ground truth X, Y; poses rotating about the given end_link axes; noisy corners."""
    rng = np.random.default_rng(seed)
    K = np.array([[913.09, 0, 627.37], [0, 913.09, 480.32], [0, 0, 1.0]])
    dist = np.zeros(5)
    # camera ~0.65 m above the table looking down; base at the image's left
    X = pose.invert(pose.rigid(pose.rot("x", math.pi) @ pose.rot("z", 0.3), [0.35, 0.05, 0.65]))   # T_cam_base
    Y = pose.rigid(pose.rot("y", -math.pi / 2) @ pose.rot("z", 0.2), [-0.06, 0.03, 0.04])           # T_ee_m1
    F0 = pose.rigid(pose.rot("y", math.pi / 2) @ pose.rot("x", math.pi), [0.25, -0.05, 0.20])        # marker faces up
    samples = []
    for i in range(n_cal + n_eval):
        ang = np.zeros(3)
        for k, a in enumerate("xyz"):
            if a in axes:
                ang[k] = rng.uniform(-0.4, 0.4)
        dR = pose.rot("x", ang[0]) @ pose.rot("y", ang[1]) @ pose.rot("z", ang[2])
        F = F0 @ pose.rigid(dR, rng.uniform(-0.05, 0.05, 3))
        O = X @ F @ Y
        c = project(O, 0.02, K, dist) + rng.normal(0, corner_px, (4, 2))
        Fm = F @ pose.rigid(pose.rot("x", rng.normal(0, joint_rad)) @ pose.rot("y", rng.normal(0, joint_rad)), [0, 0, 0])
        sols = pose.solve_marker(c, 0.02, K, dist)
        samples.append({"id": i, "F": Fm, "O": sols[0][1], "corners": c, "sigma": max(corner_px, 0.05),
                        "size": 0.02, "K": K, "dist": dist, "role": "calib" if i < n_cal else "eval"})
    res = solve(samples)
    if "T_cam_base" in res:
        Xe, Ye = np.array(res["T_cam_base"]), np.array(res["T_ee_m1"])
        res["truth_error"] = {
            "cam_base_mm": float(np.linalg.norm(pose.invert(Xe)[:3, 3] - pose.invert(X)[:3, 3]) * 1000),
            "cam_base_deg": math.degrees(pose.rotation_angle(Xe[:3, :3].T @ X[:3, :3])),
            "ee_m1_mm": float(np.linalg.norm(Ye[:3, 3] - Y[:3, 3]) * 1000),
            "ee_m1_deg": math.degrees(pose.rotation_angle(Ye[:3, :3].T @ Y[:3, :3]))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve"); s.add_argument("session", type=Path)
    sub.add_parser("synth")
    args = ap.parse_args()
    if args.cmd == "solve":
        res = solve(load_samples(args.session))
        (args.session / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps({k: v for k, v in res.items() if k not in ("fit", "holdout")}, indent=1, ensure_ascii=False))
        print("holdout:", json.dumps({k: v for k, v in res.get("holdout", {}).items() if k != "rows"}, ensure_ascii=False))
    else:
        for label, kw in (("10 poses, xyz rotations, 0.15 px", {}),
                          ("10 poses, xyz, 0.4 px (small far marker)", {"corner_px": 0.4}),
                          ("xyz, 0.15 px + joint/deflection noise 0.003 rad", {"joint_rad": 0.003}),
                          ("rotation about z only (degenerate)", {"axes": "z"}),
                          ("translation only (degenerate)", {"axes": ""})):
            r = synth(**kw)
            te = r.get("truth_error")
            h = r.get("holdout", {})
            print(f"{label:50} ok={r.get('ok')} {r.get('message', '')}")
            if te:
                print(f"{'':50} truth err: cam-base {te['cam_base_mm']:.2f} mm / {te['cam_base_deg']:.3f}°, "
                      f"ee-m1 {te['ee_m1_mm']:.2f} mm / {te['ee_m1_deg']:.3f}°; holdout reproj {h.get('reproj_px_mean', 0):.2f} px, "
                      f"rot {h.get('rot_err_deg_median', 0):.2f}°, axis spread {r['diversity']['axis_spread']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
