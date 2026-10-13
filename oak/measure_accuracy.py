#!/usr/bin/env python3
"""Recognition accuracy from recordings of a STATIC scene (record with the page's 録画 button).

    python measure_accuracy.py stats output/recordings/<A>                   # temporal stability
    python measure_accuracy.py compare <A> <B> --id 3 --known-mm 50 [--axis x|y|z|any]
                                                    # scale: the block moved by a known distance
    python measure_accuracy.py frames <dir-of-png> --intrinsics k.json       # stills (e.g. other resolutions)

stats, per marker id: detection rate, image side length (px), corner jitter (px), position spread
(cam and ID 2 world, mm), rotation spread (deg), ambiguous / flip rate, reprojection error. Each frame
is reprocessed with the current pose.py from the lossless video (the same code as the live view).

Temporal stability (spread while nothing moves) is NOT absolute accuracy. Absolute accuracy needs a
reference: compare uses known displacements / heights between two static recordings, in the ID 2
world frame (so camera motion between the two cancels as long as ID 2 stays put).
Writes <recording>/accuracy.json.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import pose
import recorder
import world

HERE = Path(__file__).resolve().parent


def make_detector(cfg):
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeStep, p.adaptiveThreshWinSizeMax = 3, 4, 53
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary)), p)


def mean_rot(Rs):
    U, _, Vt = np.linalg.svd(sum(Rs))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def analyse(frames, K, dist, cfg) -> dict:
    det = make_detector(cfg)
    obs_ = pose.Observer()
    per = {}
    n = 0
    for img, t in frames:
        n += 1
        c, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        obs = obs_.observe(c, ids, t, cfg, K, dist)
        ref = next((o for o in obs if o.id == world.REF_ID and o.valid), None)
        for o in obs:
            d = per.setdefault(o.id, {"n": 0, "status": {}, "side": [], "corners": [], "T": [], "Tw": [], "reproj": []})
            d["n"] += 1
            d["status"][o.status] = d["status"].get(o.status, 0) + 1
            d["side"].append(o.side_px)
            d["corners"].append(o.corners)
            d["reproj"].append(o.reproj_px)
            if o.T_cam_marker is not None and o.status != "rejected":
                d["T"].append(o.T_cam_marker)
                d["Tw"].append(pose.invert(ref.T_cam_marker) @ o.T_cam_marker if ref is not None and o.id != world.REF_ID else None)
    out = {"frames": n, "markers": {}}
    for mid, d in sorted(per.items()):
        C = np.array(d["corners"])
        r = {"detected": d["n"], "detection_rate": d["n"] / n, "status_counts": d["status"],
             "ambiguous_rate": d["status"].get("ambiguous", 0) / d["n"], "flip_rate": d["status"].get("flip", 0) / d["n"],
             "side_px_median": float(np.median(d["side"])),
             "corner_jitter_px": float(np.sqrt(np.mean(np.sum((C - C.mean(0)) ** 2, axis=2)))),
             "reproj_px_median": float(np.nanmedian(d["reproj"]))}
        for key, label in (("T", "cam"), ("Tw", "world")):
            Ts = [T for T in d[key] if T is not None]
            if len(Ts) < 2:
                continue
            P = np.array([T[:3, 3] for T in Ts])
            Rm = mean_rot([T[:3, :3] for T in Ts])
            ang = np.degrees([pose.rotation_angle(Rm.T @ T[:3, :3]) for T in Ts])
            normal = Rm[:, 2]
            r[label] = {"n": len(Ts), "mean_mm": (P.mean(0) * 1000).tolist(), "std_mm": (P.std(0) * 1000).tolist(),
                        "p2p_mm": ((P.max(0) - P.min(0)) * 1000).tolist(),
                        "rot_dev_deg_median": float(np.median(ang)), "rot_dev_deg_p95": float(np.percentile(ang, 95)),
                        "rot_dev_deg_max": float(np.max(ang)), "mean_normal": normal.tolist(),
                        # only frames whose pose is on the majority branch (normal within 30 deg of the mean)
                        "off_branch_rate": float(np.mean([math.degrees(math.acos(np.clip(T[:3, 2] @ normal, -1, 1))) > 30 for T in Ts]))}
            if label == "cam":
                z = float(P[:, 2].mean())
                # first-order depth sensitivity of a square of known size: dZ/Z = d(side)/side
                r["cam"]["depth_from_jitter_mm"] = z * r["corner_jitter_px"] / max(r["side_px_median"], 1) * 1000
        out["markers"][str(mid)] = r
    return out


def recording_frames(d: Path):
    src = recorder.ReplaySource(d, realtime=False)
    K, dist = np.array(src.meta["intrinsics"]["K"]), np.array(src.meta["intrinsics"]["dist_coeffs"])
    return ((f, r["t"]) for f, r in src), K, dist, src.meta


def print_report(rep: dict) -> None:
    print(f"frames {rep['frames']}")
    for mid, r in rep["markers"].items():
        print(f"ID {mid}: det {r['detection_rate'] * 100:.0f}%  side {r['side_px_median']:.1f}px  corner jitter {r['corner_jitter_px']:.3f}px  "
              f"reproj {r['reproj_px_median']:.3f}px  ambiguous {r['ambiguous_rate'] * 100:.0f}%  flip {r['flip_rate'] * 100:.0f}%")
        for k in ("cam", "world"):
            if k in r:
                c = r[k]
                print(f"   {k:5}: std xyz {np.round(c['std_mm'], 2).tolist()} mm, p-p {np.round(c['p2p_mm'], 1).tolist()} mm, "
                      f"rot dev med {c['rot_dev_deg_median']:.2f} p95 {c['rot_dev_deg_p95']:.2f} max {c['rot_dev_deg_max']:.2f} deg, "
                      f"off-branch {c['off_branch_rate'] * 100:.0f}%" + (f", depth from jitter ~{c['depth_from_jitter_mm']:.2f} mm" if k == "cam" else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("recording", type=Path)
    c = sub.add_parser("compare"); c.add_argument("a", type=Path); c.add_argument("b", type=Path)
    c.add_argument("--id", type=int, required=True); c.add_argument("--known-mm", type=float, required=True)
    c.add_argument("--axis", default="any", choices=["x", "y", "z", "any"])
    f = sub.add_parser("frames"); f.add_argument("dir", type=Path); f.add_argument("--intrinsics", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=HERE / "markers.toml")
    args = ap.parse_args()
    cfg = pose.load_config(args.config)
    if args.cmd == "stats":
        frames, K, dist, meta = recording_frames(args.recording)
        rep = analyse(frames, K, dist, cfg)
        rep.update(recording=str(args.recording), image_size=meta["image_size"], intrinsics_source=meta["intrinsics"]["source"])
        (args.recording / "accuracy.json").write_text(json.dumps(rep, indent=1))
        print_report(rep)
    elif args.cmd == "frames":
        k = json.loads(args.intrinsics.read_text())
        K, dist = np.array(k["K"]), np.array(k["dist_coeffs"] if "dist_coeffs" in k else k["dist"])
        paths = sorted(args.dir.glob("*.png"))
        rep = analyse(((cv2.imread(str(p)), i / 10) for i, p in enumerate(paths)), K, dist, cfg)
        rep.update(image_size=list(cv2.imread(str(paths[0])).shape[1::-1]))
        (args.dir / "accuracy.json").write_text(json.dumps(rep, indent=1))
        print_report(rep)
    else:
        res = []
        for d in (args.a, args.b):
            frames, K, dist, _ = recording_frames(d)
            r = analyse(frames, K, dist, cfg)["markers"].get(str(args.id))
            if not r or "world" not in r:
                sys.exit(f"{d}: ID {args.id} has no world-frame poses (ID 2 must be visible and ok)")
            res.append(np.array(r["world"]["mean_mm"]))
        dvec = res[1] - res[0]
        meas = {"x": abs(dvec[0]), "y": abs(dvec[1]), "z": abs(dvec[2]), "any": float(np.linalg.norm(dvec))}[args.axis]
        print(f"ID {args.id} moved {np.round(dvec, 2).tolist()} mm in world; measured {meas:.2f} mm vs known {args.known_mm} mm "
              f"→ scale {meas / args.known_mm:.4f} ({(meas - args.known_mm):+.2f} mm)")


if __name__ == "__main__":
    main()
