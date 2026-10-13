#!/usr/bin/env python3
"""Help decide a block marker's mount (corner + x_axis) from a saved still. Decides nothing itself.

    python mount_check.py output/snapshots/<time> --id 3
    python mount_check.py output/snapshots/<time> --pair      # ID 0 and ID 3 in the same still

--id: undistorts raw.png and warps it into the marker's own plane at true scale (4 px/mm, marker
centre in the middle, x right / y up as printed), so the face carrying the marker can be compared
with the face outlines that each mount candidate implies. Candidates with the same outline in the
marker plane (a face drawing turned by 180 deg) are shown together: one face alone cannot tell them
apart; which long edge borders the other marked face decides it (use --pair, or say which edge it
is). Writes mount_<id>.png into the still's directory.

--pair: for every combination of the ID 0 and ID 3 candidates, the block pose from ID 0 vs from
ID 3 in this still. Only combinations that agree are physically possible. Writes mount_pair.json.

The measured values (marker size, offsets, block size) come from markers.toml; only corner and
x_axis are enumerated. Nothing is written to markers.toml.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

import pose

HERE = Path(__file__).resolve().parent
PX_PER_MM = 4
HALF_MM = 95


def candidates(spec_raw: dict, cfg: pose.Config, mid: int):
    m = spec_raw["marker"][str(mid)]
    size = cfg.markers[mid].size_m
    dx, dy = float(m["offset_x_mm"]) * pose.MM, float(m["offset_y_mm"]) * pose.MM
    out = []
    for corner in pose.CORNERS:
        for ax in pose.AXIS_DIRS:
            T = pose.block_mount(m["face"], corner, ax, size, dx, dy, cfg.block_size.value)
            out.append((corner, ax, T))
    return out


def face_outline_in_marker(T_obj_marker: np.ndarray, face: str, block_size: np.ndarray) -> np.ndarray:
    """Corners of the marked face, in marker-plane coordinates (m)."""
    normal, right, up, (ia, ib) = pose.FACES[face]
    W, H = block_size[ia], block_size[ib]
    c = normal * (block_size @ np.abs(normal)) / 2
    pts = [c + su * W / 2 * right + sv * H / 2 * up for su, sv in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
    Tm = pose.invert(T_obj_marker)
    return np.array([(Tm[:3, :3] @ p + Tm[:3, 3])[:2] for p in pts])


def rectify(raw, corners, size, K, dist):
    und = cv2.undistort(raw, K, dist)
    uc = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    obj = pose.marker_object_points(size)[:, :2]          # m, x right / y up
    n = 2 * HALF_MM * PX_PER_MM
    # canvas pixel = (x*1000*ppm + n/2, -y*1000*ppm + n/2)
    dst = np.array([[x * 1000 * PX_PER_MM + n / 2, -y * 1000 * PX_PER_MM + n / 2] for x, y in obj], np.float32)
    H = cv2.getPerspectiveTransform(uc.astype(np.float32), dst)
    return cv2.warpPerspective(und, H, (n, n)), n


def to_canvas(p, n):
    return (int(round(p[0] * 1000 * PX_PER_MM + n / 2)), int(round(-p[1] * 1000 * PX_PER_MM + n / 2)))


def single(snap: Path, mid: int, cfg: pose.Config, raw_cfg: dict) -> Path:
    st = json.loads((snap / "state.json").read_text())
    m = next((m for m in st["markers"] if m["id"] == mid), None)
    if m is None:
        raise SystemExit(f"ID {mid} はこの静止画に写っていない")
    raw = cv2.imread(str(snap / "raw.png"))
    K, dist = np.array(st["intrinsics"]["K"]), np.array(st["intrinsics"]["dist_coeffs"])
    size = cfg.markers[mid].size_m
    base, n = rectify(raw, np.array(m["corners_px"], np.float32), size, K, dist)
    for k in range(-HALF_MM, HALF_MM + 1, 10):                 # 10 mm grid
        c = (60, 60, 60) if k else (120, 120, 120)
        q = int(k * PX_PER_MM + n / 2)
        cv2.line(base, (q, 0), (q, n), c, 1)
        cv2.line(base, (0, q), (n, q), c, 1)
    face = raw_cfg["marker"][str(mid)]["face"]
    groups: dict[tuple, list] = {}
    for corner, ax, T in candidates(raw_cfg, cfg, mid):
        poly = face_outline_in_marker(T, face, cfg.block_size.value)
        key = tuple(sorted(tuple(np.round(p * 1000, 1)) for p in poly))
        groups.setdefault(key, []).append((corner, ax, poly))
    tiles = []
    for i, (key, cands) in enumerate(sorted(groups.items())):
        t = base.copy()
        poly = cands[0][2]
        cv2.polylines(t, [np.array([to_canvas(p, n) for p in poly], np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        label = " / ".join(f"{c} x={a}" for c, a, _ in cands)
        cv2.rectangle(t, (0, 0), (n, 26), (0, 0, 0), -1)
        cv2.putText(t, f"#{i + 1}: {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(base))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    head = np.zeros((60, sheet.shape[1], 3), np.uint8)
    cv2.putText(head, f"ID {mid} marker plane, 4 px/mm, 10 mm grid, x right / y up as printed. Green = face outline implied by the candidate.",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, f"face {face}, block {cfg.block_size.value * 1000} mm. Pairs in one tile differ by a 180 deg turn of the face drawing.",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out = snap / f"mount_{mid}.png"
    cv2.imwrite(str(out), np.vstack([head, sheet]))
    return out


def pair(snap: Path, cfg: pose.Config, raw_cfg: dict) -> Path:
    st = json.loads((snap / "state.json").read_text())
    ms = {m["id"]: m for m in st["markers"] if m["T_cam_marker"]}
    if 0 not in ms or 3 not in ms:
        raise SystemExit("ID 0 と ID 3 の両方が写っている静止画が必要")
    rows = []
    for (c0, a0, T0), (c3, a3, T3) in itertools.product(candidates(raw_cfg, cfg, 0), candidates(raw_cfg, cfg, 3)):
        for s0, s3 in itertools.product(("T_cam_marker", "T_cam_marker_alt"), repeat=2):
            if ms[0][s0] is None or ms[3][s3] is None:
                continue
            A = np.array(ms[0][s0]) @ pose.invert(T0)
            B = np.array(ms[3][s3]) @ pose.invert(T3)
            c = pose.compare(A, B)
            rows.append({"id0": f"{c0} x={a0}", "id3": f"{c3} x={a3}", "ippe0": s0, "ippe3": s3,
                         "dt_mm": c["dt_m"] * 1000, "drot_deg": math.degrees(c["drot_rad"])})
    rows.sort(key=lambda r: r["dt_mm"] + r["drot_deg"])
    out = snap / "mount_pair.json"
    out.write_text(json.dumps({"status": {i: ms[i]["status"] for i in (0, 3)}, "ranked": rows[:20]}, indent=1))
    for r in rows[:8]:
        print(f"{r['dt_mm']:7.1f} mm {r['drot_deg']:6.1f} deg  ID0 {r['id0']:<26} ID3 {r['id3']:<26} {r['ippe0'][-3:]} {r['ippe3'][-3:]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--id", type=int)
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--config", type=Path, default=HERE / "markers.toml")
    args = ap.parse_args()
    import tomllib
    raw_cfg = tomllib.loads(args.config.read_text())
    cfg = pose.load_config(args.config)
    if args.id is not None:
        print(single(args.snapshot, args.id, cfg, raw_cfg))
    if args.pair:
        print(pair(args.snapshot, cfg, raw_cfg))


if __name__ == "__main__":
    main()
