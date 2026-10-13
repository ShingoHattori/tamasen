#!/usr/bin/env python3
"""Own intrinsic calibration of the colour camera, as an alternative to the factory EEPROM values.

    python calib_intrinsics.py board  --out calib/charuco_A4.png        # print at 100 %, measure a square
    python calib_intrinsics.py calibrate output/snapshots/*/raw.png --square-mm 30.0 --out calib/intrinsics.json
    python vision.py --intrinsics calib/intrinsics.json                  # use it

Board: ChArUco, DICT_5X5_100 (deliberately not the 4x4 dictionary of the scene markers), 7 x 5
squares. Print it, measure the real square size with a ruler and pass that with --square-mm — a
printer's scaling is not trusted. Take >= 12 stills with vision.py ("静止画を保存") at the same
--size the calibration will be used with, covering the image corners and tilting the board.

The output records the image size and camera id; vision.py refuses it for another size or
camera. Nothing here invents a focal length: if too few views are usable, it stops.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

DICT = cv2.aruco.DICT_5X5_100
SQUARES = (7, 5)
MARKER_RATIO = 0.75
MIN_VIEWS = 12
MIN_CORNERS = 8


def make_board(square_m: float = 0.030) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(SQUARES, square_m, square_m * MARKER_RATIO, cv2.aruco.getPredefinedDictionary(DICT))


def cmd_board(args) -> None:
    board = make_board()
    px_per_mm = 300 / 25.4                            # 300 dpi
    size = (int(SQUARES[0] * 30 * px_per_mm), int(SQUARES[1] * 30 * px_per_mm))
    img = board.generateImage(size, marginSize=int(10 * px_per_mm))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"{args.out}: {SQUARES[0]}x{SQUARES[1]} squares of 30 mm at 300 dpi. Print at 100 %, then MEASURE a square.")


def detect(paths, board):
    det = cv2.aruco.CharucoDetector(board)
    obj_all, img_all, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if size is None:
            size = img.shape[::-1]
        elif img.shape[::-1] != size:
            sys.exit(f"{p}: size {img.shape[::-1]} differs from {size}")
        cc, ci, _, _ = det.detectBoard(img)
        if ci is None or len(ci) < MIN_CORNERS:
            print(f"  skip {p} ({0 if ci is None else len(ci)} corners)")
            continue
        o, i = board.matchImagePoints(cc, ci)
        obj_all.append(o)
        img_all.append(i)
        used.append(str(p))
    return obj_all, img_all, used, size


def cmd_calibrate(args) -> None:
    board = make_board(args.square_mm / 1000)
    obj, img, used, size = detect(args.images, board)
    print(f"usable views: {len(used)} / {len(args.images)}")
    if len(used) < MIN_VIEWS:
        sys.exit(f"need at least {MIN_VIEWS} usable views; not calibrating")
    flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        pr, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    out = {"image_size": list(size), "K": K.tolist(), "dist": dist.ravel().tolist(),
           "model": "opencv rational (k1..k6, p1, p2)" if args.rational else "opencv (k1, k2, p1, p2, k3)",
           "rms_px": float(rms), "per_view_rms_px": per_view, "n_views": len(used), "images": used,
           "square_mm": args.square_mm, "board": f"ChArUco {SQUARES} DICT_5X5_100", "camera_id": args.camera_id,
           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "calibrated"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"rms {rms:.3f} px, fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f} -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board")
    b.add_argument("--out", type=Path, default=Path("calib/charuco_A4.png"))
    c = sub.add_parser("calibrate")
    c.add_argument("images", nargs="+", type=Path)
    c.add_argument("--square-mm", type=float, required=True, help="measured on the print")
    c.add_argument("--camera-id", default="1844301091A70A1300")
    c.add_argument("--rational", action="store_true", help="8-coefficient model (needs many varied views)")
    c.add_argument("--out", type=Path, default=Path("calib/intrinsics.json"))
    args = ap.parse_args()
    {"board": cmd_board, "calibrate": cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
