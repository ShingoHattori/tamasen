#!/usr/bin/env python3
"""OAK-D / OAK-D-Lite: capture one colour still and one depth still.

Writes into output/<device>/:
  color.png      BGR frame from CAM_A
  depth_raw.png  16-bit depth in millimetres, aligned to CAM_A, same size as color.png
  depth_vis.png  colourised depth, for looking at

Usage: python capture_still.py [--device <id-or-name-substring>]
"""
import argparse
import os

import cv2
import numpy as np
import depthai as dai

from oak_common import describe, mono_size, pick_device

SIZE = (1280, 720)  # colour and depth share this, so the two stills overlay 1:1
WARMUP_FRAMES = 30  # let auto-exposure / white balance settle before saving


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", help="id or name substring, e.g. 'lite' or '1844'")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(describe(device))
    out_dir = os.path.join(args.out, device.getDeviceName().lower())
    os.makedirs(out_dir, exist_ok=True)

    mono = mono_size(device)
    print(f"mono {mono[0]}x{mono[1]} -> depth {SIZE[0]}x{SIZE[1]}")

    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=mono_left.requestOutput(mono),
            right=mono_right.requestOutput(mono),
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
        )
        # Reproject depth into the colour camera's frame so the two stills line up.
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # Mandatory alongside setDepthAlign: without it the pipeline never emits a frame.
        stereo.setOutputSize(*SIZE)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        color_q = cam_rgb.requestOutput(SIZE, dai.ImgFrame.Type.BGR888i).createOutputQueue(
            maxSize=4, blocking=False)
        depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        color = depth = None
        for _ in range(WARMUP_FRAMES):
            color = color_q.get().getCvFrame()
            depth = depth_q.get().getFrame()

    cv2.imwrite(f"{out_dir}/color.png", color)
    cv2.imwrite(f"{out_dir}/depth_raw.png", depth.astype(np.uint16))

    valid = depth[depth > 0]
    lo, hi = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (0.0, 1.0)
    norm = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0  # black out pixels with no disparity match
    cv2.imwrite(f"{out_dir}/depth_vis.png", vis)

    print(f"color {color.shape}  depth {depth.shape}  -> {out_dir}/")
    if valid.size:
        print(f"depth {valid.min()}-{valid.max()} mm, valid {100 * valid.size / depth.size:.1f}%")


if __name__ == "__main__":
    main()
