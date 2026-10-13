#!/usr/bin/env python3
"""OAK-D / OAK-D-Lite: record colour and depth video.

Writes into output/<device>/:
  color.mp4     colour stream
  depth.mp4     colourised depth, aligned to the colour camera
  sbs.mp4       the two stacked horizontally

Usage: python record_video.py [seconds] [--device <id-or-name-substring>]
"""
import argparse
import os
import time

import cv2
import numpy as np
import depthai as dai

from oak_common import describe, mono_size, pick_device

SIZE = (1280, 720)
FPS = 30

# Fixed depth range for the colour map. Unlike the stills, a per-frame percentile
# stretch would make the video flicker as the scene changes, so pin it here.
DEPTH_MIN_MM, DEPTH_MAX_MM = 300, 6000


def colorise(depth: np.ndarray) -> np.ndarray:
    norm = np.clip((depth.astype(np.float32) - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seconds", nargs="?", type=float, default=10.0)
    ap.add_argument("--device", help="id or name substring, e.g. 'lite' or '1844'")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(describe(device))
    out_dir = os.path.join(args.out, device.getDeviceName().lower())
    os.makedirs(out_dir, exist_ok=True)

    mono = mono_size(device)
    print(f"mono {mono[0]}x{mono[1]} -> depth {SIZE[0]}x{SIZE[1]}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = SIZE
    color_out = cv2.VideoWriter(f"{out_dir}/color.mp4", fourcc, FPS, (w, h))
    depth_out = cv2.VideoWriter(f"{out_dir}/depth.mp4", fourcc, FPS, (w, h))
    sbs_out = cv2.VideoWriter(f"{out_dir}/sbs.mp4", fourcc, FPS, (w * 2, h))

    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=mono_left.requestOutput(mono, fps=FPS),
            right=mono_right.requestOutput(mono, fps=FPS),
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*SIZE)  # mandatory alongside setDepthAlign
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        color_q = cam_rgb.requestOutput(SIZE, dai.ImgFrame.Type.BGR888i, fps=FPS).createOutputQueue(
            maxSize=4, blocking=False)
        depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        # Warm up 3A before the recording starts, so frame 1 is not a dark flash.
        for _ in range(FPS):
            color_q.get(), depth_q.get()

        n, start = 0, time.monotonic()
        while time.monotonic() - start < args.seconds:
            color = color_q.get().getCvFrame()
            depth_vis = colorise(depth_q.get().getFrame())
            color_out.write(color)
            depth_out.write(depth_vis)
            sbs_out.write(np.hstack([color, depth_vis]))
            n += 1
        elapsed = time.monotonic() - start

    for writer in (color_out, depth_out, sbs_out):
        writer.release()

    print(f"{n} frames in {elapsed:.1f}s ({n / elapsed:.1f} fps) -> {out_dir}/")


if __name__ == "__main__":
    main()
