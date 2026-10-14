#!/usr/bin/env python3
"""Pictures of the ID 1 marker mount in the simulator (offscreen, nothing sent anywhere).

    MUJOCO_GL=egl python render_mount.py [--check output/mount_check_<time>.json] [--out output/mount_render]

Close-ups of the gripper with and without the mount, the whole arm at the mock pose, and the closest
approaches found by check_mount.py (mount in light blue, its collision boxes as translucent overlays).
"""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import cv2

import scene

HERE = Path(__file__).resolve().parent


def build(settings, enabled):
    s = copy.deepcopy(settings)
    s["marker_mount"]["enabled"] = enabled
    m = mujoco.MjModel.from_xml_string(scene.build_xml(s))
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "marker_mount_visual")
    if g >= 0:
        m.geom_rgba[g] = [0.35, 0.75, 1.0, 1.0]
    m.vis.scale.framelength, m.vis.scale.framewidth = 0.035, 0.0025
    return m, scene.Ids(m)


def render(m, ids, q, finger, view, collision=False, size=(960, 720), label="", obstacles=True, frames=True):
    d = mujoco.MjData(m)
    d.qpos[ids.arm_qadr] = q
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = finger, -finger
    d.qpos[ids.block_q:ids.block_q + 7] = [10, 10, 1, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, size[1], size[0])
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for k in (0, 1, 2) + ((5,) if obstacles else ()):
        opt.geomgroup[k] = 1
    if collision:
        opt.geomgroup[4] = 1
    opt.sitegroup[:] = 0
    opt.sitegroup[1] = 1
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE if frames else mujoco.mjtFrame.mjFRAME_NONE
    cam = mujoco.MjvCamera()
    target = view.get("lookat", "end")
    # "end": a point between the mount and the fingertips, the same in both models
    cam.lookat[:] = d.xpos[ids.end_body] + d.xmat[ids.end_body].reshape(3, 3) @ np.array([-0.07, 0, 0.02]) \
        if target == "end" else np.array(target)
    cam.distance, cam.azimuth, cam.elevation = view["dist"], view["az"], view["el"]
    r.update_scene(d, cam, opt)
    img = r.render()[:, :, ::-1].copy()
    r.close()
    cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="check_mount.py output (closest approaches)")
    ap.add_argument("--out", default=str(HERE / "output" / "mount_render"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = scene.load_settings()
    mA, idsA = build(s, True)
    mB, idsB = build(s, False)
    q_mock = np.array(s["mock_joints"]["q_rad"], float)
    close = [dict(dist=0.30, az=160, el=-45), dict(dist=0.30, az=60, el=-30), dict(dist=0.30, az=250, el=-20)]
    tiles = []
    for v in close:
        a = render(mA, idsA, q_mock, 0.02, v, label="with mount", obstacles=False)
        b = render(mB, idsB, q_mock, 0.02, v, label="without", obstacles=False)
        tiles.append(np.hstack([a, b]))
    cv2.imwrite(str(out / "closeup.png"), np.vstack(tiles))
    cv2.imwrite(str(out / "closeup_collision.png"),
                render(mA, idsA, q_mock, 0.02, close[0], collision=True, label="collision boxes (group 4)", obstacles=False))
    whole = [render(mA, idsA, q_mock, 0.02, dict(lookat=[0.12, 0, 0.25], dist=1.25, az=a, el=-15), label=f"mock pose, azimuth {a}",
                    frames=False) for a in (150, 230)]
    folded = [render(mA, idsA, np.zeros(6), 0.0, dict(lookat=[0.0, 0, 0.18], dist=0.6, az=a, el=-25), label="all joints 0 (folded)",
                     obstacles=False) for a in (200, 120)]
    cv2.imwrite(str(out / "whole.png"), np.vstack([np.hstack(whole), np.hstack(folded)]))
    if args.check:
        rep = json.loads(Path(args.check).read_text())
        imgs = []
        for b, v in list(rep["random"]["closest_per_body_mm"].items())[:4]:
            q = np.array(v["q"])
            for vv in (dict(dist=0.45, az=200, el=-25), dict(dist=0.45, az=110, el=-30)):
                imgs.append(render(mA, idsA, q, 0.02, vv, collision=True, size=(640, 480),
                                   label=f"closest to {b}: {v['min_mm']:.1f} mm", obstacles=b.startswith("obst")))
        rows = [np.hstack(imgs[i:i + 2]) for i in range(0, len(imgs), 2)]
        cv2.imwrite(str(out / "closest.png"), np.vstack(rows))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
