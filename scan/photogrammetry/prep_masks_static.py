"""置いた物体を iPad（背面 LiDAR + ARKit）で撮った記録から、Object Capture 用の物体マスクを作る。

各フレームの深度を ARKit 姿勢で世界座標へ戻し、「区間ごとの物体中心から --radius 以内」かつ「机の面より --above 以上の高さ」の
画素を物体とする（世界の y が上）。区間の中心は static_fuse.py の phase<i>_object.ply の重心、机の高さは同じ区間の scene の最頻の y。
使い方: scan/.venv/bin/python scan/photogrammetry/prep_masks_static.py <session> <static_dir> <out> --phases 0-228,236-376
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("static_dir")
    ap.add_argument("out")
    ap.add_argument("--phases", required=True)
    ap.add_argument("--radius", type=float, default=0.075)
    ap.add_argument("--above", type=float, default=0.003)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for pi, ph in enumerate(a.phases.split(",")):
        f0, f1 = map(int, ph.split("-"))
        obj = np.asarray(o3d.io.read_point_cloud(os.path.join(a.static_dir, f"phase{pi}_object.ply")).points)
        scene = np.asarray(o3d.io.read_point_cloud(os.path.join(a.static_dir, f"phase{pi}_scene.ply")).points)
        c = np.median(obj, 0)
        h, e = np.histogram(scene[:, 1], bins=np.arange(scene[:, 1].min(), scene[:, 1].max() + 0.002, 0.002))
        table_y = e[h.argmax()] + 0.001
        print(f"phase {pi}: 中心 {np.round(c * 1000)} mm、机の高さ y {table_y * 1000:.0f} mm")
        for i in range(f0, f1 + 1):
            b = os.path.join(a.session, "frames", f"{i:06d}")
            if not os.path.exists(b + ".json"):
                continue
            j = json.load(open(b + ".json"))
            dw, dh, vw, vh = j["depth_w"], j["depth_h"], j["video_w"], j["video_h"]
            d = np.nan_to_num(np.fromfile(b + ".depth", np.float32).reshape(dh, dw))
            K = np.array(j["K"]).reshape(3, 3).copy()
            sx, sy = dw / j["K_ref"][0], dh / j["K_ref"][1]
            fx, fy = K[0, 0] * sx, K[1, 1] * sy
            cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
            T = np.array(j["T_world_cam_arkit"]).reshape(4, 4) @ FLIP
            v, u = np.mgrid[0:dh, 0:dw]
            X = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], -1).reshape(-1, 3)
            W = X @ T[:3, :3].T + T[:3, 3]
            m = ((d.reshape(-1) > 0.05) & (np.linalg.norm(W - c, axis=1) < a.radius) & (W[:, 1] > table_y + a.above)).reshape(dh, dw)
            m = m.astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nl, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if nl <= 1:
                continue
            m = (lab == 1 + st[1:, 4].argmax()).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((3, 3), np.uint8))
            m = cv2.resize(m, (vw, vh), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(a.out, f"{i:06d}.png"), (m > 127).astype(np.uint8) * 255)
            n += 1
    print(f"masks {n} -> {a.out}")


if __name__ == "__main__":
    main()
