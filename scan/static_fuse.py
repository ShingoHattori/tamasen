"""物体を置いたまま iPad（背面 LiDAR + ARKit）を動かして撮った記録を、ARKit の世界座標で統合する（試作）。

物体は区間ごとに静止している前提（裏返しの前後で区間を分ける: --phases 0-228,234-376）。
各区間: 深度を ARKit のカメラ姿勢で世界座標へ → TSDF → 机の平面より上・物体のまわりだけ切り出す。
使い方: scan/.venv/bin/python scan/static_fuse.py <session> --phases 0-228,234-376 --center-id 3 --out DIR
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit カメラ（y 上, z 後ろ）→ OpenCV（y 下, z 前）


def load(session, i):
    b = os.path.join(session, "frames", f"{i:06d}")
    if not os.path.exists(b + ".json"):
        return None
    j = json.load(open(b + ".json"))
    dw, dh = j["depth_w"], j["depth_h"]
    d = np.fromfile(b + ".depth", np.float32).reshape(dh, dw)
    if os.path.exists(b + ".conf"):
        c = np.fromfile(b + ".conf", np.uint8).reshape(dh, dw)
        d = np.where(c >= 2, d, 0)
    d = np.nan_to_num(d, nan=0, posinf=0)
    K = np.array(j["K"]).reshape(3, 3).copy()
    sx, sy = dw / j["K_ref"][0], dh / j["K_ref"][1]
    K[0, 0] *= sx; K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy; K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    T = np.array(j["T_world_cam_arkit"]).reshape(4, 4) @ FLIP
    img = cv2.resize(cv2.imread(b + ".jpg"), (dw, dh), interpolation=cv2.INTER_AREA)
    return d, K, T, img, j


def marker_center(session, frames, mid, size_m):
    """区間内でマーカー mid の四隅を多視点三角測量（ARKit 姿勢）して中心と法線を返す。"""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    obs = []
    for i in frames:
        b = os.path.join(session, "frames", f"{i:06d}")
        co, ids, _ = det.detectMarkers(cv2.imread(b + ".jpg"))
        if ids is None or mid not in ids.ravel():
            continue
        j = json.load(open(b + ".json"))
        K = np.array(j["K"]).reshape(3, 3)
        T = np.array(j["T_world_cam_arkit"]).reshape(4, 4) @ FLIP
        P = K @ np.linalg.inv(T)[:3]
        obs.append((P, co[list(ids.ravel()).index(mid)][0]))
    if len(obs) < 3:
        return None
    C = []
    for k in range(4):
        A = []
        for P, q in obs:
            A += [q[k][0] * P[2] - P[0], q[k][1] * P[2] - P[1]]
        X = np.linalg.svd(np.array(A))[2][-1]
        C.append(X[:3] / X[3])
    C = np.array(C)
    n = np.cross(C[1] - C[0], C[3] - C[0]); n /= np.linalg.norm(n)
    side = np.mean([np.linalg.norm(C[k] - C[(k + 1) % 4]) for k in range(4)])
    return C.mean(0), n, side, len(obs), C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--phases", required=True)
    ap.add_argument("--center-id", type=int, default=3)
    ap.add_argument("--radius", type=float, default=0.12, help="物体中心まわりの切り出し半径 m")
    ap.add_argument("--voxel", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for pi, ph in enumerate(a.phases.split(",")):
        f0, f1 = map(int, ph.split("-"))
        frames = list(range(f0, f1 + 1))
        # 机の平面（ID 2 があればその平面、なければ RANSAC）
        # 物体の中心: 指定マーカーの三角測量（なければ全マーカーの中心）
        mc = None
        for mid in (a.center_id, 0, 3):
            mc = marker_center(a.session, frames[::2], mid, 0.02)
            if mc is not None:
                print(f"phase {pi}: ID {mid} 中心 {np.round(mc[0] * 1000)} 法線 {np.round(mc[1], 2)} 一辺 {mc[2] * 1000:.1f} mm（{mc[3]} 視点）")
                break
        # 点の積み上げ（ARKit 姿勢をそのまま使う）。2 mm ボクセルで 3 フレーム以上に出た所だけ残す
        allp, allc, allf = [], [], []
        used = 0
        for i in frames:
            r = load(a.session, i)
            if r is None:
                continue
            d, K, T, img, j = r
            v, u = np.nonzero((d > 0.1) & (d < 1.2))
            z = d[v, u]
            X = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
            W = X @ T[:3, :3].T + T[:3, 3]
            allp.append(W); allc.append(img[v, u, ::-1] / 255.0); allf.append(np.full(len(W), i))
            used += 1
        W = np.concatenate(allp); Cc = np.concatenate(allc); Fi = np.concatenate(allf)
        key = np.floor(W / a.voxel).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        nfr = np.zeros(len(cnt)); np.add.at(nfr, inv.ravel(), 1)
        sums = np.zeros((len(cnt), 3)); np.add.at(sums, inv.ravel(), W)
        cols = np.zeros((len(cnt), 3)); np.add.at(cols, inv.ravel(), Cc)
        good = nfr >= 3
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sums[good] / nfr[good, None]))
        pc.colors = o3d.utility.Vector3dVector(cols[good] / nfr[good, None])
        P = np.asarray(pc.points)
        c = mc[0] if mc is not None else P.mean(0)
        keep = np.linalg.norm(P - c, axis=1) < a.radius
        pc = pc.select_by_index(np.where(keep)[0])
        # 机の平面を除く（世界 y が上。最大平面を RANSAC）
        pl, inl = pc.segment_plane(0.003, 3, 2000)
        nrm = np.array(pl[:3]); nrm *= np.sign(nrm[1])
        dist = np.asarray(pc.points) @ nrm + pl[3] * np.sign(pl[1])
        obj = pc.select_by_index(np.where(dist > 0.004)[0])
        lab = np.array(obj.cluster_dbscan(0.006, 20))
        if lab.size and lab.max() >= 0:
            big = np.bincount(lab[lab >= 0]).argmax()
            obj = obj.select_by_index(np.where(lab == big)[0])
        o3d.io.write_point_cloud(os.path.join(a.out, f"phase{pi}_object.ply"), obj)
        o3d.io.write_point_cloud(os.path.join(a.out, f"phase{pi}_scene.ply"), pc)
        # 机に沿った外接箱（机の法線 + 水平面内の最小外接長方形）
        Q = np.asarray(obj.points)
        e1 = np.cross(nrm, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        uv = np.c_[Q @ e1, Q @ e2].astype(np.float32)
        (cx, cy), (bw, bh), ang = cv2.minAreaRect(uv)
        hgt = np.percentile(Q @ nrm + pl[3] * np.sign(pl[1]), 99.5)
        print(f"phase {pi}: frames {used}, 物体 {len(Q)} 点, 机上の外接箱 {bw * 1000:.1f} x {bh * 1000:.1f} x 高さ {hgt * 1000:.1f} mm")


if __name__ == "__main__":
    main()
