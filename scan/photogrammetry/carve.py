"""Object Capture のメッシュを、写真の物体マスク（GrabCut）による visual hull で削る。

Object Capture の各カメラ姿勢（poses.json、モデル座標）と記録の K でボクセル中心を投影し、
「写っているフレームのうち --max-out 割より多くでマスクの外」のボクセルを捨てる。メッシュの内側 ∩ visual hull を
マーチングキューブでメッシュに戻す（メッシュの凹みは残り、写真からはみ出した膨らみは削れる）。
手ぶれの大きいフレームはラプラシアン分散で除く。出力はモデル座標（尺度は markers_on_mesh.py で後から決める）。
使い方: carve.py <session> <model_dir> <masks> <out_dir> [--res 220]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import open3d as o3d
import trimesh
from skimage import measure

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("model_dir"); ap.add_argument("masks"); ap.add_argument("out")
    ap.add_argument("--res", type=int, default=220, help="長辺方向のボクセル数")
    ap.add_argument("--max-out", type=float, default=0.03, help="マスク外と判定されてよいフレームの割合")
    ap.add_argument("--tol-px", type=int, default=2, help="マスクを広げる画素数（輪郭の誤差の許容）")
    ap.add_argument("--sharp-pct", type=float, default=25, help="ラプラシアン分散が下位この%%のフレームは使わない")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mesh_path = sorted(glob.glob(os.path.join(a.model_dir, "baked_mesh_*.obj")))[0]
    m = trimesh.load(mesh_path, force="mesh", process=False)
    poses = json.load(open(os.path.join(a.model_dir, "poses.json")))
    lo, hi = m.bounds
    pad = 0.05 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    vox = (hi - lo).max() / a.res
    axes = [np.arange(lo[k], hi[k], vox) + vox / 2 for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    # メッシュの内側
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))))
    inside = sc.compute_occupancy(o3d.core.Tensor(G.astype(np.float32))).numpy() > 0
    # 使うフレーム（シャープなもの）
    frames = []
    for k, v in poses.items():
        mp = os.path.join(a.masks, f"{int(k):06d}.png")
        if not os.path.exists(mp):
            continue
        img = cv2.imread(os.path.join(a.session, "frames", f"{int(k):06d}.jpg"), 0)
        msk = cv2.imread(mp, 0)
        ys, xs = np.nonzero(msk)
        if len(xs) < 100:
            continue
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        frames.append((int(k), cv2.Laplacian(crop, cv2.CV_64F).var()))
    thr = np.percentile([s for _, s in frames], a.sharp_pct)
    use = [k for k, s in frames if s >= thr]
    seen = np.zeros(len(G), np.int32); out = np.zeros(len(G), np.int32)
    ious = []
    ker = np.ones((2 * a.tol_px + 1, 2 * a.tol_px + 1), np.uint8)
    for k in use:
        j = json.load(open(os.path.join(a.session, "frames", f"{k:06d}.json")))
        K = np.array(j["K"]).reshape(3, 3)
        T = np.array(poses[str(k)]).reshape(4, 4).T @ FLIP      # モデル ← OpenCV カメラ
        Tc = np.linalg.inv(T)
        msk = cv2.imread(os.path.join(a.masks, f"{k:06d}.png"), 0) > 127
        mskd = cv2.dilate(msk.astype(np.uint8), ker) > 0
        Y = G @ Tc[:3, :3].T + Tc[:3, 3]
        ok = Y[:, 2] > 1e-6
        p = (Y[:, :2] / np.where(ok, Y[:, 2], 1)[:, None]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        u, v = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        inimg = ok & (u >= 0) & (v >= 0) & (u < msk.shape[1]) & (v < msk.shape[0])
        seen += inimg
        idx = np.where(inimg)[0]
        out[idx] += ~mskd[v[idx], u[idx]]
        # メッシュ内側の投影と写真のマスクの IoU（姿勢の確認）
        sil = np.zeros(msk.shape, bool); ii = idx[inside[idx]]
        sil[v[ii], u[ii]] = True
        sil = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ious.append((sil & msk).sum() / max((sil | msk).sum(), 1))
    hull = (seen > 0) & (out <= a.max_out * np.maximum(seen, 1))
    keep = inside & hull
    vol = keep.reshape([len(x) for x in axes]).astype(np.float32)
    vol = np.pad(vol, 1)
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts = (verts - 1) * vox + lo
    cm = trimesh.Trimesh(verts, faces)
    if len(cm.faces) > 20000:
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cm.vertices), o3d.utility.Vector3iVector(cm.faces))
        om = om.simplify_quadric_decimation(20000)
        cm = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles))
    # 元メッシュの UV を最近点で引き継ぐのは省略（形の確認用）。テクスチャ付きは元メッシュを使う
    cm.export(os.path.join(a.out, "carved.obj"))
    rep = {"frames_used": len(use), "frames_total": len(frames), "pose_check_IoU_median": float(np.median(ious)),
           "voxel_units": float(vox), "inside_vox": int(inside.sum()), "kept_vox": int(keep.sum()),
           "removed_frac": float(1 - keep.sum() / max(inside.sum(), 1))}
    json.dump(rep, open(os.path.join(a.out, "carve.json"), "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
