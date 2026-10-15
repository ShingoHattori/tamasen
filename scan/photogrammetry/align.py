"""Object Capture のモデルを、ARKit の姿勢で実寸に直し、マーカー座標で書き出す。

1. 区間（物体が静止している間）ごとに、Object Capture のカメラ位置（モデル座標）と ARKit のカメラ位置（世界、m）を
   相似変換（Umeyama）で合わせる → 尺度 s と T_world_model。区間で尺度が揃っているかを確認する。
2. 各マーカーの四隅を、ARKit 姿勢で区間ごとに多視点三角測量し、モデル座標（実寸）へ移す。
3. 基準マーカーの座標系（中心が原点, x 右, y 上, z 外向き）でメッシュを書き出す（m）。
使い方: scan/.venv/bin/python scan/photogrammetry/align.py <session> <model_dir> --phases 0-228,236-376 --ref 0 --sizes 0:20,3:20
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import trimesh

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / (S ** 2).sum(1).mean()
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("model_dir")
    ap.add_argument("--phases", required=True)
    ap.add_argument("--ref", type=int, required=True, help="出力の基準マーカー ID")
    ap.add_argument("--sizes", default="", help="'ID:mm,…' 既知の一辺（照合用）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.model_dir
    poses = json.load(open(os.path.join(a.model_dir, "poses.json")))
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    sizes = {int(k): float(v) for k, v in (x.split(":") for x in a.sizes.split(",") if x)}
    report = {"phases": []}
    corners_model = {}   # id -> list of (4x3 corners in metric model frame, phase)
    scales = []
    for ph in a.phases.split(","):
        f0, f1 = map(int, ph.split("-"))
        A, B, obs = [], [], {}
        for i in range(f0, f1 + 1):
            b = os.path.join(a.session, "frames", f"{i:06d}")
            if not os.path.exists(b + ".json"):
                continue
            j = json.load(open(b + ".json"))
            Tw = np.array(j["T_world_cam_arkit"]).reshape(4, 4)
            if str(i) in poses:
                Tm = np.array(poses[str(i)]).reshape(4, 4).T   # 列優先 → 行列
                A.append(Tm[:3, 3]); B.append(Tw[:3, 3])
            co, ids, _ = det.detectMarkers(cv2.imread(b + ".jpg"))
            if ids is None:
                continue
            P = np.array(j["K"]).reshape(3, 3) @ np.linalg.inv(Tw @ FLIP)[:3]
            for mid, q in zip(ids.ravel(), co):
                obs.setdefault(int(mid), []).append((P, q[0]))
        A, B = np.array(A), np.array(B)
        s, R, t = umeyama(A, B)
        res = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        scales.append(s)
        ph_rep = {"frames": ph, "pairs": len(A), "scale_model_to_m": s, "cam_fit_mm_median": float(np.median(res) * 1000),
                  "cam_fit_mm_p90": float(np.percentile(res, 90) * 1000), "markers": {}}
        # T_world_model（相似）: x_w = s R x_m + t → 実寸モデル座標 x_mm = s x_m なので x_w = R x_mm + t
        for mid, L in obs.items():
            if len(L) < 5:
                continue
            C = []
            for k in range(4):
                M = []
                for P, q in L:
                    M += [q[k][0] * P[2] - P[0], q[k][1] * P[2] - P[1]]
                X = np.linalg.svd(np.array(M))[2][-1]
                C.append(X[:3] / X[3])
            C = np.array(C)
            err = []
            for P, q in L:
                x = (P @ np.c_[C, np.ones(4)].T).T
                err.append(np.linalg.norm(x[:, :2] / x[:, 2:] - q, axis=1).mean())
            Cm = (R.T @ (C - t).T).T   # 実寸モデル座標
            side = float(np.mean([np.linalg.norm(C[k] - C[(k + 1) % 4]) for k in range(4)]) * 1000)
            ph_rep["markers"][mid] = {"views": len(L), "side_mm": side, "reproj_px_median": float(np.median(err))}
            if np.median(err) < 6:
                corners_model.setdefault(mid, []).append(Cm)
        report["phases"].append(ph_rep)
        print(json.dumps(ph_rep, ensure_ascii=False))
    s = float(np.mean(scales))
    report["scale"] = s
    mesh_path = sorted(glob.glob(os.path.join(a.model_dir, "baked_mesh_*.obj")))[0]
    m = trimesh.load(mesh_path, process=False)
    m.apply_scale(s)
    # 基準マーカーの姿勢（区間で平均）
    Cs = corners_model.get(a.ref)
    if not Cs:
        raise SystemExit(f"基準マーカー {a.ref} を三角測量できなかった")
    if len(Cs) > 1:
        report["ref_phase_disagreement_mm"] = float(np.abs(Cs[0] - Cs[1]).max() * 1000)
    C = np.mean(Cs, 0)
    o = C.mean(0)
    x = (C[1] + C[2]) / 2 - (C[0] + C[3]) / 2; x /= np.linalg.norm(x)
    y = (C[0] + C[1]) / 2 - (C[3] + C[2]) / 2; y -= x * (x @ y); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    T_model_ref = np.eye(4); T_model_ref[:3, :3] = np.c_[x, y, z]; T_model_ref[:3, 3] = o
    m.apply_transform(np.linalg.inv(T_model_ref))
    # 体の向きが揃うよう、他のマーカーも基準座標で報告
    report["markers_in_ref_mm"] = {}
    for mid, L in corners_model.items():
        for k, Cm in enumerate(L):
            Cr = (np.linalg.inv(T_model_ref) @ np.c_[Cm, np.ones(4)].T).T[:, :3]
            report["markers_in_ref_mm"][f"{mid}#{k}"] = np.round(Cr.mean(0) * 1000, 1).tolist()
    os.makedirs(out, exist_ok=True)
    m.export(os.path.join(out, f"mesh_m{a.ref}.obj"))
    m.export(os.path.join(out, f"mesh_m{a.ref}.stl"))
    m.convex_hull.export(os.path.join(out, f"hull_m{a.ref}.stl"))
    o3 = m.bounding_box_oriented
    report["obb_mm"] = np.round(np.sort(o3.primitive.extents * 1000), 1).tolist()
    report["extents_in_ref_mm"] = np.round(m.extents * 1000, 1).tolist()
    json.dump(report, open(os.path.join(out, "align.json"), "w"), indent=1, ensure_ascii=False)
    print("scale", s, "OBB mm", report["obb_mm"], "markers", report["markers_in_ref_mm"])


if __name__ == "__main__":
    main()
