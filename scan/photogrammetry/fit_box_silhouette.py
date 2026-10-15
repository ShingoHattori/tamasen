"""写真の物体マスク（GrabCut）と ARKit のカメラ姿勢だけから、直方体の姿勢と寸法を当てはめる（深度・メッシュ不使用）。
目的関数: 各フレームで直方体を投影した凸包とマスクの 1 − IoU の平均。--fix-dims で寸法を固定した当てはめも比べる。
使い方: fit_box_silhouette.py <session> <masks> --init <mesh_in_world.stl または x,y,z> --frames 0-228
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
SC = 0.25  # 1/4 解像度で比べる


def load(session, masks, frames):
    out = []
    for i in frames:
        mp = os.path.join(masks, f"{i:06d}.png")
        jp = os.path.join(session, "frames", f"{i:06d}.json")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            continue
        j = json.load(open(jp))
        m = cv2.imread(mp, 0)
        m = cv2.resize(m, (int(m.shape[1] * SC), int(m.shape[0] * SC)), interpolation=cv2.INTER_AREA) > 127
        K = np.array(j["K"]).reshape(3, 3) * SC; K[2, 2] = 1
        Tc = np.linalg.inv(np.array(j["T_world_cam_arkit"]).reshape(4, 4) @ FLIP)
        out.append((m, K, Tc))
    return out


def corners(x):
    c = np.array([[a, b, d] for a in (-.5, .5) for b in (-.5, .5) for d in (-.5, .5)]) * x[6:9]
    return c @ Rotation.from_rotvec(x[3:6]).as_matrix().T + x[:3]


def cost(x, data):
    C = corners(x)
    tot = 0
    for m, K, Tc in data:
        Y = C @ Tc[:3, :3].T + Tc[:3, 3]
        if (Y[:, 2] <= 0.02).any():
            tot += 1; continue
        p = (Y / Y[:, 2:]) @ K.T
        h = cv2.convexHull(p[:, :2].astype(np.float32))
        b = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(b, np.round(h * 4).astype(np.int32), 1, shift=2)
        b = b > 0
        tot += 1 - (b & m).sum() / max((b | m).sum(), 1)
    return tot / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("masks")
    ap.add_argument("--frames", default="0-228")
    ap.add_argument("--center", required=True, help="初期中心 x,y,z（m、ARKit 世界）")
    ap.add_argument("--dims", default="0.08,0.07,0.035", help="初期寸法 m")
    ap.add_argument("--fix-dims", default="", help="寸法を固定して姿勢だけ当てる（並び替えて全 6 通り試す）")
    a = ap.parse_args()
    f0, f1 = map(int, a.frames.split("-"))
    data = load(a.session, a.masks, range(f0, f1 + 1, 3))
    c0 = np.array([float(v) for v in a.center.split(",")])
    best = None
    rng = np.random.default_rng(0)
    for trial in range(12):
        rv = Rotation.random(random_state=int(rng.integers(1 << 30))).as_rotvec()
        x0 = np.r_[c0, rv, [float(v) for v in a.dims.split(",")]]
        if a.fix_dims:
            dims = np.array([float(v) for v in a.fix_dims.split(",")])
            f = lambda y: cost(np.r_[y, dims], data)
            r = minimize(f, x0[:6], method="Powell", options={"maxiter": 4000, "xtol": 1e-4, "ftol": 1e-5})
            x = np.r_[r.x, dims]
        else:
            r = minimize(lambda y: cost(y, data), x0, method="Powell", options={"maxiter": 8000, "xtol": 1e-4, "ftol": 1e-5})
            x = r.x
        if best is None or r.fun < best[0]:
            best = (r.fun, x)
    fun, x = best
    print(json.dumps({"frames": len(data), "mean_IoU": round(1 - fun, 4), "dims_mm": np.round(np.sort(np.abs(x[6:9])) * 1000, 1).tolist(),
                      "center_mm": np.round(x[:3] * 1000, 1).tolist()}))


if __name__ == "__main__":
    main()
