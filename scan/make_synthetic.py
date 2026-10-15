#!/usr/bin/env python3
"""合成セッションを作り、process.py を走らせて真値と比べる（実機データなしでの試験）。

  scan/.venv/bin/python scan/make_synthetic.py [--out scan/output/synthetic] [--frames 120] [--run]

作るもの（アプリの書き出しと同じ形式）:
  - 67 x 78 x 31 mm の直方体（面ごとに色と模様）。+Z 面（67 x 78）に ID 0・20 mm のマーカー（白縁 3 mm）
  - 物体はカメラの前 約 0.33 m で回転（面内の回転 1.5 周 ＋ 2 軸の傾き ±60°）。マーカーが見えない向きも入る
  - 手: 物体の ±Y 面のそばを滑る肌色の楕円体（途中で持ち替え）＋カメラ前を横切る球
  - 背景: 1.0 m の壁。映像 1920x1440、深度 256x192（lidar）、K_ref 4032x3024、歪み中心ずらし、
    樽型の歪み（角で約 -2 %）を lensDistortionLookupTable / inverse… に 42 点で記録
  - 深度: 雑音 σ = 1 mm + 距離の 2 乗に比例、段差の画素の 60 % を前後の間の値にする（飛び画素）
真値は <session>/synthetic_truth.json（アプリの形式には含まれない）。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lens import Camera, lut_apply  # noqa: E402

BOX = np.array([0.067, 0.078, 0.031])
MARKER = 0.020
MARGIN = 0.003
VIDEO = (1920, 1440)
DEPTH = (256, 192)
REF = (4032, 3024)
FX_REF = 2940.0
K_REF = np.array([[FX_REF, 0, 2016 + 9.0], [0, FX_REF, 1512 - 6.0], [0, 0, 1]])
DIST_C = np.array([2016 + 4.0, 1512 + 3.0])
K1 = -0.02                     # 歪み: mag(r) = K1 (r / r_max)^2（歪みなし → 歪みあり）
N_LUT = 42
# マーカーの取り付け（物体座標）: +Z 面、中心 (-17, 24) mm、マーカーの x 軸 = 物体の +Y（90° 回転）
T_OBJ_MARKER = np.eye(4)
T_OBJ_MARKER[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
T_OBJ_MARKER[:3, 3] = [-0.017, 0.024, BOX[2] / 2]


def make_luts(cam_rmax: float):
    rn = np.linspace(0, 1, N_LUT)
    fwd = K1 * rn ** 2
    inv = np.zeros(N_LUT)
    for i, rd in enumerate(rn):            # 歪みありの半径 rd に対する歪みなし半径 ru: ru (1 + K1 ru^2) = rd
        ru = rd
        for _ in range(30):
            ru = ru - (ru * (1 + K1 * ru ** 2) - rd) / (1 + 3 * K1 * ru ** 2)
        inv[i] = ru / rd - 1 if rd > 0 else 0.0
    return fwd, inv


def rays_for(cam: Camera, uv_cv: np.ndarray, w: int, h: int) -> np.ndarray:
    """歪みありの画素 (cv) → 単位方向ベクトル（カメラ座標）。"""
    rect = cam.distorted_to_rect_cv(uv_cv, w, h)
    K = cam.K_for(w, h)
    x = (rect[:, 0] - K[0, 2]) / K[0, 0]
    y = (rect[:, 1] - K[1, 2]) / K[1, 1]
    d = np.stack([x, y, np.ones_like(x)], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def marker_bits() -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(d, 0, 6, borderBits=1)   # 6x6、0 = 黒


def obj_pose(i: int, n: int) -> np.ndarray:
    s = i / n
    spin = 2 * math.pi * 1.5 * s
    a = math.radians(60) * math.sin(2 * math.pi * 2 * s)
    b = math.radians(60) * math.cos(2 * math.pi * 1.3 * s)
    R = (Rotation.from_euler("y", b) * Rotation.from_euler("x", a) * Rotation.from_euler("x", math.pi)
         * Rotation.from_euler("z", spin)).as_matrix()
    t = [0.012 * math.sin(2 * math.pi * s), 0.010 * math.cos(2 * math.pi * 0.7 * s), 0.33 + 0.03 * math.sin(2 * math.pi * 1.7 * s)]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def ellipsoid(radii, T):
    m = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=24)
    v = np.asarray(m.vertices) * radii
    m.vertices = o3d.utility.Vector3dVector(v @ T[:3, :3].T + T[:3, 3])
    return m


def box_mesh(T):
    m = o3d.geometry.TriangleMesh.create_box(*BOX)
    v = np.asarray(m.vertices) - BOX / 2
    m.vertices = o3d.utility.Vector3dVector(v @ T[:3, :3].T + T[:3, 3])
    return m


def hands(i: int, n: int, T_cam_obj: np.ndarray):
    """手（物体座標で ±Y 面のそばを滑る指）とカメラ前を横切る球。カメラ座標の (メッシュ, 種類) の列。"""
    s = i / n
    side = -1 if s < 0.5 else 1
    c_obj = np.array([0.025 * math.sin(2 * math.pi * 3 * s), side * (BOX[1] / 2 + 0.013 + 0.003), -0.004])
    T = np.eye(4)
    T[:3, 3] = c_obj
    T = T_cam_obj @ T
    out = [ellipsoid([0.020, 0.012, 0.010], T)]
    if 0.25 < s < 0.45 or 0.65 < s < 0.8:
        u = (s - 0.25) / 0.2 if s < 0.5 else (s - 0.65) / 0.15
        Tb = np.eye(4)
        Tb[:3, 3] = [-0.09 + 0.18 * u, 0.02, 0.24]
        out.append(ellipsoid([0.022, 0.022, 0.022], Tb))
    return out


def shade_box(P_obj: np.ndarray, N_obj: np.ndarray, bits: np.ndarray, rng_tex: np.ndarray) -> np.ndarray:
    base = {0: (200, 120, 80), 1: (90, 170, 200), 2: (120, 200, 120), 3: (200, 200, 110), 4: (210, 210, 210),
            5: (150, 110, 190)}
    face = np.argmax(np.abs(P_obj / (BOX / 2)), 1) * 2 + (np.take_along_axis(P_obj, np.argmax(np.abs(P_obj / (BOX / 2)), 1)[:, None], 1)[:, 0] > 0)
    col = np.array([base[int(f)] for f in range(6)], np.float64)[face]
    # 4 mm の格子の模様
    g = np.floor(P_obj / 0.004).astype(int)
    tex = rng_tex[(g[:, 0] * 73856093 ^ g[:, 1] * 19349663 ^ g[:, 2] * 83492791) % len(rng_tex)]
    col = col * (0.75 + 0.25 * tex[:, None])
    # マーカー（+Z 面）
    Tm = np.linalg.inv(T_OBJ_MARKER)
    Pm = P_obj @ Tm[:3, :3].T + Tm[:3, 3]
    on = (face == 5) & (np.abs(Pm[:, 0]) < MARKER / 2 + MARGIN) & (np.abs(Pm[:, 1]) < MARKER / 2 + MARGIN)
    col[on] = 235
    inner = on & (np.abs(Pm[:, 0]) < MARKER / 2) & (np.abs(Pm[:, 1]) < MARKER / 2)
    cx = np.clip(((Pm[inner, 0] + MARKER / 2) / MARKER * 6).astype(int), 0, 5)
    cy = np.clip(((MARKER / 2 - Pm[inner, 1]) / MARKER * 6).astype(int), 0, 5)
    col[inner] = np.where(bits[cy, cx] > 128, 235, 25)[:, None]
    return col


def render(out: Path, n: int, seed: int):
    rng = np.random.default_rng(seed)
    rng_tex = rng.random(4096)
    fwd, inv = make_luts(1.0)
    cam = Camera(K_REF.copy(), float(REF[0]), float(REF[1]), DIST_C.copy(), fwd, inv)
    bits = marker_bits()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    W, H = VIDEO
    dw, dh = DEPTH
    # 背景（壁 z = 1.0 m）は静止しているので 1 回だけ
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dirs_full = rays_for(cam, np.stack([u.ravel(), v.ravel()], 1).astype(np.float64), W, H)
    Pw = dirs_full * (1.0 / dirs_full[:, 2:3])
    wall = (150 + 40 * np.sin(Pw[:, 0] * 60) * np.cos(Pw[:, 1] * 45)).reshape(H, W)
    bg = np.repeat(wall[:, :, None], 3, 2) * np.array([1.0, 0.97, 0.92])
    du, dv = np.meshgrid(np.arange(dw), np.arange(dh))
    duv_video = (np.stack([du.ravel(), dv.ravel()], 1) + 0.5) * (W / dw) - 0.5
    dirs_d = rays_for(cam, duv_video, W, H)
    truth = {"box_size_m": BOX.tolist(), "T_obj_marker": T_OBJ_MARKER.tolist(), "marker_id": 0,
             "marker_size_m": MARKER, "frames": []}
    for i in range(n):
        T_co = obj_pose(i, n)
        scene = o3d.t.geometry.RaycastingScene()
        ids = {}
        ids[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(box_mesh(T_co)))] = "box"
        for hm in hands(i, n, T_co):
            ids[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(hm))] = "hand"
        wall_m = o3d.geometry.TriangleMesh.create_box(4, 4, 0.01)
        wall_m.translate([-2, -2, 1.0])
        ids[scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wall_m))] = "wall"

        # ---- RGB: 物体と手の外接矩形だけ光線を飛ばす
        pts = [np.asarray(box_mesh(T_co).vertices)] + [np.asarray(h.vertices) for h in hands(i, n, T_co)]
        uv = cam.project_cv(np.vstack(pts), W, H)
        x0, y0 = np.floor(np.nanmin(uv, 0) - 8).astype(int)
        x1, y1 = np.ceil(np.nanmax(uv, 0) + 8).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        img = bg.copy()
        if x1 > x0 and y1 > y0:
            sub = (slice(y0, y1), slice(x0, x1))
            d = dirs_full.reshape(H, W, 3)[sub].reshape(-1, 3)
            col = shade(scene, ids, d, T_co, bits, rng_tex)
            img[sub] = np.where(np.isnan(col), img[sub].reshape(-1, 3), col).reshape(y1 - y0, x1 - x0, 3)
        img = cv2.GaussianBlur(img, (0, 0), 0.7)
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "frames" / f"{i:06d}.jpg"), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ---- 深度（z、メートル）
        r = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros_like(dirs_d), dirs_d]).astype(np.float32)))
        th = r["t_hit"].numpy()
        z = (th * dirs_d[:, 2]).reshape(dh, dw)
        z[~np.isfinite(z)] = 0
        hi = cv2.dilate(z, np.ones((3, 3), np.uint8))
        lo = cv2.erode(np.where(z > 0, z, 9).astype(np.float32), np.ones((3, 3), np.uint8))
        edge = (hi - lo > 0.02) & (rng.random(z.shape) < 0.6)
        z = np.where(edge, lo + rng.random(z.shape) * (hi - lo), z)
        z = z + rng.normal(0, 1, z.shape) * (0.001 + 0.004 * z ** 2)
        z[rng.random(z.shape) < 0.01] = 0
        z.astype("<f4").tofile(out / "frames" / f"{i:06d}.depth")
        meta = {"t": i / 10.0, "K": K_REF.ravel().tolist(), "K_ref": list(REF), "dist_center": DIST_C.tolist(),
                "lut": fwd.tolist(), "inv_lut": inv.tolist(), "depth_filtered": False}
        (out / "frames" / f"{i:06d}.json").write_text(json.dumps(meta))
        truth["frames"].append({"frame": f"{i:06d}", "T_cam_marker": (T_co @ T_OBJ_MARKER).tolist()})
    (out / "session.json").write_text(json.dumps({
        "camera": "lidar", "video_w": W, "video_h": H, "depth_w": dw, "depth_h": dh, "fps": 10,
        "device": "synthetic", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": n}, indent=1))
    (out / "synthetic_truth.json").write_text(json.dumps(truth))


def shade(scene, ids, d, T_co, bits, rng_tex):
    r = scene.cast_rays(o3d.core.Tensor(np.hstack([np.zeros_like(d), d]).astype(np.float32)))
    th = r["t_hit"].numpy()
    gid = r["geometry_ids"].numpy()
    nrm = r["primitive_normals"].numpy()
    P = d * th[:, None]
    col = np.full((len(d), 3), np.nan)
    light = np.array([0.3, -0.5, -1.0])
    light /= np.linalg.norm(light)
    lam = 0.45 + 0.55 * np.clip(np.abs(np.sum(nrm * light, 1)), 0, 1)
    kind = np.array([ids.get(int(g), "none") for g in range(max(ids) + 1)] + ["none"])
    k = kind[np.where(gid < len(kind) - 1, gid, len(kind) - 1)]
    b = k == "box"
    if b.any():
        Ti = np.linalg.inv(T_co)
        Po = P[b] @ Ti[:3, :3].T + Ti[:3, 3]
        col[b] = shade_box(Po, None, bits, rng_tex) * lam[b, None]
    h = k == "hand"
    col[h] = np.array([225, 170, 140]) * lam[h, None]          # RGB の肌色
    w = k == "wall"
    col[w] = np.nan
    return col


# --------------------------------------------------------------------------- 評価

def box_distance(P_obj: np.ndarray) -> np.ndarray:
    """直方体表面までの距離（符号なし）。"""
    q = np.abs(P_obj) - BOX / 2
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, 1), 0)
    return np.abs(outside + inside)


def evaluate(session: Path, res_dir: Path) -> dict:
    import trimesh
    truth = json.loads((session / "synthetic_truth.json").read_text())
    res = json.loads((res_dir / "result.json").read_text())
    T_om = np.array(truth["T_obj_marker"])
    T_mo = np.linalg.inv(T_om)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * BOX / 2
    cm = corners @ T_mo[:3, :3].T + T_mo[:3, 3]
    out = {"truth_bbox_m": {"min": cm.min(0).round(4).tolist(), "max": cm.max(0).round(4).tolist()},
           "result_bbox_m": res.get("bbox_m"), "frames_used": res["frames_used"],
           "frames_anchored": res.get("frames_anchored_by_marker"), "frames_tracked": res.get("frames_tracked_by_icp"),
           "frames_total": res["frames_total"]}
    if res.get("bbox_m"):
        out["bbox_err_mm"] = {"min": ((np.array(res["bbox_m"]["min"]) - cm.min(0)) * 1000).round(2).tolist(),
                              "max": ((np.array(res["bbox_m"]["max"]) - cm.max(0)) * 1000).round(2).tolist()}
    m = trimesh.load(res_dir / "mesh.ply", process=False)
    S, _ = trimesh.sample.sample_surface(m, 40000, seed=1)
    So = S @ T_om[:3, :3].T + T_om[:3, 3]
    d = box_distance(So) * 1000
    out["mesh_to_truth_mm"] = {"mean": round(float(d.mean()), 3), "rms": round(float(np.sqrt((d ** 2).mean())), 3),
                               "p95": round(float(np.percentile(d, 95)), 3), "max": round(float(d.max()), 3)}
    # 真の表面 → メッシュ（面ごと。-Z 面はマーカーの裏で、どの向きからもほぼ見えない）
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(m.vertices.astype(np.float32)),
                                                    o3d.core.Tensor(m.faces.astype(np.int32))))
    rng = np.random.default_rng(0)
    per = {}
    for ax in range(3):
        for sgn in (-1, 1):
            u = rng.uniform(-0.5, 0.5, (4000, 3)) * BOX
            u[:, ax] = sgn * BOX[ax] / 2
            Pm = u @ T_mo[:3, :3].T + T_mo[:3, 3]
            dd = scene.compute_distance(o3d.core.Tensor(Pm.astype(np.float32))).numpy() * 1000
            per[("-+"[sgn > 0]) + "XYZ"[ax]] = {"mean": round(float(dd.mean()), 2), "p95": round(float(np.percentile(dd, 95)), 2)}
    out["truth_to_mesh_mm_by_face"] = per
    # 姿勢の誤差（マーカーだけ / 最終）
    tf = {f["frame"]: np.array(f["T_cam_marker"]) for f in truth["frames"]}
    errs = {"before_icp": [], "final": []}
    for f in res["frames"]:
        if f.get("status") != "ok":
            continue
        Tt = tf[f["frame"]]
        for key, k2 in (("before_icp", "T_cam_marker_before_icp"), ("final", "T_cam_marker")):
            E = np.linalg.inv(Tt) @ np.array(f[k2])
            # 物体中心での位置誤差（マーカー中心より意味がある）
            c = T_mo[:3, 3]
            errs[key].append((np.linalg.norm(E[:3, :3] @ c + E[:3, 3] - c) * 1000,
                              math.degrees(math.acos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))))
    for k, v in errs.items():
        if v:
            v = np.array(v)
            out[f"pose_err_{k}"] = {"pos_mm_median": round(float(np.median(v[:, 0])), 3),
                                   "pos_mm_p95": round(float(np.percentile(v[:, 0], 95)), 3),
                                   "rot_deg_median": round(float(np.median(v[:, 1])), 3),
                                   "rot_deg_p95": round(float(np.percentile(v[:, 1], 95)), 3)}
    out["mesh"] = res.get("mesh")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(HERE / "output" / "synthetic" / "scan-20261015-000000"))
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run", action="store_true", help="生成後に process.py を走らせて評価する")
    p.add_argument("--no-render", action="store_true", help="既存のセッションを使う")
    p.add_argument("--process-args", default="", help="process.py へ追加で渡す引数")
    a = p.parse_args()
    out = Path(a.out)
    t0 = time.time()
    if not a.no_render:
        render(out, a.frames, a.seed)
        print(f"[synthetic] {a.frames} フレーム生成 {time.time() - t0:.1f} s → {out}", flush=True)
    if a.run:
        t1 = time.time()
        cmd = [sys.executable, str(HERE / "process.py"), str(out), "--marker-id", "0", "--marker-mm", "20"] + a.process_args.split()
        subprocess.run(cmd, check=True)
        print(f"[synthetic] process.py {time.time() - t1:.1f} s", flush=True)
        ev = evaluate(out, out / "result")
        (out / "result" / "synthetic_eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
        print(json.dumps(ev, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
