"""模様付きメッシュ（Object Capture の OBJ）をGPU（MuJoCo EGL）で多方向から描き、ArUco を検出して、マーカーの四隅をメッシュ上の 3D 点として求める。
既知の一辺（--sizes）から尺度を決め、基準マーカー（--ref）の座標系（中心が原点, x 右, y 上, z 外向き, m）でメッシュを書き出す。
使い方: scan/.venv/bin/python scan/photogrammetry/markers_on_mesh.py <model_dir> --sizes 0:20,3:20 --ref 0 [--out DIR]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import trimesh

DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())


def gpu_hits(m, mesh_path, tex_path, dirs, px, ext):
    """MuJoCo（EGL、GPU）で各方向から透視投影で描き、ArUco の四隅を深度でメッシュ上の 3D 点に戻す。"""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    c = m.centroid
    dist = 3.0 * ext
    fovy = 2 * np.degrees(np.arctan(0.6 * ext / dist))
    cams = []
    for i, d in enumerate(dirs):
        z = d / np.linalg.norm(d)            # カメラの後ろ向き（MuJoCo のカメラは -z を見る）
        x = np.cross([0, 0, 1] if abs(z[2]) < 0.9 else [1, 0, 0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pos = c + z * dist
        cams.append((pos, x, y, z))
    cam_xml = "".join(f'<camera name="c{i}" pos="{p[0]} {p[1]} {p[2]}" xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" fovy="{fovy}"/>'
                      for i, (p, x, y, z) in enumerate(cams))
    xml = f"""<mujoco><compiler angle="degree"/>
<visual><global offwidth="{px}" offheight="{px}"/><headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
<map znear="0.01" zfar="50"/></visual>
<asset><texture name="t" type="2d" file="{tex_path}"/><material name="mt" texture="t" emission="0" specular="0" shininess="0"/>
<mesh name="obj" file="{mesh_path}"/></asset>
<worldbody><geom type="mesh" mesh="obj" material="mt" contype="0" conaffinity="0"/>{cam_xml}</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, px, px)
    f = (px / 2) / np.tan(np.radians(fovy) / 2)
    hits = {}
    for i, (pos, x, y, z) in enumerate(cams):
        r.disable_depth_rendering(); r.update_scene(data, camera=i); img = r.render()[:, :, ::-1].copy()
        r.enable_depth_rendering(); r.update_scene(data, camera=i); dep = r.render().copy()
        co, ids, _ = DET.detectMarkers(img)
        if ids is None:
            continue
        for mid, q in zip(ids.ravel(), co):
            q = q[0]
            C = []
            for qu, qv in q:
                iu, iv = int(round(qu)), int(round(qv))
                win = dep[max(iv - 2, 0):iv + 3, max(iu - 2, 0):iu + 3]
                win = win[win < 0.9 * 50]
                if win.size == 0:
                    break
                zd = np.median(win)
                C.append(pos + zd * ((qu - px / 2 + 0.5) / f * x - (qv - px / 2 + 0.5) / f * y - z))
            if len(C) < 4:
                continue
            C = np.array(C)
            n = np.cross(C[1] - C[0], C[3] - C[0]); n /= np.linalg.norm(n)
            hits.setdefault(int(mid), []).append((C, float(n @ z), float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))))
        if i == 0:
            cv2.imwrite(os.path.join(os.path.dirname(mesh_path), "..", "gpu_view0.jpg"), img)
    return hits


def directions(n=40):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--sizes", required=True, help="'ID:mm,…' マーカーの黒い正方形の一辺")
    ap.add_argument("--ref", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", type=int, default=60)
    ap.add_argument("--px", type=int, default=1400)
    a = ap.parse_args()
    out = a.out or os.path.join(a.model_dir, "..", "aligned")
    os.makedirs(out, exist_ok=True)
    sizes = {int(k): float(v) / 1000 for k, v in (x.split(":") for x in a.sizes.split(","))}
    m = trimesh.load(sorted(glob.glob(os.path.join(a.model_dir, "baked_mesh_*.obj")))[0], process=False)
    ext = float(np.linalg.norm(m.extents))
    mesh_path = sorted(glob.glob(os.path.join(a.model_dir, "baked_mesh_*.obj")))[0]
    tex_path = mesh_path[:-4] + "_tex0.png"
    hits = gpu_hits(m, os.path.abspath(mesh_path), os.path.abspath(tex_path), directions(a.views), a.px, ext)
    # マーカーごとに、正面に近い視点（n·z 大）の上位を平均。位置が離れた同じ ID は別個体として分ける
    found = {}
    for mid, L in hits.items():
        groups = []
        for C, cosv, spx in sorted(L, key=lambda t: -t[1]):
            for g in groups:
                if np.linalg.norm(g[0][0].mean(0) - C.mean(0)) < 0.3 * np.linalg.norm(C[0] - C[2]):
                    g.append((C, cosv, spx)); break
            else:
                groups.append([(C, cosv, spx)])
        found[mid] = []
        for g in groups:
            best = [t for t in g if t[1] > 0.8][:8] or g[:3]
            C = np.mean([t[0] for t in best], 0)
            side = float(np.mean([np.linalg.norm(C[k] - C[(k + 1) % 4]) for k in range(4)]))
            found[mid].append({"C": C, "side_units": side, "views": len(g), "best_cos": best[0][1]})
    rep = {"markers": {}}
    scales = []
    for mid, gs in found.items():
        for k, g in enumerate(gs):
            rep["markers"][f"{mid}#{k}"] = {"side_units": g["side_units"], "views": g["views"], "center_units": g["C"].mean(0).tolist()}
            if mid in sizes:
                scales.append(sizes[mid] / g["side_units"])
    print(json.dumps(rep, indent=1))
    if not scales:
        raise SystemExit("寸法既知のマーカーが見つからない")
    s = float(np.median(scales))
    rep["scale_units_to_m"] = s
    rep["scale_spread"] = [float(min(scales)), float(max(scales))]
    gs = found.get(a.ref)
    if not gs:
        raise SystemExit(f"基準マーカー {a.ref} が見つからない")
    if len(gs) > 1:
        print(f"注意: ID {a.ref} が {len(gs)} か所にある。最も多くの視点で見えたものを基準にする")
    C = max(gs, key=lambda g: g["views"])["C"] * s
    m.apply_scale(s)
    o = C.mean(0)
    xv = (C[1] + C[2]) / 2 - (C[0] + C[3]) / 2; xv /= np.linalg.norm(xv)
    yv = (C[0] + C[1]) / 2 - (C[3] + C[2]) / 2; yv -= xv * (xv @ yv); yv /= np.linalg.norm(yv)
    T = np.eye(4); T[:3, :3] = np.c_[xv, yv, np.cross(xv, yv)]; T[:3, 3] = o
    Ti = np.linalg.inv(T)
    m.apply_transform(Ti)
    rep["markers_in_ref_mm"] = {}
    for mid, gs2 in found.items():
        for k, g in enumerate(gs2):
            Cr = (Ti @ np.c_[g["C"] * s, np.ones(4)].T).T[:, :3]
            n = np.cross(Cr[1] - Cr[0], Cr[3] - Cr[0]); n /= np.linalg.norm(n)
            rep["markers_in_ref_mm"][f"{mid}#{k}"] = {"center": np.round(Cr.mean(0) * 1000, 1).tolist(), "normal": np.round(n, 3).tolist(),
                                                      "side_mm": round(g["side_units"] * s * 1000, 2)}
    m.export(os.path.join(out, f"mesh_m{a.ref}.obj"))
    m.export(os.path.join(out, f"mesh_m{a.ref}.stl"))
    m.convex_hull.export(os.path.join(out, f"hull_m{a.ref}.stl"))
    rep["obb_mm"] = np.round(np.sort(m.bounding_box_oriented.primitive.extents * 1000), 1).tolist()
    json.dump(rep, open(os.path.join(out, "markers_on_mesh.json"), "w"), indent=1, ensure_ascii=False)
    print("scale", s, rep["scale_spread"], "OBB mm", rep["obb_mm"])
    print(json.dumps(rep["markers_in_ref_mm"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
