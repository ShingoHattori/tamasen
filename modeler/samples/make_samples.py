#!/usr/bin/env python3
"""合成データ（実物の写真ではない）のサンプルを作る。

  modeler/.venv/bin/python modeler/samples/make_samples.py [--only plate_synthetic] [--no-project]

作るもの（samples/<name>/）:
  images/*.jpg     合成画像（1280×960、K 既知、弱い樽型の歪みを掛けてある。背景は固定カメラなので全画像で同じ）
  truth.json       真値（T_cam_marker・T_marker_obj・形状の寸法・マーカー四隅 raw / und・手の有無）。テスト用。プロジェクトには入れない
  project/         取り込み済みのプロジェクト（サーバーと同じ前処理を通したもの。形状は空）。git には入れない（.gitignore）

サンプル:
  plate_synthetic      板 120×60×3 mm、貫通穴 2 つ（円 φ12、長穴 20×8）。上面中央に DICT_4X4_50 ID 7（30 mm、Z 軸まわり 90°）。
                       上面に黒い塗装の円（φ10、穴ではない）。裏側の画像 2 枚（マーカーが見えない）、背景のみの画像 1 枚
  container_synthetic  カップ（回転体: 外半径 35、高さ 80、肉厚 3、底 3、上が開口）。側面に平らなカードで DICT_4X4_50 ID 11（30 mm）。
                       開口を覗く画像、マーカーが見えない画像を含む

描画: 三角形の z バッファ（2 倍の超標本化）＋ Lambert の陰影、マーカーは平面への光線交差で貼る（深度判定つき）、
雑然とした背景、一部の画像で肌色の「手」が物体の一部を隠す、軽いぼけ・雑音・JPEG。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import manifold3d as mf
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
import preprocess as pp  # noqa: E402
import project as pj  # noqa: E402

W, H = 1280, 960
K = np.array([[913.1, 0.0, 640.3], [0.0, 912.7, 478.6], [0.0, 0.0, 1.0]])
DIST = np.array([-0.06, 0.02, 0.0, 0.0, 0.0])     # 弱い樽型（四隅で約 3 %）
SS = 2            # 超標本化
MARGIN = 60       # 歪みで外側から来る画素のための余白（最終画素）
LIGHT = np.array([-0.35, -0.55, -0.75])     # 光源への向き（cam 座標。カメラの左上手前）
LIGHT = LIGHT / np.linalg.norm(LIGHT)
NOTE = "合成データ（実物ではない）。make_samples.py が生成"


# --------------------------------------------------------------------------- 幾何

def Rx(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def T_of(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def mesh_of(m: mf.Manifold):
    me = m.to_mesh()
    return np.asarray(me.vert_properties)[:, :3].astype(float), np.asarray(me.tri_verts).astype(int)


def plate_manifold() -> mf.Manifold:
    plate = mf.Manifold.cube([120.0, 60.0, 3.0], True)
    hole = mf.Manifold.cylinder(10.0, 6.0, 6.0, 64, True).translate([-40.0, 0.0, 0.0])
    slot2d = mf.CrossSection.square([12.0, 8.0], True) + mf.CrossSection.circle(4.0, 48).translate([-6.0, 0.0]) \
        + mf.CrossSection.circle(4.0, 48).translate([6.0, 0.0])
    slot = slot2d.extrude(10.0).translate([40.0, 0.0, -5.0])
    return plate - hole - slot


def cup_manifold() -> mf.Manifold:
    prof = [(0.0, 0.0), (35.0, 0.0), (35.0, 80.0), (32.0, 80.0), (32.0, 3.0), (0.0, 3.0)]
    return mf.CrossSection([prof]).revolve(96, 360.0)


# --------------------------------------------------------------------------- 描画

def K_canvas(s: int) -> np.ndarray:
    """余白つき・s 倍の描画面の K（画素中心が整数の規約を保つ）。"""
    Kc = K.copy()
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = s * (K[0, 2] + MARGIN + 0.5) - 0.5
    Kc[1, 2] = s * (K[1, 2] + MARGIN + 0.5) - 0.5
    return Kc


def rasterize(meshes, T_cam_obj, Kc, shape, img):
    """meshes: [(V_obj, F, bgr)]。img（float、描画面）に描き、逆深度バッファを返す。"""
    h, w = shape
    zinv = np.zeros((h, w), np.float32)
    R, t = T_cam_obj[:3, :3], T_cam_obj[:3, 3]
    for V, F, col in meshes:
        Vc = V @ R.T + t
        uv = Vc @ Kc.T
        uv = uv[:, :2] / uv[:, 2:3]
        col = np.asarray(col, float)
        for f in F:
            p0, p1, p2 = Vc[f[0]], Vc[f[1]], Vc[f[2]]
            n = np.cross(p1 - p0, p2 - p0)
            if np.dot(n, p0) >= 0:          # 裏向き
                continue
            n = n / (np.linalg.norm(n) + 1e-12)
            q = uv[f]
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
            if x0 > x1 or y0 > y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1, dtype=float), np.arange(y0, y1 + 1, dtype=float))
            (ax, ay), (bx, by), (cx, cy) = q
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-9:
                continue
            w0 = ((bx - xs) * (cy - ys) - (by - ys) * (cx - xs)) / area
            w1 = ((cx - xs) * (ay - ys) - (cy - ys) * (ax - xs)) / area
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            zi = (w0 / p0[2] + w1 / p1[2] + w2 / p2[2]).astype(np.float32)
            sub = zinv[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (zi > sub)
            if not upd.any():
                continue
            sub[upd] = zi[upd]
            shade = 0.38 + 0.62 * max(0.0, float(np.dot(n, LIGHT)))
            img[y0:y1 + 1, x0:x1 + 1][upd] = col * shade
    return zinv


def decal(img, zinv, Kc, T_cam_decal, size_mm, tex_bgra, tol_mm=0.4):
    """平面のテクスチャ（decal 座標: x 右・y 上、中心が原点、一辺 size_mm）を深度判定つきで貼る。"""
    h, w = zinv.shape
    R, t = T_cam_decal[:3, :3], T_cam_decal[:3, 3]
    nrm = R[:, 2]
    if np.dot(nrm, t) >= 0:
        return
    s = size_mm / 2
    corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]) @ R.T + t
    q = corners @ Kc.T
    q = q[:, :2] / q[:, 2:3]
    x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - 1, 0)
    x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + 1, [w - 1, h - 1])
    if x0 > x1 or y0 > y1:
        return
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1, dtype=float), np.arange(y0, y1 + 1, dtype=float))
    d = np.stack([(xs - Kc[0, 2]) / Kc[0, 0], (ys - Kc[1, 2]) / Kc[1, 1], np.ones_like(xs)], -1)
    lam = np.dot(nrm, t) / (d @ nrm)
    P = d * lam[..., None]
    loc = (P - t) @ R
    th, tw = tex_bgra.shape[:2]
    mu = ((loc[..., 0] / size_mm + 0.5) * tw - 0.5).astype(np.float32)
    mv = ((0.5 - loc[..., 1] / size_mm) * th - 0.5).astype(np.float32)
    smp = cv2.remap(tex_bgra.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    z = P[..., 2]
    zb = zinv[y0:y1 + 1, x0:x1 + 1]
    front = (zb == 0) | (z <= 1.0 / np.maximum(zb, 1e-12) + tol_mm)
    a = smp[..., 3:4] / 255.0 * front[..., None]
    shade = 0.38 + 0.62 * max(0.0, float(np.dot(nrm, LIGHT)))
    sub = img[y0:y1 + 1, x0:x1 + 1]
    sub[:] = sub * (1 - a) + smp[..., :3] * shade * a


def marker_texture(dictionary: str, mid: int, px_per_module: int = 48) -> tuple[np.ndarray, float]:
    """白縁 1 モジュールつきの ArUco（BGRA）。戻り値の 2 つ目は白縁を含む一辺 ÷ 黒い正方形の一辺。"""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    bits = d.markerSize + 2
    m = cv2.aruco.generateImageMarker(d, mid, bits * px_per_module)
    m = cv2.copyMakeBorder(m, px_per_module, px_per_module, px_per_module, px_per_module, cv2.BORDER_CONSTANT, value=255)
    m = np.where(m > 127, 235, 18).astype(np.uint8)     # 紙の白・印刷の黒
    bgra = np.dstack([m, m, m, np.full_like(m, 255)])
    return bgra, (bits + 2) / bits


def disc_texture(n: int = 128, bgr=(22, 22, 26)) -> np.ndarray:
    t = np.zeros((n * 4, n * 4, 4), np.uint8)
    cv2.circle(t, (2 * n, 2 * n), 2 * n - 4, (*bgr, 255), -1, cv2.LINE_AA)
    return cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA)


def backdrop(seed: int) -> np.ndarray:
    """固定カメラの背景（余白つきの最終解像度）。机の木目風＋雑然とした物。真っ黒な物は置かない。"""
    rng = np.random.default_rng(seed)
    h, w = H + 2 * MARGIN, W + 2 * MARGIN
    y, x = np.mgrid[0:h, 0:w].astype(float)
    base = np.array([96, 128, 160], float)        # 木の色（BGR）
    grain = 10 * np.sin(x / 23.0 + 3 * np.sin(y / 97.0)) + 6 * np.sin(y / 7.0 + x / 300.0)
    img = base[None, None] + grain[..., None] + y[..., None] * 0.03
    img = img.astype(np.float32)
    for _ in range(36):
        col = rng.uniform(70, 230, 3)
        kind = rng.integers(3)
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        if kind == 0:
            sx, sy = int(rng.uniform(30, 160)), int(rng.uniform(20, 120))
            cv2.rectangle(img, (cx, cy), (cx + sx, cy + sy), col.tolist(), -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(img, (cx, cy), int(rng.uniform(12, 70)), col.tolist(), -1, cv2.LINE_AA)
        else:
            cv2.line(img, (cx, cy), (cx + int(rng.uniform(-300, 300)), cy + int(rng.uniform(-200, 200))),
                     col.tolist(), int(rng.uniform(3, 14)), cv2.LINE_AA)
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255)


def draw_hand(img, tip_uv, out_dir_uv, rng):
    """肌色の手（掌の楕円＋指 3 本）。tip_uv（最終画素・余白込み）に指先、out_dir_uv の向きに掌。"""
    skin = np.array([128, 160, 214], float)
    o = np.asarray(out_dir_uv, float)
    o /= np.linalg.norm(o) + 1e-9
    perp = np.array([-o[1], o[0]])
    palm = np.asarray(tip_uv, float) + o * 120
    ov = img.copy()
    ang = math.degrees(math.atan2(o[1], o[0]))
    cv2.ellipse(ov, tuple(int(v) for v in palm), (75, 58), ang, 0, 360, (skin * 0.92).tolist(), -1, cv2.LINE_AA)
    for k in (-1, 0, 1):
        a = palm - o * 40 + perp * 26 * k
        b = np.asarray(tip_uv, float) + perp * 22 * k + o * (8 * abs(k))
        cv2.line(ov, tuple(int(v) for v in a), tuple(int(v) for v in b), (skin * (1.0 - 0.05 * abs(k))).tolist(), 24,
                 cv2.LINE_AA)
        cv2.circle(ov, tuple(int(v) for v in b), 12, skin.tolist(), -1, cv2.LINE_AA)
    img[:] = ov


def distort_to_raw(canvas: np.ndarray) -> np.ndarray:
    """ピンホールの描画面（余白つき）→ 歪みのある原画像。"""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    und = cv2.undistortPoints(np.c_[u.ravel(), v.ravel()].reshape(-1, 1, 2), K, DIST, None, None, K,
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)).reshape(H, W, 2)
    mx = (und[..., 0] + MARGIN).astype(np.float32)
    my = (und[..., 1] + MARGIN).astype(np.float32)
    return cv2.remap(canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def render(bg, meshes, decals, T_cam_obj, hand=None, seed=0):
    """→ (原画像 BGR uint8)。decals: [(T_obj_decal, size_mm, tex)]。hand: (obj 点, obj での外向き)。"""
    rng = np.random.default_rng(seed)
    Kc = K_canvas(SS)
    big = cv2.resize(bg, None, fx=SS, fy=SS, interpolation=cv2.INTER_LINEAR)
    if meshes:
        zinv = rasterize(meshes, T_cam_obj, Kc, big.shape[:2], big)
        for T_od, size, tex in decals:
            decal(big, zinv, Kc, T_cam_obj @ T_od, size, tex)
    canvas = cv2.resize(big, (W + 2 * MARGIN, H + 2 * MARGIN), interpolation=cv2.INTER_AREA)
    if hand is not None:
        K1 = K_canvas(1)
        p, outward = hand
        pc = T_cam_obj[:3, :3] @ np.asarray(p, float) + T_cam_obj[:3, 3]
        qc = T_cam_obj[:3, :3] @ (np.asarray(p, float) + np.asarray(outward, float) * 30) + T_cam_obj[:3, 3]
        a = (K1 @ pc)[:2] / pc[2]
        b = (K1 @ qc)[:2] / qc[2]
        draw_hand(canvas, a, b - a, rng)
    gain = rng.uniform(0.94, 1.06)
    canvas = canvas * gain
    raw = distort_to_raw(canvas.astype(np.float32))
    raw = cv2.GaussianBlur(raw, (0, 0), 0.7)
    raw = raw + rng.normal(0, 2.0, raw.shape).astype(np.float32)
    return np.clip(raw, 0, 255).astype(np.uint8)


def corners_truth(T_cam_marker, size_mm):
    obj = pp.marker_points(size_mm)
    pc = obj @ T_cam_marker[:3, :3].T + T_cam_marker[:3, 3]
    und = (pc @ K.T)
    und = und[:, :2] / und[:, 2:3]
    raw, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
    return und, raw.reshape(-1, 2)


def marker_visible(T_cam_marker, size_mm) -> bool:
    n = T_cam_marker[:3, 2]
    if np.dot(n, T_cam_marker[:3, 3]) >= -1e-6 * np.linalg.norm(T_cam_marker[:3, 3]):
        return False
    view = -T_cam_marker[:3, 3] / np.linalg.norm(T_cam_marker[:3, 3])
    if np.dot(n, view) < math.cos(math.radians(70)):
        return False
    und, raw = corners_truth(T_cam_marker, size_mm)
    return bool(np.all((raw >= 10) & (raw <= [W - 11, H - 11])))


# --------------------------------------------------------------------------- サンプル定義

def plate_spec():
    V, F = mesh_of(plate_manifold())
    T_obj_marker = T_of(Rz(90), [0.0, 0.0, 1.5])
    T_obj_disc = T_of(np.eye(3), [-26.0, -19.0, 1.5])
    R0 = Rx(180)                                 # 上面（+z）がカメラ向き、obj +y が画像の上
    views = [  # (名前, R, 距離, tx, ty, 手)
        ("top_a", Rx(22) @ Ry(4) @ Rz(8) @ R0, 320, 5, -5, None),
        ("top_b", Rx(-18) @ Ry(16) @ Rz(-28) @ R0, 340, -20, 10, None),
        ("top_c_hand", Rx(12) @ Ry(-22) @ Rz(58) @ R0, 330, 15, 0, ([60.0, -30.0, 0.0], [1.0, -1.0, 0.0])),
        ("top_d", Rx(34) @ Ry(20) @ Rz(100) @ R0, 350, 0, 15, None),
        ("top_e_steep", Rx(52) @ Ry(-8) @ Rz(-12) @ R0, 330, 10, -10, None),
        ("back_a", Rx(18) @ Ry(10) @ Rz(14), 330, 0, 0, None),
        ("back_b_hand", Rx(-24) @ Ry(-14) @ Rz(-38), 340, -10, 5, ([-60.0, 30.0, 0.0], [-1.0, 1.0, 0.0])),
    ]
    return {
        "name": "plate_synthetic",
        "title": "合成データ: 穴あき板（120×60×3 mm）",
        "marker": {"dictionary": "DICT_4X4_50", "id": 7, "size_mm": 30.0},
        "meshes": [(V, F, (205, 190, 172))],         # 薄い青灰（BGR）
        "T_obj_marker": T_obj_marker,
        "decals": [(T_obj_disc, 10.0, disc_texture())],
        "center_obj": [0.0, 0.0, 0.0],
        "views": views,
        "truth_shape": {
            "kind": "extrude", "outer": [[-60, -30], [60, -30], [60, 30], [-60, 30]], "thickness": 3.0,
            "symmetric": True,
            "holes": [{"shape": "circle", "center": [-40, 0], "d": 12.0},
                      {"shape": "slot", "center": [40, 0], "l": 20.0, "w": 8.0, "axis": "x"}],
            "through_holes": 2,
            "dark_disc_not_hole": {"center_obj": [-26.0, -19.0, 1.5], "d": 10.0,
                                   "note": "上面に塗った黒い円。穴ではない（穴と誤認しないかの試験用）"},
        },
        "background": True,
    }


def cup_spec():
    V, F = mesh_of(cup_manifold())
    card = mf.Manifold.cube([0.8, 40.0, 40.0], True).translate([35.6, 0.0, 45.0])   # マーカーを貼る平らなカード
    Vc, Fc = mesh_of(card)
    # マーカー: z 外向き = obj +x、y 上 = obj +z、x 右 = obj +y
    T_obj_marker = T_of(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float), [36.0, 0.0, 45.0])
    R0 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # obj x→cam x、obj y→cam z、obj z→cam -y（直立）
    views = [
        ("side_marker", Rx(5) @ R0 @ Rz(-90), 400, 0, 0, None),
        ("side_oblique", Rx(25) @ R0 @ Rz(-60), 410, 10, -5, None),
        ("opening_marker", Rx(42) @ R0 @ Rz(-115), 400, -10, 10, None),
        ("opening_steep", Rx(62) @ R0 @ Rz(-90), 380, 0, 0, None),
        ("opening_no_marker", Rx(35) @ R0 @ Rz(90), 400, 5, 0, None),
        ("side_no_marker", Rx(10) @ R0 @ Rz(170), 410, -5, 5, None),
        ("side_hand", Rx(15) @ R0 @ Rz(-45), 400, 0, 0, ([-35.0, 0.0, 30.0], [-1.0, 0.0, -0.3])),
    ]
    return {
        "name": "container_synthetic",
        "title": "合成データ: 容器（カップ、外半径 35・高さ 80・肉厚 3 mm）",
        "marker": {"dictionary": "DICT_4X4_50", "id": 11, "size_mm": 30.0},
        "meshes": [(V, F, (150, 140, 60)), (Vc, Fc, (225, 225, 225))],
        "T_obj_marker": T_obj_marker,
        "decals": [],
        "center_obj": [0.0, 0.0, 40.0],
        "views": views,
        "truth_shape": {
            "kind": "revolve", "profile": [[0, 0], [35, 0], [35, 80], [32, 80], [32, 3], [0, 3]],
            "outer_radius": 35.0, "height": 80.0, "wall": 3.0, "bottom": 3.0, "open_top": True,
            "marker_card": {"size_mm": [0.8, 40.0, 40.0], "center_obj": [35.6, 0.0, 45.0],
                            "note": "マーカーを貼った平らなカード（側面に接して固定）。容器の形状には含めない"},
        },
        "background": True,
    }


def generate(spec: dict, out: Path, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir()
    bg = backdrop(seed)
    mk = spec["marker"]
    tex, ratio = marker_texture(mk["dictionary"], mk["id"])
    decals = [(spec["T_obj_marker"], mk["size_mm"] * ratio, tex)] + spec["decals"]
    truth = {"synthetic": True, "note": NOTE, "title": spec["title"], "units": {"length": "mm"},
             "camera": {"width": W, "height": H, "K": K.ravel().tolist(), "dist": DIST.tolist(),
                        "K_und": K.ravel().tolist()},
             "marker": mk, "T_marker_obj": np.linalg.inv(spec["T_obj_marker"]).ravel().round(9).tolist(),
             "T_obj_marker": spec["T_obj_marker"].ravel().round(9).tolist(),
             "shape": spec["truth_shape"], "background_image": None, "images": []}
    if spec.get("background"):
        img = render(bg, [], [], np.eye(4), seed=seed + 999)
        cv2.imwrite(str(img_dir / "synthetic_background.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        truth["background_image"] = "synthetic_background.jpg"
    c = np.asarray(spec["center_obj"], float)
    for k, (name, R, dist, tx, ty, hand) in enumerate(spec["views"]):
        t = np.array([tx, ty, dist], float) - R @ c
        T_cam_obj = T_of(R, t)
        img = render(bg, spec["meshes"], decals, T_cam_obj, hand, seed=seed + k)
        fn = f"synthetic_{k + 1:02d}_{name}.jpg"
        cv2.imwrite(str(img_dir / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        T_cam_marker = T_cam_obj @ spec["T_obj_marker"]
        und, raw = corners_truth(T_cam_marker, mk["size_mm"])
        vis = marker_visible(T_cam_marker, mk["size_mm"])
        truth["images"].append({
            "file": fn, "view": name, "T_cam_obj": T_cam_obj.ravel().round(9).tolist(),
            "T_cam_marker": T_cam_marker.ravel().round(9).tolist(), "marker_visible": vis,
            "corners_und": und.round(4).tolist() if vis else None, "corners_raw": raw.round(4).tolist() if vis else None,
            "hand": hand is not None})
        print(f"  {fn}: marker {'見える' if vis else '見えない'}{'・手あり' if hand else ''}", flush=True)
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    (out / "README.md").write_text(
        f"# {spec['title']}\n\n**合成データ（実物の写真ではない）**。`make_samples.py` が生成した。\n\n"
        "- `images/`: 合成画像（K・歪み既知）。`truth.json`: 真値（テスト用。プロジェクトには入れない）\n"
        "- `project/`: 取り込み済みのプロジェクト（形状は空）。サーバーの「サンプルを開く」で複製して使う\n")
    return truth


def build_project(sample_dir: Path, dest: Path | None = None) -> Path:
    """samples/<name>/images をサーバーと同じ前処理で取り込んだプロジェクトを dest（既定 sample_dir/project）に作る。"""
    truth = json.loads((sample_dir / "truth.json").read_text())
    dest = dest or sample_dir / "project"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images").mkdir(parents=True)
    p = pj.new_project(f"{truth['title']}")
    p["marker"] = dict(truth["marker"])
    cam = truth["camera"]
    p["cameras"]["oak"] = pp.make_camera(cam["K"], cam["dist"], cam["width"], cam["height"], "file")
    p["attributes"]["notes"] = f"{NOTE}。真値は samples/{sample_dir.name}/truth.json（プロジェクトには入れていない）"
    files = ([truth["background_image"]] if truth.get("background_image") else []) + [i["file"] for i in truth["images"]]
    for f in files:
        role = "background" if f == truth.get("background_image") else "object"
        iid = pj.next_image_id(p)
        cams: dict = {}
        im = pp.import_image(p, dest / "images" / iid, iid, (sample_dir / "images" / f).read_bytes(), f, role, cams)
        p["cameras"].update(cams)
        p["images"].append(im)
        if role == "background":
            p["background_image"] = iid
    pj.validate(p)
    pj.atomic_write(dest / "project.json", pj.dump(p))
    return dest


SPECS = {"plate_synthetic": plate_spec, "container_synthetic": cup_spec}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(SPECS))
    ap.add_argument("--no-project", action="store_true", help="画像と真値だけ作る（取り込み済みプロジェクトは作らない）")
    a = ap.parse_args()
    for i, (name, fn) in enumerate(SPECS.items()):
        if a.only and name != a.only:
            continue
        print(f"{name}（合成データ）", flush=True)
        generate(fn(), HERE / name, seed=100 * (i + 1))
        if not a.no_project:
            print(f"  プロジェクト: {build_project(HERE / name)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
