"""非破壊の前処理（DESIGN.md §2）と姿勢推定。

- 取り込み: EXIF の向きを読んで画素に適用する（記録は Image.import_info.exif_orientation、適用後 orientation = 1）。
- 歪み補正: **K_und = K**（原画像と同じ焦点距離・主点。getOptimalNewCameraMatrix は使わない）。
  画素の大きさが原画像と同じで、校正なしなら undistorted.png は original と同じ画素になる。
  原画像の外から来る画素（樽型なら出ない、糸巻き型なら四隅）は黒で、ラベルは 0（未定義）。
- マーカー: 原画像で検出（サブピクセル補正）→ 四隅を undistortPoints で歪み補正画像の座標へ（corners_raw / corners_und の両方を持つ）。
- ラベル候補: 背景のみの画像があれば差分、なければ GrabCut（中央の広い矩形＋マーカーを前景の種）。
  穴を埋める処理はしない（3×3 の open/close だけ）。物体に囲まれた背景らしい領域だけをラベル 3（穴）にする。
  暗いだけの領域は穴にしない（差分が大きい・周囲の背景の色と似ていなければ物体 2 のまま）。
- 姿勢: マーカー四隅から IPPE_SQUARE（2 解とも返す）。対応点 PnP は全解と残差・縮退・曖昧さを返す（保存はしない）。
長さは mm、画素は画素中心が整数（OpenCV と同じ）。
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from project import new_image, next_contour_id, no_camera

LABEL_UNKNOWN, LABEL_BG, LABEL_OBJ, LABEL_HOLE, LABEL_MARKER = 0, 1, 2, 3, 4
ALL_STEPS = ["undistort", "markers", "quality", "mask", "edges", "contours"]
IMREAD_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION   # 向きは自前で適用する

DICTS = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


class PreprocessError(Exception):
    pass


# --------------------------------------------------------------------------- 読み込み・EXIF

def sniff_ext(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def exif_orientation(data: bytes) -> int:
    """JPEG の EXIF Orientation（0x0112）。なければ 1。"""
    if data[:2] != b"\xff\xd8":
        return 1
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return 1
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):                   # EOI / SOS 以降は画素
            return 1
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + seg_len]
        if marker == 0xE1 and seg[:6] == b"Exif\x00\x00":
            t = seg[6:]
            if len(t) < 8:
                return 1
            e = "<" if t[:2] == b"II" else ">"
            ifd = struct.unpack(e + "I", t[4:8])[0]
            if ifd + 2 > len(t):
                return 1
            n = struct.unpack(e + "H", t[ifd:ifd + 2])[0]
            for k in range(n):
                o = ifd + 2 + 12 * k
                if o + 12 > len(t):
                    break
                tag, typ = struct.unpack(e + "HH", t[o:o + 4])
                if tag == 0x0112:
                    v = struct.unpack(e + "H", t[o + 8:o + 10])[0]
                    return v if 1 <= v <= 8 else 1
            return 1
        i += 2 + seg_len
    return 1


def apply_orientation(img: np.ndarray, o: int) -> np.ndarray:
    if o == 2:
        return cv2.flip(img, 1)
    if o == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if o == 4:
        return cv2.flip(img, 0)
    if o == 5:
        return cv2.transpose(img)
    if o == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if o == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if o == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """画像バイト列 → (BGR, EXIF の向き)。向きは適用済み。"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), IMREAD_FLAGS)
    if img is None:
        raise PreprocessError("画像として読めない（JPEG / PNG / BMP / TIFF / WebP に対応）")
    o = exif_orientation(data)
    return apply_orientation(img, o), o


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), IMREAD_FLAGS)
    if img is None:
        raise PreprocessError(f"{path.name} が読めない")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise PreprocessError(f"{path.name} を書けない")
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


# --------------------------------------------------------------------------- カメラ・歪み補正

def mat3(v) -> np.ndarray:
    a = np.asarray(v, float)
    return a.reshape(3, 3)


def cam_arrays(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = mat3(cam["K"])
    dist = np.asarray(cam.get("dist") or [], float).ravel()
    K_und = mat3(cam.get("K_und") or cam["K"])
    return K, dist, K_und


def has_dist(cam: dict) -> bool:
    return any(abs(float(x)) > 0 for x in cam.get("dist") or [])


def make_camera(K, dist, width: int, height: int, source: str) -> dict:
    """K_und = K（モジュールの説明を参照）。"""
    K = mat3(K)
    if not (K[0, 0] > 0 and K[1, 1] > 0):
        raise PreprocessError("K の焦点距離が正でない")
    d = [float(x) for x in np.asarray(dist if dist is not None else [], float).ravel()]
    if len(d) not in (0, 4, 5, 8, 12, 14):
        raise PreprocessError(f"歪み係数の数 {len(d)} は OpenCV の形式（4, 5, 8, 12, 14）でない")
    k = [float(x) for x in K.ravel()]
    return {"width": int(width), "height": int(height), "K": k, "dist": d, "K_und": list(k), "source": source}


_maps_cache: dict = {}


def und_maps(cam: dict):
    key = (tuple(cam["K"]), tuple(cam.get("dist") or []), tuple(cam.get("K_und") or cam["K"]),
           cam["width"], cam["height"])
    m = _maps_cache.get(key)
    if m is None:
        K, dist, K_und = cam_arrays(cam)
        m = cv2.initUndistortRectifyMap(K, dist, None, K_und, (cam["width"], cam["height"]), cv2.CV_32FC1)
        if len(_maps_cache) > 8:
            _maps_cache.clear()
        _maps_cache[key] = m
    return m


def undistort(img: np.ndarray, cam: dict, interp=cv2.INTER_LINEAR) -> np.ndarray:
    if not has_dist(cam) and cam["K"] == (cam.get("K_und") or cam["K"]):
        return img.copy()
    mx, my = und_maps(cam)
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_und(cam: dict) -> np.ndarray:
    """歪み補正画像のうち原画像の内側から来た画素（bool）。"""
    w, h = cam["width"], cam["height"]
    if not has_dist(cam):
        return np.ones((h, w), bool)
    mx, my = und_maps(cam)
    return (mx >= -0.5) & (mx <= w - 0.5) & (my >= -0.5) & (my <= h - 0.5)


def raw_to_und(pts, cam: dict) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    if not has_dist(cam) and cam["K"] == (cam.get("K_und") or cam["K"]):
        return p.reshape(-1, 2).copy()
    K, dist, K_und = cam_arrays(cam)
    return cv2.undistortPoints(p, K, dist, None, None, K_und,
                                   (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)).reshape(-1, 2)


def und_to_raw(pts, cam: dict) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    if not has_dist(cam) and cam["K"] == (cam.get("K_und") or cam["K"]):
        return p.copy()
    K, dist, K_und = cam_arrays(cam)
    n = np.linalg.solve(K_und, np.c_[p, np.ones(len(p))].T).T
    out, _ = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return out.reshape(-1, 2)


def remap_und_points(pts, old_cam: dict, new_cam: dict) -> np.ndarray:
    """古い校正の歪み補正座標 → 新しい校正の歪み補正座標（原画像を経由）。"""
    return raw_to_und(und_to_raw(pts, old_cam), new_cam)


def remap_und_image(img: np.ndarray, old_cam: dict, new_cam: dict) -> np.ndarray:
    """古い校正の歪み補正画像（ラベル・深度）を新しい校正の歪み補正画像へ（最近傍）。"""
    w, h = new_cam["width"], new_cam["height"]
    if has_dist(new_cam):
        mx, my = und_maps(new_cam)                  # 新 und 画素 → 原画像
        raw = np.dstack([mx, my]).reshape(-1, 2)
    else:
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        raw = und_to_raw(np.c_[u.ravel(), v.ravel()], new_cam)
    old = raw_to_und(raw, old_cam).astype(np.float32).reshape(h, w, 2)
    return cv2.remap(img, old[..., 0], old[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --------------------------------------------------------------------------- マーカー

def aruco_dict(name: str):
    if name not in DICTS:
        raise PreprocessError(f"ArUco 辞書 {name} は OpenCV にない")
    return cv2.aruco.getPredefinedDictionary(DICTS[name])


def refine_quad_edges(gray: np.ndarray, quad: np.ndarray, half: float = 4.0) -> np.ndarray | None:
    """四辺それぞれで輝度勾配の極大（法線方向、放物線で補間）を拾って直線を当てはめ、隣り合う直線の交点を四隅にする。

    歪み補正画像（直線が直線のまま）で使う。cornerSubPix は凸な角（黒い正方形の角）でぼけると内側へ 0.2 px ほど寄るので、
    辺の当てはめで置き換える。うまくいかない辺があれば None。
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
    h, w = g.shape
    q = np.asarray(quad, float).reshape(4, 2)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        L = np.linalg.norm(b - a)
        if L < 12:
            return None
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        ts = np.linspace(0.12, 0.88, max(8, int(L / 2)))
        ss = np.arange(-half, half + 1e-9, 0.25)
        base = a[None] + ts[:, None] * (b - a)[None]
        pts = base[:, None, :] + ss[None, :, None] * n[None, None, :]
        if pts[..., 0].min() < 1 or pts[..., 1].min() < 1 or pts[..., 0].max() > w - 2 or pts[..., 1].max() > h - 2:
            return None
        prof = cv2.remap(g, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR)
        grad = np.abs(np.gradient(prof, axis=1))
        k = grad.argmax(1)
        ok = (k > 0) & (k < len(ss) - 1)
        if ok.sum() < 6:
            return None
        r = np.arange(len(k))[ok]
        k = k[ok]
        y0, y1, y2 = grad[r, k - 1], grad[r, k], grad[r, k + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0).clip(-0.5, 0.5)
        sk = ss[k] + off * 0.25
        e = base[r] + sk[:, None] * n[None]
        if np.median(grad[r, k]) < 4:              # コントラストが低い
            return None
        vx, vy, x0, y0_ = cv2.fitLine(e.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0_], float), np.array([vx, vy], float)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return None
        s1, _ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s1 * d1)
    out = np.array(out)
    if np.abs(out - q).max() > 2.5:
        return None
    return out


def detect_markers(bgr: np.ndarray, cam: dict, dictionary: str, only_id: int | None = None,
                   und: np.ndarray | None = None) -> list[dict]:
    """原画像で検出し、四隅（0..3 = 左上・右上・右下・左下）を raw / und の両方で返す。

    und（歪み補正画像）を渡すと、四隅を und 上の辺の当てはめ（refine_quad_edges）で磨き、raw はそこから求め直す。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    det = cv2.aruco.ArucoDetector(aruco_dict(dictionary), params)
    corners, ids, _ = det.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    gu = None
    if und is not None:
        gu = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY) if und.ndim == 3 else und
    for c, i in zip(corners, ids.ravel()):
        if only_id is not None and int(i) != only_id:
            continue
        raw = c.reshape(4, 2).astype(np.float64)
        u = raw_to_und(raw, cam)
        refined = "subpix"
        if gu is not None:
            r = refine_quad_edges(gu, u)
            if r is not None:
                u = r
                raw = und_to_raw(u, cam)
                refined = "edges"
        out.append({"id": int(i), "corners_raw": raw.round(3).tolist(), "corners_und": u.round(3).tolist(),
                    "source": "auto", "refine": refined})
    out.sort(key=lambda m: m["id"])
    return out


def marker_points(size_mm: float) -> np.ndarray:
    s = size_mm / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)


# --------------------------------------------------------------------------- 品質

def quality(bgr: np.ndarray, valid: np.ndarray | None = None, markers_found: int | None = None,
            role: str = "object") -> dict:
    """blur_var = エッジ近傍（Canny の ±2 px）のラプラシアンの分散。画面全体の分散は模様の量で大きく変わるので、
    エッジのある所だけで測る（ぼけるとエッジの鋭さが落ちて小さくなる）。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = valid if valid is not None else np.ones(gray.shape, bool)
    g = gray[m]
    near = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((5, 5), np.uint8)).astype(bool) & m
    lap = cv2.Laplacian(gray, cv2.CV_64F)[near]
    blur = float(lap.var()) if lap.size >= 0.001 * gray.size else 0.0
    over = float((g >= 250).mean() * 100) if g.size else 0.0
    under = float((g <= 5).mean() * 100) if g.size else 0.0
    notes, ok = [], True
    if blur < 40:
        ok = False
        notes.append(f"ぶれ・ピンぼけの可能性（エッジ近傍のラプラシアン分散 {blur:.0f} < 40）")
    if over > 5:
        ok = False
        notes.append(f"白飛び {over:.1f}%")
    if under > 30:
        ok = False
        notes.append(f"黒つぶれ {under:.1f}%")
    if markers_found == 0 and role == "object":
        notes.append("基準マーカーが見えない（裏側の画像なら正常。姿勢は手動か対応点で合わせる）")
    return {"blur_var": round(blur, 2), "overexposed_pct": round(over, 3), "underexposed_pct": round(under, 3),
            "ok": ok, "notes": notes}


# --------------------------------------------------------------------------- ラベル候補

K3 = np.ones((3, 3), np.uint8)


def marker_mask(shape, markers: list[dict]) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for mk in markers:
        q = np.asarray(mk["corners_und"], np.float64)
        cv2.fillConvexPoly(m, np.round(q * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    return m.astype(bool)


def tidy(fg: np.ndarray) -> np.ndarray:
    """3×3 の open → close を 1 回ずつ。穴埋めはしない。"""
    u = fg.astype(np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, K3)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, K3)
    return u.astype(bool)


def keep_main(fg: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """物体の連結成分だけを残す: マーカーが見えていればマーカーに触れる成分、なければ最大の成分。

    背景の雑多な物が別の成分として前景に残るのを防ぐ（物体が複数の成分に割れたときは人がブラシで足す）。
    """
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n <= 1:
        return fg
    keep = np.zeros(n, bool)
    if seeds.any():
        keep[np.unique(lab[seeds])] = True
    else:
        keep[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))] = True
    keep[0] = False
    return keep[lab]


def lab_float(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _sample(a: np.ndarray, n: int, rng) -> np.ndarray:
    if len(a) <= n:
        return a
    return a[rng.choice(len(a), n, replace=False)]


def _centers(px: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(px)))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, c = cv2.kmeans(px.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return c


def _min_dist(px: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sqrt(((px[:, None, :] - centers[None]) ** 2).sum(-1)).min(1)


def enclosed_regions(fg: np.ndarray, valid: np.ndarray):
    """物体（fg）に完全に囲まれた非物体の連結成分（4 連結）。画像の端・無効域に触れるものは除く。"""
    inv = (~fg).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    border = np.zeros(n, bool)
    border[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = True
    border[np.unique(lab[~valid])] = True
    out = []
    for i in range(1, n):
        if not border[i] and inv[lab == i].any():
            out.append((lab == i, int(st[i, cv2.CC_STAT_AREA])))
    return out


def classify_holes(und: np.ndarray, fg: np.ndarray, valid: np.ndarray, diff: np.ndarray | None,
                   diff_thr: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """囲まれた領域を「穴（背景が見えている）」と「物体（暗い・模様などで物体から外れただけ）」に分ける。

    差分があるとき: 領域の差分の中央値がしきい値未満なら穴。
    差分がないとき: 物体の外周のすぐ外（幅 ~20 px の帯）の背景の色（k-means 8 色）に近く、
    物体の色より背景の色に近いときだけ穴。暗いだけでは穴にしない。
    """
    holes = np.zeros(fg.shape, bool)
    to_obj = np.zeros(fg.shape, bool)
    notes: list[str] = []
    regions = enclosed_regions(fg, valid)
    if not regions:
        return holes, to_obj, np.zeros(fg.shape, bool), notes
    lab = None
    if diff is None:
        lab = lab_float(und)
        filled = fg.copy()
        for r, _ in regions:
            filled |= r
        ring = cv2.dilate(filled.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
        ring &= ~filled & valid
        bg_px = _sample(lab[ring], 20000, rng)
        fg_px = _sample(lab[fg], 20000, rng)
        if len(bg_px) < 50 or len(fg_px) < 50:
            notes.append("穴の判定に使う背景・物体の画素が少ない。囲まれた領域は未定義（0）のまま")
            unknown = np.zeros(fg.shape, bool)
            for r, _ in regions:
                unknown |= r
            return holes, to_obj, unknown, notes
        cb, cf = _centers(bg_px, 8), _centers(fg_px, 8)
    n_hole = n_rej = 0
    for r, area in regions:
        if diff is not None:
            is_hole = float(np.median(diff[r])) < diff_thr
        else:
            px = _sample(lab[r], 4000, rng)
            db, df = np.median(_min_dist(px, cb)), np.median(_min_dist(px, cf))
            is_hole = db < 12.0 and db < df
        if is_hole:
            holes |= r
            n_hole += 1
        else:
            to_obj |= r
            n_rej += 1
    if n_hole:
        notes.append(f"穴の候補 {n_hole} 箇所（物体に囲まれ、背景と同じに見える領域）")
    if n_rej:
        notes.append(f"物体に囲まれているが背景に見えない領域 {n_rej} 箇所は物体のまま（暗い・模様の可能性。穴なら穴ブラシで直す）")
    return holes, to_obj, np.zeros(fg.shape, bool), notes


def grabcut_roi(img: np.ndarray, mask: np.ndarray, iters: int, margin: int = 40) -> np.ndarray:
    """確定背景で囲まれた範囲だけで GrabCut を回す（速くするため）。mask は GC_* 値で書き換えて返す。"""
    unsure = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD, cv2.GC_PR_BGD)) if (mask == cv2.GC_BGD).any() else np.ones(mask.shape, bool)
    ys, xs = np.nonzero(unsure)
    if len(xs) == 0:
        return mask
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1])
    has_f = np.isin(sub, (cv2.GC_FGD, cv2.GC_PR_FGD)).any()
    has_b = np.isin(sub, (cv2.GC_BGD, cv2.GC_PR_BGD)).any()
    if not (has_f and has_b):
        return mask
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(np.ascontiguousarray(img[y0:y1, x0:x1]), sub, None, bgm, fgm, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        raise PreprocessError(f"GrabCut に失敗: {e}") from None
    out = mask.copy()
    out[y0:y1, x0:x1] = sub
    return out


def fg_by_grabcut(und: np.ndarray, valid: np.ndarray, mk: np.ndarray, rect=None, iters: int = 4) -> np.ndarray:
    """初期値: 画像の外周 6 %（または rect の外）を確定背景、内側を「たぶん前景」、マーカーを確定前景。

    1/2 に縮小して回し、元の解像度で境界の帯（±4 px）だけをもう一度回す。
    """
    h, w = valid.shape
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    if rect:
        x, y, rw, rh = [int(round(v)) for v in rect]
    else:
        x, y, rw, rh = int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88)
    mask[max(0, y):y + rh, max(0, x):x + rw] = cv2.GC_PR_FGD
    mask[~valid] = cv2.GC_BGD
    mask[mk] = cv2.GC_FGD
    s = 0.5
    small = cv2.resize(und, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    ms = grabcut_roi(small, ms, iters, margin=20)
    fg0 = cv2.resize(np.isin(ms, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_LINEAR).astype(bool)
    band = cv2.dilate(fg0.astype(np.uint8), K3, iterations=4).astype(bool) & \
        ~cv2.erode(fg0.astype(np.uint8), K3, iterations=4).astype(bool)
    m2 = np.where(fg0, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
    m2[band] = np.where(fg0[band], cv2.GC_PR_FGD, cv2.GC_PR_BGD)
    m2[mk] = cv2.GC_FGD
    m2[~valid] = cv2.GC_BGD
    m2 = grabcut_roi(und, m2, 2, margin=8)
    return np.isin(m2, (cv2.GC_FGD, cv2.GC_PR_FGD))


def diff_image(und: np.ndarray, bg_und: np.ndarray) -> np.ndarray:
    a = lab_float(cv2.GaussianBlur(und, (5, 5), 0))
    b = lab_float(cv2.GaussianBlur(bg_und, (5, 5), 0))
    return np.sqrt(((a - b) ** 2).sum(-1))


def auto_labels(und: np.ndarray, valid: np.ndarray, markers: list[dict], bg_und: np.ndarray | None = None,
                options: dict | None = None) -> tuple[np.ndarray, list[str], str]:
    """ラベル候補（0..4）、注記、方式（"background_diff" / "grabcut"）。"""
    options = options or {}
    rng = np.random.default_rng(0)
    mk = marker_mask(und.shape, markers) & valid
    notes: list[str] = []
    diff = None
    thr = 0.0
    if bg_und is not None and bg_und.shape == und.shape:
        method = "background_diff"
        diff = diff_image(und, bg_und)
        thr = float(options.get("diff_threshold") or max(10.0, 4.0 * float(np.median(diff[valid]))))
        fg = (diff > thr) & valid
        notes.append(f"背景画像との差分（ΔE > {thr:.1f}）")
    else:
        method = "grabcut"
        if bg_und is not None:
            notes.append("背景画像の大きさが違うので GrabCut を使った")
        fg = fg_by_grabcut(und, valid, mk, options.get("rect"), int(options.get("grabcut_iters", 4)))
        notes.append("GrabCut（中央の広い矩形とマーカーを前景の種に）")
    fg = tidy(fg) | mk
    fg = keep_main(fg, mk)
    if not fg.any():
        notes.append("物体の領域が見つからない")
    holes, to_obj, unknown, hn = classify_holes(und, fg, valid, diff, thr * 0.6 if diff is not None else 0, rng)
    notes += hn
    labels = np.full(und.shape[:2], LABEL_BG, np.uint8)
    labels[fg | to_obj] = LABEL_OBJ
    labels[holes] = LABEL_HOLE
    labels[unknown] = LABEL_UNKNOWN
    labels[mk] = LABEL_MARKER
    labels[~valid] = LABEL_UNKNOWN
    return labels, notes, method


def strokes_mask(shape, strokes, value_mask: np.ndarray | None = None) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for s in strokes or []:
        pts = np.asarray(s.get("points") or s.get("pts") or [], np.float64).reshape(-1, 2)
        r = max(1, int(round(float(s.get("radius", 5)))))
        if len(pts) == 0:
            continue
        p = np.round(pts).astype(np.int32)
        if len(p) == 1:
            cv2.circle(m, tuple(int(v) for v in p[0]), r, 1, -1)
        else:
            cv2.polylines(m, [p], False, 1, 2 * r + 1)
            for q in p[[0, -1]]:
                cv2.circle(m, tuple(int(v) for v in q), r, 1, -1)
    return m.astype(bool)


def grabcut_with_seeds(und: np.ndarray, seed_labels: np.ndarray | None, fg_strokes, bg_strokes,
                       iters: int = 5) -> np.ndarray:
    """前景・背景のストローク（歪み補正画素、太さ = 半径）と既存ラベルを種に GrabCut。ラベル 3・4・0 はそのまま残す。"""
    h, w = und.shape[:2]
    if seed_labels is None:
        seed_labels = np.full((h, w), LABEL_BG, np.uint8)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[seed_labels == LABEL_OBJ] = cv2.GC_PR_FGD
    mask[seed_labels == LABEL_HOLE] = cv2.GC_BGD
    mask[seed_labels == LABEL_MARKER] = cv2.GC_FGD
    mask[seed_labels == LABEL_UNKNOWN] = cv2.GC_BGD
    f, b = strokes_mask(und.shape, fg_strokes), strokes_mask(und.shape, bg_strokes)
    keep = np.isin(seed_labels, (LABEL_HOLE, LABEL_MARKER, LABEL_UNKNOWN))
    mask[f & ~keep] = cv2.GC_FGD
    mask[b & ~keep] = cv2.GC_BGD
    if not np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).any():
        raise PreprocessError("前景の種がない（前景ブラシで物体をなぞる）")
    # 前景候補から離れた背景は確定背景にして範囲を絞る
    near = cv2.dilate(np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
    mask[~near & (mask == cv2.GC_PR_BGD)] = cv2.GC_BGD
    mask = grabcut_roi(und, mask, iters)
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    fg = tidy(fg) | f & ~keep
    out = np.where(fg, LABEL_OBJ, LABEL_BG).astype(np.uint8)
    out[keep] = seed_labels[keep]
    return out


def edges(und: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), lo, hi)


def contours_from_labels(labels: np.ndarray, epsilon_px: float = 1.0, min_area: float = 30.0) -> list[dict]:
    """外周 = 物体（2・3・4 を合わせた領域）の外側、穴 = 物体の内側のラベル 3。頂点は approxPolyDP（ε ≈ 1 px）で間引く。"""
    out = []
    solid = np.isin(labels, (LABEL_OBJ, LABEL_HOLE, LABEL_MARKER)).astype(np.uint8)
    cs, hier = cv2.findContours(solid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outers = []
    if hier is not None:
        for c, hh in zip(cs, hier[0]):
            if hh[3] != -1 or cv2.contourArea(c) < min_area:
                continue
            outers.append(c)
            a = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
            out.append({"kind": "outer", "points_und": a.astype(float).tolist()})
    hole = (labels == LABEL_HOLE).astype(np.uint8)
    hs, hier = cv2.findContours(hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        for c, hh in zip(hs, hier[0]):
            if hh[3] != -1 or cv2.contourArea(c) < min_area:
                continue
            p = c.reshape(-1, 2)[0]
            if not any(cv2.pointPolygonTest(o, (float(p[0]), float(p[1])), False) >= 0 for o in outers):
                continue
            a = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
            out.append({"kind": "hole", "points_und": a.astype(float).tolist()})
    return out


# --------------------------------------------------------------------------- 姿勢

def T_from_rt(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, float).ravel()
    return T


def T_list(T: np.ndarray) -> list[float]:
    return [float(round(v, 9)) for v in np.asarray(T, float).ravel()]


def T_arr(v) -> np.ndarray:
    return np.asarray(v, float).reshape(4, 4)


def project(T_cam_x: np.ndarray, pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    p = (T_cam_x[:3, :3] @ np.asarray(pts, float).T).T + T_cam_x[:3, 3]
    uv = (K @ p.T).T
    return uv[:, :2] / uv[:, 2:3]


def marker_pose(corners_und, K_und: np.ndarray, size_mm: float) -> dict:
    """IPPE_SQUARE の 2 解。best が 0 番目。ambiguity = 誤差(best) / 誤差(2 番目)（1 に近いほど曖昧）。"""
    img = np.asarray(corners_und, np.float64).reshape(4, 1, 2)
    obj = marker_points(size_mm)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K_und, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    sols = []
    for r, t in zip(rvecs, tvecs):
        T = T_from_rt(r, t)
        res = np.linalg.norm(project(T, obj, K_und) - img.reshape(4, 2), axis=1)
        sols.append({"T_cam_marker": T_list(T), "rms_px": float(np.sqrt((res ** 2).mean())),
                     "residuals_px": res.round(4).tolist()})
    if not sols:
        raise PreprocessError("マーカーの姿勢が解けない")
    sols.sort(key=lambda s: s["rms_px"])
    amb = sols[0]["rms_px"] / sols[1]["rms_px"] if len(sols) > 1 and sols[1]["rms_px"] > 0 else 0.0
    side = float(np.mean([np.linalg.norm(img[i, 0] - img[(i + 1) % 4, 0]) for i in range(4)]))
    notes = []
    if amb > 0.6:
        notes.append("2 通りの向きの誤差が近い（マーカーが小さい・正面向き）。他の画像や形状と見比べて確認する")
    if side < 25:
        notes.append(f"マーカーが小さい（一辺 {side:.0f} px）")
    if sols[0]["rms_px"] > 1.5:
        notes.append(f"再投影誤差が大きい（{sols[0]['rms_px']:.2f} px）。四隅の位置を確認する")
    return {"solutions": sols, "reproj_px": sols[0]["rms_px"], "ambiguity": float(amb), "side_px": side,
            "note": "。".join(notes)}


def _rot_angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def pnp(uv_und, p_marker, K_und: np.ndarray, ambiguity_factor: float = 1.5, noise_floor_px: float = 0.5) -> dict:
    """対応点（歪み補正画素 ↔ マーカー座標 mm）から T_cam_marker の候補を全部返す。

    - 4 点以上・非平面: SQPNP（＋ITERATIVE で磨く）。平面: IPPE の 2 解。3〜4 点: P3P / AP3P の解も足す。
    - degenerate: 4 点未満、3D 点がほぼ一直線、画像上の広がりが小さい（第 2 特異値 < 3 px）
    - ambiguous: 2 番目の解の RMS が best の ambiguity_factor 倍以内（best は noise_floor_px で下駄を履かせる）
    """
    uv = np.asarray(uv_und, np.float64).reshape(-1, 2)
    P = np.asarray(p_marker, np.float64).reshape(-1, 3)
    n = len(uv)
    if n != len(P):
        raise PreprocessError("uv_und と p_obj の数が違う")
    notes, degenerate = [], False
    if n < 3:
        raise PreprocessError("対応点が 3 点未満では姿勢を解けない")
    if n < 4:
        degenerate = True
        notes.append("対応点が 4 点未満（解が一意に決まらない）")
    sP = np.linalg.svd(P - P.mean(0), compute_uv=False)
    su = np.linalg.svd(uv - uv.mean(0), compute_uv=False)
    if sP[0] <= 1e-9 or sP[1] / sP[0] < 0.02:
        degenerate = True
        notes.append("3D の点がほぼ一直線")
    if su[1] < 3.0 * math.sqrt(n):
        degenerate = True
        notes.append("画像上の点の広がりが小さい（ほぼ一直線か一点に集中）")
    coplanar = sP[0] > 0 and sP[2] / sP[0] < 1e-3
    dist0 = np.zeros(5)
    cands = []

    def add(method, rvecs, tvecs):
        for r, t in zip(rvecs, tvecs):
            cands.append((method, np.asarray(r, float).reshape(3), np.asarray(t, float).reshape(3)))

    o3, i2 = P.reshape(-1, 1, 3), uv.reshape(-1, 1, 2)
    methods = []
    if n >= 4:
        methods.append(("IPPE", cv2.SOLVEPNP_IPPE) if coplanar else ("SQPNP", cv2.SOLVEPNP_SQPNP))
        if not coplanar:
            methods.append(("EPNP", cv2.SOLVEPNP_EPNP))
    if n in (3, 4):
        methods += [("P3P", cv2.SOLVEPNP_P3P), ("AP3P", cv2.SOLVEPNP_AP3P)]
    for name, flag in methods:
        try:
            k, rv, tv, _ = cv2.solvePnPGeneric(o3, i2, K_und, dist0, flags=flag)
            add(name, rv, tv)
        except cv2.error:
            pass
    if n >= 4 and coplanar:
        try:
            k, rv, tv, _ = cv2.solvePnPGeneric(o3, i2, K_und, dist0, flags=cv2.SOLVEPNP_SQPNP)
            add("SQPNP", rv, tv)
        except cv2.error:
            pass
    sols = []
    for method, r, t in cands:
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
            continue
        if n >= 4:
            try:
                r2, t2 = cv2.solvePnPRefineLM(o3, i2, K_und, dist0, r.reshape(3, 1).copy(), t.reshape(3, 1).copy())
                r, t = r2.ravel(), t2.ravel()
            except cv2.error:
                pass
        T = T_from_rt(r, t)
        pc = (T[:3, :3] @ P.T).T + T[:3, 3]
        if np.any(pc[:, 2] <= 0):
            continue                       # カメラの後ろ
        res = np.linalg.norm(project(T, P, K_und) - uv, axis=1)
        dup = False
        for s in sols:
            Ts = T_arr(s["T_cam_marker"])
            if _rot_angle_deg(Ts[:3, :3], T[:3, :3]) < 0.5 and np.linalg.norm(Ts[:3, 3] - T[:3, 3]) < 0.5:
                dup = True
                if np.sqrt((res ** 2).mean()) < s["rms_px"]:
                    s.update(T_cam_marker=T_list(T), rms_px=float(np.sqrt((res ** 2).mean())),
                             max_px=float(res.max()), residuals_px=res.round(4).tolist())
                s["methods"] = sorted(set(s["methods"]) | {method})
                break
        if not dup:
            sols.append({"T_cam_marker": T_list(T), "rms_px": float(np.sqrt((res ** 2).mean())),
                         "max_px": float(res.max()), "residuals_px": res.round(4).tolist(), "methods": [method]})
    sols.sort(key=lambda s: s["rms_px"])
    if not sols:
        return {"solutions": [], "degenerate": True, "ambiguous": False, "coplanar": bool(coplanar),
                "notes": notes + ["解が見つからない"]}
    ambiguous = len(sols) > 1 and sols[1]["rms_px"] <= ambiguity_factor * max(sols[0]["rms_px"], noise_floor_px)
    if ambiguous:
        notes.append(f"2 番目の解の誤差（{sols[1]['rms_px']:.2f} px）が best（{sols[0]['rms_px']:.2f} px）と近い。"
                     "対応点を増やすか、奥行きの違う点を足す")
    for s in sols:
        s["ambiguous"] = bool(ambiguous)
        s["degenerate"] = bool(degenerate)
    return {"solutions": sols, "degenerate": bool(degenerate), "ambiguous": bool(ambiguous),
            "coplanar": bool(coplanar), "notes": notes}


# --------------------------------------------------------------------------- 深度

def load_depth(data: bytes, filename: str = "") -> tuple[np.ndarray, str, str]:
    """(配列, 単位 "mm"/"m", 保存名)。PNG は 16 bit mm、NPY は float32 m。"""
    if data[:6] == b"\x93NUMPY" or filename.lower().endswith(".npy"):
        try:
            a = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            raise PreprocessError(f"npy が読めない: {e}") from None
        if a.ndim != 2:
            raise PreprocessError(f"深度は 2 次元配列が必要（受け取った形 {a.shape}）")
        a = a.astype(np.float32)
        a[~np.isfinite(a)] = 0
        a[a < 0] = 0
        return a, "m", "depth.npy"
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise PreprocessError("深度が読めない（16 bit PNG（mm）か .npy（float32、m）を送る）")
    if a.ndim != 2 or a.dtype != np.uint16:
        raise PreprocessError(f"深度 PNG は 1 チャンネル 16 bit（mm）でなければならない（受け取った {a.dtype}、{a.shape}）")
    return a, "mm", "depth.png"


def depth_points(depth: np.ndarray, unit: str, invalid: float, K_und: np.ndarray, step: int = 4,
                 colors: np.ndarray | None = None, max_points: int = 300000) -> dict:
    step = max(1, int(step))
    d = depth[::step, ::step].astype(np.float64)
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = d * (1000.0 if unit == "m" else 1.0)
    ok = np.isfinite(z) & (z > 0) & (d != invalid)
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    if len(z) > max_points:
        k = np.linspace(0, len(z) - 1, max_points).astype(int)
        u, v, z = u[k], v[k], z[k]
    fx, fy, cx, cy = K_und[0, 0], K_und[1, 1], K_und[0, 2], K_und[1, 2]
    pts = np.c_[(u - cx) * z / fx, (v - cy) * z / fy, z]
    out = {"frame": "cam", "unit": "mm", "step": step, "count": int(len(z)), "points": pts.round(2).tolist()}
    if colors is not None:
        c = colors[v.astype(int), u.astype(int)][:, ::-1]   # BGR → RGB
        out["colors"] = c.astype(int).tolist()
    return out


# --------------------------------------------------------------------------- 画像単位の処理（ファイル）

def choose_camera(project: dict, width: int, height: int) -> tuple[str, dict | None]:
    """画像に使うカメラの名前と、新しく足すカメラ（不要なら None）。大きさの合うカメラを使い、なければ校正なしで作る。"""
    cams = project.get("cameras", {})
    for key in ["oak"] + sorted(cams):
        c = cams.get(key)
        if c and c["width"] == width and c["height"] == height:
            return key, None
    key = "oak" if "oak" not in cams else f"cam_{width}x{height}"
    return key, no_camera(width, height)


def labels_file(img_dir: Path) -> Path:
    f = img_dir / "labels.png"
    return f if f.is_file() else img_dir / "labels_auto.png"


def read_labels(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.uint8)


def run_steps(project: dict, im: dict, img_dir: Path, steps=None, options: dict | None = None) -> dict:
    """前処理をやり直し、更新した Image（コピー）を返す。ファイル（undistorted / labels_auto / edges）を書く。

    labels.png（人の修正）は変えない。手動のマーカー四隅（source = manual）は options.replace_manual が無い限り残す。
    """
    options = dict(options or {})
    steps = [s for s in (steps or ALL_STEPS)]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise PreprocessError(f"知らない前処理 {bad}（使えるもの: {ALL_STEPS}）")
    im = copy.deepcopy(im)
    cam = project["cameras"][im["camera"]]
    raw = read_bgr(img_dir / im["file"])
    if raw.shape[1] != cam["width"] or raw.shape[0] != cam["height"]:
        raise PreprocessError(f"画像 {raw.shape[1]}×{raw.shape[0]} とカメラ {im['camera']} の大きさが違う")
    info = dict(im.get("preprocess_info") or {})
    notes = []
    und_path = img_dir / "undistorted.png"
    if "undistort" in steps or not und_path.is_file():
        und = undistort(raw, cam)
        write_png(und_path, und)
    else:
        und = read_bgr(und_path)
    valid = valid_und(cam)
    mk_spec = project.get("marker", {})
    if "markers" in steps:
        dictionary = options.get("dictionary") or mk_spec.get("dictionary", "DICT_4X4_50")
        only = options.get("marker_id")
        found = detect_markers(raw, cam, dictionary, None if only is None else int(only), und)
        manual = [] if options.get("replace_manual") else [m for m in im.get("markers", []) if m.get("source") == "manual"]
        man_ids = {m["id"] for m in manual}
        for m in manual:                       # 手動の四隅は原画像の座標から und を作り直す
            m["corners_und"] = raw_to_und(m["corners_raw"], cam).round(3).tolist()
        im["markers"] = manual + [m for m in found if m["id"] not in man_ids]
        ref = int(mk_spec.get("id", -1))
        notes.append(f"マーカー検出（{dictionary}）: {[m['id'] for m in found] or 'なし'}"
                     + ("" if any(m["id"] == ref for m in im["markers"]) else f"。基準マーカー ID {ref} は見えない"))
    elif "undistort" in steps:
        for m in im.get("markers", []):
            m["corners_und"] = raw_to_und(m["corners_raw"], cam).round(3).tolist()
    if "quality" in steps:
        ref = int(mk_spec.get("id", -1))
        im["quality"] = quality(und, valid, sum(1 for m in im["markers"] if m["id"] == ref), im.get("role", "object"))
    if "mask" in steps:
        if im.get("role") == "background":
            lab = np.where(valid, LABEL_BG, LABEL_UNKNOWN).astype(np.uint8)
            info["mask_method"] = "background_image"
            notes.append("背景のみの画像（全体を背景 1 に）")
        else:
            bg = None
            bid = project.get("background_image")
            if bid and bid != im["id"] and options.get("use_background", True):
                f = img_dir.parent / bid / "undistorted.png"
                if f.is_file():
                    bg = read_bgr(f)
            lab, mn, method = auto_labels(und, valid, im["markers"], bg, options)
            info["mask_method"] = method
            notes += mn
        write_png(img_dir / "labels_auto.png", lab)
    if "edges" in steps:
        write_png(img_dir / "edges.png", edges(und, int(options.get("canny_lo", 50)), int(options.get("canny_hi", 150))))
    if "contours" in steps:
        lab = read_labels(labels_file(img_dir))
        if lab is not None:
            im["contours"] = [c for c in im.get("contours", []) if c.get("status") == "confirmed"]
            for c in contours_from_labels(lab, float(options.get("epsilon_px", 1.0))):
                c["id"] = next_contour_id(im)
                c["status"] = "candidate"
                im["contours"].append(c)
            notes.append(f"輪郭候補 外周 {sum(c['kind'] == 'outer' for c in im['contours'])}・"
                         f"穴 {sum(c['kind'] == 'hole' for c in im['contours'])}")
    info["notes"] = notes
    info["steps"] = steps
    info["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    im["preprocess_info"] = info
    return im


def import_image(project: dict, img_dir: Path, iid: str, data: bytes, filename: str, role: str,
                 cameras_out: dict) -> dict:
    """取り込み: 元画像を保存（向きの適用が要るときは original.png に回した画素、元のバイト列は upload.<ext>）→ 全前処理。

    新しく作ったカメラは cameras_out に入れる（呼び出し側で project に足す）。project は読み取りだけ（カメラを足した写しを渡す）。
    """
    ext = sniff_ext(data)
    if ext is None:
        raise PreprocessError(f"{filename}: 画像として読めない（JPEG / PNG / BMP / TIFF / WebP に対応）")
    bgr, o = decode(data)
    img_dir.mkdir(parents=True, exist_ok=True)       # 呼び出し側が id を確保して作っていることがある
    if o == 1:
        file = f"original.{ext}"
        (img_dir / file).write_bytes(data)
    else:
        (img_dir / f"upload.{ext}").write_bytes(data)
        file = "original.png"
        write_png(img_dir / file, bgr)
    h, w = bgr.shape[:2]
    cams = dict(project.get("cameras", {}), **cameras_out)
    key, new = choose_camera({"cameras": cams}, w, h)
    if new is not None:
        cameras_out[key] = new
        cams[key] = new
    im = new_image(iid, file, key, w, h, role)
    im["import_info"] = {"filename": filename, "exif_orientation": o, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()}
    return run_steps(dict(project, cameras=cams), im, img_dir)
