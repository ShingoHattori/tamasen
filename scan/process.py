#!/usr/bin/env python3
"""iPad 深度スキャン（RGB + 位置合わせ済み深度）から、指定マーカー座標で表した物体の 3D モデルを作る。

  scan/.venv/bin/python scan/process.py <session_dir> --marker-id 0 --marker-mm 20 [--out DIR]

手順:
  1. 各フレームで ArUco を検出し、四隅を Apple の歪み LUT で歪みなしへ戻して IPPE_SQUARE で 2 解を求める。
     マーカー内の深度に平面を当て、法線（傾き）と奥行きを深度で決め直す（RGB だけの PnP は小さいマーカーで
     傾きと z が弱い）。深度で決まらないときは誤差比で曖昧・反転を棄却する。小さいマーカーも棄却。
     ほかのマーカー（--marker-sizes）が写っていれば、同時に写ったフレームから基準マーカーとの相対姿勢を求め、
     基準が見えないフレームの姿勢にも使う。
  2. 深度の段差（飛び画素）を除き、歪みを除いた深度を点群にしてマーカー座標へ移し、箱で切り出す。
  3. 手の除去: 2 mm のボクセルごとに、各フレームの深度へ投影して「表面として見えた」「その手前が空いて
     見えた（素通し）」を数え、見えたときに十分な割合で占有だったボクセルだけを残す（手は物体に対して
     動くので素通しの票が集まる）。肌色マスクは任意。
  4. 残った点を模型にして各フレームの姿勢を点-面 ICP で少しだけ補正（事前分布つき・上限つき）し、
     補正の平均は差し引く（マーカー座標の定義はマーカーが決める）。投票をやり直す。
  5. マーカー座標の TSDF に統合し、マーチングキューブでメッシュ化。穴を塞ぎ、三角形数を減らす。
  出力は README.md を参照。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lens import Camera, FrameMeta, Session  # noqa: E402
import viz  # noqa: E402

DICT = cv2.aruco.DICT_4X4_50


# --------------------------------------------------------------------------- 小道具

def rigid(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a を b へ回す最小回転。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * _perp(a)).as_matrix()
    return Rotation.from_rotvec(v / s * math.atan2(s, c)).as_matrix()


def _perp(a):
    p = np.cross(a, [1, 0, 0])
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(a, [0, 1, 0])
    return p / np.linalg.norm(p)


def angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def marker_object_points(size: float) -> np.ndarray:
    """cv2.aruco の四隅順（左上・右上・右下・左下）、マーカー座標: x 右, y 上, z 面から外向き。"""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def line_intersection(q: np.ndarray) -> np.ndarray:
    """四角形の対角線の交点（透視投影でのマーカー中心の像）。"""
    p1, p2, p3, p4 = q
    d1, d2 = p3 - p1, p4 - p2
    A = np.array([d1, -d2]).T
    try:
        s = np.linalg.solve(A, p2 - p1)
        return p1 + s[0] * d1
    except np.linalg.LinAlgError:
        return q.mean(0)


def backproject(D: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None):
    """深度画像（歪みなし）→ カメラ座標の点 (N,3) と画素添字 (N,)。"""
    h, w = D.shape
    m = D > 0 if mask is None else (mask & (D > 0))
    idx = np.flatnonzero(m)
    v, u = np.divmod(idx, w)
    z = D.ravel()[idx].astype(np.float64)
    y = (v - K[1, 2]) / K[1, 1] * z
    x = (u - K[0, 2] - K[0, 1] * y / z) / K[0, 0] * z
    return np.stack([x, y, z], 1), idx


# --------------------------------------------------------------------------- フレーム

@dataclass
class MarkerObs:
    id: int
    side_px: float
    corners_ref: np.ndarray          # 歪みなし ref cc
    T: np.ndarray | None = None      # T_cam_marker（採用した解）
    T_rgb: np.ndarray | None = None  # RGB だけの最良解
    reproj_px: float = math.nan      # 映像画素
    ambiguity: float = math.nan
    plane: dict | None = None
    status: str = "ok"               # ok | rejected
    reasons: list = field(default_factory=list)
    depth_refined: bool = False


@dataclass
class Frame:
    meta: FrameMeta
    K_d: np.ndarray                  # 歪みなし深度画像の K（cv 添字）
    D: np.ndarray                    # 歪みなし・段差除去済みの深度 (m)、無効は 0
    Dmin: np.ndarray                 # 3x3 の最小（無効は inf）
    C: np.ndarray                    # 深度解像度の歪みなし RGB
    obs: dict = field(default_factory=dict)
    T: np.ndarray | None = None      # 採用した T_cam_marker（基準マーカー）
    T_marker_only: np.ndarray | None = None
    source: str = ""
    status: str = "ok"
    reasons: list = field(default_factory=list)
    icp: dict | None = None
    track: dict | None = None
    obj_mask: np.ndarray | None = None
    n_edge: int = 0


class Prep:
    """フレームごとの前処理（検出・深度の歪み除去・段差除去）。"""

    def __init__(self, S: Session, a):
        self.S, self.a = S, a
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
        self.maps: dict = {}
        # 深度が大きいとき（TrueDepth 640x480）は 2x2 などでまとめて処理を軽くする
        self.ds = max(1, int(math.ceil(S.depth_w / a.depth_max_w)))
        self.dw, self.dh = S.depth_w // self.ds, S.depth_h // self.ds

    def _maps(self, cam: Camera):
        k = cam.key()
        if k not in self.maps:
            S = self.S
            self.maps[k] = cam.rectify_maps(self.dw, self.dh, self.dw, self.dh)
        return self.maps[k]

    def frame(self, f: FrameMeta, sizes: dict[int, float]) -> tuple[Frame | None, str | None]:
        S, a = self.S, self.a
        img = S.rgb(f)
        if img is None:
            return None, "RGB が読めない"
        if img.shape[1] != S.video_w or img.shape[0] != S.video_h:
            return None, f"RGB の寸法 {img.shape[1]}x{img.shape[0]} が session.json と違う"
        Draw = S.depth(f)
        if Draw is None:
            return None, "深度がない・寸法が違う"
        conf = S.confidence(f)
        if conf is not None and a.min_conf > 0:
            Draw[conf < a.min_conf] = 0
        Draw[(Draw < a.min_depth) | (Draw > a.max_depth)] = 0
        Df, n_edge = edge_filter(Draw, a.edge_mm / 1000, a.edge_deg, f.cam.K_for(S.depth_w, S.depth_h)[0, 0])
        if self.ds > 1:
            Df = block_reduce(Df, self.ds)
        erode = a.edge_erode_px if a.edge_erode_px is not None else (2 if S.kind == "truedepth" else 0)
        if erode > 0:
            # 輪郭の「太り」（前景の深度が背景側へにじむ。TrueDepth で数 mm）を削る: 無効画素から n 画素以内を捨てる
            k = 2 * erode + 1
            Df[cv2.erode((Df > 0).astype(np.uint8), np.ones((k, k), np.uint8)) == 0] = 0
        mx, my = self._maps(f.cam)
        D = remap_depth(Df, mx, my)
        small = cv2.resize(img, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        C = cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if a.skin_mask:
            D[skin(C, a.skin_dilate_px)] = 0
        K_d = f.cam.K_for(self.dw, self.dh)
        Dinf = np.where(D > 0, D, np.inf).astype(np.float32)
        Dmin = cv2.erode(Dinf, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)
        fr = Frame(f, K_d, D, Dmin, C, n_edge=n_edge)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        ids = [] if ids is None else [int(i) for i in np.ravel(ids)]
        for c, mid in zip(corners, ids):
            if mid not in sizes:
                continue
            if ids.count(mid) > 1:
                o = MarkerObs(mid, 0.0, np.zeros((4, 2)), status="rejected", reasons=["同じ ID が複数"])
                fr.obs[mid] = o
                continue
            fr.obs[mid] = self.solve(fr, np.asarray(c, np.float64).reshape(4, 2), mid, sizes[mid])
        return fr, None

    def solve(self, fr: Frame, c_cv: np.ndarray, mid: int, size: float) -> MarkerObs:
        S, a, cam = self.S, self.a, fr.meta.cam
        side = float(min(np.linalg.norm(c_cv - np.roll(c_cv, 1, 0), axis=1)))
        sv = np.array([cam.ref_w / S.video_w, cam.ref_h / S.video_h])
        ref = cam.undistort((c_cv + 0.5) * sv)
        o = MarkerObs(mid, side, ref)
        obj = marker_object_points(size)
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, ref, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = []
        for i in range(n):
            R, _ = cv2.Rodrigues(rvecs[i])
            T = rigid(R, tvecs[i])
            uv, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], cam.K, None)
            err = float(np.linalg.norm(uv.reshape(-1, 2) - ref, axis=1).mean() / sv.mean())
            sols.append((err, T))
        sols.sort(key=lambda s: s[0])
        if not sols:
            o.status = "rejected"
            o.reasons.append("PnP 失敗")
            return o
        o.reproj_px, o.T_rgb = sols[0]
        o.ambiguity = sols[0][0] / sols[1][0] if len(sols) > 1 and sols[1][0] > 0 else 0.0
        if side < a.min_side_px:
            o.status = "rejected"
            o.reasons.append(f"マーカーが小さい（一辺 {side:.0f} px < {a.min_side_px}）")
            return o
        if o.reproj_px > a.max_reproj_px:
            o.status = "rejected"
            o.reasons.append(f"再投影誤差 {o.reproj_px:.2f} px > {a.max_reproj_px}")
            return o
        if o.T_rgb[2, 3] <= 0:
            o.status = "rejected"
            o.reasons.append("カメラの後ろに解")
            return o
        T = o.T_rgb
        pl = None if a.no_depth_refine else self.marker_plane(fr, ref)
        o.plane = pl
        if pl is not None and pl.get("ok"):
            n_ = pl["n"]
            cand = [(math.degrees(math.acos(np.clip(Ts[:3, 2] @ n_, -1, 1))), e, Ts)
                    for e, Ts in sols if e <= a.max_reproj_px]
            ang, e, Ts = min(cand, key=lambda x: x[0])
            pl["angle_rgb_deg"] = round(ang, 2)
            pl["switched_ippe"] = Ts is not sols[0][1]
            # 法線は RGB（IPPE）と深度平面を分散の逆数で重みづけして合わせる。
            #   深度: σ² ≈ rms² / (N·面内の広がり²)、RGB: σ ≈ rgb_tilt_px / 一辺 px（合成データで 85 px → 約 0.6°）
            sig_d = pl["sigma_rad"]
            sig_r = a.rgb_tilt_px / max(side, 1.0)
            w_d, w_r = 1 / sig_d ** 2, 1 / sig_r ** 2
            n_f = w_d * n_ + w_r * Ts[:3, 2]
            n_f /= np.linalg.norm(n_f)
            pl["w_depth"] = round(w_d / (w_d + w_r), 3)
            ray = np.linalg.solve(cam.K, np.r_[line_intersection(ref), 1.0])
            s = float(n_f @ pl["m"]) / float(n_f @ ray)
            t_new = s * ray
            dz = float(np.linalg.norm(t_new - Ts[:3, 3]))
            pl["shift_mm"] = round(dz * 1000, 2)
            if ang <= a.plane_max_deg and dz <= a.plane_max_mm / 1000:
                T = rigid(rot_between(Ts[:3, 2], n_f) @ Ts[:3, :3], t_new)
                o.depth_refined = True
            else:
                o.reasons.append(f"深度平面が RGB と合わない（傾き {ang:.0f}°、位置 {dz * 1000:.0f} mm）。RGB の解を使う")
        elif pl is not None:
            o.reasons.append(f"深度平面なし（{pl.get('why')}）")
        if not o.depth_refined:
            if o.ambiguity > a.ambiguity:
                o.status = "rejected"
                o.reasons.append(f"IPPE 2 解の誤差比 {o.ambiguity:.2f} > {a.ambiguity}（傾きの向きが決まらず、深度でも決められない）")
                return o
        if T[:3, 2] @ T[:3, 3] > 0:
            o.status = "rejected"
            o.reasons.append("マーカー面がカメラと反対を向いた解")
            return o
        o.T = T
        return o

    def marker_plane(self, fr: Frame, ref_quad: np.ndarray) -> dict:
        """マーカーの四角形の内側（85 % に縮小）の深度に平面を当てる。n はカメラ向き、n·X = d。"""
        S, cam = self.S, fr.meta.cam
        q = ref_quad * np.array([self.dw / cam.ref_w, self.dh / cam.ref_h]) - 0.5
        c = line_intersection(q)
        q = c + 0.85 * (q - c)
        mask = np.zeros(fr.D.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(q * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        P, _ = backproject(fr.D, fr.K_d, mask.astype(bool))
        if len(P) < self.a.plane_min_pts:
            return {"ok": False, "why": f"深度点 {len(P)} 個", "n_pts": len(P)}
        keep = np.ones(len(P), bool)
        for _ in range(4):
            m = P[keep].mean(0)
            _, _, Vt = np.linalg.svd(P[keep] - m, full_matrices=False)
            n = Vt[2]
            r = (P - m) @ n
            sig = max(1.4826 * np.median(np.abs(r[keep])), 0.0007)
            keep_new = np.abs(r) < 3 * sig
            if keep_new.sum() < self.a.plane_min_pts or np.array_equal(keep_new, keep):
                break
            keep = keep_new
        if n @ m > 0:
            n = -n
        rms = float(np.sqrt(np.mean(((P[keep] - m) @ n) ** 2)))
        lam = np.linalg.svd(P[keep] - m, compute_uv=False) ** 2 / keep.sum()
        # 理論値の 2.5 倍・下限 0.5°（縁にかかる画素や相関した雑音で、理論値より実際の誤差が大きい。合成データで確認）
        sigma = max(2.5 * max(rms, 0.0005) / math.sqrt(keep.sum() * max(lam[1], 1e-8)), math.radians(0.5))
        out = {"n": n, "m": m, "d": float(n @ m), "n_pts": int(keep.sum()), "rms_mm": round(rms * 1000, 2),
               "sigma_rad": float(sigma),
               "depth_m": round(float(m[2]), 4)}
        if rms > self.a.plane_max_rms_mm / 1000:
            out.update(ok=False, why=f"平面の残差 {rms * 1000:.1f} mm")
            return out
        out["ok"] = True
        return out


def edge_filter(D: np.ndarray, abs_m: float, max_deg: float, fx: float):
    """飛び画素・輪郭の除去。4 近傍のどれかとの差が max(abs_m, d·tan(max_deg)/fx) を超える画素を無効にする。
    （傾いた面の正常な勾配は残し、面の視線に対する傾きが max_deg を超える段差だけを切る）"""
    valid = D > 0
    thr = np.maximum(abs_m, D * math.tan(math.radians(max_deg)) / fx)
    bad = np.zeros_like(valid)
    P = np.pad(D, 1, mode="edge")
    h, w = D.shape
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = P[dy:dy + h, dx:dx + w]
        bad |= valid & (nb > 0) & (np.abs(nb - D) > thr)
    out = D.copy()
    out[bad] = 0
    return out, int(bad.sum())


def block_reduce(D: np.ndarray, k: int) -> np.ndarray:
    """k x k の画素をまとめる。全部有効で幅 4 mm 以内なら平均、そうでなければ無効。"""
    h, w = D.shape[0] // k * k, D.shape[1] // k * k
    B = D[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, k * k)
    ok = np.all(B > 0, -1) & ((B.max(-1) - B.min(-1)) < 0.004)
    return np.where(ok, B.mean(-1), 0).astype(np.float32)


def remap_depth(D: np.ndarray, mx, my) -> np.ndarray:
    """歪みを除いた深度。4 近傍が揃っていれば線形補間、そうでなければ最近傍（段差をまたいで混ぜない）。"""
    near = cv2.remap(D, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lin = cv2.remap(D, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    val = cv2.remap((D > 0).astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ok = (val > 0.999) & (np.abs(lin - near) < 0.004)
    return np.where(ok, lin, near).astype(np.float32)


def skin(C: np.ndarray, dilate_px: int = 2) -> np.ndarray:
    ycc = cv2.cvtColor(C, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 135, 85), (255, 180, 135))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = 2 * dilate_px + 1
    return cv2.dilate(m, np.ones((k, k), np.uint8)) > 0


# --------------------------------------------------------------------------- 複数マーカー

def marker_relations(frames: list[Frame], ref_id: int) -> dict[int, dict]:
    """基準マーカーと同時に写ったフレームから T_ref_k を求める（回転は外れ値を除いた平均、並進は中央値）。"""
    pairs: dict[int, list[np.ndarray]] = {}
    for fr in frames:
        r = fr.obs.get(ref_id)
        if r is None or r.T is None or not r.depth_refined:
            continue
        for k, o in fr.obs.items():
            if k == ref_id or o.T is None or not o.depth_refined:
                continue
            pairs.setdefault(k, []).append(inv(r.T) @ o.T)
    out = {}
    for k, Ts in pairs.items():
        if len(Ts) < 3:
            out[k] = {"n": len(Ts), "ok": False, "why": "同時に写ったフレームが 3 未満"}
            continue
        Rs = Rotation.from_matrix(np.array([T[:3, :3] for T in Ts]))
        ts = np.array([T[:3, 3] for T in Ts])
        Rm = Rs.mean()
        dev = np.degrees((Rm.inv() * Rs).magnitude())
        tm = np.median(ts, 0)
        tdev = np.linalg.norm(ts - tm, axis=1)
        keep = (dev < max(3 * np.median(dev), 2.0)) & (tdev < max(3 * np.median(tdev), 0.002))
        Rm = Rs[keep].mean()
        tm = np.median(ts[keep], 0)
        out[k] = {"n": len(Ts), "n_used": int(keep.sum()), "ok": True, "T_ref_k": rigid(Rm.as_matrix(), tm),
                  "rot_spread_deg": round(float(np.median(dev[keep])), 2),
                  "pos_spread_mm": round(float(np.median(tdev[keep]) * 1000), 2)}
    return out


def choose_poses(frames: list[Frame], ref_id: int, rel: dict, a) -> None:
    last: tuple[float, np.ndarray] | None = None
    for fr in frames:
        cands = []
        r = fr.obs.get(ref_id)
        if r is not None and r.T is not None:
            cands.append((1e9 + r.side_px, r.T, f"ID {ref_id}"))
        for k, o in fr.obs.items():
            if k != ref_id and o.T is not None and rel.get(k, {}).get("ok") and o.depth_refined:
                cands.append((o.side_px, o.T @ inv(rel[k]["T_ref_k"]), f"ID {k} 経由"))
        if not cands:
            fr.status = "rejected"
            if not fr.obs:
                fr.reasons.append("登録したマーカーが検出されない")
            for k, o in fr.obs.items():
                fr.reasons += [f"ID {k}: {x}" for x in o.reasons] or [f"ID {k}: 使えない"]
            continue
        _, T, src = max(cands, key=lambda c: c[0])
        if last is not None and fr.meta.t - last[0] < a.flip_window_s:
            jump = math.degrees(math.acos(np.clip(last[1][:3, 2] @ T[:3, 2], -1, 1)))
            if jump > a.flip_deg:
                fr.status = "rejected"
                fr.reasons.append(f"直前のフレームから法線が {jump:.0f}° 跳んだ（反転とみなす）")
                continue
        fr.T, fr.T_marker_only, fr.source = T, T.copy(), src
        last = (fr.meta.t, T)


# --------------------------------------------------------------------------- 点群・投票

class Grid:
    def __init__(self, crop_m: np.ndarray, vox: float):
        self.lo = crop_m[[0, 2, 4]]
        self.hi = crop_m[[1, 3, 5]]
        self.vox = vox
        self.n = np.ceil((self.hi - self.lo) / vox).astype(int)

    def index(self, P: np.ndarray):
        ijk = np.floor((P - self.lo) / self.vox).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < self.n), 1)
        return ijk, ok

    def lin(self, ijk):
        return np.ravel_multi_index(ijk.T, self.n)

    def centres(self, lin):
        return self.lo + (np.stack(np.unravel_index(lin, self.n), 1) + 0.5) * self.vox


def frame_points(fr: Frame, grid: Grid, mask: np.ndarray | None = None):
    """フレームの点をマーカー座標へ。切り出し箱の中だけ。返り値: 点, 画素添字, ボクセル線形添字。"""
    P, idx = backproject(fr.D, fr.K_d)
    Tm = inv(fr.T)
    Pm = P @ Tm[:3, :3].T + Tm[:3, 3]
    ijk, ok = grid.index(Pm)
    Pm, idx, ijk = Pm[ok], idx[ok], ijk[ok]
    lin = grid.lin(ijk)
    if mask is not None:
        keep = mask.ravel()[lin]
        Pm, idx, lin = Pm[keep], idx[keep], lin[keep]
    return Pm, idx, lin


def project_depth(fr: Frame, Pm: np.ndarray):
    """マーカー座標の点を歪みなし深度画像へ。返り値: 画素 (u,v) 整数, z, 画像内か。"""
    Pc = Pm @ fr.T[:3, :3].T + fr.T[:3, 3]
    z = Pc[:, 2]
    K = fr.K_d
    zz = np.where(z > 1e-3, z, np.nan)
    u = K[0, 0] * Pc[:, 0] / zz + K[0, 1] * Pc[:, 1] / zz + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / zz + K[1, 2]
    h, w = fr.D.shape
    inside = (z > 1e-3) & (u > -0.5) & (u < w - 0.5) & (v > -0.5) & (v < h - 0.5)
    ui = np.clip(np.round(np.nan_to_num(u)), 0, w - 1).astype(np.int64)
    vi = np.clip(np.round(np.nan_to_num(v)), 0, h - 1).astype(np.int64)
    return ui, vi, z, inside, u, v


def vote(frames: list[Frame], grid: Grid, a) -> tuple[np.ndarray, dict]:
    """可視性つき投票。返り値: 残すボクセルの密な bool 格子と統計。"""
    lins = [frame_points(fr, grid)[2] for fr in frames]
    cand = np.unique(np.concatenate(lins)) if lins else np.zeros(0, np.int64)
    C = grid.centres(cand)
    hits = np.zeros(len(cand), np.int32)
    free = np.zeros(len(cand), np.int32)
    info = np.zeros(len(cand), np.int32)
    th, tf = a.tau_hit_mm / 1000, a.tau_free_mm / 1000
    for fr in frames:
        ui, vi, z, inside, _, _ = project_depth(fr, C)
        d = fr.D[vi, ui]
        dmin = fr.Dmin[vi, ui]
        valid = inside & (d > 0)
        hits += valid & (np.abs(z - d) <= th)
        free += inside & np.isfinite(dmin) & (z < dmin - tf)
        info += valid
    keep = (hits >= a.min_hits) & (hits >= a.vote_ratio * (hits + free)) & (hits >= a.min_hit_frac * info)
    occ = np.zeros(grid.n, bool)
    occ.ravel()[cand[keep]] = True
    lab, nlab = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    stats = {"candidates": int(len(cand)), "voted": int(keep.sum())}
    if nlab:
        sizes = np.bincount(lab.ravel())[1:]
        big = sizes.max()
        good = np.flatnonzero(sizes >= a.min_component * big) + 1
        occ = np.isin(lab, good)
        stats.update(components=int(nlab), kept_components=int(len(good)), kept=int(occ.sum()))
    else:
        stats.update(components=0, kept_components=0, kept=0)
    return occ, stats


def object_masks(frames: list[Frame], grid: Grid, occ: np.ndarray) -> np.ndarray:
    """各フレームの物体画素（残したボクセルに入る点）を記録し、全フレームの点を返す。"""
    allP, allC = [], []
    for fr in frames:
        Pm, idx, _ = frame_points(fr, grid, occ)
        m = np.zeros(fr.D.size, bool)
        m[idx] = True
        fr.obj_mask = m.reshape(fr.D.shape)
        allP.append(Pm)
        allC.append(fr.C.reshape(-1, 3)[idx][:, ::-1] / 255.0)
    if not allP:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(allP), np.concatenate(allC)


# --------------------------------------------------------------------------- ICP

def model_normals(P: np.ndarray, k: int = 20):
    tree = cKDTree(P)
    _, nn = tree.query(P, k=min(k, len(P)))
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", Q, Q)
    w, V = np.linalg.eigh(cov)
    return tree, V[:, :, 0], w[:, 0] / np.maximum(w.sum(1), 1e-18)


def icp_frame(src: np.ndarray, tree, MP, MN, a) -> tuple[np.ndarray, dict]:
    """点-面 ICP（ガウス・ニュートン）。マーカーの姿勢を事前分布にして、形が決めない方向は動かさない。"""
    Lam = np.diag([1 / math.radians(a.icp_prior_deg) ** 2] * 3 + [1 / (a.icp_prior_mm / 1000) ** 2] * 3)
    xi = np.zeros(6)
    T = np.eye(4)
    sig2 = (a.icp_sigma_mm / 1000) ** 2
    huber = a.icp_sigma_mm / 1000
    st = {}
    for max_d in (a.icp_max_corr_mm / 1000, a.icp_max_corr_mm / 2000):
        for _ in range(12):
            p = src @ T[:3, :3].T + T[:3, 3]
            dist, j = tree.query(p, distance_upper_bound=max_d)
            ok = np.isfinite(dist)
            if ok.sum() < 30:
                return T, {"ok": False, "why": f"対応点 {int(ok.sum())} 個"}
            pp, n = p[ok], MN[j[ok]]
            r = np.sum(n * (pp - MP[j[ok]]), 1)
            J = np.hstack([np.cross(pp, n), n])
            w = huber / np.maximum(np.abs(r), huber)
            scale = min(len(r), a.icp_neff) / w.sum() / sig2
            H = (J * w[:, None]).T @ J * scale + Lam
            g = (J * w[:, None]).T @ r * scale + Lam @ xi
            dx = -np.linalg.solve(H, g)
            xi += dx
            dT = rigid(Rotation.from_rotvec(dx[:3]).as_matrix(), dx[3:])
            T = dT @ T
            if np.linalg.norm(dx[3:]) < 1e-5 and np.linalg.norm(dx[:3]) < 1e-4:
                break
        st = {"fitness": float(ok.mean()), "rmse_mm": float(np.sqrt(np.mean(r ** 2)) * 1000)}
    st["ok"] = True
    return T, st


def run_icp(frames: list[Frame], grid: Grid, occ: np.ndarray, a) -> dict:
    P, _ = object_masks(frames, grid, occ)
    if len(P) < 200:
        return {"skipped": "模型の点が少ない"}
    P = voxel_down(P, a.voxel_mm / 1000 * 0.75)
    tree, MN, curv = model_normals(P)
    MP = P
    deltas = {}
    for fr in frames:
        src, _, _ = frame_points(fr, grid, occ)
        if len(src) < 100:
            fr.icp = {"ok": False, "why": "点が少ない"}
            continue
        src = voxel_down(src, a.voxel_mm / 1000)
        dT, st = icp_frame(src, tree, MP, MN, a)
        dt = float(np.linalg.norm(dT[:3, 3]) * 1000)
        da = angle_deg(dT[:3, :3])
        st.update(dt_mm=round(dt, 2), dr_deg=round(da, 2))
        if st["ok"] and (dt > a.icp_max_mm or da > a.icp_max_deg or st["fitness"] < a.icp_min_fitness):
            st["ok"] = False
            st["why"] = "補正が上限を超えた / 一致率が低い（マーカーの姿勢のまま）"
        fr.icp = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}
        if st["ok"]:
            deltas[id(fr)] = dT
    if not deltas:
        return {"accepted": 0, "frames": len(frames)}
    # 補正の平均（共通の並進・回転）は差し引く。マーカー座標の定義はマーカーが決める
    rv = np.array([Rotation.from_matrix(d[:3, :3]).as_rotvec() for d in deltas.values()])
    tv = np.array([d[:3, 3] for d in deltas.values()])
    M = rigid(Rotation.from_rotvec(rv.mean(0)).as_matrix(), tv.mean(0))
    Mi = inv(M)
    dts, das = [], []
    for fr in frames:
        d = deltas.get(id(fr))
        if d is None:
            continue
        d = Mi @ d
        fr.T = fr.T @ inv(d)    # T_marker_cam' = d · T_marker_cam
        dts.append(np.linalg.norm(d[:3, 3]) * 1000)
        das.append(angle_deg(d[:3, :3]))
    return {"accepted": len(deltas), "frames": len(frames),
            "removed_mean_shift_mm": round(float(np.linalg.norm(tv.mean(0)) * 1000), 3),
            "removed_mean_rot_deg": round(float(np.degrees(np.linalg.norm(rv.mean(0)))), 3),
            "correction_mm_median": round(float(np.median(dts)), 3),
            "correction_mm_max": round(float(np.max(dts)), 3),
            "correction_deg_median": round(float(np.median(das)), 3),
            "correction_deg_max": round(float(np.max(das)), 3)}


def voxel_down(P: np.ndarray, vox: float) -> np.ndarray:
    if len(P) == 0:
        return P
    k = np.floor(P / vox).astype(np.int64)
    _, inv_, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv_.ravel(), P)
    return out / cnt[:, None]


# --------------------------------------------------------------------------- TSDF

def tsdf_fuse(frames: list[Frame], grid: Grid, occ: np.ndarray, a):
    trunc = a.tsdf_trunc_mm / 1000
    r = int(math.ceil(trunc / grid.vox)) + 1
    active = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3)), iterations=r)
    lin = np.flatnonzero(active)
    C = grid.centres(lin)
    tsdf = np.zeros(len(lin), np.float32)
    wsum = np.zeros(len(lin), np.float32)
    for fr in frames:
        ui, vi, z, inside, u, v = project_depth(fr, C)
        d = sample_depth(fr.D, u, v, ui, vi)
        valid = inside & (d > 0)
        sdf = d - z
        near = valid & (np.abs(sdf) < trunc) & fr.obj_mask[vi, ui]
        front = valid & (sdf >= trunc)                 # 表面より十分手前 = 空いている（手・背景の画素でも可）
        upd = near | front
        val = np.where(front, 1.0, np.clip(sdf / trunc, -1, 1)).astype(np.float32)
        tsdf[upd] = (tsdf[upd] * wsum[upd] + val[upd]) / (wsum[upd] + 1)
        wsum[upd] += 1
    vol = np.ones(grid.n, np.float32)
    W = np.zeros(grid.n, np.float32)
    vol.ravel()[lin] = tsdf
    W.ravel()[lin] = wsum
    return vol, W


def sample_depth(D, u, v, ui, vi):
    """双線形（4 近傍が揃って差 5 mm 以内のとき）、それ以外は最近傍。"""
    h, w = D.shape
    u0 = np.clip(np.floor(np.nan_to_num(u)), 0, w - 2).astype(np.int64)
    v0 = np.clip(np.floor(np.nan_to_num(v)), 0, h - 2).astype(np.int64)
    fu = np.clip(np.nan_to_num(u) - u0, 0, 1)
    fv = np.clip(np.nan_to_num(v) - v0, 0, 1)
    a, b, c, d = D[v0, u0], D[v0, u0 + 1], D[v0 + 1, u0], D[v0 + 1, u0 + 1]
    st = np.stack([a, b, c, d])
    ok = np.all(st > 0, 0) & ((st.max(0) - st.min(0)) < 0.005)
    lin = (a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv)
    return np.where(ok, lin, D[vi, ui])


def extract_mesh(vol, W, grid: Grid, a, occ: np.ndarray | None = None):
    """TSDF → メッシュ。

    既定（閉じる）: 観測したボクセルは TSDF の値、観測していないボクセルは「表面の凸包の内側なら中身、
    外なら空き」とみなしてマーチングキューブにかける。見えなかった面（持っていた底面など）は
    凸包の面で塞がるので、必ず閉じた（watertight な）メッシュになる。凹みが見えていなければ埋まる。
    --no-fill: 観測したボクセルだけで面を張る（見えなかった所は穴のまま）。
    """
    import trimesh
    from scipy.spatial import Delaunay
    from skimage import measure
    mask = W >= a.tsdf_min_weight
    if mask.sum() < 8 or vol[mask].min() > 0 or vol[mask].max() < 0:
        return None, {"error": "TSDF に表面がない"}
    info = {}
    if a.no_fill:
        v, f, _, _ = measure.marching_cubes(np.where(mask, vol, 1.0), level=0.0, mask=mask, allow_degenerate=False)
    else:
        surf = mask & (np.abs(vol) < 0.5)
        if occ is not None:
            surf &= ndimage.binary_dilation(occ, iterations=1)
        idx = np.flatnonzero(surf)
        F = np.where(mask, vol, 1.0).astype(np.float32)
        if len(idx) >= 4:
            S = grid.centres(idx)
            lo_, hi_ = S.min(0), S.max(0)
            un = np.flatnonzero(~mask)
            C = grid.centres(un)
            inb = np.all((C >= lo_) & (C <= hi_), 1)
            hull = Delaunay(S[np.random.default_rng(0).choice(len(S), min(len(S), 20000), replace=False)])
            ins = np.zeros(len(un), bool)
            ins[inb] = hull.find_simplex(C[inb]) >= 0
            F.ravel()[un[ins]] = -1.0
            info["unobserved_filled_voxels"] = int(ins.sum())
        F = np.pad(F, 1, constant_values=1.0)
        v, f, _, _ = measure.marching_cubes(F, level=0.0, allow_degenerate=False)
        v = v - 1
    v = grid.lo + (v + 0.5) * grid.vox
    # skimage の向き（勾配が増える = 外向き）に合わせて三角形の向きを揃える
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    info["mc_faces"] = int(len(m.faces))
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        areas = np.array([p.area for p in parts])
        m = trimesh.util.concatenate([p for p, ar in zip(parts, areas) if ar >= a.min_component * areas.max()])
        info["dropped_parts"] = int((areas < a.min_component * areas.max()).sum())
    m = decimate(m, a.max_tris)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    info.update(faces=int(len(m.faces)), vertices=int(len(m.vertices)), watertight=bool(m.is_watertight))
    if m.is_watertight:
        info["volume_cm3"] = round(float(m.volume) * 1e6, 2)
    return m, info


def decimate(m, max_tris: int):
    import open3d as o3d
    import trimesh
    if len(m.faces) <= max_tris:
        return m
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    om = om.simplify_quadric_decimation(max_tris)
    om.remove_degenerate_triangles()
    om.remove_duplicated_triangles()
    om.remove_non_manifold_edges()
    return trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True)


def vertex_colours(m, frames: list[Frame], S: Session, n_frames: int = 24) -> np.ndarray:
    """頂点色: 見えているフレーム（深度で可視判定）の元解像度 RGB を、向きで重みづけして平均。"""
    V, N = m.vertices, m.vertex_normals
    acc = np.zeros((len(V), 3))
    ws = np.zeros(len(V))
    for fr in frames[:: max(1, len(frames) // n_frames)]:
        img = S.rgb(fr.meta)
        ui, vi, z, inside, _, _ = project_depth(fr, V)
        d = fr.D[vi, ui]
        Pc = V @ fr.T[:3, :3].T + fr.T[:3, 3]
        Nc = N @ fr.T[:3, :3].T
        cosv = np.abs(np.sum(Nc * Pc, 1)) / np.linalg.norm(Pc, axis=1)   # 法線の向きが揃っていなくてもよいように
        vis = inside & (d > 0) & (np.abs(d - z) < 0.006) & (cosv > 0.2)
        if img is None or vis.sum() == 0:
            continue
        uv = fr.meta.cam.project_cv(Pc[vis], S.video_w, S.video_h).astype(np.float32)
        col = cv2.remap(img, uv[:, 0].reshape(-1, 1), uv[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3)
        acc[vis] += col[:, ::-1] * cosv[vis, None]
        ws[vis] += cosv[vis]
    out = np.full((len(V), 3), 160.0)
    out[ws > 0] = acc[ws > 0] / ws[ws > 0, None]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 本体

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--marker-id", type=int, required=True, help="基準マーカー（出力はこのマーカー座標）")
    p.add_argument("--marker-mm", type=float, required=True, help="基準マーカーの黒い正方形の一辺 mm")
    p.add_argument("--marker-sizes", default="", help="ほかのマーカー 'ID:mm,ID:mm'（基準が見えないフレームの姿勢に使う）")
    p.add_argument("--out", default=None, help="出力先（既定 <session>/result）")
    p.add_argument("--crop-mm", default="-150,150,-150,150,-150,150", help="マーカー座標の切り出し箱 x0,x1,y0,y1,z0,z1")
    p.add_argument("--voxel-mm", type=float, default=2.0)
    p.add_argument("--step", type=int, default=1, help="n フレームごとに使う")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--min-depth", type=float, default=0.0, help="これより近い深度は使わない (m)。LiDAR は 0.25 未満が不安定")
    p.add_argument("--max-depth", type=float, default=5.0)
    p.add_argument("--min-conf", type=int, default=1, help="ARKit の信頼度（.conf）がこれ未満の深度は捨てる（0 で無効、2 = high のみ）")
    p.add_argument("--depth-max-w", type=int, default=320, help="深度の幅がこれを超えれば整数分の 1 にまとめる")
    p.add_argument("--min-side-px", type=float, default=25.0)
    p.add_argument("--max-reproj-px", type=float, default=3.0)
    p.add_argument("--ambiguity", type=float, default=0.6, help="IPPE 誤差比がこれを超え、深度でも決まらなければ棄却")
    p.add_argument("--no-depth-refine", action="store_true", help="マーカー姿勢を深度平面で補正しない")
    p.add_argument("--rgb-tilt-px", type=float, default=0.9,
                   help="RGB の法線の不確かさ = これ / マーカーの一辺 px（rad）。深度平面の法線との重みづけに使う")
    p.add_argument("--plane-min-pts", type=int, default=12)
    p.add_argument("--plane-max-rms-mm", type=float, default=3.0)
    p.add_argument("--plane-max-deg", type=float, default=25.0, help="深度平面と RGB の法線がこれ以上違えば深度を使わない")
    p.add_argument("--plane-max-mm", type=float, default=30.0, help="深度での位置が RGB とこれ以上違えば深度を使わない")
    p.add_argument("--flip-deg", type=float, default=60.0)
    p.add_argument("--flip-window-s", type=float, default=0.5)
    p.add_argument("--edge-mm", type=float, default=4.0, help="段差除去: 隣の画素との差の下限の閾値 mm")
    p.add_argument("--edge-deg", type=float, default=78.0, help="段差除去: 視線に対してこれより寝た面の段差を切る")
    p.add_argument("--edge-erode-px", type=int, default=None,
                   help="深度の輪郭（無効画素の縁）から n 画素を捨てる（処理解像度）。既定: TrueDepth 2、LiDAR 0")
    p.add_argument("--skin-mask", action="store_true", help="肌色の画素を除く（物体が肌色に近いときは使わない）")
    p.add_argument("--skin-dilate-px", type=int, default=2, help="肌色マスクを広げる画素数（処理解像度。指先の縁を残さない）")
    p.add_argument("--tau-hit-mm", type=float, default=4.0)
    p.add_argument("--tau-free-mm", type=float, default=8.0)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--min-hit-frac", type=float, default=0.02, help="深度が有効に写ったフレーム数（隠れていた回を含む）に対する占有の割合の下限")
    p.add_argument("--vote-ratio", type=float, default=0.5)
    p.add_argument("--min-component", type=float, default=0.2, help="最大の塊に対する比がこれ未満の塊は捨てる")
    p.add_argument("--icp-iters", type=int, default=2, help="0 で ICP しない")
    p.add_argument("--icp-max-mm", type=float, default=8.0)
    p.add_argument("--icp-max-deg", type=float, default=4.0)
    p.add_argument("--icp-min-fitness", type=float, default=0.5)
    p.add_argument("--icp-max-corr-mm", type=float, default=6.0)
    p.add_argument("--icp-prior-mm", type=float, default=3.0)
    p.add_argument("--icp-prior-deg", type=float, default=2.0)
    p.add_argument("--icp-sigma-mm", type=float, default=1.5)
    p.add_argument("--icp-neff", type=int, default=300)
    p.add_argument("--no-track", action="store_true", help="マーカーのないフレームを ICP で追跡しない")
    p.add_argument("--track-passes", type=int, default=2, help="追跡のパス数（2 回目は 1 回目の結果で作ったモデルで追い直す）")
    p.add_argument("--track-near-mm", type=float, default=15.0, help="予測姿勢でモデル表面からこれ以内の点だけ使う")
    p.add_argument("--track-min-pts", type=int, default=300)
    p.add_argument("--track-min-inliers", type=int, default=150)
    p.add_argument("--track-min-fitness", type=float, default=0.35,
                   help="3 mm 以内に対応がある点の割合（モデルにまだない面が写るので低め。裏返しは素通し矛盾で弾く）")
    p.add_argument("--track-max-rmse-mm", type=float, default=2.0)
    p.add_argument("--track-max-move-mm", type=float, default=40.0, help="予測からの 1 フレームの動きの上限")
    p.add_argument("--track-max-move-deg", type=float, default=30.0)
    p.add_argument("--track-loop-mm", type=float, default=15.0, help="ループ補正で配分するずれの上限")
    p.add_argument("--track-loop-deg", type=float, default=10.0)
    p.add_argument("--track-search-mm", type=float, default=40.0, help="再捕捉: 前回の物体中心からこの範囲（＋物体の半径）を探す")
    p.add_argument("--track-max-free-viol", type=float, default=0.08,
                   help="モデル点のうち観測面より手前に出る割合の上限（裏返し・取り違えの検出）")
    p.add_argument("--track-global-tries", type=int, default=1, help="再捕捉の RANSAC 回数（0 で再捕捉しない）")
    p.add_argument("--track-max-gap", type=int, default=10, help="連続して失敗したらこのフレーム数で再捕捉をやめる")
    p.add_argument("--no-track-color", dest="track_color", action="store_false", help="色を使わない点-面 ICP にする")
    p.add_argument("--tsdf-trunc-mm", type=float, default=8.0)
    p.add_argument("--tsdf-min-weight", type=float, default=2.0)
    p.add_argument("--max-tris", type=int, default=20000)
    p.add_argument("--no-fill", action="store_true", help="見えなかった所を凸包で塞がない（開いたメッシュになる）")
    p.add_argument("--overlays", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    S = Session(Path(a.session))
    out = Path(a.out) if a.out else S.root / "result"
    out.mkdir(parents=True, exist_ok=True)
    sizes = {a.marker_id: a.marker_mm / 1000}
    for tok in filter(None, a.marker_sizes.split(",")):
        k, v = tok.split(":")
        sizes.setdefault(int(k), float(v) / 1000)
    crop = np.array([float(x) for x in a.crop_mm.split(",")]) / 1000
    if len(crop) != 6 or np.any(crop[[1, 3, 5]] <= crop[[0, 2, 4]]):
        raise SystemExit("--crop-mm は x0,x1,y0,y1,z0,z1（x0<x1 …）")
    grid = Grid(crop, a.voxel_mm / 1000)

    metas = S.frames[:: a.step]
    if a.max_frames:
        metas = metas[: a.max_frames]
    print(f"[scan] {S.root.name}: {S.kind} {S.video_w}x{S.video_h} / 深度 {S.depth_w}x{S.depth_h}, "
          f"{len(metas)} フレーム（ファイル {len(S.frames)}）", flush=True)
    prep = Prep(S, a)
    frames: list[Frame] = []
    record: list[dict] = []
    for f in metas:
        fr, why = prep.frame(f, sizes)
        if fr is None:
            record.append({"frame": f.stem, "status": "rejected", "reasons": [why]})
            continue
        frames.append(fr)
    rel = marker_relations(frames, a.marker_id) if len(sizes) > 1 else {}
    choose_poses(frames, a.marker_id, rel, a)
    n_anchor = sum(fr.status == "ok" for fr in frames)
    print(f"[scan] マーカーで姿勢 {n_anchor} / {len(metas)} フレーム（{time.time() - t0:.1f} s）", flush=True)
    track_stats = []
    if not a.no_track and a.min_hits <= n_anchor < len(frames):
        track_stats = track_frames(frames, grid, a, t0)
    used = [fr for fr in frames if fr.status == "ok"]

    result = {"session": str(S.root), "camera": S.kind, "source": S.info.get("source"),
              "marker_id": a.marker_id, "marker_size_m": a.marker_mm / 1000,
              "units": "m（座標はすべて基準マーカー座標: 原点 = 黒い正方形の中心、x 右, y 上, z 面から外向き）",
              "frames_total": len(metas), "frames_used": len(used),
              "frames_anchored_by_marker": n_anchor,
              "frames_tracked_by_icp": sum(1 for fr in used if fr.source.startswith("ICP")),
              "frames_rejected": len(metas) - len(used),
              "tracking": track_stats,
              "params": {k: v for k, v in vars(a).items() if k not in ("session",)},
              "marker_relations": {str(k): {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                            for kk, vv in r.items()} for k, r in rel.items()}}
    warnings = []

    def finish(status: str):
        result["status"] = status
        result["warnings"] = warnings
        result["frames"] = record + [frame_record(fr) for fr in frames]
        result["rejected_reasons"] = reason_summary(result["frames"])
        result["elapsed_s"] = round(time.time() - t0, 1)
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"[scan] {status}。{out / 'result.json'}", flush=True)
        return result

    if len(used) < a.min_hits:
        warnings.append(f"姿勢が求まったフレームが {len(used)} しかない。マーカー ID・寸法・写り方を確認する")
        for k, n in reason_summary(record + [frame_record(fr) for fr in frames]).items():
            print(f"    {n:4d}  {k}")
        return finish("failed")

    if S.kind == "lidar":
        dz = np.median([fr.T[2, 3] for fr in used])
        if dz < 0.25:
            warnings.append(f"マーカーまでの距離の中央値 {dz * 100:.0f} cm。LiDAR は 25 cm 未満で不安定")

    occ, vstats = vote(used, grid, a)
    result["vote"] = [vstats]
    print(f"[scan] 投票: {vstats}", flush=True)
    result["icp"] = []
    for it in range(a.icp_iters):
        if occ.sum() == 0:
            break
        st = run_icp(used, grid, occ, a)
        result["icp"].append(st)
        print(f"[scan] ICP {it + 1}: {st}", flush=True)
        occ, vstats = vote(used, grid, a)
        result["vote"].append(vstats)
    if occ.sum() == 0:
        warnings.append("投票で残ったボクセルがない（切り出し箱・深度・姿勢を確認）")
        return finish("failed")
    P, Ccol = object_masks(used, grid, occ)
    write_points(out / "points.ply", P, Ccol, a.voxel_mm / 2000)

    vol, W = tsdf_fuse(used, grid, occ, a)
    mesh, minfo = extract_mesh(vol, W, grid, a, occ)
    result["mesh"] = minfo
    if mesh is None:
        warnings.append(minfo.get("error", "メッシュなし"))
        return finish("failed")
    mesh.visual.vertex_colors = np.c_[vertex_colours(mesh, used, S), np.full(len(mesh.vertices), 255)]
    mesh.export(out / "mesh.ply")
    mesh.export(out / "mesh.stl")
    mesh.export(out / "mesh.obj")
    hull = mesh.convex_hull
    hull.export(out / "hull.stl")
    lo, hi = mesh.bounds
    result["bbox_m"] = {"min": lo.round(5).tolist(), "max": hi.round(5).tolist(),
                        "size": (hi - lo).round(5).tolist()}
    result["hull"] = {"faces": int(len(hull.faces)), "volume_cm3": round(float(hull.volume) * 1e6, 2)}
    result["box_fit"] = fit_box(mesh)
    print(f"[scan] 直方体当てはめ: {result['box_fit']['size_sorted_mm']} mm、RMS {result['box_fit']['rms_mm']} mm", flush=True)
    if not minfo.get("watertight"):
        warnings.append("メッシュが閉じていない（MuJoCo の衝突には hull.stl を使う）")

    (out / "overlay").mkdir(exist_ok=True)
    pick = used[:: max(1, len(used) // max(1, a.overlays))][: a.overlays]
    for fr in pick:
        img = viz.overlay(S.rgb(fr.meta), mesh, fr.T, fr.meta.cam, S.video_w, S.video_h, a.marker_mm / 1000)
        cv2.imwrite(str(out / "overlay" / f"{fr.meta.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(out / "preview.png"), viz.preview(mesh, a.marker_mm / 1000))
    print(f"[scan] メッシュ {minfo}  bbox {result['bbox_m']['size']} m", flush=True)
    return finish("ok")


def track_frames(frames: list[Frame], grid: Grid, a, t0: float) -> list[dict]:
    """マーカーのないフレームを ICP 追跡で埋める（track.py）。パスごとにモデルを作り直す。"""
    import track
    anchors = {i: fr.T for i, fr in enumerate(frames) if fr.status == "ok"}
    for i, fr in enumerate(frames):
        if i not in anchors:
            fr.marker_reasons = list(fr.reasons)
    out = []
    for p in range(a.track_passes):
        posed = [fr for fr in frames if fr.T is not None]
        occ, vst = vote(posed, grid, a)
        P, C = object_masks(posed, grid, occ)
        if len(P) < 500:
            out.append({"pass": p + 1, "skipped": f"モデルの点 {len(P)} 個", "vote": vst})
            break
        model, tree = track.make_model(P, C, a.voxel_mm / 1000)
        a.track_global_now = a.track_global_tries if p == a.track_passes - 1 else 0   # 再捕捉は最後のパスだけ（時間）
        res, st = track.track_all(frames, anchors, model, tree, grid, a)
        n_ok = 0
        for i, (T, src, info) in res.items():
            fr = frames[i]
            fr.track = info
            if T is None:
                fr.T = fr.T_marker_only = None
                fr.status, fr.source = "rejected", ""
                fr.reasons = list(fr.marker_reasons) + [f"追跡: {w}" for w in info.get("why", [])]
            else:
                fr.T, fr.T_marker_only = T, T.copy()
                fr.status, fr.source, fr.reasons = "ok", src, []
                n_ok += 1
        st.update({"pass": p + 1, "model_points": int(len(model.points)), "tracked": n_ok,
                   "elapsed_s": round(time.time() - t0, 1)})
        print(f"[scan] 追跡 {p + 1}: {n_ok} フレームを追跡（モデル {len(model.points)} 点、"
              f"ループ補正 {st['loop_closed']} 区間・大きすぎ {st['loop_too_large']}）{time.time() - t0:.1f} s", flush=True)
        out.append(st)
    return out


def fit_box(mesh, n: int = 30000) -> dict:
    """メッシュ表面に向きつき直方体を当てはめる（頑健な最小二乗）。検収用: 寸法と点の距離の RMS。"""
    import trimesh
    from scipy.optimize import least_squares
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    obb = mesh.bounding_box_oriented
    T0 = np.asarray(obb.primitive.transform)
    x0 = np.r_[Rotation.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3], np.asarray(obb.primitive.extents) / 2]

    def dist(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        Q = (P - x[3:6]) @ R
        q = np.abs(Q) - np.abs(x[6:9])
        out = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        return out

    r = least_squares(dist, x0, loss="soft_l1", f_scale=0.002, x_scale=np.r_[[0.1] * 3, [0.01] * 6])
    d = np.abs(dist(r.x))
    ext = np.abs(r.x[6:9]) * 2
    R = Rotation.from_rotvec(r.x[:3]).as_matrix()
    order = np.argsort(ext)
    return {"size_m": ext.round(5).tolist(), "size_sorted_mm": np.sort(ext * 1000).round(2).tolist(),
            "centre_m": r.x[3:6].round(5).tolist(), "R_marker_box": R.round(5).tolist(),
            "axes_in_marker_frame": {f"{ext[i] * 1000:.1f}mm": R[:, i].round(3).tolist() for i in order},
            "rms_mm": round(float(np.sqrt(np.mean(d ** 2))) * 1000, 3), "median_mm": round(float(np.median(d)) * 1000, 3),
            "p95_mm": round(float(np.percentile(d, 95)) * 1000, 3),
            "obb_sorted_mm": np.sort(np.asarray(obb.primitive.extents) * 1000).round(2).tolist()}


def frame_record(fr: Frame) -> dict:
    d = {"frame": fr.meta.stem, "t": round(fr.meta.t, 4), "status": fr.status, "reasons": fr.reasons,
         "pose_source": fr.source, "edge_pixels_removed": fr.n_edge, "markers": {}}
    for k, o in fr.obs.items():
        m = {"status": o.status, "side_px": round(o.side_px, 1), "reproj_px": round(o.reproj_px, 3),
             "ambiguity": round(o.ambiguity, 3), "depth_refined": o.depth_refined, "reasons": o.reasons}
        if o.plane:
            m["plane"] = {k2: (v.round(4).tolist() if isinstance(v, np.ndarray) else v) for k2, v in o.plane.items()}
        d["markers"][str(k)] = m
    if fr.T is not None:
        d["T_cam_marker"] = np.round(fr.T, 6).tolist()
        d["T_cam_marker_before_icp"] = np.round(fr.T_marker_only, 6).tolist()
    if fr.icp is not None:
        d["icp"] = fr.icp
    if fr.track is not None:
        d["track"] = fr.track
    return d


def reason_summary(recs: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in recs:
        if r["status"] == "ok":
            continue
        for x in r["reasons"] or ["不明"]:
            key = x.split("（")[0]
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def write_points(path: Path, P: np.ndarray, C: np.ndarray, vox: float):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(C)
    pc = pc.voxel_down_sample(vox)
    o3d.io.write_point_cloud(str(path), pc)


if __name__ == "__main__":
    main()
