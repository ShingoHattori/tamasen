"""iPad 深度スキャンのセッション読み込みとカメラモデル（Apple の歪み LUT）。

座標の約束（このディレクトリのコード全体で共通）:
  連続座標 (cc)  画素 i は [i, i+1) を占め、中心は i + 0.5。解像度の違う画像どうしは単純な倍率で対応する
                 （Apple の intrinsicMatrix / lensDistortionCenter をそのまま倍率で縮める扱いと同じ）。
  OpenCV 添字 (cv) 画素中心が整数。cv = cc - 0.5。cv2 の検出結果・remap・Open3D の内部パラメータはこちら。
  ref            K_ref（intrinsicMatrixReferenceDimensions）の画素。歪み LUT もこの画素で適用する。
  rect           歪みを除いた（ピンホールの）座標。K はこの座標に対する行列。

Apple の lensDistortionPointForPoint（AVCameraCalibrationData.h のリファレンス実装）:
  r_max = |(max(cx, W - cx), max(cy, H - cy))|（歪み中心から最も遠い画像の角まで、W,H = 参照寸法）
  r = |p - c|、r < r_max なら val = r (n-1) / r_max で表を線形補間、それ以外は表の最後の値
  p' = c + (p - c)(1 + mag)
  lensDistortionLookupTable        は 歪みなし → 歪みあり（表の添字は歪みなしの半径）
  inverseLensDistortionLookupTable は 歪みあり → 歪みなし（表の添字は歪みありの半径）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def lut_apply(p: np.ndarray, table: np.ndarray, center: np.ndarray, r_max: float) -> np.ndarray:
    """Apple の lensDistortionPointForPoint を (N,2) の点にまとめて適用する（ref 画素）。"""
    v = p - center
    r = np.linalg.norm(v, axis=-1)
    n = len(table)
    val = np.clip(r * (n - 1) / r_max, 0, n - 1)        # r >= r_max は表の最後の値（np.interp も端で止まる）
    mag = np.interp(val, np.arange(n), table)
    return center + v * (1.0 + mag)[..., None]


@dataclass
class Camera:
    """1 フレーム分のカメラ（K と歪み表は ref 画素）。"""
    K: np.ndarray                 # 3x3, ref cc
    ref_w: float
    ref_h: float
    center: np.ndarray            # 歪み中心, ref cc
    lut: np.ndarray               # 歪みなし → 歪みあり
    inv_lut: np.ndarray           # 歪みあり → 歪みなし
    r_max: float = field(init=False)

    def __post_init__(self):
        cx, cy = self.center
        dx = max(cx, self.ref_w - cx)
        dy = max(cy, self.ref_h - cy)
        self.r_max = float(np.hypot(dx, dy))

    @property
    def has_distortion(self) -> bool:
        return len(self.lut) > 1 or len(self.inv_lut) > 1

    # ---- ref cc での歪み付け・歪み取り（片方の表しかないときは反復で逆を解く）
    def distort(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if len(self.lut) > 1:
            return lut_apply(p, self.lut, self.center, self.r_max)
        if len(self.inv_lut) > 1:
            return self._invert(p, self.inv_lut)
        return p.copy()

    def undistort(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if len(self.inv_lut) > 1:
            return lut_apply(p, self.inv_lut, self.center, self.r_max)
        if len(self.lut) > 1:
            return self._invert(p, self.lut)
        return p.copy()

    def _invert(self, target: np.ndarray, table: np.ndarray) -> np.ndarray:
        x = target.copy()
        for _ in range(20):
            x = x + (target - lut_apply(x, table, self.center, self.r_max))
        return x

    # ---- 解像度 (w, h) の画像に合わせた K（cv 添字、歪みなし）
    def K_for(self, w: int, h: int) -> np.ndarray:
        s = np.array([w / self.ref_w, h / self.ref_h])
        K = self.K.copy()
        K[0, 0] *= s[0]
        K[0, 1] *= s[0]
        K[1, 1] *= s[1]
        K[0, 2] = K[0, 2] * s[0] - 0.5
        K[1, 2] = K[1, 2] * s[1] - 0.5
        return K

    def rect_to_distorted_cv(self, rect_cv: np.ndarray, w: int, h: int) -> np.ndarray:
        """解像度 (w,h) の歪みなし cv 座標 → 同じ解像度の歪みあり cv 座標。"""
        s = np.array([self.ref_w / w, self.ref_h / h])
        ref = (np.asarray(rect_cv, np.float64) + 0.5) * s
        return self.distort(ref) / s - 0.5

    def distorted_to_rect_cv(self, dist_cv: np.ndarray, w: int, h: int) -> np.ndarray:
        s = np.array([self.ref_w / w, self.ref_h / h])
        ref = (np.asarray(dist_cv, np.float64) + 0.5) * s
        return self.undistort(ref) / s - 0.5

    def project_cv(self, P_cam: np.ndarray, w: int, h: int) -> np.ndarray:
        """カメラ座標の点 (N,3) → 解像度 (w,h) の歪みあり画像の cv 座標 (N,2)。z<=0 は nan。"""
        K = self.K_for(w, h)
        z = P_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * P_cam[:, 0] / z + K[0, 1] * P_cam[:, 1] / z + K[0, 2]
            v = K[1, 1] * P_cam[:, 1] / z + K[1, 2]
        uv = np.stack([u, v], 1)
        uv[z <= 1e-6] = np.nan
        return self.rect_to_distorted_cv(uv, w, h)

    def rectify_maps(self, w: int, h: int, src_w: int, src_h: int):
        """歪みなし (w,h) 画像を作るための cv2.remap 用マップ（元画像は歪みあり (src_w,src_h)）。"""
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        rect = np.stack([u.ravel(), v.ravel()], 1)
        d = self.rect_to_distorted_cv(rect, w, h)
        d = (d + 0.5) * np.array([src_w / w, src_h / h]) - 0.5
        return d[:, 0].reshape(h, w).astype(np.float32), d[:, 1].reshape(h, w).astype(np.float32)

    def key(self) -> tuple:
        return (self.K.tobytes(), self.ref_w, self.ref_h, self.center.tobytes(),
                self.lut.tobytes(), self.inv_lut.tobytes())


@dataclass
class FrameMeta:
    index: int
    stem: str
    t: float
    cam: Camera
    depth_filtered: bool
    T_world_cam_arkit: np.ndarray | None = None   # ARKit の world←camera（カメラ軸 x 右・y 上・z 後ろ）。今は記録だけ
    tracking: str | None = None


class Session:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "session.json").read_text())
        self.kind = self.info.get("camera", "lidar")
        self.video_w, self.video_h = int(self.info["video_w"]), int(self.info["video_h"])
        self.depth_w, self.depth_h = int(self.info["depth_w"]), int(self.info["depth_h"])
        fdir = self.root / "frames"
        self.frames: list[FrameMeta] = []
        for i, jp in enumerate(sorted(fdir.glob("*.json"))):
            m = json.loads(jp.read_text())
            K = np.array(m["K"], np.float64).reshape(3, 3)
            ref_w, ref_h = (float(x) for x in m.get("K_ref") or (self.video_w, self.video_h))
            c = m.get("dist_center") or [K[0, 2], K[1, 2]]
            if m.get("K_video"):
                # 映像画素の K があればそれを優先する（参照寸法へ倍率で戻し、K_for で映像寸法にすると元に戻る）
                Kv = np.array(m["K_video"], np.float64).reshape(3, 3)
                sx, sy = ref_w / self.video_w, ref_h / self.video_h
                K = Kv.copy()
                K[0, 0] *= sx; K[0, 1] *= sx; K[0, 2] = Kv[0, 2] * sx
                K[1, 1] *= sy; K[1, 2] = Kv[1, 2] * sy
                # 注: Apple の K と同じく主点は連続座標 (cc) とみなす（0.5 px の差は実用上無視できる）
            cam = Camera(K, ref_w, ref_h, np.array(c, np.float64),
                         np.array(m.get("lut") or [], np.float64),
                         np.array(m.get("inv_lut") or [], np.float64))
            Tw = m.get("T_world_cam_arkit")
            self.frames.append(FrameMeta(i, jp.stem, float(m.get("t", i / float(self.info.get("fps", 30)))),
                                         cam, bool(m.get("depth_filtered", False)),
                                         np.array(Tw, np.float64).reshape(4, 4) if Tw else None,
                                         m.get("tracking")))
        # session.json の "frames" は 0 のことがある（アプリ側の不具合）。ファイルを数えた数を使う
        if not self.frames:
            raise SystemExit(f"{fdir} にフレーム（*.json）がない")
        self.check()

    def check(self):
        va = self.video_w / self.video_h
        da = self.depth_w / self.depth_h
        if abs(va - da) > 2e-3:
            raise SystemExit(f"映像 {self.video_w}x{self.video_h} と深度 {self.depth_w}x{self.depth_h} の縦横比が違う"
                             "（同じ視野という前提が崩れる）")
        for f in self.frames[:1]:
            ra = f.cam.ref_w / f.cam.ref_h
            if abs(ra - va) > 2e-3:
                raise SystemExit(f"K_ref {f.cam.ref_w:.0f}x{f.cam.ref_h:.0f} と映像の縦横比が違う"
                                 "（K を倍率だけで縮められない。クロップした映像は未対応）")

    def rgb(self, f: FrameMeta) -> np.ndarray | None:
        return cv2.imread(str(self.root / "frames" / f"{f.stem}.jpg"), cv2.IMREAD_COLOR)

    def depth(self, f: FrameMeta) -> np.ndarray | None:
        p = self.root / "frames" / f"{f.stem}.depth"
        if not p.exists():
            return None
        d = np.fromfile(p, dtype="<f4")
        if d.size != self.depth_w * self.depth_h:
            return None
        d = d.reshape(self.depth_h, self.depth_w).astype(np.float32)
        d[~np.isfinite(d) | (d <= 0)] = 0.0
        return d

    def confidence(self, f: FrameMeta) -> np.ndarray | None:
        """ARKit の深度信頼度（0 低 / 1 中 / 2 高）。ファイルがなければ None。"""
        p = self.root / "frames" / f"{f.stem}.conf"
        if not p.exists():
            return None
        c = np.fromfile(p, dtype=np.uint8)
        if c.size != self.depth_w * self.depth_h:
            return None
        return c.reshape(self.depth_h, self.depth_w)
