"""確認用の画像: RGB へのモデルの重ね描き（overlay）と 3 方向の正射影（preview）。"""
from __future__ import annotations

import cv2
import numpy as np


def _shade(normals: np.ndarray, view: np.ndarray) -> np.ndarray:
    light = np.array([0.3, -0.4, 1.0])
    light = light / np.linalg.norm(light)
    vd = np.abs(np.sum(normals * view, -1)) if view.ndim == 2 else np.abs(normals @ view)
    return 0.35 + 0.65 * np.clip(vd * 0.6 + np.clip(normals @ light, 0, 1) * 0.4, 0, 1)


def overlay(img: np.ndarray, mesh, T_cam_marker: np.ndarray, cam, w: int, h: int, marker_size: float) -> np.ndarray:
    """メッシュを歪みつきの映像へ投影して半透明で重ね、マーカーの正方形と軸を描く。"""
    V = mesh.vertices @ T_cam_marker[:3, :3].T + T_cam_marker[:3, 3]
    F = mesh.faces
    uv = cam.project_cv(V, w, h)
    Fn = mesh.face_normals @ T_cam_marker[:3, :3].T
    ctr = V[F].mean(1)
    front = np.sum(Fn * ctr, 1) < 0                          # 裏向きの三角形は描かない
    good = front & np.all(np.isfinite(uv[F]).all(2), 1)
    order = np.argsort(-ctr[good, 2])                        # 遠いものから描く
    fid = np.flatnonzero(good)[order]
    sh = _shade(Fn[fid], -ctr[fid] / np.linalg.norm(ctr[fid], axis=1, keepdims=True))
    layer = img.copy()
    pts = np.round(uv[F[fid]] * 4).astype(np.int32)
    for p, s in zip(pts, sh):
        c = (int(60 * s), int(220 * s), int(255 * s))
        cv2.fillPoly(layer, [p], c, lineType=cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
    hs = marker_size / 2
    sq = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]])
    ax = np.array([[0, 0, 0], [marker_size, 0, 0], [0, marker_size, 0], [0, 0, marker_size]])
    P = np.vstack([sq, ax]) @ T_cam_marker[:3, :3].T + T_cam_marker[:3, 3]
    q = cam.project_cv(P, w, h)
    if np.all(np.isfinite(q)):
        q = np.round(q).astype(int)
        cv2.polylines(out, [q[:4]], True, (0, 0, 255), 2, cv2.LINE_AA)
        for k, col in ((5, (0, 0, 255)), (6, (0, 200, 0)), (7, (255, 80, 0))):
            cv2.line(out, tuple(q[4]), tuple(q[k]), col, 3, cv2.LINE_AA)
    if out.shape[1] > 1280:
        s = 1280 / out.shape[1]
        out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return out


VIEWS = [  # (名前, 画面右の軸, 画面上の軸, 視線の反対 = 手前の軸)
    ("top (+Z から)", np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
    ("front (-Y から)", np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, -1, 0])),
    ("side (+X から)", np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
]


def preview(mesh, marker_size: float, panel: int = 520) -> np.ndarray:
    V, F = mesh.vertices, mesh.faces
    hs = marker_size / 2
    sq = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]])
    allp = np.vstack([V, sq, [[0, 0, 0]]])
    span = float(np.max(np.ptp(allp, 0))) * 1.15 + 1e-3
    centre = (allp.min(0) + allp.max(0)) / 2
    scale = (panel - 60) / span
    cols = mesh.visual.vertex_colors[:, :3].astype(float) if hasattr(mesh.visual, "vertex_colors") else None
    panels = []
    for name, ex, ey, ez in VIEWS:
        img = np.full((panel, panel, 3), 250, np.uint8)

        def to2(P):
            x = (P - centre) @ ex * scale + panel / 2
            y = panel / 2 - (P - centre) @ ey * scale
            return np.stack([x, y], -1)

        # 10 mm の格子（マーカー原点を通る）
        step = 0.01 * scale
        o = to2(np.zeros(3))
        for k in range(-60, 61):
            x, y = o[0] + k * step, o[1] + k * step
            if 0 <= x < panel:
                cv2.line(img, (int(x), 0), (int(x), panel), (232, 232, 232), 1)
            if 0 <= y < panel:
                cv2.line(img, (0, int(y)), (panel, int(y)), (232, 232, 232), 1)
        depth = (V[F].mean(1)) @ ez
        order = np.argsort(depth)
        sh = _shade(mesh.face_normals, ez)
        P2 = np.round(to2(V)[F] * 4).astype(np.int32)
        for i in order:
            s = sh[i]
            if cols is not None:
                c = cols[F[i]].mean(0)[::-1] * (0.5 + 0.5 * s)
            else:
                c = np.array([200, 180, 150]) * s
            cv2.fillPoly(img, [P2[i]], tuple(int(x) for x in c), lineType=cv2.LINE_AA, shift=2)
        q = np.round(to2(sq)).astype(int)
        cv2.polylines(img, [q], True, (0, 0, 230), 2, cv2.LINE_AA)
        L = marker_size
        for axis, col in ((np.array([1, 0, 0]), (0, 0, 230)), (np.array([0, 1, 0]), (0, 170, 0)),
                          (np.array([0, 0, 1]), (230, 90, 0))):
            a2 = to2(axis * L)
            if np.linalg.norm(a2 - o) > 3:
                cv2.arrowedLine(img, tuple(np.round(o).astype(int)), tuple(np.round(a2).astype(int)), col, 2,
                                cv2.LINE_AA, tipLength=0.2)
        lo, hi = V.min(0), V.max(0)
        w_ = (hi - lo) @ np.abs(ex) * 1000
        h_ = (hi - lo) @ np.abs(ey) * 1000
        cv2.putText(img, name.split(" ")[0], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(img, f"{w_:.1f} x {h_:.1f} mm  (grid 10 mm)", (10, panel - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (panel - 1, panel - 1), (180, 180, 180), 1)
        panels.append(img)
    return np.hstack(panels)
