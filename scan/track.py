"""マーカーが見えないフレームの姿勢を、物体モデルへの ICP 追跡で埋める。

  - マーカーで姿勢が決まったフレーム（アンカー）から前向き・後ろ向きにたどる。
  - 各フレームは直前の姿勢（＋等速の予測）から、モデルへの色つき ICP（Open3D colored ICP、点-面＋色の勾配）。
    使う点は「予測した姿勢で切り出し箱に入り、モデル表面から近い」ものだけ（手・顔・背景を混ぜない）。
  - 判定（一致点数・一致率・RMSE・1 フレームの動き）に落ちたらそこで鎖を切る。
  - アンカーに挟まれた区間で前向きの鎖が次のアンカーまで届いたら、届いた姿勢とアンカーのずれ E を
    区間に比例配分する（ループ補正）。E が大きければ、各フレームで近い側のアンカーからの鎖を使う。
"""
from __future__ import annotations

import math
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from process import Frame, Grid, angle_deg, backproject, inv, rigid

REG = o3d.pipelines.registration
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def make_model(P: np.ndarray, C: np.ndarray, vox: float):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.colors = o3d.utility.Vector3dVector(C)
    pc = pc.voxel_down_sample(vox)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 4, max_nn=30))
    return pc, cKDTree(np.asarray(pc.points))


def frame_cloud(fr: Frame, T_cam_marker: np.ndarray, grid: Grid, tree: cKDTree, near: float):
    """予測姿勢で、箱の中かつモデルから near 以内の点（マーカー座標）と色。"""
    P, idx = backproject(fr.D, fr.K_d)
    Tm = inv(T_cam_marker)
    Pm = P @ Tm[:3, :3].T + Tm[:3, 3]
    ok = np.all((Pm >= grid.lo) & (Pm < grid.hi), 1)
    Pm, idx = Pm[ok], idx[ok]
    if len(Pm) == 0:
        return Pm, np.zeros((0, 3))
    d, _ = tree.query(Pm, distance_upper_bound=near)
    ok = np.isfinite(d)
    return Pm[ok], fr.C.reshape(-1, 3)[idx[ok]][:, ::-1] / 255.0


def free_violation(fr: Frame, T_cam_marker: np.ndarray, model, a) -> float:
    """モデルの点を深度へ投影し、観測された表面より tau_free 以上手前にある割合（見えているはずの空間に
    モデルがある = 姿勢の矛盾）。箱の裏返し（180° の取り違え）をこれで弾く。"""
    P = np.asarray(model.points)
    if len(P) > 4000:
        P = P[:: len(P) // 4000]
    Pc = P @ T_cam_marker[:3, :3].T + T_cam_marker[:3, 3]
    z = Pc[:, 2]
    K = fr.K_d
    ok = z > 0.01
    u = np.round(K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 1] * Pc[ok, 1] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
    h, w = fr.D.shape
    ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = fr.Dmin[v[ins], u[ins]]                      # 3x3 の最小（手前側）で判定 = 甘め
    zz = z[ok][ins]
    valid = np.isfinite(d)
    if valid.sum() < 50:
        return 0.0
    return float(np.mean(zz[valid] < d[valid] - a.tau_free_mm / 1000))


def icp_track(fr: Frame, T_pred: np.ndarray, model, tree, grid: Grid, a, check_move: bool = True):
    """1 フレーム分。返り値: (T_cam_marker or None, 統計)。"""
    vox = a.voxel_mm / 1000 * 0.75
    P, C = frame_cloud(fr, T_pred, grid, tree, a.track_near_mm / 1000)
    st = {"n_src": int(len(P))}
    if len(P) < a.track_min_pts:
        st["why"] = f"モデル付近の点 {len(P)} 個"
        return None, st
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    src.colors = o3d.utility.Vector3dVector(C)
    src = src.voxel_down_sample(vox)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 4, max_nn=30))
    T = np.eye(4)
    try:
        for dist, it in ((a.track_near_mm / 1000 * 0.6, 30), (0.005, 30), (0.0025, 30)):
            if a.track_color and dist <= 0.0025:     # 色つきは最後の段だけ（モデル側の色勾配の計算が重い）
                r = REG.registration_colored_icp(src, model, dist, T, REG.TransformationEstimationForColoredICP(
                    lambda_geometric=0.9), REG.ICPConvergenceCriteria(1e-6, 1e-6, it))
            else:
                r = REG.registration_icp(src, model, dist, T, REG.TransformationEstimationPointToPlane(),
                                         REG.ICPConvergenceCriteria(1e-6, 1e-6, it))
            T = r.transformation
    except RuntimeError as e:
        st["why"] = f"ICP 失敗 {e}"
        return None, st
    ev = REG.evaluate_registration(src, model, 0.003, T)
    st.update(fitness=round(float(ev.fitness), 3), rmse_mm=round(float(ev.inlier_rmse) * 1000, 2),
              n_inl=int(round(ev.fitness * len(src.points))))
    T_new = T_pred @ inv(T)
    mv = T_new @ inv(T_pred)
    st.update(move_mm=round(float(np.linalg.norm(T_new[:3, 3] - T_pred[:3, 3]) * 1000), 1),
              move_deg=round(angle_deg(mv[:3, :3]), 1))
    st["free_viol"] = round(free_violation(fr, T_new, model, a), 3)
    bad = []
    if st["free_viol"] > a.track_max_free_viol:
        bad.append(f"素通し矛盾 {st['free_viol']:.2f}（モデルが観測面より手前に出る。裏返しの疑い）")
    if st["n_inl"] < a.track_min_inliers:
        bad.append(f"一致点 {st['n_inl']}")
    if ev.fitness < a.track_min_fitness:
        bad.append(f"一致率 {ev.fitness:.2f}")
    if ev.inlier_rmse * 1000 > a.track_max_rmse_mm:
        bad.append(f"RMSE {ev.inlier_rmse * 1000:.1f} mm")
    if check_move and (st["move_mm"] > a.track_max_move_mm or st["move_deg"] > a.track_max_move_deg):
        bad.append(f"予測から {st['move_mm']} mm / {st['move_deg']}°")
    if bad:
        st["why"] = "、".join(bad)
        return None, st
    return T_new, st


class Global:
    """局所 ICP が外れたときの再捕捉: 前回の物体中心のまわりの点を FPFH + RANSAC でモデルへ当て、色つき ICP で詰める。"""

    def __init__(self, model, a):
        self.a = a
        self.vox = 0.003
        self.model = model
        P = np.asarray(model.points)
        self.centre = P.mean(0)
        self.radius = float(np.percentile(np.linalg.norm(P - self.centre, axis=1), 98))
        _, _, Vt = np.linalg.svd(P - self.centre, full_matrices=False)
        self.flips = [np.eye(4)]
        for ax in Vt:
            R = Rotation.from_rotvec(np.pi * ax).as_matrix()
            self.flips.append(rigid(R, self.centre - R @ self.centre))
        self.tgt = model.voxel_down_sample(self.vox)
        self.tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 3, max_nn=30))
        self.tgt_f = REG.compute_fpfh_feature(self.tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 5,
                                                                                              max_nn=100))

    calls = 0
    seconds = 0.0

    def __call__(self, fr: Frame, T_last: np.ndarray, model, tree, grid, a):
        t0 = time.time()
        try:
            return self._run(fr, T_last, model, tree, grid, a)
        finally:
            Global.calls += 1
            Global.seconds += time.time() - t0

    def _run(self, fr: Frame, T_last: np.ndarray, model, tree, grid, a):
        c = T_last[:3, :3] @ self.centre + T_last[:3, 3]
        P, idx = backproject(fr.D, fr.K_d)
        ok = np.linalg.norm(P - c, axis=1) < self.radius + a.track_search_mm / 1000
        if ok.sum() < a.track_min_pts:
            return None, {"why": f"再捕捉: 前回位置の近くに点 {int(ok.sum())} 個"}
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[ok]))
        src.colors = o3d.utility.Vector3dVector(fr.C.reshape(-1, 3)[idx[ok]][:, ::-1] / 255.0)
        sd = src.voxel_down_sample(self.vox)
        sd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 3, max_nn=30))
        sf = REG.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 5, max_nn=100))
        best = None
        for seed in range(getattr(a, "track_global_now", a.track_global_tries)):
            o3d.utility.random.seed(seed + 1)
            r = REG.registration_ransac_based_on_feature_matching(
                sd, self.tgt, sf, self.tgt_f, True, self.vox * 1.5, REG.TransformationEstimationPointToPoint(False), 3,
                [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9), REG.CorrespondenceCheckerBasedOnDistance(self.vox * 1.5)],
                REG.RANSACConvergenceCriteria(20000, 0.999))
            T_mc = r.transformation                      # カメラ → マーカー
            # 直方体などの対称形は 180° 裏返しと取り違えやすいので、主軸まわりの 180° 回転も候補にする
            for F in self.flips:
                T_pred = inv(F @ T_mc)
                T, st = icp_track(fr, T_pred, model, tree, grid, a, check_move=False)
                if T is not None and (best is None or score(st) > score(best[1])):
                    best = (T, st)
        if best is None:
            return None, {"why": "再捕捉: RANSAC + ICP が判定に通らない"}
        T, st = best
        d = float(np.linalg.norm((T[:3, :3] @ self.centre + T[:3, 3]) - c) * 1000)
        st["reacquired"] = True
        st["jump_mm"] = round(d, 1)
        if d > a.track_search_mm + 20:
            return None, {"why": f"再捕捉: 前回位置から {d:.0f} mm"}
        return T, st


def score(st: dict) -> float:
    return st["fitness"] - 4 * st["free_viol"]


def chain(frames: list[Frame], order: list[int], anchors: dict[int, np.ndarray], model, tree, grid, a):
    """order の順にたどる。返り値: {i: (T, 統計, 起点アンカー i)}。アンカーに着いたら、そこで追跡した姿勢も記録。"""
    out, arrive = {}, {}
    prev = prevprev = None
    last = None                              # 最後に姿勢が分かったフレームの姿勢（再捕捉の探索中心）
    start = None
    broken = False                           # 再捕捉した鎖はループ補正に使わない
    glob = Global(model, a) if getattr(a, "track_global_now", a.track_global_tries) > 0 else None
    gap = 0
    for i in order:
        if i in anchors:
            if prev is not None and not broken:
                T, st = icp_track(frames[i], predict(prev, prevprev), model, tree, grid, a)
                if T is not None:
                    arrive[i] = (T, start)
            prev, prevprev, start, last, broken, gap = anchors[i], None, i, anchors[i], False, 0
            continue
        if last is None:
            continue
        T, st = (None, {"why": "直前が失敗"}) if prev is None else icp_track(frames[i], predict(prev, prevprev),
                                                                            model, tree, grid, a)
        if T is None and glob is not None and gap < a.track_max_gap:
            T2, st2 = glob(frames[i], last, model, tree, grid, a)
            if T2 is not None:
                T, st, broken = T2, dict(st2, local_why=st.get("why")), True
            else:
                st = dict(st, why=st.get("why", "") + " / " + st2["why"])
        if T is None:
            out[i] = (None, st, start)
            prev = prevprev = None
            gap += 1
            continue
        out[i] = (T, dict(st, chain_broken=broken), start)
        prevprev, prev, last, gap = prev, T, T, 0
    return out, arrive


def predict(prev, prevprev):
    if prevprev is None:
        return prev
    v = prev @ inv(prevprev)                      # 等速（半分だけ効かせる）
    rv = Rotation.from_matrix(v[:3, :3]).as_rotvec() * 0.5
    return rigid(Rotation.from_rotvec(rv).as_matrix(), v[:3, 3] * 0.5) @ prev


def interp_right(T: np.ndarray, E: np.ndarray, alpha: float) -> np.ndarray:
    """T · E^alpha（E はマーカー座標側の補正）。"""
    rv = Rotation.from_matrix(E[:3, :3]).as_rotvec() * alpha
    return T @ rigid(Rotation.from_rotvec(rv).as_matrix(), E[:3, 3] * alpha)


def track_all(frames: list[Frame], anchors: dict[int, np.ndarray], model, tree, grid, a) -> dict:
    n = len(frames)
    fwd, arr_f = chain(frames, list(range(n)), anchors, model, tree, grid, a)
    bwd, _ = chain(frames, list(range(n - 1, -1, -1)), anchors, model, tree, grid, a)
    result, stats = {}, {"loop_closed": 0, "loop_too_large": 0, "closure_mm": [], "closure_deg": []}
    akeys = sorted(anchors)
    nxt = {}
    for k, j in zip(akeys, akeys[1:]):
        nxt[k] = j
    for i in range(n):
        if i in anchors:
            continue
        f, b = fwd.get(i), bwd.get(i)
        fo = f is not None and f[0] is not None
        bo = b is not None and b[0] is not None
        if fo:
            s = f[2]
            j = nxt.get(s)
            if j is not None and j in arr_f and arr_f[j][1] == s and not f[1].get("chain_broken") and all(
                    fwd.get(k) is not None and fwd[k][0] is not None for k in range(s + 1, j)):
                E = inv(arr_f[j][0]) @ anchors[j]
                emm, edeg = float(np.linalg.norm(E[:3, 3]) * 1000), angle_deg(E[:3, :3])
                if emm <= a.track_loop_mm and edeg <= a.track_loop_deg:
                    alpha = (i - s) / (j - s)
                    result[i] = (interp_right(f[0], E, alpha), "ICP 追跡（前後のアンカー間、ループ補正）",
                                 dict(f[1], closure_mm=round(emm, 2), closure_deg=round(edeg, 2)))
                    continue
        if fo and bo:
            use_f = (i - f[2]) <= (b[2] - i)
            c = f if use_f else b
            result[i] = (c[0], "ICP 追跡（" + ("前" if use_f else "後ろ") + "のアンカーから、近い側）", c[1])
        elif fo:
            result[i] = (f[0], "ICP 追跡（前のアンカーから）", f[1])
        elif bo:
            result[i] = (b[0], "ICP 追跡（後ろのアンカーから）", b[1])
        else:
            why = []
            for nm, c in (("前向き", f), ("後ろ向き", b)):
                if c is None:
                    why.append(f"{nm}: 鎖が届かない（手前で切れた / アンカーなし）")
                else:
                    why.append(f"{nm}: {c[1].get('why', '?')}")
            result[i] = (None, "", {"why": why})
    # ループ補正の統計（アンカー区間ごと）
    for s, j in nxt.items():
        if j in arr_f and arr_f[j][1] == s:
            E = inv(arr_f[j][0]) @ anchors[j]
            emm, edeg = float(np.linalg.norm(E[:3, 3]) * 1000), angle_deg(E[:3, :3])
            stats["closure_mm"].append(round(emm, 2))
            stats["closure_deg"].append(round(edeg, 2))
            if emm <= a.track_loop_mm and edeg <= a.track_loop_deg:
                stats["loop_closed"] += 1
            else:
                stats["loop_too_large"] += 1
    return result, stats
