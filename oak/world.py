"""Provisional world frame from the ID 2 reference card.

world = the ID 2 marker frame itself: origin at the marker centre, x / y along the printed marker,
z out of the card, Z=0 on the card surface (the floor is at Z = -card thickness, not measured).

Two states, never mixed up:

  tracking  T_cam_world is the ID 2 pose of *this* frame. If ID 2 is not seen (or its pose is
            ambiguous / rejected) in this frame, the world frame is invalid for this frame —
            an old observation is never used as the current one.
  fixed     T_cam_world was registered from N consecutive good ID 2 observations and saved.
            Assumes neither the camera nor the card has moved since. While ID 2 is visible the
            assumption is checked every frame; a drift beyond tolerance invalidates the fixed
            reference until it is re-registered or released. While ID 2 is hidden it stays valid
            but is reported as unverified.

The saved reference is a provisional world frame only. It is not an arm-base calibration and is
written to a file that says so.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

import pose

REGISTER_FRAMES = 30
REGISTER_TIMEOUT_S = 5.0
REGISTER_MAX_POS_STD = 1.0e-3      # m
REGISTER_MAX_ROT_DEV = math.radians(1.0)
DRIFT_POS = 5.0e-3                 # m
DRIFT_ROT = math.radians(2.0)
REF_ID = 2


def mean_pose(Ts: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Average of rigid transforms (chordal mean for rotation) and the spread around it."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    pos_std = float(np.sqrt(np.mean([np.sum((T[:3, 3] - t) ** 2) for T in Ts])))
    rot_dev = max(pose.rotation_angle(R.T @ T[:3, :3]) for T in Ts)
    return pose.rigid(R, t), pos_std, rot_dev


class WorldRef:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.mode = "tracking"
        self.fixed: dict | None = None         # saved registration (T_cam_world etc.)
        self.fixed_verified = False            # ID 2 seen consistent since load / registration
        self.drift: dict | None = None
        self.registering: dict | None = None
        self.message = ""
        if path.exists():
            try:
                self.fixed = json.loads(path.read_text())
                self.fixed["T"] = np.array(self.fixed["T_cam_world"])
                self.mode = "fixed"
                self.message = f"前回登録した基準を読み込んだ（{self.fixed['registered_at']}）。ID 2 が見えたら検証する"
            except Exception as e:
                self.message = f"{path.name} を読めない: {e}"
                self.fixed = None

    # ---- commands from the web page
    def start_register(self) -> None:
        self.registering = {"started": time.time(), "poses": []}
        self.message = "基準を登録中: ID 2 とカメラを動かさないでください"

    def release(self) -> None:
        self.mode, self.drift, self.registering = "tracking", None, None
        self.message = "固定を解除した（毎フレーム追従）。保存済みファイルは残す"

    def use_fixed(self) -> None:
        if self.fixed is not None:
            self.mode, self.fixed_verified, self.drift = "fixed", False, None
            self.message = "保存済みの基準を使用（ID 2 が見えたら検証する）"

    # ---- per frame
    def update(self, obs: list[pose.Observation], t: float, cfg: pose.Config, meta: dict) -> dict:
        ref = next((o for o in obs if o.id == REF_ID), None)
        ok = ref is not None and ref.valid
        if self.registering is not None:
            self._register_step(ref if ok else None, cfg, meta)

        state = {"ref_id": REF_ID, "mode": self.mode, "message": self.message,
                 "id2_visible": ref is not None, "id2_status": None if ref is None else ref.status,
                 "card_physically_fixed": bool(cfg.markers.get(REF_ID) and cfg.markers[REF_ID].physically_fixed),
                 "registering": None if self.registering is None else len(self.registering["poses"]),
                 "register_frames": REGISTER_FRAMES, "saved": None, "valid": False, "T_cam_world": None,
                 "verified": None, "drift": None, "reason": ""}
        if self.fixed is not None:
            state["saved"] = {k: v for k, v in self.fixed.items() if k != "T"}

        if self.mode == "tracking":
            if ok:
                state.update(valid=True, T_cam_world=ref.T_cam_marker.tolist(), verified=True)
            else:
                state["reason"] = ("ID 2 がこのフレームで見えない" if ref is None
                                   else f"ID 2 の観測が {ref.status}（{'; '.join(ref.reasons)}）")
            return state

        T = self.fixed["T"]
        if ok:
            c = pose.compare(T, ref.T_cam_marker)
            self.drift = {"dt_m": c["dt_m"], "drot_rad": c["drot_rad"], "t": t}
            if c["dt_m"] > DRIFT_POS or c["drot_rad"] > DRIFT_ROT:
                self.fixed_verified = False
                state.update(reason=(f"固定基準と現在の ID 2 が {c['dt_m'] * 1000:.1f} mm / "
                                     f"{math.degrees(c['drot_rad']):.1f}° ずれた。カメラかカードが動いた可能性。"
                                     "再登録するか固定を解除する"), drift=self.drift)
                return state
            self.fixed_verified = True
        state.update(valid=True, T_cam_world=T.tolist(), drift=self.drift,
                     verified=self.fixed_verified and ok,
                     reason="" if ok else "ID 2 が見えないため固定基準を検証できない（カメラが動いていない前提で使用中）")
        return state

    def _register_step(self, ref, cfg: pose.Config, meta: dict) -> None:
        r = self.registering
        if ref is None:
            if r["poses"]:
                self.message = "登録中断: ID 2 の良好な観測が途切れた。もう一度押す"
                self.registering = None
            elif time.time() - r["started"] > REGISTER_TIMEOUT_S:
                self.message = "登録失敗: ID 2 が良好に観測できない"
                self.registering = None
            return
        r["poses"].append(ref.T_cam_marker)
        if len(r["poses"]) < REGISTER_FRAMES:
            return
        T, pos_std, rot_dev = mean_pose(r["poses"])
        self.registering = None
        if pos_std > REGISTER_MAX_POS_STD or rot_dev > REGISTER_MAX_ROT_DEV:
            self.message = (f"登録失敗: ばらつきが大きい（位置 {pos_std * 1000:.2f} mm, "
                            f"回転 {math.degrees(rot_dev):.2f}°）")
            return
        spec = cfg.markers.get(REF_ID)
        rec = {
            "purpose": "provisional world frame from the ID 2 card. NOT an arm-base calibration.",
            "frames": "T_cam_world: world (= ID 2 marker frame, Z=0 on the card surface) -> cam, metres",
            "T_cam_world": T.tolist(),
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_frames": REGISTER_FRAMES, "pos_std_m": pos_std, "rot_max_dev_rad": rot_dev,
            "card_physically_fixed": bool(spec and spec.physically_fixed),
            "card_thickness": spec.card_thickness.json() if spec and spec.card_thickness else None,
            "marker_size": spec.size.json() if spec else None,
            "intrinsics_source": meta.get("intrinsics_source"), "image_size": meta.get("image_size"),
            "camera": meta.get("camera"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        self.fixed = dict(rec, T=T)
        self.mode, self.fixed_verified, self.drift = "fixed", True, None
        self.message = (f"基準を登録・保存した（{self.path.name}、位置ばらつき {pos_std * 1000:.2f} mm, "
                        f"回転 {math.degrees(rot_dev):.2f}°）")
