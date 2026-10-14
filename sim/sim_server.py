#!/usr/bin/env python3
"""MuJoCo simulator for the B601-DM and the block, driven from the web page. Never commands the arm.

    MUJOCO_GL=egl python sim_server.py            # http://127.0.0.1:8092 (proxied by vision.py at /sim/)

Two modes:

  sync   (現実同期) no physics. Every tick copies the read-only joint angles (rebot/arm_state.py)
         and the observed block pose (oak/vision.py) into the model and runs mj_forward only, so
         the simulator cannot drift away from what was observed. Missing inputs are shown as
         missing (mock joints are labelled; a block that is not currently observed is hidden).
  trial  (シミュレーション試行) a copy of the current state; live inputs no longer overwrite it.
         The block is a free dynamic body; approach, close, lift and hold are simulated with
         contacts. Reset returns to sync.

Block placement: "observed" maps the vision block pose (ID 2 world) into the base frame with
scene.toml [placement]; unless that is calibrated, it uses the explicit sample placement and is
labelled 模擬（未校正）. "mock" places the block from scene.toml [mock_block]. A trial records which
basis it used, so a mock success is never reported as a success on the real layout.
"""
import argparse
import io
import json
import math
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mujoco
import numpy as np

import planner
import scene

HERE = Path(__file__).resolve().parent
BASIS_LABEL = {"mock": "模擬配置（scene.toml の mock_block、観測を使わない）",
               "observed_sample": "観測姿勢 + 仮の台座配置（カメラ⇔台座 未校正。実配置ではない）",
               "observed_calibrated": "観測姿勢 + 校正済みの台座配置"}


def rpy(r):
    cx, sx, cy, sy, cz, sz = math.cos(r[0]), math.sin(r[0]), math.cos(r[1]), math.sin(r[1]), math.cos(r[2]), math.sin(r[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def fetch(url: str, timeout=0.3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"available": False, "error": str(e)}


class Sim:
    def __init__(self, args) -> None:
        self.args = args
        self.settings = scene.load_settings()
        self.m, _ = scene.load_model(self.settings)
        self.d = mujoco.MjData(self.m)
        self.ids = scene.Ids(self.m)
        self.gap = scene._gap_model(self.m)
        self.lock = threading.RLock()
        self.mode = "sync"
        self.placement_source = "mock"
        self.sync_info = {}
        self.joint_source = ["—"] * 6
        self.block_visible = False
        self.block_basis = None
        self.plan = None
        self.plan_basis = None
        self.trial = None
        self.trial_thread = None
        self.stop_trial = threading.Event()
        self.view = {"cam": "overview", "geom": "visual"}
        self.fk_vs_vision = ""
        self.real_fk = None
        mujoco.mj_forward(self.m, self.d)

    # ------------------------------------------------------------------ placement
    def handeye(self):
        f = HERE.parent / "oak" / "calib" / "handeye_active.json"
        if f.exists():
            try:
                h = json.loads(f.read_text())
                if h.get("ok") and h.get("status") == "calibrated":
                    return h
            except Exception:
                return None
        return None

    def T_base_world(self):
        p = self.settings["placement"]
        if p["status"] == "calibrated":
            return rigid(rpy(p["rotation_rpy_rad"]), p["translation_m"]), "observed_calibrated"
        return rigid(rpy(p["sample_rotation_rpy_rad"]), p["sample_translation_m"]), "observed_sample"

    def mock_block(self):
        mb = self.settings["mock_block"]
        return rigid(rpy([0, 0, mb["yaw_rad"]]), mb["position_m"])

    def observed_block(self, vs: dict):
        """Block pose in base from the current vision frame, or (None, reason). Never averaged."""
        if vs.get("available") is False or "block" not in vs:
            return None, "認識サーバーに接続できない"
        if time.time() - vs.get("t", 0) > 0.5:
            return None, f"観測が古い（{time.time() - vs.get('t', 0):.1f} s 前）"
        if not vs["world"]["valid"] and self.handeye() is None:
            return None, "ID 2 world が無効: " + vs["world"].get("reason", "")
        bt = vs.get("block_table") or {}
        he = self.handeye()
        if bt.get("status") == "on_table" and he is not None:
            # resting on the table: (x, y, yaw) fitted to all block-marker corners, height and tilt from the
            # table plane (oak/table_fit.py) -- no IPPE ambiguity, smoothed while static
            return (np.array(bt["T_base_obj"]), "observed_calibrated",
                    f"机の面に拘束（ID {', '.join(map(str, bt['markers']))}、{bt['face']}、{bt['rms_px']:.2f} px）、校正 {he.get('session')}",
                    False, None), None
        side = {str(mk["id"]): mk["side_px"] for mk in vs["markers"]}
        cands = [(k, b) for k, b in vs["block"].items() if b["T_world_obj"] and b["status"] in ("ok", "pair")]
        uncertain = False
        if not cands:
            # Show an ambiguous / flipped estimate, clearly marked, but never plan or simulate on it.
            cands = [(k, b) for k, b in vs["block"].items() if b["T_world_obj"]]
            uncertain = True
            if not cands:
                return None, "ブロックの観測がない（ID 0 / ID 3 が見えていない、または取り付け未登録）"
        k, b = max(cands, key=lambda kb: side.get(kb[0], 0))      # the larger marker in the image
        he = self.handeye()
        if he is not None:     # camera <-> base calibrated: map the camera-frame pose directly
            T = np.array(he["T_base_cam"]) @ np.array(b["T_cam_obj"])
            return (T, "observed_calibrated", f"ID {k} から（{b['status']}）、校正 {he.get('session')}", uncertain,
                    np.array(b["T_world_obj"]) if b["T_world_obj"] else None), None
        T, basis = self.T_base_world()
        return (T @ np.array(b["T_world_obj"]), basis, f"ID {k} から（{b['status']}）", uncertain, np.array(b["T_world_obj"])), None

    def set_block(self, T):
        q = self.ids.block_q
        if T is None:
            self.d.qpos[q:q + 7] = [10, 10, -1, 1, 0, 0, 0]       # out of sight
            return
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
        self.d.qpos[q:q + 3], self.d.qpos[q + 3:q + 7] = T[:3, 3], quat
        self.d.qvel[self.ids.block_d:self.ids.block_d + 6] = 0

    def block_T(self):
        q = self.ids.block_q
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[q + 3:q + 7])
        return rigid(R.reshape(3, 3), self.d.qpos[q:q + 3].copy())

    # ------------------------------------------------------------------ sync
    def sync_tick(self) -> None:
        arm = fetch(self.args.arm_url + "/state.json")
        vs = fetch(self.args.vision_url + "/state.json")
        with self.lock:
            if self.mode != "sync":
                return
            info = {}
            q = np.array(self.settings["mock_joints"]["q_rad"], float)
            src = ["模擬姿勢（scene.toml）"] * 6
            grip = None
            if arm.get("ok"):
                by = {j["name"]: j for j in arm["joints"]}
                for i, n in enumerate(scene.ARM_JOINTS):
                    if by.get(n) and by[n]["pos"] is not None and by[n]["age_s"] is not None and by[n]["age_s"] < 0.5:
                        q[i], src[i] = by[n]["pos"], "実機（読み取り専用）"
                g = by.get("gripper")
                if g and g["pos"] is not None:
                    grip = g["pos"]
                info["joints"] = "実機の関節角を反映中（読み取り専用）" + ("" if all(s.startswith("実機") for s in src) else "、一部は模擬")
                self.real_fk = arm.get("fk")
            else:
                info["joints"] = "実機の関節角なし → 模擬姿勢: " + (arm.get("error") or "arm_state の応答なし / 値が古い")
                self.real_fk = None
            self.d.qpos[self.ids.arm_qadr] = q
            self.joint_source = src
            jaw = self.settings["jaw"]
            if grip is not None:
                gap = float(np.clip(grip / jaw["gripper_rad_full_open"] * jaw["max_gap_m"], 0, jaw["max_gap_m"]))
                half = float(np.clip((gap - self.gap.gap0) / 2, *self.ids.fl_range))
                info["gripper"] = (f"実機グリッパー角 {grip:+.3f} rad → 隙間 {gap * 1000:.0f} mm（換算式: "
                                   + ("実測済み、開閉の向きで約 1 mm ずれる）" if jaw.get("gripper_map_status") in ("measured", "calibrated")
                                      else "未検証）"))
            else:
                half = 0.0
                info["gripper"] = "グリッパー角なし → 閉"
            self.d.qpos[self.ids.fl_q], self.d.qpos[self.ids.fr_q] = half, -half
            self.block_uncertain = False
            if self.placement_source == "mock":
                self.set_block(self.mock_block())
                self.block_visible, self.block_basis = True, "mock"
                info["block"] = BASIS_LABEL["mock"]
            else:
                ob, why = self.observed_block(vs)
                if ob is None:
                    self.set_block(None)
                    self.block_visible, self.block_basis = False, None
                    info["block"] = "非表示: " + why
                else:
                    T, basis, which, uncertain, T_world_obj = ob
                    self.set_block(T)
                    self.block_visible, self.block_basis = True, basis
                    self.block_uncertain = uncertain
                    self.block_obs = {"T_world_obj": None if T_world_obj is None else T_world_obj.tolist(),
                                      "T_base_obj": T.tolist(), "which": which}
                    info["block"] = f"{BASIS_LABEL[basis]}、{which}" + ("。観測が曖昧なため表示のみ（計画・試行には使わない）" if uncertain else "")
            self.m.geom_rgba[self.ids.block_geom] = [1.0, 0.5, 0.1, 0.45] if self.block_uncertain else [0.95, 0.85, 0.15, 1]
            mujoco.mj_forward(self.m, self.d)        # kinematics only, no physics in sync mode
            self.sync_info = info
            self.fk_vs_vision = self._fk_vs_vision(vs)

    def _fk_vs_vision(self, vs: dict) -> str:
        he = self.handeye()
        cal = he is not None or self.settings["placement"]["status"] == "calibrated"
        g = (vs.get("config") or {}).get("markers", {}).get("1", {}).get("grip", {})
        grip_cal = g.get("status") in ("measured", "calibrated")
        if not (cal and grip_cal) or he is None:
            return ("関節角からの手先と、ID 1 の観測は別々に保持している。カメラ⇔台座"
                    + ("" if cal else "（未校正）") + "・ID 1→把持中心" + ("" if grip_cal else "（未校正）")
                    + " が揃うまで比較しない。")
        # ID 1 where the joint angles (FK, model angles incl. zero offsets) and the adopted calibration
        # put it, vs where the camera sees it now. Only with fresh joints and an unambiguous observation.
        m1 = next((m for m in vs.get("markers", []) if m.get("id") == 1 and m.get("status") == "ok"), None)
        if m1 is None or self.sync_info.get("joints", "").startswith("実機の関節角なし"):
            return "校正済み。ID 1 が ok で見えていないか関節角がないため、今は比較していない"
        F = np.eye(4)
        F[:3, :3], F[:3, 3] = self.d.xmat[self.ids.end_body].reshape(3, 3), self.d.xpos[self.ids.end_body]
        P = np.array(he["T_cam_base"]) @ F @ np.array(he["T_ee_m1"])
        O = np.array(m1["T_cam_marker"])
        dp = (P[:3, 3] - O[:3, 3]) * 1000
        Rd = P[:3, :3].T @ O[:3, :3]
        ang = math.degrees(math.acos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
        return (f"ID 1: 関節角＋校正の予測と観測の差 {np.linalg.norm(dp):.1f} mm（画像の面内 {np.linalg.norm(dp[:2]):.1f} mm、"
                f"奥行き {dp[2]:+.1f} mm）、向き {ang:.1f}°（校正 {he.get('session')}、採用時の評価: 位置 中央値 "
                f"{he.get('holdout', {}).get('pos_err_mm_median', float('nan')):.1f} mm）")

    # ------------------------------------------------------------------ plan / trial
    def do_plan(self) -> str:
        with self.lock:
            if not self.block_visible:
                return "ブロックが配置されていないため計画できない"
            if getattr(self, "block_uncertain", False):
                return "ブロックの観測が曖昧（IPPE の 2 解が近い / 反転）なため計画しない"
            T = self.block_T()
            q0 = self.d.qpos[self.ids.arm_qadr].copy()
            basis = self.block_basis
        cands = planner.plan(self.m, self.settings, T, q0, view=self.view_model())
        with self.lock:
            self.plan = {"basis": basis, "basis_label": BASIS_LABEL[basis], "T_base_obj": T.tolist(),
                         "candidates": cands, "t": time.time()}
        ok = [c for c in cands if c.ok]
        return f"候補 {len(cands)} 件中 {len(ok)} 件が条件を満たす（{BASIS_LABEL[basis]}）"

    def set_mode(self, mode: str) -> str:
        if mode == "sync":
            return self.reset()
        with self.lock:
            if self.mode == "sync":
                self.mode = "trial"
                self.trial = {"phase": "待機（状態をコピー済み、ライブ観測による上書きを停止）", "log": [],
                              "running": False, "done": False, "success": False, "result": "",
                              "basis": self.block_basis, "basis_label": BASIS_LABEL.get(self.block_basis, "ブロックなし")}
        return "シミュレーション試行モード（現在の状態をコピーした）"

    def reset(self) -> str:
        self.stop_trial.set()
        if self.trial_thread:
            self.trial_thread.join(timeout=5)
        with self.lock:
            self.mode = "sync"
            self.d.qvel[:] = 0
            self.d.ctrl[:] = 0
            self.stop_trial.clear()
        return "現実同期モードへ戻した（試行の状態は破棄。結果の記録は output/trials に残る）"

    def start_trial(self, cand_index: int | None) -> str:
        if self.trial_thread and self.trial_thread.is_alive():
            return "試行中"
        if self.mode == "sync":
            self.set_mode("trial")
        with self.lock:
            if not self.block_visible:
                return "ブロックが配置されていないため試行できない"
            if getattr(self, "block_uncertain", False):
                self.mode = "sync"
                return "ブロックの観測が曖昧なため試行しない"
            if self.plan is None or self.plan["basis"] != self.trial["basis"]:
                pass
        if self.plan is None or self.plan["basis"] != self.trial["basis"]:
            self.do_plan()
        ok = [c for c in self.plan["candidates"] if c.ok]
        if cand_index is not None:
            cand = self.plan["candidates"][cand_index]
        elif ok:
            cand = ok[0]
        else:
            return "条件を満たす把持候補がない（試行しない）"
        self.trial_thread = threading.Thread(target=self._run_trial, args=(cand,), daemon=True)
        self.trial_thread.start()
        return f"試行開始: {cand.name}"

    def _log(self, msg: str) -> None:
        self.trial["log"].append(f"{self.d.time:6.2f}s {msg}")

    def _run_trial(self, cand: planner.Candidate) -> None:
        m, d, ids, tr = self.m, self.d, self.ids, self.trial
        jaw = self.settings["jaw"]
        max_gap = min(jaw["max_gap_m"], self.gap(ids.fl_range[1]))
        with self.lock:
            tr.update(running=True, done=False, success=False, result="", candidate=cand.name, phase="初期化")
            d.time = 0.0
            d.qvel[:] = 0
            # The copied block pose may float above or sink into the floor (observed pose with an
            # uncalibrated placement). Rest it on the floor, and say by how much it was moved.
            T = self.block_T()
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * \
                np.array(self.settings["block"]["size_m"]) / 2
            zmin = float(np.min((T[:3, :3] @ corners.T)[2] + T[2, 3]))
            if abs(zmin) > 1e-4:
                d.qpos[ids.block_q + 2] -= zmin
                self._log(f"ブロックの最下点 z={zmin * 1000:+.1f} mm → 床に接地させた（補正 {-zmin * 1000:+.1f} mm）")
            q_now = d.qpos[ids.arm_qadr].copy()
            d.ctrl[ids.act_arm] = q_now
            open_half = float(np.clip((min(cand.width_m + 0.01, max_gap) - self.gap.gap0) / 2, *ids.fl_range))
            d.ctrl[ids.act_grip] = d.qpos[ids.fl_q]
            z0 = float(d.qpos[ids.block_q + 2])
        self._log(f"候補: {cand.name}、必要開口 {cand.width_m * 1000:.0f} mm、開く量 {2 * open_half * 1000 + self.gap.gap0 * 1000:.0f} mm")
        try:
            self._move(None, open_half, 0.6, "顎を開く")
            self._move(cand.q_pre, open_half, 2.0, "接近前の姿勢へ")
            self._move(cand.q_grasp, open_half, 1.5, "把持位置へ下降")
            self._move(None, 0.0, 1.2, "顎を閉じる")
            self._settle(0.5, "保持力の安定待ち")
            fl_closed = float(d.qpos[ids.fl_q])
            gap_closed = self.gap(fl_closed)
            self._log(f"閉じた後の隙間 {gap_closed * 1000:.1f} mm")
            self._move(cand.q_lift, 0.0, 2.0, "持ち上げ")
            self._settle(1.5, "保持（1.5 s）")
            with self.lock:
                res = self._evaluate(z0, gap_closed)
                tr.update(res)
        except InterruptedError:
            with self.lock:
                tr.update(result="中断（リセット）", success=False)
        finally:
            with self.lock:
                tr.update(running=False, done=True, phase="終了")
            self._save_result(cand)

    def _step_until(self, t_end: float) -> None:
        while d_time(self) < t_end:
            if self.stop_trial.is_set():
                raise InterruptedError
            t0 = time.monotonic()
            with self.lock:
                for _ in range(10):
                    mujoco.mj_step(self.m, self.d)
            time.sleep(max(0.0, 0.010 - (time.monotonic() - t0)))   # about real time

    def _move(self, q_target, finger_target, duration: float, label: str) -> None:
        d, ids = self.d, self.ids
        with self.lock:
            self.trial["phase"] = label
            q0 = d.ctrl[ids.act_arm].copy()
            f0 = float(d.ctrl[ids.act_grip])
            t0 = d.time
        self._log(label)
        qt = q0 if q_target is None else np.asarray(q_target)
        while True:
            with self.lock:
                s = min(1.0, (d.time - t0) / duration)
                w = 0.5 - 0.5 * math.cos(math.pi * s)
                d.ctrl[ids.act_arm] = q0 + (qt - q0) * w
                d.ctrl[ids.act_grip] = f0 + (finger_target - f0) * w
            if s >= 1.0:
                break
            self._step_until(d.time + 0.02)
        self._settle(0.3, None)

    def _settle(self, duration: float, label: str | None) -> None:
        if label:
            with self.lock:
                self.trial["phase"] = label
            self._log(label)
        self._step_until(self.d.time + duration)

    def _contacts(self) -> dict:
        m, d, ids = self.m, self.d, self.ids
        out = {"left": 0, "right": 0, "floor_block": 0, "arm_floor": 0, "other": []}
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if ids.block_body in pair:
                other = (pair - {ids.block_body}).pop() if len(pair) > 1 else ids.block_body
                if other in ids.finger_left:
                    out["left"] += 1
                elif other in ids.finger_right:
                    out["right"] += 1
                elif ids.floor_geom in (c.geom1, c.geom2):
                    out["floor_block"] += 1
                else:
                    out["other"].append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other))
            elif ids.floor_geom in (c.geom1, c.geom2):
                out["arm_floor"] += 1
        return out

    def _evaluate(self, z0: float, gap_closed: float) -> dict:
        c = self._contacts()
        dz = float(self.d.qpos[self.ids.block_q + 2] - z0)
        size = np.array(self.settings["block"]["size_m"])
        lifted = dz > 0.8 * planner.LIFT
        held = c["left"] > 0 and c["right"] > 0
        reasons = []
        if not lifted:
            reasons.append(f"持ち上がっていない（上昇 {dz * 1000:.1f} mm / 目標 {planner.LIFT * 1000:.0f} mm）")
        if not held:
            reasons.append(f"両顎で接触していない（左 {c['left']}、右 {c['right']} 点）")
        if c["floor_block"]:
            reasons.append("ブロックが床に接したまま")
        if gap_closed < min(size) * 0.5:
            reasons.append(f"顎がほぼ閉じきった（隙間 {gap_closed * 1000:.1f} mm）＝物体を挟めていない")
        ok = lifted and held and not c["floor_block"]
        res = {"success": ok, "lift_mm": dz * 1000, "contacts": c, "gap_closed_mm": gap_closed * 1000,
               "result": ("成功: 持ち上げて両顎で保持（シミュレーション内、" + BASIS_LABEL.get(self.trial["basis"], "") + "）")
               if ok else "失敗: " + "; ".join(reasons)}
        self._log(res["result"])
        return res

    def _save_result(self, cand) -> None:
        out = HERE / "output" / "trials"
        out.mkdir(parents=True, exist_ok=True)
        with self.lock:
            rec = {k: v for k, v in self.trial.items() if k != "log"}
            rec.update(log=self.trial["log"], candidate=cand.json(), settings=self.settings,
                       saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        (out / (time.strftime("%Y%m%d-%H%M%S") + ".json")).write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))

    # ------------------------------------------------------------------ real-arm path checks
    # The arm service executes only paths stored here, fetched by id; the browser never sends raw
    # joint targets to the arm. The check covers the floor, self-collision (beyond the hull
    # tolerance) and joint limits along the straight joint-space path. The block and the camera
    # stand are NOT checked: their positions relative to the base are not calibrated.
    def check_path(self, target: str, cand_index: int | None) -> dict:
        """Simulate the move the real arm would make, and accept it only if the simulation is clean.

        The profile (times, joint targets) computed here is exactly what the arm service executes:
        straight joint-space path, max joint speed VMAX. The simulation runs the model's dynamics
        with gravity and the sim position actuators tracking that profile (block removed: its real
        position is unknown until calibrated). Checks: floor clearance >= FLOOR_CLEARANCE for every
        arm / jaw geom (so a real arm that sags or lags inside the execution corridor still clears
        it), self-collision beyond the hull tolerance, joint limits, and the simulated tracking error.
        Not checked: the block and the camera stand (positions relative to the base not calibrated).
        """
        arm = fetch(self.args.arm_url + "/state.json")
        if not arm.get("ok"):
            return {"ok": False, "message": "実機の関節角が読めない: " + str(arm.get("error", "応答なし"))}
        by = {j["name"]: j["pos"] for j in arm["joints"]}
        q_from = np.array([by[n] for n in scene.ARM_JOINTS])
        if target.startswith("calib:"):
            cp = getattr(self, "calib_poses", None)
            if not cp:
                return {"ok": False, "message": "校正姿勢が未生成"}
            p = cp["poses"][int(target.split(":")[1])]
            if not p["ok"]:
                return {"ok": False, "message": "この校正姿勢は静的チェック不合格: " + "; ".join(p["reasons"])}
            q_to, label = np.array(p["q"]), f"校正姿勢 {p['i']}（{p['role']}）{p['label']}"
        elif target == "zero":
            q_to, label = np.zeros(6), "ゼロ姿勢（全関節 0 rad = モーター零点の姿勢）"
        elif target == "engage_pose":
            qe = (arm.get("control") or {}).get("q_engage")
            if not qe:
                return {"ok": False, "message": "有効化時の姿勢が記録されていない"}
            q_to, label = np.array([qe[n] for n in scene.ARM_JOINTS]), "有効化時の姿勢へ戻る"
        else:
            if not self.plan:
                return {"ok": False, "message": "把持候補が未計算"}
            ok = [c for c in self.plan["candidates"] if c.ok]
            cand = self.plan["candidates"][cand_index] if cand_index is not None else (ok[0] if ok else None)
            if cand is None or not cand.ok:
                return {"ok": False, "message": "条件を満たす候補がない"}
            q_to = {"pre": cand.q_pre, "near": cand.q_near, "grasp": cand.q_grasp, "lift": cand.q_lift}[target]
            if q_to is None:
                return {"ok": False, "message": "この候補には近接の姿勢がない（計画し直す）"}
            label = f"{cand.name}: " + {"pre": "接近前の姿勢", "near": "近接（把持の 3 cm 手前）", "grasp": "把持姿勢",
                                        "lift": "持ち上げ姿勢"}[target]
            if cand.correction:
                label += f"（視覚補正 {np.linalg.norm(cand.correction['offset_mm']):.1f} mm 済み）"
        vmax = VMAX_FINAL if target == "grasp" else None
        res = simulate_profile(self.m, self.ids, q_from, np.asarray(q_to, float), self.gap.gap0, vmax)
        import uuid
        pid = uuid.uuid4().hex[:12]
        rec = dict(res, id=pid, label=label, checked_at=time.time(), q_from=q_from.tolist(),
                   message=("シミュレーションで問題なし（床との距離・登録済み障害物・自己衝突・関節制限・追従）。"
                            + ("未登録の障害物: " + ", ".join(o["name"] for o in scene.obstacle_report(self.settings) if not o["in_scene"]) + "。" 
                               if any(not o["in_scene"] for o in scene.obstacle_report(self.settings)) else "")
                            + "実ブロックは camera↔base 未校正のため未確認"
                            if res["ok"] else "実行不可: " + "; ".join(res["reasons"])))
        with self.lock:
            self.paths = {k: v for k, v in getattr(self, "paths", {}).items() if time.time() - v["checked_at"] < 600}
            if rec["ok"]:
                self.paths[pid] = rec
            self.last_path = rec
            self.preview = {"traj": res["sim_q"], "dt": res["sim_dt"], "t0": time.monotonic()}
        return {k: v for k, v in rec.items() if k not in ("sim_q",)}

    def _grip_ok(self) -> bool:
        import tomllib
        g = tomllib.loads((HERE.parent / "oak" / "markers.toml").read_text()).get("marker", {}).get("1", {}).get("grip", {})
        return g.get("status") in ("measured", "calibrated")

    # ------------------------------------------------------------------ calibration poses
    def make_calib_poses(self, n_calib: int = 10, n_eval: int = 4) -> dict:
        """Static poses for the eye-to-hand calibration, around the current real pose.

        Rotations of end_link about its own x, y and z axes (single and combined, 17-30 deg) plus a
        few cm of translation, so that relative motions excite several rotation axes. Every pose is
        checked statically (IK, limits, floor / registered obstacles, self-collision); the path to it
        is simulated separately before it may be executed. If the arm is folded at rest, the centre
        is the rest orientation lifted by 10 cm. Whether ID 1 stays visible is only known when the
        sample is captured.
        """
        arm = fetch(self.args.arm_url + "/state.json")
        if not arm.get("ok"):
            return {"ok": False, "message": "実機の関節角が読めない"}
        q0 = np.array([j["pos"] for j in arm["joints"][:6]])
        m, ids = self.m, self.ids
        d = mujoco.MjData(m)
        d.qpos[ids.arm_qadr] = q0
        mujoco.mj_kinematics(m, d)
        R0 = d.xmat[ids.end_body].reshape(3, 3).copy()
        p0 = d.xpos[ids.end_body].copy()
        folded = q0[1] > -0.25 and q0[2] > -0.25
        if folded:
            p0 = p0 + np.array([0, 0, 0.10])
        site_off = np.array([-0.025, 0, 0])
        a = math.radians
        specs = [("x", 25, [0, 0, 0]), ("x", -25, [0.03, 0, 0]), ("y", 25, [0, 0.03, 0]), ("y", -25, [0, -0.03, 0]),
                 ("z", 30, [0, 0, 0.03]), ("z", -30, [-0.03, 0, 0]), ("xy", 20, [0.02, 0.02, 0]), ("yz", 20, [0, 0, -0.02]),
                 ("xz", -20, [0.02, -0.02, 0.02]), ("xyz", 17, [-0.02, 0.02, 0.0]),
                 # evaluation poses: different combinations, never used for fitting
                 ("xy", -22, [-0.02, 0, 0.02]), ("yz", -25, [0.03, 0.02, 0]), ("xz", 22, [0, -0.03, -0.02]), ("z", 15, [0.02, 0.03, 0.01]),
                 # extra calibration poses (joint zero offsets need more samples)
                 ("x", 15, [0, 0.03, 0.02]), ("x", -15, [-0.02, 0, 0.02]), ("y", 15, [0.02, -0.02, 0]), ("y", -15, [-0.02, 0.02, 0.02]),
                 ("z", 20, [0.03, 0, -0.02]), ("z", -20, [0, -0.03, 0.02]), ("xy", -15, [0.03, 0.03, 0]), ("yz", 15, [-0.03, 0, 0])]
        out = []
        chosen = [(sp, "calib") for sp in specs[:n_calib]] + [(sp, "eval") for sp in specs[10:10 + n_eval]] + \
                 [(sp, "calib") for sp in specs[14:]]
        for k, ((axes, deg, dt), role) in enumerate(chosen):
            R = R0.copy()
            for ax in axes:
                R = R @ planner_rot(ax, a(deg))
            p_end = p0 + np.array(dt)
            p_site = p_end + R @ site_off
            best = None
            for seed in (q0, np.array(q0) + [0, -0.5, -0.6, 0, 0, 0]):
                q, pe, re = planner.solve_ik(m, d, ids, p_site, R, np.asarray(seed, float))
                if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
                    best = (q, pe, re)
            q, pe, re = best
            reasons = []
            if pe > planner.IK_POS_TOL or re > planner.IK_ROT_TOL:
                reasons.append(f"IK 解なし（{pe * 1000:.1f} mm / {math.degrees(re):.1f}°）")
            k_ = planner.static_contacts(m, d, ids, q, self.gap.gap0, block_away=True)
            if k_["floor"]:
                reasons.append("床・登録障害物（余裕込み）と干渉")
            if [x for x in k_["self"] if x[0] > planner.SELF_PEN_TOL]:
                reasons.append("自己衝突")
            big = float(np.max(np.abs(np.degrees(q - q0) * ([1, 1, 1, 1, 1, 1]))))
            if big > CALIB_MAX_JOINT_DEG:
                reasons.append(f"関節変化が大きい（最大 {big:.0f}° > {CALIB_MAX_JOINT_DEG}°）")
            out.append({"i": k, "role": role, "label": f"{axes} {deg:+d}° / {np.round(np.array(dt) * 100, 1).tolist()} cm",
                        "q": q.tolist(), "ok": not reasons, "reasons": reasons})
        with self.lock:
            self.calib_poses = {"q0": q0.tolist(), "folded_centre_lifted": folded, "poses": out, "t": time.time()}
        n_ok = sum(p["ok"] for p in out)
        return {"ok": n_ok > 0, "message": f"校正姿勢 {len(out)} 個を生成（静的チェック合格 {n_ok}）。"
                                           "各姿勢へは経路シミュレーション → 実機実行（依頼者）→ サンプル記録の順"}

    def make_calib_poses_workspace(self, n: int = 8, n_eval: int = 2, tries: int = 30000) -> dict:
        """Calibration poses spread over the workspace, chosen where ID 1 is predicted to be seen.

        Random joint configurations in a broad working region (joint2 -120..-3 deg, joint3 -150..-3, joint4/5 within
        +-100, joint1 within +-90 of the current pose; end_link 12-45 cm from the base axis and >= 6 cm high; the
        full joint ranges produced folded-over and behind-the-base poses); kept if the static check passes (floor / registered
        obstacles with their margins, self-collision, end_link >= 5 cm above the floor) and the ID 1 marker,
        placed with the latest hand-eye estimate (T_cam_base, T_ee_m1 from oak/calib/handeye/<newest>/result.json,
        used for this prediction even if not adopted) and the camera intrinsics from the vision service,
        projects fully inside the image (60 px margin), at least 30 px a side, facing the camera within 50 deg.
        From those, n are taken that differ most in end_link orientation (what the calibration needs) and in arm
        configuration (joints 2-5, for the joint zero offsets) -- farthest-point on a combined distance -- and
        ordered as a short tour from the current pose; no step changes a joint by more than CALIB_STEP_MAX_DEG
        (bridging candidates are inserted only where needed). Occlusion by the arm itself is not predicted. Paths are simulated before execution as usual.
        """
        arm = fetch(self.args.arm_url + "/state.json")
        if not arm.get("ok"):
            return {"ok": False, "message": "実機の関節角が読めない"}
        q0 = np.array([j["pos"] for j in arm["joints"][:6]])
        vs = fetch(self.args.vision_url + "/state.json")
        intr = (vs or {}).get("intrinsics") or {}
        if not intr.get("K"):
            return {"ok": False, "message": "カメラの内部パラメータが読めない（vision）"}
        K, dist = np.array(intr["K"]), np.array(intr.get("dist_coeffs") or np.zeros(5))
        W, H = intr["image"]
        root = HERE.parent / "oak" / "calib" / "handeye"
        res = None
        for dd in sorted([x for x in root.iterdir() if x.is_dir()], reverse=True) if root.exists() else []:
            f = dd / "result.json"
            if f.exists():
                r = json.loads(f.read_text())
                if r.get("T_cam_base") and r.get("T_ee_m1"):
                    res, src = r, f"{dd.name}/result.json（{'採用条件を満たす' if r.get('ok') else '未採用'}の推定、見え方の予測だけに使う）"
                    break
        if res is None:
            return {"ok": False, "message": "hand-eye の推定結果がない（先に既存の方法でサンプルを集めて「解く」）"}
        X, Y = np.array(res["T_cam_base"]), np.array(res["T_ee_m1"])
        cam_base = np.linalg.inv(X)[:3, 3]                 # camera position in the base frame (not a registered obstacle)
        size = (self.settings.get("_marker1_size_m") or 0.03)
        h = size / 2
        corners_m = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]], float)
        m, ids = self.m, self.ids
        d = mujoco.MjData(m)
        r_ = np.radians
        lo = np.maximum(ids.arm_range[:, 0], [q0[0] - r_(90), r_(-120), r_(-150), r_(-100), r_(-100), ids.arm_range[5, 0]])
        hi = np.minimum(ids.arm_range[:, 1], [q0[0] + r_(90), r_(-3), r_(-3), r_(100), r_(100), ids.arm_range[5, 1]])
        rng = np.random.default_rng(int(time.time()))
        cands = []
        for _ in range(tries):
            q = rng.uniform(lo, hi)
            d.qpos[ids.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            p_end = d.xpos[ids.end_body]
            if not (p_end[2] >= 0.06 and 0.12 <= math.hypot(p_end[0], p_end[1]) <= 0.45):
                continue
            if np.linalg.norm(p_end - cam_base) < CAM_CLEAR_M:       # keep the hand away from the camera itself
                continue
            F = np.eye(4)
            F[:3, :3], F[:3, 3] = d.xmat[ids.end_body].reshape(3, 3), p_end
            P = X @ F @ Y                                   # marker in the camera frame
            if P[2, 3] < CAM_CLEAR_M:
                continue
            to_cam = -P[:3, 3] / np.linalg.norm(P[:3, 3])
            facing = float(np.degrees(np.arccos(np.clip(P[:3, 2] @ to_cam, -1, 1))))
            if facing > 55:
                continue
            c = (P @ corners_m.T).T[:, :3]
            uv, _ = cv2.projectPoints(c, np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(-1, 2)
            side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1)))
            if side < 28 or uv[:, 0].min() < 60 or uv[:, 1].min() < 60 or uv[:, 0].max() > W - 60 or uv[:, 1].max() > H - 60:
                continue
            k_ = planner.static_contacts(m, d, ids, q, self.gap.gap0, block_away=True)
            if k_["floor"] or [x for x in k_["self"] if x[0] > planner.SELF_PEN_TOL]:
                continue
            cands.append((q, uv.mean(0), side, facing, F[:3, :3].copy()))
        if len(cands) < n:
            return {"ok": False, "message": f"条件を満たす姿勢が {len(cands)} 個しかない（{tries} 個から）。アームをカメラの視野の中ほどに動かしてから生成し直す"}
        Q = np.array([c[0] for c in cands])
        Rs = np.array([c[4] for c in cands])
        w = np.array([1, 1, 1, 1, 1, 0.3])                  # joint6 matters less for diversity (tour distance)
        wj = np.array([0, 0.5, 0.5, 0.5, 0.5, 0])           # arm configuration part of the selection distance

        def dist(i):                                        # end_link rotation angle + configuration (rad)
            tr = np.einsum("nij,ij->n", Rs, Rs[i])          # trace(R_n^T R_i)
            ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
            return np.sqrt(ang ** 2 + np.sum(((Q - Q[i]) * wj) ** 2, axis=1))
        chosen = [int(np.argmin(np.linalg.norm((Q - q0) * w, axis=1)))]
        dmin = dist(chosen[0])
        while len(chosen) < n:
            i = int(np.argmax(dmin))
            chosen.append(i)
            dmin = np.minimum(dmin, dist(i))
        # Tour from the current pose. Every move (straight joint-space line, as executed) is checked here
        # kinematically: floor clearance >= FLOOR_CLEARANCE + 10 mm, registered obstacles, self-collision.
        # (Only the poses were checked before; the move to one dipped to 7.5 mm above the table.)
        env = [ids.floor_geom] + list(ids.obstacle_geoms)
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        arm_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids.arm_bodies and m.geom_bodyid[g] != base
                 and (m.geom_contype[g] | m.geom_conaffinity[g])]
        edge_cache = {}

        def move_ok(qa, qb, key):
            if key in edge_cache:
                return edge_cache[key]
            nstep = max(8, int(np.max(np.abs(qb - qa)) / math.radians(3)))
            ok = True
            for t in np.linspace(0, 1, nstep + 1)[1:]:
                d.qpos[ids.arm_qadr] = qa + (qb - qa) * t
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                if any(c.dist < -planner.SELF_PEN_TOL and m.geom_bodyid[c.geom1] in ids.arm_bodies
                       and m.geom_bodyid[c.geom2] in ids.arm_bodies for c in d.contact[:d.ncon]):
                    ok = False; break
                for g in arm_g:
                    if mujoco.mj_geomDistance(m, d, g, ids.floor_geom, 0.05, None) < FLOOR_CLEARANCE + 0.010:
                        ok = False; break
                    if any(mujoco.mj_geomDistance(m, d, g, og, 0.05, None) < 0.0 for og in env[1:]):
                        ok = False; break
                if not ok:
                    break
            edge_cache[key] = ok
            return ok

        tour, cur, cur_key, left = [], q0, "start", set(chosen)
        bridges, dropped = set(), []
        step_max = math.radians(CALIB_STEP_MAX_DEG)
        while left:
            near = sorted([j for j in left if np.max(np.abs(Q[j] - cur)) <= step_max],
                          key=lambda j: np.linalg.norm((Q[j] - cur) * w))
            nxt = next((j for j in near if move_ok(cur, Q[j], (cur_key, j))), None)
            if nxt is not None:
                tour.append(nxt); left.remove(nxt); cur, cur_key = Q[nxt], nxt
                continue
            # no chosen pose reachable directly: a bridging candidate that is, nearest to the closest target
            tgt = min(left, key=lambda j: np.linalg.norm((Q[j] - cur) * w))
            bridge = sorted([j for j in range(len(Q)) if j not in tour and j not in left
                             and np.max(np.abs(Q[j] - cur)) <= step_max],
                            key=lambda b: np.linalg.norm((Q[b] - Q[tgt]) * w))[:15]
            b = next((j for j in bridge if np.linalg.norm((Q[j] - Q[tgt]) * w) < np.linalg.norm((cur - Q[tgt]) * w)
                      and move_ok(cur, Q[j], (cur_key, j))), None)
            if b is None:                                   # unreachable from here: drop that target
                left.remove(tgt); dropped.append(tgt)
                continue
            tour.append(b); bridges.add(b); cur, cur_key = Q[b], b
        out = []
        for k, i in enumerate(tour):
            q, c_uv, side, facing, _ = cands[i]
            br = i in bridges
            out.append({"i": k, "role": "calib" if br else ("eval" if sum(o["role"] == "calib" and not o.get("bridge") for o in out) % 4 == 3
                                                            and sum(o["role"] == "eval" for o in out) < n_eval else "calib"),
                        "bridge": br,
                        "label": ("（中継・記録は任意）" if br else "") + f"作業範囲 {k}: 予測 画像 ({c_uv[0]:.0f}, {c_uv[1]:.0f}) px、一辺 {side:.0f} px、向き {facing:.0f}° / 関節 "
                                 + ", ".join(f"{math.degrees(v):.0f}" for v in q),
                        "q": q.tolist(), "ok": True, "reasons": []})
        with self.lock:
            self.calib_poses = {"q0": q0.tolist(), "folded_centre_lifted": False, "poses": out, "t": time.time(),
                                "mode": "workspace", "prediction_source": src}
        if not out:
            return {"ok": False, "message": "今の姿勢から安全に移動できる校正姿勢が見つからない"}
        return {"ok": True, "message": f"作業範囲の校正姿勢 {len(out)} 個を生成（候補 {len(cands)} / {tries}、"
                                       f"移動経路の確認で外した目標 {len(dropped)}、予測: {src}）。"
                                       "各姿勢へは経路シミュレーション → 実機実行（依頼者）→ サンプル記録の順。自分の腕による遮蔽は予測しない"}

    def view_model(self) -> dict | None:
        """Adopted camera calibration + intrinsics + ID 1 size, for predicting where ID 1 is seen."""
        he = self.handeye()
        vs = fetch(self.args.vision_url + "/state.json")
        intr = (vs or {}).get("intrinsics") or {}
        size = (((vs or {}).get("config") or {}).get("markers", {}).get("1", {}).get("size") or {}).get("value")
        if he is None or not intr.get("K") or not size:
            return None
        return {"X": np.array(he["T_cam_base"]), "Y": np.array(he["T_ee_m1"]), "K": np.array(intr["K"]),
                "dist": np.array(intr.get("dist_coeffs") or np.zeros(5)), "image": intr["image"], "size": float(size)}

    def visual_correct(self, cand_index=None) -> dict:
        """Measure where the grip site really is (ID 1 seen now) against where the joints say it is, and shift
        the candidate's remaining targets by that difference.

        The block was located through the same camera and calibration as ID 1 is now, so the calibration's
        position error cancels: offset o = FK(q_now) grip site - observed grip site (base frame), and every
        target t is commanded as t + o. Uses 10 frames; only ID 1 seen "ok". Nothing is sent to the arm."""
        if not self.plan:
            return {"ok": False, "message": "把持計画がない"}
        ok = [c for c in self.plan["candidates"] if c.ok]
        cand = self.plan["candidates"][cand_index] if cand_index is not None else (ok[0] if ok else None)
        if cand is None or not cand.ok or not cand.targets:
            return {"ok": False, "message": "補正できる候補がない（計画し直す）"}
        he = self.handeye()
        if he is None:
            return {"ok": False, "message": "カメラ⇔台座の校正が採用されていない"}
        if not str(self.sync_info.get("joints", "")).startswith("実機の関節角を反映中"):
            return {"ok": False, "message": "実機の関節角が読めていない: " + str(self.sync_info.get("joints"))}
        X, Y = np.array(he["T_cam_base"]), np.array(he["T_ee_m1"])
        Xi, Yi = np.linalg.inv(X), np.linalg.inv(Y)
        m, ids = self.m, self.ids
        site_local = np.r_[m.site_pos[ids.grip_site], 1.0]           # grip site in end_link coordinates
        obs, qs = [], []
        for _ in range(10):
            vs = fetch(self.args.vision_url + "/state.json")
            m1 = next((mk for mk in (vs or {}).get("markers", []) if mk.get("id") == 1), None)
            if m1 and m1.get("status") == "ok" and time.time() - vs.get("t", 0) < 0.3:
                T = Xi @ np.array(m1["T_cam_marker"]) @ Yi                # end_link as seen
                obs.append(T)
                with self.lock:
                    qs.append(self.d.qpos[ids.arm_qadr].copy())
            time.sleep(0.06)
        if len(obs) < 6:
            return {"ok": False, "message": f"ID 1 が ok で見えたのは {len(obs)} / 10 フレーム。見える姿勢で行う"}
        q_now = np.mean(qs, axis=0)
        d = mujoco.MjData(m)
        d.qpos[ids.arm_qadr] = q_now
        mujoco.mj_kinematics(m, d)
        fk_site = d.site_xpos[ids.grip_site].copy()
        R_fk = d.xmat[ids.end_body].reshape(3, 3)
        seen = np.mean([(T @ site_local)[:3] for T in obs], axis=0)
        spread = float(np.max([np.linalg.norm((T @ site_local)[:3] - seen) for T in obs]) * 1000)
        o = fk_site - seen
        R_obs = obs[len(obs) // 2][:3, :3]
        rot = math.degrees(math.acos(np.clip((np.trace(R_fk.T @ R_obs) - 1) / 2, -1, 1)))
        if np.linalg.norm(o) > 0.03:
            return {"ok": False, "message": f"関節角と観測の差が {np.linalg.norm(o) * 1000:.0f} mm と大きすぎるので補正しない（校正・観測を確認）"}
        probs = planner.retarget(m, self.settings, cand, o, q_now)
        if probs:
            return {"ok": False, "message": "補正後の目標が解けない: " + "; ".join(probs)}
        cand.correction = {"offset_mm": (o * 1000).round(1).tolist(), "rot_err_deg": round(rot, 1), "frames": len(obs),
                           "spread_mm": round(spread, 1), "t": time.time()}
        return {"ok": True, "message": f"視覚補正: 関節角による把持点と観測のずれ {np.linalg.norm(o) * 1000:.1f} mm "
                                       f"(x {o[0] * 1000:+.1f}, y {o[1] * 1000:+.1f}, z {o[2] * 1000:+.1f})、向き {rot:.1f}°、"
                                       f"{len(obs)} フレーム（ばらつき {spread:.1f} mm）。接近前・近接・把持・持ち上げの目標を直した。"
                                       "次の移動は経路シミュレーションから", "correction": cand.correction}

    def preview_calib(self) -> dict:
        cp = getattr(self, "calib_poses", None)
        if not cp:
            return {"message": "校正姿勢がない"}
        traj, q = [], np.array(cp["q0"])
        for p in [p for p in cp["poses"] if p["ok"]]:
            r = simulate_profile(self.m, self.ids, q, np.array(p["q"]), self.gap.gap0)
            traj += r["sim_q"]
            q = np.array(p["q"])
        self.preview = {"traj": traj, "dt": 0.02, "t0": time.monotonic()}
        return {"message": f"全 {sum(p['ok'] for p in cp['poses'])} 姿勢の巡回をツインで再生中（{len(traj) * 0.02:.0f} s）"}

    def real_grasp_blockers(self) -> list[str]:
        b = []
        if self.settings["placement"]["status"] != "calibrated" and self.handeye() is None:
            b.append("カメラ⇔アーム台座が未校正")
        if not self._grip_ok():
            b.append("ID 1 → 把持中心（実際の顎の中心）が未確認（markers.toml [marker.1.grip]）")
        if not self.plan or self.plan["basis"] != "observed_calibrated":
            b.append("把持計画が校正済みの実物配置に基づいていない")
        if self.settings["jaw"]["gripper_map_status"] not in ("measured", "calibrated"):
            b.append("グリッパーのモーター角と開口の対応が未確認")
        if self.settings["jaw"]["max_gap_status"] not in ("measured", "calibrated"):
            b.append("実機の最大開口が未確認")
        return b

    # ------------------------------------------------------------------ state
    def state(self) -> dict:
        with self.lock:
            m, d, ids = self.m, self.d, self.ids
            joints = []
            for i, n in enumerate(scene.ARM_JOINTS):
                lo, hi = ids.arm_range[i]
                q = float(d.qpos[ids.arm_qadr[i]])
                src = self.joint_source[i] if self.mode == "sync" else "シミュレーション"
                if q < lo - 1e-3 or q > hi + 1e-3:
                    src += f"（モデルの可動範囲外 {math.degrees(q - (lo if q < lo else hi)):+.1f}°）"
                joints.append({"name": n, "q": q, "lo": float(lo), "hi": float(hi), "source": src})
            fl = float(d.qpos[ids.fl_q])
            joints.append({"name": "finger_left（顎）", "q": fl, "lo": float(ids.fl_range[0]), "hi": float(ids.fl_range[1]),
                           "source": "slide [m]（角度欄は rad 表示の換算なし）"})
            contacts = []
            if self.mode == "trial":
                c = self._contacts()
                contacts = [f"ブロック–左顎 {c['left']} 点、ブロック–右顎 {c['right']} 点、ブロック–床 {c['floor_block']} 点、"
                            f"アーム–床 {c['arm_floor']} 点" + (f"、その他 {c['other']}" if c["other"] else "")]
            jaw = self.settings["jaw"]
            plan = None
            if self.plan:
                plan = {"basis": self.plan["basis"], "basis_label": self.plan["basis_label"], "t": self.plan["t"],
                        "candidates": [c.json() for c in self.plan["candidates"]]}
            mT = d.xmat[ids.end_body].reshape(3, 3)
            return {
                "mode": self.mode,
                "placement": {"source": self.placement_source,
                              "label": BASIS_LABEL.get(self.block_basis, "ブロックなし") if self.mode == "sync" or not self.trial
                              else self.trial["basis_label"],
                              "calibrated": self.block_basis == "observed_calibrated"},
                "sync": self.sync_info, "joints": joints,
                "block_observation": getattr(self, "block_obs", None) if self.placement_source == "observed" else None,
                "jaw": {"gap_m": self.gap(fl), "max_gap_model_m": self.gap(ids.fl_range[1]),
                        "max_gap_config": {"value": jaw["max_gap_m"], "status": jaw["max_gap_status"]},
                        "note": "物理的な把持 = 指の衝突メッシュ（front/mid/rear）、表示 = 着色メッシュ。"
                                "最大開口はモデル 100 mm・実機 100 mm（2026-10-14 実測）。顎の形状そのものの対応は未確認。"},
                "fk": {"frame": "end_link（台座座標、MuJoCo）", "p": d.xpos[ids.end_body].tolist(), "R": mT.tolist()},
                "real_fk": self.real_fk,
                "fk_vs_vision": self.fk_vs_vision,
                "plan": plan, "trial": self.trial,
                "physics": {k: v.json() for k, v in scene.physics_params(self.settings).items()},
                "contacts": contacts,
                "last_path": getattr(self, "last_path", None) and {k: v for k, v in self.last_path.items() if k not in ("sim_q", "q_plan", "times")},
                "real_grasp_blockers": self.real_grasp_blockers(),
                "obstacles": scene.obstacle_report(self.settings),
                "calib_poses": getattr(self, "calib_poses", None),
                "handeye_active": None if self.handeye() is None else {k: self.handeye().get(k) for k in ("session", "adopted_at", "holdout")},
                "notes": [
                    "公開モデル Seeed-Projects/Borot-Arm_Mujoco @ 3f3e98d（rebotarm_b601_colored.xml）を読み込み、"
                    "床・ブロック・リンクの凸包衝突・アクチュエータを追加（scene.py）。",
                    f"自己衝突の確認から除外: {', '.join('–'.join(p) for p in scene.EXCLUDED_SELF_PAIRS)}（凸包が常に重なるため）。",
                    "実機へは何も送らない。sync モードは物理演算なし（mj_forward のみ）。",
                    "シミュレーションでの把持成功は、実機で同じ把持力で成功することを意味しない。",
                ],
            }


LIMIT_TOL = 0.03              # rad
CALIB_MAX_JOINT_DEG = 60      # calibration poses must stay near the current pose (camera stand etc.)
CALIB_STEP_MAX_DEG = 100      # workspace calibration poses: largest joint change of one move (each move is simulated)
CAM_CLEAR_M = 0.25            # workspace calibration poses: end_link and ID 1 at least this far from the camera
VMAX = 0.25                  # rad/s — the same profile is executed by rebot/arm_state.py
VMAX_FINAL = 0.10            # rad/s for the last approach (near -> grasp): shallower stop if the jaws hit something
FLOOR_CLEARANCE = 0.015      # m, every arm / jaw geom along the simulated motion
SIM_TRACK_MAX = 0.05         # rad, the simulated arm must follow the profile this closely


def profile(q_from, q_to, dt=0.02, vmax=None):
    """Straight joint-space path, all joints arriving together, smooth (cosine) start and stop.

    The cosine profile peaks at pi/2 times the mean speed, so T is chosen to keep that peak <= VMAX.
    """
    T = max(math.pi / 2 * float(np.max(np.abs(q_to - q_from))) / (vmax or VMAX), 0.5)
    n = int(np.ceil(T / dt)) + 1
    t = np.linspace(0, T, n)
    s = 0.5 - 0.5 * np.cos(np.pi * t / T)
    return t, q_from + np.outer(s, q_to - q_from)


def simulate_profile(m, ids, q_from, q_to, gap0, vmax=None) -> dict:
    # Obstacles are judged by distance below; in this dynamics run they must not push the arm around
    # (a path that starts inside an inflated box would otherwise be blocked by it).
    saved = [(g, m.geom_contype[g], m.geom_conaffinity[g]) for g in ids.obstacle_geoms]
    for g, _, _ in saved:
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
    try:
        return _simulate_profile(m, ids, q_from, q_to, gap0, vmax)
    finally:
        for g, ct, ca in saved:
            m.geom_contype[g], m.geom_conaffinity[g] = ct, ca


def _simulate_profile(m, ids, q_from, q_to, gap0, vmax=None) -> dict:
    lo, hi = ids.arm_range[:, 0], ids.arm_range[:, 1]
    d = mujoco.MjData(m)
    d.qpos[ids.arm_qadr] = q_from
    d.qpos[ids.block_q:ids.block_q + 7] = [10, 10, 1, 1, 0, 0, 0]
    d.ctrl[ids.act_arm] = q_from
    mujoco.mj_forward(m, d)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")   # stands on the floor by design
    arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids.arm_bodies and m.geom_bodyid[g] != base
                 and m.geom_contype[g] | m.geom_conaffinity[g]]
    reasons, sim_q, track, min_clear, worst = [], [], 0.0, math.inf, None
    min_obst, worst_obst = math.inf, None
    start_obst = None             # clearance at the start: a path that begins inside the (inflated) box may leave it
    # The real arm rests slightly past some model limits (joint2 reads +0.5 deg at the folded rest; the
    # hand-eye fit found a ~2 deg zero offset there). Within LIMIT_TOL the target is clamped to the
    # limit; beyond it the target is refused.
    if np.any(q_to < lo - LIMIT_TOL) or np.any(q_to > hi + LIMIT_TOL):
        reasons.append("目標が関節制限の外")
    q_to = np.clip(q_to, lo, hi)
    t, qp = profile(q_from, q_to, vmax=vmax)
    steps = int(round((t[1] - t[0]) / m.opt.timestep)) if len(t) > 1 else 1
    for k in range(len(t) + 25):                       # +0.5 s settle
        d.ctrl[ids.act_arm] = qp[min(k, len(t) - 1)]
        for _ in range(steps):
            mujoco.mj_step(m, d)
        q = d.qpos[ids.arm_qadr].copy()
        sim_q.append(q.tolist())
        track = max(track, float(np.max(np.abs(q - qp[min(k, len(t) - 1)]))))
        for g in arm_geoms:
            dist = mujoco.mj_geomDistance(m, d, g, ids.floor_geom, 0.05, None)
            if dist < min_clear:
                min_clear, worst = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            for og in ids.obstacle_geoms:            # obstacles are already inflated by margin (+ uncertainty)
                dist = mujoco.mj_geomDistance(m, d, g, og, 0.05, None)
                if k == 0:
                    start_obst = dist if start_obst is None else min(start_obst, dist)
                if dist < min_obst:
                    min_obst, worst_obst = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, og)
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in ids.arm_bodies and b2 in ids.arm_bodies and -c.dist > planner.SELF_PEN_TOL:
                reasons.append(f"自己衝突 {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1)}–"
                               f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)}（t={k * 0.02:.1f} s）")
                break
        if reasons:
            break
    if min_clear < FLOOR_CLEARANCE:
        reasons.append(f"床との距離 {min_clear * 1000:.1f} mm < {FLOOR_CLEARANCE * 1000:.0f} mm（{worst}）")
    allowed = min(0.0, (start_obst if start_obst is not None else 0.0)) - 0.001
    if min_obst < allowed:
        reasons.append(f"障害物 {worst_obst}（余裕込みの箱）に入る（{min_obst * 1000:.1f} mm"
                       + (f"、始点 {start_obst * 1000:.1f} mm より深い" if start_obst is not None and start_obst < 0 else "") + "）")
    if track > SIM_TRACK_MAX:
        reasons.append(f"シミュレーション上の追従誤差 {track:.3f} rad > {SIM_TRACK_MAX}")
    return {"ok": not reasons, "reasons": reasons, "times": t.tolist(), "q_plan": qp.tolist(),
            "duration_s": float(t[-1]), "max_step_rad": float(np.max(np.abs(q_to - q_from))),
            "min_floor_clearance_m": float(min_clear), "sim_track_max_rad": track,
            "min_obstacle_clearance_m": None if not math.isfinite(min_obst) else float(min_obst),
            "obstacles_checked": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ids.obstacle_geoms],
            "sim_q": sim_q, "sim_dt": 0.02, "vmax": VMAX}


def planner_rot(axis: str, a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]), "y": np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]),
            "z": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])}[axis]


def d_time(sim):
    with sim.lock:
        return sim.d.time


CAMS = {"overview": dict(lookat=[0.2, 0.0, 0.1], distance=1.0, azimuth=135, elevation=-30),
        "side": dict(lookat=[0.2, 0.0, 0.12], distance=0.8, azimuth=90, elevation=-10),
        "gripper": None}


# Minimal stand-alone viewer (the full UI is the vision page, which proxies only the main sim on 8092).
VIEWER_HTML = """<!doctype html><meta charset="utf-8"><title>sim viewer</title>
<style>body{font:14px sans-serif;margin:8px;background:#222;color:#ddd}img{width:960px;cursor:grab;display:block;margin-top:6px}
button{margin-right:4px}</style>
<div>カメラ: <button onclick="cam('overview')">overview</button><button onclick="cam('side')">side</button>
<button onclick="cam('gripper')">gripper</button> 表示: <button onclick="geom('visual')">visual</button>
<button onclick="geom('collision')">collision</button><button onclick="geom('both')">both</button>
<span id="st"></span></div>
<img id="v" src="stream.mjpg?cam=gripper&geom=visual" draggable="false">
<div>ドラッグ = 回転、ホイール = 距離。シミュレーター内の表示だけで、アームには何も送らない。</div>
<script>
let c='gripper', g='visual', az=150, el=-35, dist=0.35, drag=null;
const v=document.getElementById('v');
function reload(){ v.src='stream.mjpg?cam='+c+'&geom='+g+'&t='+Date.now(); }
function cam(x){ c=x; reload(); }
function geom(x){ g=x; reload(); }
function send(){ fetch('api/view',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({azimuth:az,elevation:el,distance:dist})}); }
v.onmousedown=e=>{drag=[e.clientX,e.clientY]};
window.onmouseup=()=>{drag=null};
window.onmousemove=e=>{ if(!drag) return; az-=(e.clientX-drag[0])*0.4; el=Math.max(-89,Math.min(89,el-(e.clientY-drag[1])*0.4));
  drag=[e.clientX,e.clientY]; send(); };
v.onwheel=e=>{ e.preventDefault(); dist=Math.max(0.1,Math.min(3,dist*(e.deltaY>0?1.1:0.9))); send(); };
setInterval(async()=>{ try{ const s=await (await fetch('api/state')).json();
  document.getElementById('st').textContent=' | '+s.sync.joints; }catch(e){} }, 2000);
</script>"""


class RenderLoop:
    def __init__(self, sim: Sim, w=640, h=480, fps=15) -> None:
        self.sim, self.w, self.h, self.fps = sim, w, h, fps
        self.jpeg = None
        self.cond = threading.Condition()
        self.seq = 0
        self.stop = threading.Event()

    def run(self) -> None:
        r = mujoco.Renderer(self.sim.m, self.h, self.w)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        try:
            while not self.stop.is_set():
                t0 = time.monotonic()
                v = self.sim.view
                geom = v["geom"]
                opt.geomgroup[:] = 0
                opt.geomgroup[0] = 1
                opt.geomgroup[1] = 1
                if geom in ("visual", "both"):
                    opt.geomgroup[2] = 1
                if geom in ("collision", "both"):
                    opt.geomgroup[3] = 1
                    opt.geomgroup[4] = 1
                opt.geomgroup[5] = 1                       # obstacles always visible
                opt.sitegroup[:] = 0
                opt.sitegroup[1] = 1
                opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.sim.mode == "trial"
                c = CAMS.get(v["cam"])
                pv = getattr(self.sim, "preview", None)
                data = self.sim.d
                if pv is not None:
                    k = int((time.monotonic() - pv["t0"]) / pv["dt"])
                    if k < len(pv["traj"]) + 25:
                        if not hasattr(self, "pd"):
                            self.pd = mujoco.MjData(self.sim.m)
                        with self.sim.lock:
                            self.pd.qpos[:] = self.sim.d.qpos
                        self.pd.qpos[self.sim.ids.arm_qadr] = pv["traj"][min(k, len(pv["traj"]) - 1)]
                        mujoco.mj_forward(self.sim.m, self.pd)
                        data = self.pd
                    else:
                        self.sim.preview = None
                with self.sim.lock:
                    if c is None:
                        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        cam.lookat[:] = self.sim.d.site_xpos[self.sim.ids.grip_site]
                        cam.distance = v.get("distance", 0.35)
                        cam.azimuth = v.get("azimuth", 150)
                        cam.elevation = v.get("elevation", -35)
                    else:
                        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        cam.lookat[:] = c["lookat"]
                        cam.distance = v.get("distance", c["distance"])
                        cam.azimuth = v.get("azimuth", c["azimuth"])
                        cam.elevation = v.get("elevation", c["elevation"])
                    r.update_scene(data, cam, opt)
                img = r.render()
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                label = ("PATH PREVIEW (simulated)" if data is not self.sim.d else
                         "SYNC (no physics)" if self.sim.mode == "sync" else "TRIAL (simulation only)") + \
                        ("" if self.sim.block_basis == "observed_calibrated" else " | placement: NOT calibrated")
                cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.cond:
                    self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                    self.cond.notify_all()
                time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))
        finally:
            r.close()

    def wait(self, seq):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq or self.stop.is_set(), 5)
            return self.jpeg, self.seq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--arm-url", default="http://127.0.0.1:8091")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8090")
    ap.add_argument("--rate", type=float, default=20.0, help="sync rate (Hz)")
    args = ap.parse_args()

    sim = Sim(args)
    rl = RenderLoop(sim)
    stop = threading.Event()

    def sync_loop():
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                sim.sync_tick()
            except Exception as e:
                sim.sync_info = {"joints": f"同期エラー: {e}", "block": ""}
            stop.wait(max(0.0, 1.0 / args.rate - (time.monotonic() - t0)))

    threading.Thread(target=sync_loop, daemon=True).start()
    threading.Thread(target=rl.run, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/", "/view"):
                body = VIEWER_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                return self._json(sim.state())
            if path == "/stream.mjpg":
                params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                if params.get("cam") in CAMS and params["cam"] != sim.view["cam"]:
                    sim.view = {"cam": params["cam"], "geom": sim.view["geom"]}   # drop dragged angles
                if params.get("geom") in ("visual", "collision", "both"):
                    sim.view["geom"] = params["geom"]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while not stop.is_set():
                        jpeg, seq = rl.wait(seq)
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if path.startswith("/api/path/"):
                rec = getattr(sim, "paths", {}).get(path.rsplit("/", 1)[-1])
                return self._json(rec or {"ok": False, "message": "未確認または期限切れの経路 id"})
            if path == "/api/real_grasp_check":
                return self._json({"blockers": sim.real_grasp_blockers()})
            if path == "/snapshot.jpg":
                jpeg, _ = rl.wait(-1)
                body = jpeg or b""
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            path = self.path.split("?")[0]
            if path == "/api/mode":
                return self._json({"message": sim.set_mode(body.get("mode", "sync"))})
            if path == "/api/placement":
                src = body.get("source")
                if src not in ("observed", "mock"):
                    return self._json({"message": "source は observed / mock"}, 400)
                if sim.mode != "sync":
                    return self._json({"message": "試行中は配置を変えない（リセットしてから）"})
                sim.placement_source = src
                sim.plan = None
                return self._json({"message": f"ブロックの配置: {src}"})
            if path == "/api/plan":
                return self._json({"message": sim.do_plan()})
            if path == "/api/trial":
                return self._json({"message": sim.start_trial(body.get("candidate"))})
            if path == "/api/reset":
                return self._json({"message": sim.reset()})
            if path == "/api/visual_correct":
                return self._json(sim.visual_correct(body.get("candidate")))
            if path == "/api/check_path":
                return self._json(sim.check_path(body.get("target", "pre"), body.get("candidate")))
            if path == "/api/preview":
                lp = getattr(sim, "last_path", None)
                if not lp:
                    return self._json({"message": "シミュレートした経路がない"})
                sim.preview = {"traj": lp["sim_q"], "dt": lp["sim_dt"], "t0": time.monotonic()}
                return self._json({"message": "経路のシミュレーションを再生中"})
            if path == "/api/calib_poses":
                if body.get("mode") == "workspace":
                    return self._json(sim.make_calib_poses_workspace())
                return self._json(sim.make_calib_poses())
            if path == "/api/preview_calib":
                return self._json(sim.preview_calib())
            if path == "/api/view":
                for k in ("azimuth", "elevation", "distance"):
                    if k in body:
                        sim.view[k] = float(body[k])
                return self._json({"ok": True})
            if path == "/api/reload":
                if sim.mode != "sync":
                    return self._json({"message": "リセットしてから"})
                with sim.lock:
                    sim.settings = scene.load_settings()
                return self._json({"message": "scene.toml を読み直した（モデル形状の変更は再起動が必要）"})
            self.send_error(404)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"sim_server on http://{args.host}:{args.port}/ (MuJoCo {mujoco.__version__})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sim.stop_trial.set()
        rl.stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
