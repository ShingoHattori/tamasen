#!/usr/bin/env python3
"""Joint state of the reBot Arm B601-DM (read-only by default) and user-operated motion from the web UI.

    python arm_state.py --port /dev/ttyACM0                    # read-only (what run.sh starts)
    python arm_state.py --port /dev/ttyACM0 --allow-motion     # the USER starts this, never an agent
    python arm_state.py --fake --allow-motion                  # no hardware: simulated motors, for tests

Read-only path — verified in motorbridge v0.5.6 source (tag c652ce4):
  opening the bridge / add_damiao_motor(): no frames; request_feedback(): the Damiao refresh frame
  (0x7FF, [id_lo, id_hi, 0xCC, 0...]); poll_feedback_once() / get_state(): receive only.
  Never used: Controller.shutdown() or `with Controller(...)` (= DISABLE every motor). We release
  with close_bus() + close().

Motion (only with --allow-motion, only from the web page, only by the user):
  * engage   — needs the typed confirmation "MOVE". Writes the SDK's POS_VEL gains (registers
               25..28 from rebotarm_dm.yaml), switches the six ARM motors to POS_VEL while still
               disabled, enables those six motors individually (not enable_all, which would also
               energise the gripper) and immediately commands the current position: the arm holds.
  * execute  — runs a path that sim_server.py has SIMULATED (dynamics, floor clearance, self-
               collision, limits) and accepted; this process fetches the exact simulated profile
               (times, joint targets) by id — the browser cannot send raw joint targets. While engaged
               (moving AND holding) an outer feedback loop corrects the command from the measured
               joints (gravity sag), and leaving the validated corridor stops the motion. Also stops
               on: missing heartbeat from the page (dead-man), stop button, start pose mismatch.
  * stop     — hold the current position.
  * release  — needs "RELEASE": disables the motors (torque off; the arm falls unless supported).
  * grasp    — runs pre-grasp / grasp / close / lift on the real arm. Refused unless the camera-base
               placement is calibrated, the plan is based on a calibrated observed placement, and the
               gripper motor-angle mapping is measured. (Not yet true, so it is refused today.)
Every command is logged to run/arm_commands.log with its source address.
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SDK = HERE / "reBotArm_control_py"
sys.path.insert(0, str(SDK))

DAMIAO_STATUS = {0x0: "無効（トルクなし）", 0x1: "有効", 0x8: "過電圧", 0x9: "低電圧", 0xA: "過電流",
                 0xB: "MOS 過熱", 0xC: "ロータ過熱", 0xD: "通信断", 0xE: "過負荷"}
STALE_S = 0.5
VMAX = 0.25                 # rad/s, joint speed limit (the simulator builds the profile with the same value)
# Outer feedback loop on (plan - measured), per joint. Wrist joints carry little gravity load, and
# correcting them made joint6 hunt on the real arm (2026-10-13): lower gains on 4/5, none on 6.
KP = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.0])
KI = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.0])
DEADBAND = 0.003            # rad: inside it the correction is frozen (no P, no integration)
I_MAX, CORR_MAX = 0.15, 0.20   # rad, limits of the integral and of the total correction
CORRIDOR_RAD = 0.08         # rad, per-joint deviation from the simulated plan that stops the motion
CORRIDOR_M = 0.020          # m, end_link deviation (FK) that stops the motion
LAG_FREE, LAG_STOP = 0.01, 0.04   # rad: plan runs at full speed below LAG_FREE lag, stops at LAG_STOP
STALL_S = 2.0               # s the plan may stand still before it is a fault
LAG_KI_GAIN = 3.0           # integral gain multiplier while the plan clock is fully held back by lag
CAN_LOSS_S = 0.3            # s without fresh feedback while engaged = CAN / bus fault
ENGAGE_GRACE_S = 0.5
# Motor-internal POS_VEL gains that differ from the SDK yaml (written at engage). joint6 (light wrist
# roll) hunted ~0.0008 rad with torque cycling 0 -> 0.1 Nm under a constant command (2026-10-13): the
# motor's own position integral; drop it there.
PV_OVERRIDE = {"joint6": {"pos_ki": 0.0}}
# joint6 still hunted ~0.02 rad in POS_VEL with pos_ki = 0 (velocity loop vs wrist friction). It is held
# in MIT mode instead: torque = kp (q_cmd - q) - kd q_dot, no integrator. kp / kd from the SDK yaml.
# joint4 (wrist pitch, carries ~2 Nm of gravity) stick-slipped in POS_VEL: 11 mrad jumps every ~3.3 s while
# holding (2026-10-14, the motor's own position integral against friction). It is held in MIT mode too,
# with the gravity torque g(q) of the SDK's URDF as feedforward (the official SDK's gravity-compensation
# method, examples 9/10). Checked on the real arm at one pose: model -1.88 Nm vs motor -1.69 Nm (joint4),
# -6.69 vs -7.18 (joint3). The outer feedback loop removes the remaining static error.
# joint5 then showed the same stick-slip at a pose where its gravity torque crosses zero (11.8 mrad jumps every
# ~3 s at calibration pose 17, torque -0.20..+0.16 N m). Rather than finding the next joint/pose one at a
# time, every arm joint is held this way (requester, 2026-10-14). kp / kd per joint from the SDK yaml
# (joint1-3: 120 / 8, joint4-6: 18 / 2).
MIT_JOINTS = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}
# N m, clamp of the gravity feedforward: |g| over 20000 random poses inside the limits is at most
# 0 / 15.5 / 7.2 / 1.9 / 0.7 / 0 (joint1 and joint6 axes carry no gravity torque).
TAU_FF_MAX = {"joint1": 1.0, "joint2": 18.0, "joint3": 9.0, "joint4": 3.0, "joint5": 1.5, "joint6": 0.5}
# s, the gravity feedforward fades in after engage. At the folded rest pose gravity pushes joint2/3 onto their
# 0 rad stops, so the arm rests there until the term is in; an unsupported joint3 would sag at most
# g/kp = 7 / 120 = 0.06 rad during the ramp (kept short for that reason).
GRAV_RAMP_S = 0.5
TUNING_PATH = Path(__file__).resolve().parent / "mit_tuning.yaml"   # friction and gain overrides, read at engage
TRACE_AFTER_S = 2.0         # s of holding still traced after a move ends
# Gripper (motor "gripper", MIT): position mode for open / release, torque-limited closing for grip.
# Opening is proportional to the motor angle (2026-10-14, ruler: 0 / 50 / 100 mm at -0.022 / -2.929 / -5.896 rad,
# ~1 mm backlash); same values as sim/scene.toml [jaw]. Overridable in mit_tuning.yaml "gripper:".
GRIPPER_DEFAULTS = {"full_open_rad": -5.896, "max_gap_m": 0.100, "grip_torque_nm": 0.5, "speed_rad_s": 2.0,
                    "kp": 8.0, "kd": 1.0, "grip_kd": 0.3, "miss_margin_m": 0.004}
GRIP_SETTLE_S, GRIP_STILL_RAD_S, GRIP_TIMEOUT_S = 0.3, 0.05, 6.0
# Joint zero offsets (software only, never written to the motors): model angle = motor angle + offset.
# Everything this service publishes and plans with is in model angles; commands are converted back.
OFFSETS_PATH = Path(__file__).resolve().parent / "joint_offsets.yaml"
OFFSET_MAX = math.radians(10.0)
FAULT_POLICY = {
    "ページ切断（生存確認なし）": "移動中なら停止し、その場で保持（トルクは維持）。保持中はそのまま保持。",
    "サービス終了要求（SIGTERM）": "有効中は 1 回目を拒否し記録。5 s 以内の 2 回目で終了 — その場合モーターは最後の指令を保持したまま有効で残る"
                                 "（CAN タイムアウトが 0 以外なら、その時間後にモーター側で無効化＝アームが落ちる）。先にページで「トルクを切る」こと。",
    "CAN 断（フィードバックなし）": f"{CAN_LOSS_S} s で異常。移動を中止し、計画上の現在点を保持目標に固定して送り続ける（復帰しても経路は再開しない）。",
    "モーター異常コード / 予期しない無効化": "異常。移動を中止し、他の関節は計画上の現在点で保持。無効化された関節は支えがない — 手で支えてトルクを切る。",
    "計画経路からの逸脱": "異常。計画上の現在点で保持（実測位置へ目標を移さない）。新しい移動は異常解消まで拒否。",
    "トルク解除": "依頼者の操作（RELEASE 入力）でのみ。サービスが自動でトルクを切ることはない。",
}
START_TOL = 0.05            # rad, path start must match the measured pose
HEARTBEAT_S = 0.6           # page must post a heartbeat this often while engaged
PATH_MAX_AGE_S = 120
CTRL_HZ = 50
ARM = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def load_joints(hw_yaml: Path) -> list[dict]:
    cfg = yaml.safe_load(hw_yaml.read_text())
    out = []
    for j in cfg["joints"]:
        if j.get("vendor", "damiao") != "damiao":
            raise SystemExit(f"{hw_yaml}: only damiao joints are supported here ({j['name']})")
        pv = j.get("POS_VEL", {})
        out.append({"name": j["name"], "motor_id": int(j["motor_id"]), "feedback_id": int(j["feedback_id"]),
                    "model": str(j["model"]), "pv": {k: float(pv[k]) for k in ("vel_kp", "vel_ki", "pos_kp", "pos_ki") if k in pv},
                    "mit": j.get("MIT", {})})
    return out


# --------------------------------------------------------------------------- buses

class Bus:
    """motorbridge, Damiao over the serial bridge. Motion methods refuse unless allow_motion."""

    def __init__(self, port: str, joints: list[dict], allow_motion: bool) -> None:
        from motorbridge import Controller, Mode
        self.Mode = Mode
        self.allow_motion = allow_motion
        self.joints = joints
        self._ctrl = Controller.from_dm_serial(port, 921600)
        self._m = {j["name"]: self._ctrl.add_damiao_motor(j["motor_id"], j["feedback_id"], j["model"]) for j in joints}

    def poll(self, settle_s: float = 0.006) -> dict:
        for m in self._m.values():
            m.request_feedback()                 # 0xCC refresh frame only
        t_end = time.monotonic() + settle_s
        while time.monotonic() < t_end:
            self._ctrl.poll_feedback_once()      # receive only
            time.sleep(0.001)
        return {n: m.get_state() for n, m in self._m.items()}

    def read_params(self) -> dict:
        """Register READS only (0x7FF [id, 0x33, rid]): CAN timeout (rid 9) and control mode (rid 10)."""
        out = {}
        for n, m in self._m.items():
            row = {}
            for rid, key in ((9, "can_timeout"), (10, "ctrl_mode")):
                try:
                    row[key] = int(m.get_register_u32(rid, 300))
                except Exception as e:
                    row[key] = f"ERR {e}"
            out[n] = row
        return out

    def _check(self):
        if not self.allow_motion:
            raise PermissionError("read-only process")

    def prepare_pos_vel(self, names) -> None:
        self._check()
        by = {j["name"]: j for j in self.joints}
        for n in names:
            m, pv = self._m[n], dict(by[n]["pv"], **PV_OVERRIDE.get(n, {}))
            for reg, key in ((25, "vel_kp"), (26, "vel_ki"), (27, "pos_kp"), (28, "pos_ki")):
                if key in pv:
                    m.write_register_f32(reg, pv[key])
            m.ensure_mode(self.Mode.POS_VEL, 1000)
            time.sleep(0.02)

    def prepare_mit(self, name) -> None:
        self._check()
        self._m[name].ensure_mode(self.Mode.MIT, 1000)

    def enable(self, names) -> None:
        self._check()
        for n in names:
            self._m[n].enable()

    def disable(self, names) -> None:
        self._check()
        for n in names:
            self._m[n].disable()

    def send_pos_vel(self, q: dict, vlim: float) -> None:
        self._check()
        for n, v in q.items():
            self._m[n].send_pos_vel(float(v), float(vlim))

    def send_mit(self, name, pos, kp, kd, vel=0.0, tau=0.0) -> None:
        self._check()
        self._m[name].send_mit(float(pos), float(vel), float(kp), float(kd), float(tau))

    def close(self) -> None:
        try:
            self._ctrl.close_bus()               # no frames; never shutdown()
        finally:
            self._ctrl.close()


class FakeState:
    def __init__(self, pos):
        self.pos, self.vel, self.torq, self.status_code, self.t_mos, self.t_rotor = pos, 0.0, 0.0, 0, 30.0, 30.0


class FakeBus:
    """No hardware. Motors follow POS_VEL targets at the commanded speed limit."""

    def __init__(self, joints, allow_motion, q0=None, sag=0.0, speed=1.0):
        self.allow_motion = allow_motion
        self.sag = {"joint2": sag, "joint3": -sag}     # gravity sag: a POS_VEL motor settles off target
        self.joints = joints
        self.q = {j["name"]: 0.0 for j in joints}
        if q0:
            self.q.update(q0)
        self.target, self.vlim, self.enabled, self.t = dict(self.q), 0.0, set(), time.monotonic()
        self.log = []
        self.inject = {"no_feedback": False, "status": {}}     # test hooks (fake only)
        self.speed = speed                                     # <1: motors slower than commanded (lag)

    def read_params(self):
        return {n: {"can_timeout": 0, "ctrl_mode": 2, "fake": True} for n in self.q}

    def _advance(self):
        dt, self.t = time.monotonic() - self.t, time.monotonic()
        for n in self.enabled:
            e = self.target[n] + self.sag.get(n, 0.0) - self.q[n]
            self.q[n] += math.copysign(min(abs(e), self.vlim * self.speed * dt), e)

    def poll(self, settle_s=0.0):
        self._advance()
        if self.inject["no_feedback"]:
            return {}
        out = {}
        for n, v in self.q.items():
            s = FakeState(v + np.random.normal(0, 1e-5))
            s.status_code = self.inject["status"].get(n, 1 if n in self.enabled else 0)
            out[n] = s
        return out

    def _check(self):
        if not self.allow_motion:
            raise PermissionError("read-only process")

    def prepare_pos_vel(self, names):
        self._check(); self.log.append(("pos_vel", list(names)))

    def prepare_mit(self, name):
        self._check(); self.log.append(("mit", name))

    def enable(self, names):
        self._check(); self._advance(); self.enabled |= set(names); self.log.append(("enable", list(names)))

    def disable(self, names):
        self._check(); self._advance(); self.enabled -= set(names); self.log.append(("disable", list(names)))

    def send_pos_vel(self, q, vlim):
        self._check(); self._advance(); self.target.update(q); self.vlim = vlim

    def send_mit(self, name, pos, kp, kd, vel=0.0, tau=0.0):
        self._check(); self._advance(); self.target[name] = pos; self.vlim = 1.0
        self.log.append(("mit_cmd", name, round(float(pos), 4), round(float(vel), 4), round(float(tau), 3))) \
            if len(self.log) < 5000 else None

    def close(self):
        pass


# --------------------------------------------------------------------------- service

class FK:
    """end_link pose in the arm-base frame from the SDK's Pinocchio model (same as its examples)."""

    def __init__(self) -> None:
        self.error = None
        try:
            import pinocchio as pin
            from reBotArm_control_py.kinematics import load_robot_model, get_end_effector_frame_id
            self.pin, self.model = pin, load_robot_model()
            self.data = self.model.createData()
            self.fid = get_end_effector_frame_id(self.model)
            self.names = [self.model.names[i] for i in range(1, self.model.njoints)]
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def __call__(self, q_by_name: dict) -> dict | None:
        if self.error:
            return None
        q = np.zeros(self.model.nq)
        for i, n in enumerate(self.names):
            if q_by_name.get(n) is None:
                return None
            q[self.model.joints[i + 1].idx_q] = q_by_name[n]
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[self.fid]
        return {"frame": "end_link", "p": M.translation.tolist(), "R": M.rotation.tolist()}


class Gravity:
    """g(q) of the SDK's URDF via the SDK's own dynamics module (official gravity-compensation examples)."""

    def __init__(self) -> None:
        self.error = None
        try:
            from reBotArm_control_py.dynamics import load_dynamics_model, compute_generalized_gravity
            self.model = load_dynamics_model()
            self.data = self.model.createData()
            self._g = compute_generalized_gravity
            self.names = [self.model.names[i] for i in range(1, self.model.njoints)]
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def __call__(self, q_by_name: dict) -> dict | None:
        if self.error:
            return None
        q = np.zeros(self.model.nq)
        for i, n in enumerate(self.names):
            if n in q_by_name:
                q[self.model.joints[i + 1].idx_q] = q_by_name[n]
        g = np.asarray(self._g(self.model, q, self.data)).ravel()
        return {n: float(g[self.model.joints[i + 1].idx_v]) for i, n in enumerate(self.names)}


class ArmService:
    def __init__(self, args) -> None:
        self.args = args
        self.joints = load_joints(args.hw_yaml)
        self.fk = FK()
        self.gravity = Gravity()
        self.ff = {}                 # last MIT feedforward sent (shown on the page)
        self.tuning, self.tuning_error = self.load_tuning()
        self.offsets, self.offsets_error = self.load_offsets()
        self.last_rows = []
        self.trace = None            # open trace file while a move (and TRACE_AFTER_S after it) is recorded
        self.lock = threading.Lock()
        self.cmds: list[tuple] = []
        self.stop_evt = threading.Event()
        self.json = b"{}"
        self.last_vals, self.last_change = {}, {}
        self.control = {"state": "read_only", "message": "", "engaged_at": None, "q_engage": None,
                        "motion": None, "last_heartbeat": 0.0}
        self.log_path = HERE.parent / "run" / ("arm_commands_fake.log" if args.fake else "arm_commands.log")
        self.log_path.parent.mkdir(exist_ok=True)
        self.motor_params = {}
        self.bus = None
        self.exit_requests = []
        self.statuses = {}
        self.lost_since = None
        self.notes = [
            "読み取り専用の間に送信するのは Damiao のフィードバック要求（0x7FF, 0xCC）だけ。",
            "経過 s = 値が最後に変化してからの時間（motorbridge のキャッシュに受信時刻がないため、変化で鮮度を推定）。",
            "関節 0 rad とモデルの 0 rad の一致は、Pinocchio と MuJoCo の FK 一致までは確認済み。実機の零点位置そのものは未検証。",
            "グリッパーのモーター角と顎の開口の対応は未確認。",
        ]

    @staticmethod
    def load_offsets() -> tuple[dict, str | None]:
        """rebot/joint_offsets.yaml -> {joint: rad} (model = motor + offset). Missing file = all 0."""
        off = {n: 0.0 for n in ARM}
        try:
            raw = yaml.safe_load(OFFSETS_PATH.read_text()) or {}
        except FileNotFoundError:
            return off, None
        except Exception as e:
            return off, f"{OFFSETS_PATH.name} を読めない: {e}"
        try:
            for n, v in (raw.get("offsets_deg") or {}).items():
                r = math.radians(float(v))
                if n not in ARM or abs(r) > OFFSET_MAX:
                    return {k: 0.0 for k in ARM}, f"offsets_deg.{n} = {v} は不正（関節名、|値| <= 10 deg）"
                off[n] = r
        except (TypeError, ValueError) as e:
            return {k: 0.0 for k in ARM}, f"{OFFSETS_PATH.name} の値が不正: {e}"
        return off, None

    @staticmethod
    def load_tuning() -> tuple[dict, str | None]:
        """rebot/mit_tuning.yaml -> {"friction": {joint: N m}, "friction_v": rad/s, "kp": {...}, "kd": {...}}."""
        t = {"friction": {n: 0.0 for n in ARM}, "friction_v": 0.02, "kp": {}, "kd": {}, "file": str(TUNING_PATH),
             "gripper": dict(GRIPPER_DEFAULTS)}
        try:
            raw = yaml.safe_load(TUNING_PATH.read_text()) or {}
        except FileNotFoundError:
            return t, None
        except Exception as e:
            return t, f"{TUNING_PATH.name} を読めない: {e}"
        try:
            for n, v in raw.get("friction_nm", {}).items():
                if n not in ARM or not 0.0 <= float(v) <= 5.0:
                    return t, f"friction_nm.{n} = {v} は不正（関節名、0〜5 N m）"
                t["friction"][n] = float(v)
            opt = raw.get("options", {})
            t["friction_v"] = float(opt.get("friction_v", 0.02))
            if not 0.001 <= t["friction_v"] <= 0.5:
                return t, "options.friction_v は 0.001〜0.5 rad/s"
            for k, v in opt.items():
                for g, lim in (("kp", 500.0), ("kd", 50.0)):
                    if k.startswith(g + "_"):
                        n = k[len(g) + 1:]
                        if n not in ARM or not 0.0 < float(v) <= lim:
                            return t, f"options.{k} = {v} は不正"
                        t[g][n] = float(v)
            for k, v in (raw.get("gripper") or {}).items():
                if k not in GRIPPER_DEFAULTS:
                    return t, f"gripper.{k} は不明な項目"
                t["gripper"][k] = float(v)
            g = t["gripper"]
            if not (g["full_open_rad"] < -1.0 and 0.03 <= g["max_gap_m"] <= 0.2 and 0.0 < g["grip_torque_nm"] <= 1.0
                    and 0.1 <= g["speed_rad_s"] <= 6.0):
                return t, "gripper の値が範囲外（full_open_rad < -1、max_gap_m 0.03〜0.2、grip_torque_nm 0〜1、speed_rad_s 0.1〜6）"
        except (TypeError, ValueError) as e:
            return t, f"{TUNING_PATH.name} の値が不正: {e}"
        return t, None

    def _trace(self, c, qp, qm, q_cmd) -> None:
        """One line per control cycle while moving and TRACE_AFTER_S after: run/arm_trace_last.jsonl."""
        if self.trace is None:
            return
        if c["state"] != "moving" and time.monotonic() - self.trace["end"] > TRACE_AFTER_S:
            self.trace["f"].close()
            self.trace = None
            return
        if c["state"] == "moving":
            self.trace["end"] = time.monotonic()
        by = {r["name"]: r for r in self.last_rows}
        rec = {"t": round(time.monotonic() - self.trace["t0"], 3), "state": c["state"],
               "plan": [round(float(v), 5) for v in qp], "q": [round(float(v), 5) for v in qm],
               "vel": [by.get(n, {}).get("vel") for n in ARM], "torq": [by.get(n, {}).get("torq") for n in ARM],
               "cmd": [round(float(v), 5) for v in q_cmd], "ff": self.ff, "rate": (c.get("motion") or {}).get("rate")}
        self.trace["f"].write(json.dumps(rec) + "\n")

    def audit(self, what: str, src: str) -> None:
        with self.log_path.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {src} {what}\n")

    def submit(self, cmd: str, src: str, **kw) -> dict:
        if not self.args.allow_motion:
            return {"ok": False, "message": "このプロセスは読み取り専用で起動されている（--allow-motion なし）。実機操作は受け付けない。"}
        if cmd != "heartbeat":
            self.audit(f"{cmd} {json.dumps({k: v for k, v in kw.items() if k != 'path'} | ({'path': kw['path']['label']} if 'path' in kw else {}), ensure_ascii=False)}", src)
        if cmd == "heartbeat":
            self.control["last_heartbeat"] = time.monotonic()
            return {"ok": True}
        with self.lock:
            self.cmds.append((cmd, kw))
        return {"ok": True, "message": f"{cmd} を受け付けた"}

    # ---- worker (owns the bus)
    def run(self) -> None:
        bus = None
        try:
            while not self.stop_evt.is_set():
                if bus is None:
                    try:
                        bus = (FakeBus(self.joints, self.args.allow_motion, {"joint1": -1.7, "joint2": 0.004, "joint3": -0.021}, self.args.fake_sag, self.args.fake_speed)
                               if self.args.fake else Bus(self.args.port, self.joints, self.args.allow_motion))
                        self.bus = bus
                        self.motor_params = bus.read_params()
                    except Exception as e:
                        self.publish({"mode": "fake" if self.args.fake else "serial", "port": self.args.port, "ok": False,
                                      "joints": [], "error": f"{self.args.port} を開けない: {e}"})
                        self.stop_evt.wait(2.0)
                        continue
                t0 = time.monotonic()
                states = bus.poll(0.004 if self.control["state"] != "read_only" else 0.006)
                q = self.update_state(states)
                self.control_step(bus, q)
                self.stop_evt.wait(max(0.0, 1.0 / CTRL_HZ - (time.monotonic() - t0)))
        finally:
            if bus is not None:
                if self.control["state"] != "read_only":
                    # Leaving the motors enabled with nobody commanding them is worse than holding;
                    # but disabling drops the arm. Hold the measured pose once more and say so.
                    self.audit("process exit while engaged: last hold command sent, motors left enabled", "service")
                bus.close()

    def update_state(self, states) -> dict:
        now = time.time()
        rows, q = [], {}
        for j in self.joints:
            s = states.get(j["name"])
            if s is None:
                rows.append({"name": j["name"], "pos": None, "vel": None, "torq": None, "status": None, "age_s": None})
                continue
            vals = (s.pos, s.vel, s.torq, s.status_code, s.t_mos)
            if self.last_vals.get(j["name"]) != vals:
                self.last_vals[j["name"]], self.last_change[j["name"]] = vals, now
            age = now - self.last_change[j["name"]]
            pos = s.pos + self.offsets.get(j["name"], 0.0)          # model angle
            rows.append({"name": j["name"], "pos": pos, "pos_motor": s.pos, "vel": s.vel, "torq": s.torq,
                         "status": f"{s.status_code:#x} {DAMIAO_STATUS.get(s.status_code, '不明')}",
                         "status_code": s.status_code, "t_mos": s.t_mos, "t_rotor": s.t_rotor, "age_s": age})
            if age < STALE_S:
                q[j["name"]] = pos
        ok = all(r["age_s"] is not None and r["age_s"] < STALE_S for r in rows)
        self.last_rows = rows
        self.statuses = {r["name"]: r.get("status_code") for r in rows}
        c = self.control
        self.publish({"mode": ("fake" if self.args.fake else "serial") + ("+motion" if self.args.allow_motion else " read-only"),
                      "motion_allowed": self.args.allow_motion, "port": self.args.port, "ok": ok, "t": now, "joints": rows,
                      "fk": self.fk(q) if ok else None, "fk_model": "SDK reBot-DevArm_fixend.urdf (Pinocchio)",
                      "joint_offsets": {"deg": {n: round(math.degrees(v), 3) for n, v in self.offsets.items()},
                                        "rad": self.offsets, "error": self.offsets_error, "file": str(OFFSETS_PATH),
                                        "note": "pos = モーター角 + offset（モデルの角度）。pos_motor が生の値。モーターには書き込まない"},
                      "control": {k: v for k, v in c.items() if k not in ("last_heartbeat",)} | {
                          "heartbeat_age_s": (time.monotonic() - c["last_heartbeat"]) if c["last_heartbeat"] else None},
                      "limits": {"vmax_rad_s": VMAX, "corridor_rad": CORRIDOR_RAD, "corridor_m": CORRIDOR_M,
                                 "heartbeat_s": HEARTBEAT_S, "kp": KP.tolist(), "ki": KI.tolist(), "deadband_rad": DEADBAND,
                                 "can_loss_s": CAN_LOSS_S},
                      "motor_params": self.motor_params, "pv_override": PV_OVERRIDE,
                      "compensation": {
                          "joint_feedback": "関節角（エンコーダ）の追従補正: 有効中は常時（joint1-3 強め、joint4-5 弱め、joint6 なし、不感帯 0.003 rad）。モーター軸の角度を計画に合わせるだけ。",
                          "mit_gravity": (f"{', '.join(sorted(MIT_JOINTS))}: MIT 保持（kp・kd は SDK yaml）＋ 重力 g(q)（SDK URDF、公式 dynamics）の"
                                          f"フィードフォワード（有効化後 {GRAV_RAMP_S} s で立ち上げ、上限 {TAU_FF_MAX} N m）＋ 移動中は計画速度。"
                                          + (f" 重力モデルの読み込み失敗: {self.gravity.error}" if self.gravity.error else "")),
                          "mit_ff_now": self.ff,
                          "mit_tuning": dict(self.tuning, error=self.tuning_error),
                          "visual_tcp_correction": "視覚（ID 1）による手先誤差補正: 未実装（camera↔base と 手先リンク↔ID 1 の校正後に検討）。",
                          "deflection": "リンク・減速機・取り付けのたわみ: 補償していない（関節角が計画どおりでも手先はずれ得る）。"},
                      "faults_policy": FAULT_POLICY,
                      "notes": self.notes})
        return q if ok else {}

    def publish(self, st: dict) -> None:
        self.json = json.dumps(st, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)).encode()

    def _hold(self, bus, q, why):
        """Hold where the arm IS now (never return to an older hold target)."""
        here = {n: q[n] for n in ARM}
        self._send(bus, here, VMAX, qm=q)
        self.control.update(state="holding", motion=None, message=why, hold_q=here)

    def control_step(self, bus, q: dict) -> None:
        """Every cycle while engaged: feedback from the measured joints to the command.

        The motors' own POS_VEL loops do not fully cancel gravity, so the measured path differs from
        the planned (simulated) one. Outer loop, always on while engaged (moving and holding):
            e = q_plan(t) - q_measured,  I += KI * e * dt  (|I| <= I_MAX),
            q_cmd = q_plan(t) + KP * e + I                 (|q_cmd - q_plan| <= CORR_MAX)
        and the deviation is checked against the corridor the simulator validated:
            |e| <= CORRIDOR_RAD per joint, end_link deviation <= CORRIDOR_M (Pinocchio FK).
        Leaving the corridor stops the motion and holds where the arm is.
        """
        c = self.control
        with self.lock:
            cmds, self.cmds = self.cmds, []
        for cmd, kw in cmds:
            try:
                self.handle(bus, q, cmd, kw)
            except Exception as e:
                c["message"] = f"{cmd} 失敗: {type(e).__name__}: {e}"
                self.audit(f"{cmd} failed: {e}", "service")
        if c["state"] not in ("holding", "moving"):
            return
        if not all(n in q for n in ARM):
            self.lost_since = self.lost_since or time.monotonic()
            lost = time.monotonic() - self.lost_since
            if lost > CAN_LOSS_S:
                self._fault(f"フィードバックなし {lost:.1f} s（CAN 断 / 電源断の可能性）", keep_target=True)
            if c.get("last_cmd"):
                # keep holding the last commanded point; the gravity term from the last known pose
                self._send(bus, c["last_cmd"], VMAX, qm=c.get("last_q"))
                self._gripper_step(bus, {})
            return
        self.lost_since = None
        bad = {n: self.statuses.get(n) for n in ARM if self.statuses.get(n) != 0x1}
        c["cause_now"] = bool(bad)
        if bad and time.monotonic() - c.get("engaged_mono", 0) > ENGAGE_GRACE_S:
            desc = ", ".join(f"{n}: {DAMIAO_STATUS.get(v, hex(v) if v is not None else '?')}" for n, v in bad.items())
            self._fault(f"モーター状態の異常（{desc}）。無効化された関節は支えがない", keep_target=True)
        dt = 1.0 / CTRL_HZ
        qm = np.array([q[n] for n in ARM])
        c["last_q"] = {n: q[n] for n in ARM}
        if c["state"] == "moving":
            mv = c["motion"]
            hb = time.monotonic() - c["last_heartbeat"]
            if hb > HEARTBEAT_S:
                qs = self._plan_at(mv, mv["tp"])
                c.update(state="holding", motion=None, hold_q=dict(zip(ARM, qs)),
                         message=f"ページからの生存確認が {hb:.1f} s 途絶えたため停止（計画上の現在点で保持）")
                self.audit("stopped: heartbeat lost", "service")
                return
            # Time scaling: the plan clock advances only as fast as the arm follows it. Real motors lag
            # their commands; comparing against plan(wall time) turned normal lag into a corridor stop.
            lag_now = float(np.max(np.abs(self._plan_at(mv, mv["tp"]) - qm)[:5]))
            rate = float(np.clip(1.0 - (lag_now - LAG_FREE) / (LAG_STOP - LAG_FREE), 0.0, 1.0))
            mv["tp"] = min(mv["T"], mv["tp"] + dt * rate)
            mv["rate"] = rate
            if rate > 0.05:
                mv["stalled_since"] = None
            elif mv.get("stalled_since") is None:
                mv["stalled_since"] = time.monotonic()
            elif time.monotonic() - mv["stalled_since"] > STALL_S:
                self._fault(f"追従できない: 計画が {STALL_S:.0f} s 進まない（遅れ {lag_now:.3f} rad）", keep_target=True)
                return
            t = mv["tp"]
            qp = self._plan_at(mv, t)
            vdir = (self._plan_at(mv, min(mv["T"], t + dt)) - qp) / dt     # planned velocity (direction of travel)
            vp = vdir * rate                                                   # ... as the slowed plan clock runs
            mv["progress"] = min(1.0, t / mv["T"])
        else:
            qp = np.array([c["hold_q"][n] for n in ARM])
            vp = vdir = np.zeros(6)
        e = qp - qm
        fb = c.setdefault("fb", {"I": [0.0] * 6, "max_err": 0.0, "max_tcp_dev": 0.0})
        active = np.abs(e) > DEADBAND
        e_ctl = np.where(active, e - np.sign(e) * DEADBAND, 0.0)
        # Integrate while holding (static gravity / cable / friction error), and while moving only as far as
        # the plan clock is held back by lag (weight 1 - rate, up to LAG_KI_GAIN x faster). A free-running
        # plan is not integrated: under POS_VEL that wound up and overshot the end of the path. In MIT, the
        # gravity model is off by ~1-2 N m at some poses (2026-10-14, calibration pose "作業範囲 4"), which
        # stalled the plan with a frozen integral.
        # time scaling above; integrating it wound up and overshot the end of the path.
        w_i = (1.0 - mv["rate"]) * LAG_KI_GAIN if c["state"] == "moving" else 1.0
        I = np.clip(np.array(fb["I"]) + KI * e_ctl * dt * w_i, -I_MAX, I_MAX)
        q_cmd = qp + np.clip(KP * e_ctl + I, -CORR_MAX, CORR_MAX)
        tcp_dev = self._tcp_dev(qp, qm)
        fb.update(I=I.tolist(), err=e.tolist(), err_max_now=float(np.max(np.abs(e))), tcp_dev=tcp_dev,
                  max_err=max(fb["max_err"], float(np.max(np.abs(e)))),
                  max_tcp_dev=max(fb["max_tcp_dev"], tcp_dev or 0.0))
        if np.max(np.abs(e)) > CORRIDOR_RAD or (tcp_dev or 0.0) > CORRIDOR_M:
            j = ARM[int(np.argmax(np.abs(e)))]
            why = (f"計画経路から外れた: 最大 {np.max(np.abs(e)):.3f} rad（{j}）、手先 "
                   f"{0 if tcp_dev is None else tcp_dev * 1000:.0f} mm（許容 {CORRIDOR_RAD} rad / {CORRIDOR_M * 1000:.0f} mm）")
            if c["state"] == "moving" or not c.get("fault"):
                # Freeze the target at the PLANNED point of this instant. Never re-target to the
                # measured pose: with an uncorrectable sag that would follow the arm downwards.
                c.update(state="holding", motion=None, hold_q=dict(zip(ARM, qp)),
                         fault=why, message="停止（異常）: " + why + "。計画上の位置を保持目標に固定。停止・トルク切りで対処")
                self.audit("fault: " + why, "service")
        # Faults latch: they are cleared only by the user ("ack_fault") once the cause is gone, or by release.
        c["last_cmd"] = dict(zip(ARM, q_cmd))
        self._send(bus, c["last_cmd"], VMAX * 1.5, qm=q, vd=dict(zip(ARM, vp)), vdir=dict(zip(ARM, vdir)))
        self._gripper_step(bus, q)
        self._trace(c, qp, qm, q_cmd)
        if c["state"] == "moving" and t >= mv["T"] and lag_now < LAG_FREE * 2:
            end = dict(zip(ARM, np.array(mv["q_plan"][-1])))
            c.update(state="holding", motion=None, hold_q=end,
                     message=f"移動完了（最大偏差 {fb['max_err']:.3f} rad / 手先 {fb['max_tcp_dev'] * 1000:.0f} mm）。保持中もフィードバック継続")

    def _send(self, bus, qd: dict, vlim: float, qm: dict | None = None, vd: dict | None = None,
              vdir: dict | None = None) -> None:
        """POS_VEL targets, except MIT_JOINTS: tau = kp (q_cmd - q) + kd (v_plan - v) + g(q_measured)
        + friction * tanh(v_dir / friction_v). The gravity term fades in over GRAV_RAMP_S after engage and is
        clamped to TAU_FF_MAX; friction (mit_tuning.yaml) is applied only while a path is followed."""
        motor = lambda n: float(qd[n]) - self.offsets.get(n, 0.0)     # model -> motor angle
        bus.send_pos_vel({n: motor(n) for n in qd if n not in MIT_JOINTS}, vlim)
        by = {j["name"]: j for j in self.joints}
        g = self.gravity(qm) if qm else None
        ramp = float(np.clip((time.monotonic() - self.control.get("engaged_mono", 0.0)) / GRAV_RAMP_S, 0.0, 1.0))
        for n in MIT_JOINTS:
            if n in qd:
                mit = by[n]["mit"]
                if g is not None:
                    tau = float(np.clip(ramp * g[n], -TAU_FF_MAX[n], TAU_FF_MAX[n]))
                else:                                   # no pose (feedback lost): keep the last gravity term
                    tau = float(self.ff.get(n, {}).get("tau_ff", 0.0))
                fr = 0.0
                if vdir and g is not None:
                    fr = self.tuning["friction"][n] * math.tanh(float(vdir.get(n, 0.0)) / self.tuning["friction_v"])
                v = 0.0 if not vd else float(vd.get(n, 0.0))
                kp = self.tuning["kp"].get(n, float(mit.get("kp", 18.0)))
                kd = self.tuning["kd"].get(n, float(mit.get("kd", 2.0)))
                bus.send_mit(n, motor(n), kp, kd, vel=v, tau=tau + fr)
                self.ff[n] = {"tau_ff": round(tau, 3), "fric": round(fr, 3), "v_ff": round(v, 4), "ramp": round(ramp, 2)}

    # ---- gripper
    def _gap(self, rad: float) -> float:
        g = self.tuning["gripper"]
        return float(np.clip(rad / g["full_open_rad"] * g["max_gap_m"], 0.0, g["max_gap_m"]))

    def _rad(self, gap: float) -> float:
        g = self.tuning["gripper"]
        return float(np.clip(gap, 0.0, g["max_gap_m"]) / g["max_gap_m"] * g["full_open_rad"])

    def _gripper_cmd(self, bus, q, kw) -> None:
        c = self.control
        if c["state"] != "holding" or c.get("fault"):
            c["message"] = "グリッパー: アームが保持中（異常なし）のときだけ操作する"
            return
        if q.get("gripper") is None:
            c["message"] = "グリッパー: 角度が読めない"
            return
        g = c.get("gripper") or {}
        if not c.get("gripper_enabled"):
            bus.prepare_mit("gripper")
            bus.enable(["gripper"])
            c["gripper_enabled"] = True
            g = {"cmd_rad": q["gripper"]}
        act = kw.get("action")
        tu = self.tuning["gripper"]
        if act in ("open", "release"):
            gap = float(np.clip(float(kw.get("gap_mm", 0)) / 1000, 0.0, tu["max_gap_m"]))
            g.update(mode="position", target_rad=self._rad(gap), target_gap_m=gap, cmd_rad=g.get("cmd_rad", q["gripper"]),
                     event=f"{'開く' if act == 'open' else '離す'}: 隙間 {gap * 1000:.0f} mm へ")
        elif act == "grip":
            expect = float(kw.get("width_mm", 0)) / 1000
            if not 0.005 <= expect <= tu["max_gap_m"]:
                c["message"] = "グリッパー: 挟む物の幅 width_mm が範囲外"
                return
            g.update(mode="grip", expect_m=expect, t0=time.monotonic(), still_since=None,
                     event=f"挟み中（予定の幅 {expect * 1000:.0f} mm、トルク {tu['grip_torque_nm']:.2f} N m）")
        elif act == "hold":
            g.update(mode="position", target_rad=q["gripper"], target_gap_m=self._gap(q["gripper"]), cmd_rad=q["gripper"],
                     event="その場で保持")
        else:
            c["message"] = f"グリッパー: 不明な操作 {act}"
            return
        if act != "hold":
            g["result"] = None
        c["gripper"] = g
        c["message"] = "グリッパー: " + g["event"]

    def _gripper_step(self, bus, q) -> None:
        """Every cycle while the gripper is enabled: ramp a position target, or close with a torque limit."""
        c = self.control
        g = c.get("gripper")
        if not c.get("gripper_enabled") or not g or "mode" not in g:
            return
        tu, dt = self.tuning["gripper"], 1.0 / CTRL_HZ
        qg = q.get("gripper")
        if qg is None:                                   # no feedback: repeat the last command
            if g.get("last_send"):
                bus.send_mit("gripper", *g["last_send"])
            return
        g["gap_now_m"] = self._gap(qg)
        if g["mode"] == "position":
            step = tu["speed_rad_s"] * dt
            g["cmd_rad"] = float(g["cmd_rad"] + np.clip(g["target_rad"] - g["cmd_rad"], -step, step))
            send = (g["cmd_rad"], tu["kp"], tu["kd"], 0.0, 0.0)
        else:                                            # grip: no spring, closing torque (towards 0 rad)
            send = (qg, 0.0, tu["grip_kd"], 0.0, tu["grip_torque_nm"])
            vel = next((r.get("vel") for r in self.last_rows if r["name"] == "gripper"), None)
            now = time.monotonic()
            if vel is not None and abs(vel) < GRIP_STILL_RAD_S and now - g["t0"] > 0.4:
                g["still_since"] = g.get("still_since") or now
            else:
                g["still_since"] = None
            closed = g["gap_now_m"] < 0.003
            settled = g["still_since"] is not None and now - g["still_since"] > GRIP_SETTLE_S
            if g.get("result") is None and (closed or settled or now - g["t0"] > GRIP_TIMEOUT_S):
                if not closed and g["gap_now_m"] > g["expect_m"] + 0.015:
                    # stopped well wider than the object: not closing (torque direction, jam) -- do not call it a grip
                    g.update(result="not_closing", mode="position", target_rad=qg, cmd_rad=qg, target_gap_m=g["gap_now_m"],
                             event=f"閉じない: 隙間 {g['gap_now_m'] * 1000:.1f} mm で止まった（予定 {g['expect_m'] * 1000:.0f} mm）。"
                                   "トルクの向き・引っかかりを確認。その場で保持")
                    send = (qg, tu["kp"], tu["kd"], 0.0, 0.0)
                elif not closed and g["gap_now_m"] >= g["expect_m"] - tu["miss_margin_m"]:
                    g.update(result="holding", event=f"把持中: 隙間 {g['gap_now_m'] * 1000:.1f} mm（予定 {g['expect_m'] * 1000:.0f} mm）、トルクを維持")
                else:
                    g.update(result="missed", mode="position", target_rad=qg, cmd_rad=qg, target_gap_m=g["gap_now_m"],
                             event=f"つかみ損ね: 隙間 {g['gap_now_m'] * 1000:.1f} mm まで閉じた（予定 {g['expect_m'] * 1000:.0f} mm）。その場で保持")
                    send = (qg, tu["kp"], tu["kd"], 0.0, 0.0)
                c["message"] = "グリッパー: " + g["event"]
                self.audit("gripper: " + g["event"], "service")
            elif g.get("result") == "holding" and g["gap_now_m"] < g["expect_m"] - tu["miss_margin_m"]:
                g["event"] = f"警告: 把持中に隙間が {g['gap_now_m'] * 1000:.1f} mm まで減った（滑り・落下の疑い）"
        g["last_send"] = send
        bus.send_mit("gripper", *send)

    def _fault(self, why: str, keep_target: bool) -> None:
        """Stop any motion; hold the planned point of this instant (or the last command)."""
        c = self.control
        if c.get("fault") == why:
            return
        if c["state"] == "moving":
            qp = self._plan_at(c["motion"], c["motion"]["tp"])
            c.update(hold_q=dict(zip(ARM, qp)))
        c.update(state="holding", motion=None, fault=why, message="停止（異常）: " + why)
        self.audit("fault: " + why, "service")

    def request_exit(self) -> bool:
        """SIGTERM policy: refuse the first request while engaged; a second one within 5 s exits."""
        now = time.monotonic()
        self.exit_requests = [t for t in self.exit_requests if now - t < 5.0] + [now]
        if self.control["state"] == "read_only" or len(self.exit_requests) >= 2:
            if self.control["state"] != "read_only":
                self.audit("exit forced while engaged: motors left enabled with the last command", "service")
            return True
        self.control["message"] = "有効中のため終了要求を拒否した。ページで「トルクを切る」か、5 s 以内に再度 SIGTERM"
        self.audit("exit refused while engaged", "service")
        return False

    def _tcp_dev(self, qp, qm):
        a = self.fk(dict(zip(ARM, qp)))
        b = self.fk(dict(zip(ARM, qm)))
        if a is None or b is None:
            return None
        return float(np.linalg.norm(np.array(a["p"]) - np.array(b["p"])))

    @staticmethod
    def _plan_at(mv, t):
        """Planned joints at plan time t. A start offset (measured - planned start, <= START_TOL) is
        blended out over the first T_blend seconds, so the arm joins the simulated path smoothly
        instead of starting with a step it cannot follow."""
        ts, qp = mv["times"], mv["q_plan"]
        if t >= ts[-1]:
            q = np.array(qp[-1])
        else:
            i = int(np.searchsorted(ts, t, side="right")) - 1
            a = (t - ts[i]) / (ts[i + 1] - ts[i])
            q = (1 - a) * np.array(qp[i]) + a * np.array(qp[i + 1])
        off = mv.get("offset")
        if off is not None and t < mv["T_blend"]:
            w = 0.5 + 0.5 * math.cos(math.pi * t / mv["T_blend"])     # 1 -> 0, smooth
            q = q + w * np.array(off)
        return q

    def handle(self, bus, q, cmd, kw) -> None:
        c = self.control
        if cmd == "engage":
            if c["state"] != "read_only":
                c["message"] = "既に有効"
                return
            if not all(n in q for n in ARM):
                c["message"] = "関節角が読めないため有効化しない"
                return
            off, off_err = self.load_offsets()
            if off_err:
                c["message"] = "零点補正ファイルの誤りのため有効化しない: " + off_err
                return
            if off != self.offsets:
                # offsets changed while torque was off: re-read the pose in the new model angles first
                self.offsets, self.offsets_error = off, None
                c["message"] = "零点補正ファイルが変わったので読み直した。もう一度「有効化」する"
                self.audit("joint offsets reloaded: " + json.dumps({n: round(math.degrees(v), 3) for n, v in off.items()}), "service")
                return
            self.tuning, self.tuning_error = self.load_tuning()
            if self.tuning_error:
                c["message"] = "調整値ファイルの誤りのため有効化しない: " + self.tuning_error
                return
            if self.gravity.error:
                c["message"] = f"重力モデルが読めないため有効化しない（{', '.join(sorted(MIT_JOINTS))} は重力補償つき MIT 保持）: {self.gravity.error}"
                return
            bus.prepare_pos_vel([n for n in ARM if n not in MIT_JOINTS])     # while still disabled
            for n in MIT_JOINTS:
                bus.prepare_mit(n)
            hold = {n: q[n] for n in ARM}
            c["engaged_mono"] = time.monotonic()           # gravity ramp starts here (tau_ff = 0 at enable)
            bus.enable(ARM)                                # the six arm motors only
            self._send(bus, hold, VMAX, qm=q)
            c.update(state="holding", engaged_at=time.strftime("%H:%M:%S"), engaged_mono=c["engaged_mono"],
                     q_engage=hold, hold_q=hold, fault=None, last_cmd=hold,
                     fb={"I": [0.0] * 6, "max_err": 0.0, "max_tcp_dev": 0.0},
                     message="有効化した（現在位置で保持）。グリッパーは無効のまま")
            c["last_heartbeat"] = time.monotonic()
        elif cmd == "stop":
            if c["state"] in ("holding", "moving") and all(n in q for n in ARM):
                c["hold_q"] = {n: q[n] for n in ARM}
                self._hold(bus, q, "停止操作（その場で保持）")
        elif cmd == "release":
            names = ARM + (["gripper"] if c.get("gripper_enabled") else [])
            bus.disable(names)
            c.update(state="read_only", motion=None, hold_q=None, gripper_enabled=False, gripper=None, fault=None,
                     message="トルクを切った（読み取り専用に戻った）")
        elif cmd == "execute":
            if c["state"] != "holding":
                c["message"] = "保持状態でないため実行しない（先に有効化）"
                return
            if c.get("fault"):
                c["message"] = "異常が解消していないため実行しない: " + c["fault"]
                return
            path = kw["path"]
            q_now = np.array([q[n] for n in ARM])
            start = np.array(path["q_plan"][0])
            if np.max(np.abs(start - q_now)) > START_TOL:
                c["message"] = (f"経路の始点と現在の姿勢が {np.max(np.abs(start - q_now)):.3f} rad ずれている"
                                "（シミュレーション後に動いた）。シミュレートし直す")
                return
            # The integral belongs to the previous pose (gravity-model error there); start the move with half of it
            # (zeroing it would step the command).
            c["fb"] = {"I": (0.5 * np.array(c.get("fb", {}).get("I", [0.0] * 6))).tolist(), "max_err": 0.0, "max_tcp_dev": 0.0}
            offset = (q_now - start).tolist()
            if self.trace is not None:
                self.trace["f"].close()
            tp = self.log_path.parent / ("arm_trace_last_fake.jsonl" if self.args.fake else "arm_trace_last.jsonl")
            self.trace = {"f": tp.open("w"), "t0": time.monotonic(), "end": time.monotonic()}
            self.trace["f"].write(json.dumps({"path": path["label"], "path_id": path["id"], "time": time.strftime("%H:%M:%S"),
                                              "joints": ARM, "tuning": self.tuning}) + "\n")
            c.update(state="moving", message=f"移動中: {path['label']}",
                     motion={"label": path["label"], "path_id": path["id"], "times": path["times"],
                             "q_plan": path["q_plan"], "T": float(path["times"][-1]), "t0": time.monotonic(), "tp": 0.0,
                             "progress": 0.0, "rate": 1.0, "offset": offset,
                             "T_blend": min(2.0, 0.3 * float(path["times"][-1]))})
        elif cmd == "ack_fault":
            if not c.get("fault"):
                c["message"] = "異常はない"
            elif not all(n in q for n in ARM) or c.get("cause_now"):
                c["message"] = "原因が続いているため解除しない: " + c["fault"]
            else:
                e = max(abs(q[n] - c["hold_q"][n]) for n in ARM) if c.get("hold_q") else 0.0
                if e > CORRIDOR_RAD:
                    c["message"] = f"保持目標から {e:.3f} rad 離れているため解除しない（トルクを切って姿勢を確認）"
                else:
                    self.audit("fault acknowledged: " + c["fault"], "user")
                    c.update(fault=None, message="異常を確認・解除した（保持中）")
        elif cmd == "gripper":
            self._gripper_cmd(bus, q, kw)
        elif cmd == "grasp":
            c["message"] = "実機での把持実行は拒否: " + "; ".join(kw.get("blockers") or ["条件未確認"])

    # ---- HTTP
    def serve(self):
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.split("?")[0] != "/state.json":
                    return self.send_error(404)
                self._json(svc.json)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                src = self.headers.get("X-Forwarded-For", self.client_address[0]) + " via " + self.headers.get("X-Origin-Page", "?")
                path = self.path.split("?")[0]
                if path == "/api/heartbeat":
                    return self._json(svc.submit("heartbeat", src))
                if path == "/api/engage":
                    if body.get("confirm") != "MOVE":
                        return self._json({"ok": False, "message": "確認文字列 MOVE が必要"})
                    return self._json(svc.submit("engage", src))
                if path == "/api/stop":
                    return self._json(svc.submit("stop", src))
                if path == "/api/ack_fault":
                    return self._json(svc.submit("ack_fault", src))
                if path == "/api/release":
                    if body.get("confirm") != "RELEASE":
                        return self._json({"ok": False, "message": "確認文字列 RELEASE が必要"})
                    return self._json(svc.submit("release", src))
                if path == "/api/execute":
                    if not svc.args.allow_motion:
                        return self._json(svc.submit("execute", src))
                    p = fetch_json(f"{svc.args.sim_url}/api/path/{body.get('path_id', '')}")
                    if not p or not p.get("ok"):
                        return self._json({"ok": False, "message": "シミュレーターで確認済みの経路が見つからない: " + str((p or {}).get("message"))})
                    if "q_plan" not in p or "times" not in p:
                        return self._json({"ok": False, "message": "シミュレーション結果に計画プロファイルがない"})
                    if time.time() - p["checked_at"] > PATH_MAX_AGE_S:
                        return self._json({"ok": False, "message": "経路の確認から時間が経ちすぎた。確認し直す"})
                    return self._json(svc.submit("execute", src, path=p))
                if path == "/api/test/inject" and svc.args.fake:
                    svc.bus.inject.update(body)
                    return self._json({"ok": True, "inject": svc.bus.inject})
                if path == "/api/gripper":
                    if body.get("action") not in ("open", "release", "grip", "hold"):
                        return self._json({"ok": False, "message": "action は open / release / grip / hold"})
                    return self._json(svc.submit("gripper", src, **{k: body[k] for k in ("action", "gap_mm", "width_mm") if k in body}))
                if path == "/api/grasp":
                    g = fetch_json(f"{svc.args.sim_url}/api/real_grasp_check") or {}
                    return self._json(svc.submit("grasp", src, blockers=g.get("blockers", ["シミュレーターに接続できない"])))
                self.send_error(404)

        server = ThreadingHTTPServer((self.args.host, self.args.http_port), Handler)
        server.daemon_threads = True
        return server


def fetch_json(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "message": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--hw-yaml", type=Path, default=SDK / "config" / "rebotarm_dm.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8091)
    ap.add_argument("--sim-url", default="http://127.0.0.1:8092")
    ap.add_argument("--allow-motion", action="store_true", help="accept motion commands from the web page (user only)")
    ap.add_argument("--fake", action="store_true", help="simulated motors; never opens the serial port")
    ap.add_argument("--fake-speed", type=float, default=1.0, help="fake motors: fraction of the commanded speed")
    ap.add_argument("--fake-sag", type=float, default=0.03, help="fake motors: steady offset on joint2/3 (rad)")
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    svc = ArmService(args)
    th = threading.Thread(target=svc.run, daemon=True)
    th.start()
    server = svc.serve()
    mode = ("FAKE motors" if args.fake else args.port) + (" | MOTION ALLOWED from the web page" if args.allow_motion else " | read-only")
    print(f"arm_state on http://{args.host}:{args.http_port}/state.json — {mode}", flush=True)
    svc.audit(f"service start: {mode}", "service")
    signal.signal(signal.SIGTERM, lambda *_: svc.request_exit() and threading.Thread(target=server.shutdown).start())
    try:
        if args.seconds:
            threading.Timer(args.seconds, server.shutdown).start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop_evt.set()
        th.join(timeout=5)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
