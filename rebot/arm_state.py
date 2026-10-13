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
CAN_LOSS_S = 0.3            # s without fresh feedback while engaged = CAN / bus fault
ENGAGE_GRACE_S = 0.5        # after enabling, status 0x1 must be reported within this time
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
            m, pv = self._m[n], by[n]["pv"]
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

    def send_mit(self, name, pos, kp, kd) -> None:
        self._check()
        self._m[name].send_mit(float(pos), 0.0, float(kp), float(kd), 0.0)

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

    def send_mit(self, name, pos, kp, kd):
        self._check(); self._advance(); self.target[name] = pos; self.vlim = 1.0

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


class ArmService:
    def __init__(self, args) -> None:
        self.args = args
        self.joints = load_joints(args.hw_yaml)
        self.fk = FK()
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
                        bus = (FakeBus(self.joints, self.args.allow_motion, {"joint1": -1.7, "joint2": 0.0}, self.args.fake_sag, self.args.fake_speed)
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
            rows.append({"name": j["name"], "pos": s.pos, "vel": s.vel, "torq": s.torq,
                         "status": f"{s.status_code:#x} {DAMIAO_STATUS.get(s.status_code, '不明')}",
                         "status_code": s.status_code, "t_mos": s.t_mos, "t_rotor": s.t_rotor, "age_s": age})
            if age < STALE_S:
                q[j["name"]] = s.pos
        ok = all(r["age_s"] is not None and r["age_s"] < STALE_S for r in rows)
        self.statuses = {r["name"]: r.get("status_code") for r in rows}
        c = self.control
        self.publish({"mode": ("fake" if self.args.fake else "serial") + ("+motion" if self.args.allow_motion else " read-only"),
                      "motion_allowed": self.args.allow_motion, "port": self.args.port, "ok": ok, "t": now, "joints": rows,
                      "fk": self.fk(q) if ok else None, "fk_model": "SDK reBot-DevArm_fixend.urdf (Pinocchio)",
                      "control": {k: v for k, v in c.items() if k not in ("last_heartbeat",)} | {
                          "heartbeat_age_s": (time.monotonic() - c["last_heartbeat"]) if c["last_heartbeat"] else None},
                      "limits": {"vmax_rad_s": VMAX, "corridor_rad": CORRIDOR_RAD, "corridor_m": CORRIDOR_M,
                                 "heartbeat_s": HEARTBEAT_S, "kp": KP.tolist(), "ki": KI.tolist(), "deadband_rad": DEADBAND,
                                 "can_loss_s": CAN_LOSS_S},
                      "motor_params": self.motor_params,
                      "compensation": {
                          "joint_feedback": "関節角（エンコーダ）の追従補正: 有効中は常時（joint1-3 強め、joint4-5 弱め、joint6 なし、不感帯 0.003 rad）。モーター軸の角度を計画に合わせるだけ。",
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
        bus.send_pos_vel(here, VMAX)
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
                bus.send_pos_vel(c["last_cmd"], VMAX)       # keep holding the last commanded point
            return
        self.lost_since = None
        bad = {n: self.statuses.get(n) for n in ARM if self.statuses.get(n) != 0x1}
        c["cause_now"] = bool(bad)
        if bad and time.monotonic() - c.get("engaged_mono", 0) > ENGAGE_GRACE_S:
            desc = ", ".join(f"{n}: {DAMIAO_STATUS.get(v, hex(v) if v is not None else '?')}" for n, v in bad.items())
            self._fault(f"モーター状態の異常（{desc}）。無効化された関節は支えがない", keep_target=True)
        dt = 1.0 / CTRL_HZ
        qm = np.array([q[n] for n in ARM])
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
            mv["progress"] = min(1.0, t / mv["T"])
        else:
            qp = np.array([c["hold_q"][n] for n in ARM])
        e = qp - qm
        fb = c.setdefault("fb", {"I": [0.0] * 6, "max_err": 0.0, "max_tcp_dev": 0.0})
        active = np.abs(e) > DEADBAND
        e_ctl = np.where(active, e - np.sign(e) * DEADBAND, 0.0)
        # Integrate only while holding (static gravity sag). During motion the lag is handled by the
        # time scaling above; integrating it wound up and overshot the end of the path.
        I = np.array(fb["I"]) if c["state"] == "moving" else np.clip(np.array(fb["I"]) + KI * e_ctl * dt, -I_MAX, I_MAX)
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
        bus.send_pos_vel(c["last_cmd"], VMAX * 1.5)
        if c["state"] == "moving" and t >= mv["T"] and lag_now < LAG_FREE * 2:
            end = dict(zip(ARM, np.array(mv["q_plan"][-1])))
            c.update(state="holding", motion=None, hold_q=end,
                     message=f"移動完了（最大偏差 {fb['max_err']:.3f} rad / 手先 {fb['max_tcp_dev'] * 1000:.0f} mm）。保持中もフィードバック継続")

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
            bus.prepare_pos_vel(ARM)                       # while still disabled
            hold = {n: q[n] for n in ARM}
            bus.enable(ARM)                                # the six arm motors only
            bus.send_pos_vel(hold, VMAX)
            c.update(state="holding", engaged_at=time.strftime("%H:%M:%S"), engaged_mono=time.monotonic(),
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
            c.update(state="read_only", motion=None, hold_q=None, gripper_enabled=False, fault=None,
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
            c["fb"] = {"I": c.get("fb", {}).get("I", [0.0] * 6), "max_err": 0.0, "max_tcp_dev": 0.0}
            offset = (q_now - start).tolist()
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
