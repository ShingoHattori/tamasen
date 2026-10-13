#!/usr/bin/env python3
"""Read-only joint state of the reBot Arm B601-DM, served as JSON on 127.0.0.1:8091.

    python arm_state.py --port /dev/ttyACM0

What goes on the CAN bus — verified in motorbridge v0.5.6 source (tag c652ce4):
  * opening the serial bridge and add_damiao_motor(): nothing
  * request_feedback(): the Damiao "refresh" frame only — id 0x7FF, data [id_lo, id_hi, 0xCC, 0...]
  * poll_feedback_once() / get_state(): receive only
  * NOT used here: enable / disable / ensure_mode / send_* / set_zero / store_parameters, and
    Controller.shutdown() or `with Controller(...)` — shutdown() sends DISABLE to every motor,
    which would drop an arm that something else is holding. We release with close_bus() + close().
So this process can neither energize nor de-energize a motor. The ReadOnlyArm class below only
exposes the read calls, so a later edit cannot reach the command API by accident.

motorbridge's cached state carries no receive time. Freshness is therefore estimated from the
time the values last changed (encoder / current noise changes them on every real frame), and is
reported as such.

Endpoint: GET /state.json — joints (rad, rad/s, Nm, status code, age), the forward kinematics of
end_link in the arm-base frame computed with the SDK's own Pinocchio model, and notes.
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
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


class ReadOnlyArm:
    """Only the feedback path of motorbridge. Deliberately has no command methods."""

    def __init__(self, port: str, joints: list[dict]) -> None:
        from motorbridge import Controller
        self._ctrl = Controller.from_dm_serial(port, 921600)
        self._motors = [self._ctrl.add_damiao_motor(j["motor_id"], j["feedback_id"], j["model"]) for j in joints]

    def poll(self, settle_s: float = 0.006) -> list:
        for m in self._motors:
            m.request_feedback()                 # 0xCC refresh frame only
        t_end = time.monotonic() + settle_s
        while time.monotonic() < t_end:
            self._ctrl.poll_feedback_once()      # receive only
            time.sleep(0.001)
        return [m.get_state() for m in self._motors]

    def close(self) -> None:
        # close_bus(): stop the receive thread and flush, no frames. Never shutdown() (= disable all).
        try:
            self._ctrl.close_bus()
        finally:
            self._ctrl.close()


def load_joints(hw_yaml: Path) -> list[dict]:
    cfg = yaml.safe_load(hw_yaml.read_text())
    out = []
    for j in cfg["joints"]:
        if j.get("vendor", "damiao") != "damiao":
            raise SystemExit(f"{hw_yaml}: only damiao joints are supported here ({j['name']})")
        out.append({"name": j["name"], "motor_id": int(j["motor_id"]), "feedback_id": int(j["feedback_id"]),
                    "model": str(j["model"])})
    return out


class FK:
    """end_link pose in the arm-base frame from the SDK's Pinocchio model (same as its examples)."""

    def __init__(self) -> None:
        self.error = None
        try:
            import pinocchio as pin
            from reBotArm_control_py.kinematics import load_robot_model, get_end_effector_frame_id
            self.pin = pin
            self.model = load_robot_model()
            self.data = self.model.createData()
            self.fid = get_end_effector_frame_id(self.model)
            self.names = [self.model.names[i] for i in range(1, self.model.njoints)]
        except Exception as e:  # FK is optional; joint reading must still work
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--hw-yaml", type=Path, default=SDK / "config" / "rebotarm_dm.yaml")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8091)
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    joints = load_joints(args.hw_yaml)
    fk = FK()
    lock = threading.Lock()
    stop = threading.Event()
    shared = {"json": b"{}"}
    last_vals: dict[str, tuple] = {}
    last_change: dict[str, float] = {}
    notes = [
        "読み取り専用: 送信するのは Damiao のフィードバック要求フレーム（0x7FF, 0xCC）だけ。有効化・無効化・指令は送らない。",
        "経過 s = 値が最後に変化してからの時間（motorbridge のキャッシュに受信時刻がないため、変化で鮮度を推定）。",
        "関節 0 rad とモデル（URDF/MJCF）の 0 rad が一致しているかは未確認（モーター零点は過去に手動設定）。",
        "グリッパーのモーター角と顎の開口の対応は未確認。",
    ]
    if fk.error:
        notes.append(f"順運動学を計算できない: {fk.error}")

    def publish(st: dict) -> None:
        with lock:
            shared["json"] = json.dumps(st, ensure_ascii=False).encode()

    def worker() -> None:
        arm = None
        try:
            while not stop.is_set():
                if arm is None:
                    try:
                        arm = ReadOnlyArm(args.port, joints)
                    except Exception as e:
                        publish({"mode": "read-only", "port": args.port, "ok": False, "joints": [],
                                 "error": f"{args.port} を開けない: {e}", "notes": notes, "t": time.time()})
                        stop.wait(2.0)
                        continue
                t0 = time.monotonic()
                states = arm.poll()
                now = time.time()
                rows, q = [], {}
                for j, s in zip(joints, states):
                    if s is None:
                        rows.append({"name": j["name"], "pos": None, "vel": None, "torq": None,
                                     "status": None, "age_s": None})
                        continue
                    vals = (s.pos, s.vel, s.torq, s.status_code, s.t_mos)
                    if last_vals.get(j["name"]) != vals:
                        last_vals[j["name"]], last_change[j["name"]] = vals, now
                    age = now - last_change[j["name"]]
                    rows.append({"name": j["name"], "pos": s.pos, "vel": s.vel, "torq": s.torq,
                                 "status": f"{s.status_code:#x} {DAMIAO_STATUS.get(s.status_code, '不明')}",
                                 "status_code": s.status_code, "t_mos": s.t_mos, "t_rotor": s.t_rotor,
                                 "age_s": age})
                    if age < STALE_S:
                        q[j["name"]] = s.pos
                ok = all(r["age_s"] is not None and r["age_s"] < STALE_S for r in rows)
                publish({"mode": "read-only", "port": args.port, "ok": ok, "t": now, "joints": rows,
                         "fk": fk(q) if ok else None, "fk_model": "SDK reBot-DevArm_fixend.urdf (Pinocchio)",
                         "hw_yaml": str(args.hw_yaml), "notes": notes})
                stop.wait(max(0.0, 1.0 / args.rate - (time.monotonic() - t0)))
        finally:
            if arm is not None:
                arm.close()

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/state.json":
                return self.send_error(404)
            with lock:
                body = shared["json"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.http_port), Handler)
    server.daemon_threads = True
    print(f"arm_state (read-only) on http://{args.host}:{args.http_port}/state.json, port {args.port}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        if args.seconds:
            threading.Timer(args.seconds, server.shutdown).start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        th.join(timeout=5)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
