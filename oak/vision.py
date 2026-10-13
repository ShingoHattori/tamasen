#!/usr/bin/env python3
"""Live ArUco recognition of the block, the gripper marker and the ID 2 reference card.

    python vision.py --device 18443                       # live, http://127.0.0.1:8090/
    python vision.py --replay output/recordings/<time>    # same processing on a recording

RGB only (no stereo depth). Detection runs on the full colour frame; the browser gets a separate,
downscaled JPEG stream. Binds to 127.0.0.1 — view it through an SSH port forward.

Shows marker corners / ids / axes on the video, every marker pose in the camera frame, the
provisional world frame of the ID 2 card (tracking or fixed), the block from each registered block
marker separately (never averaged), and the ID 1 pose. Stills and recordings are saved only on
request. Sends nothing to the arm; the arm and simulator panels are proxied from their own
read-only / simulation-only services (rebot/arm_state.py, sim/sim_server.py).

Units in state.json: metres and radians. T_a_b maps frame-b points into frame a.
"""
import argparse
import json
import math
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import pose
import recorder
import world
from oak import Latest, Ticker, announce

HERE = Path(__file__).resolve().parent
WARMUP_FRAMES = 30
BLOCK_COLOURS = {0: (255, 255, 0), 3: (255, 0, 255)}   # BGR: from ID 0 cyan, from ID 3 magenta
STATUS_COLOURS = {"ok": (0, 200, 0), "ambiguous": (0, 200, 255), "flip": (0, 0, 255), "rejected": (0, 0, 255)}


# --------------------------------------------------------------------------- sources

def camera_source(selector: str, size: tuple[int, int], fps: float):
    """Yield (BGR frame, host capture time, ImgTransformation). Close the generator when done."""
    import depthai as dai
    from oak_common import describe, pick_device
    device = pick_device(selector)
    info = {"camera": describe(device)}
    for f in device.getConnectedCameraFeatures():
        if f.socket == dai.CameraBoardSocket.CAM_A:
            info.update(sensor=f.sensorName, sensor_size=[f.width, f.height],
                        has_autofocus=bool(f.hasAutofocus))
    cal = device.readCalibration()
    _, cw, ch = cal.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
    info["calibrated_at"] = [cw, ch]
    info["naive_K"] = np.array(cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, *size)).tolist()
    yield info
    # Linux: the Device is closed together with the Pipeline; never build a second Pipeline on it.
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput(size, dai.ImgFrame.Type.BGR888i, fps=fps).createOutputQueue(
            maxSize=4, blocking=False)
        pipeline.start()
        for _ in range(WARMUP_FRAMES):
            q.get()
        offset = None
        while True:
            f = q.get()
            # Device timestamps are on the host's steady clock; convert to wall time once.
            ts = f.getTimestamp().total_seconds()
            if offset is None:
                offset = time.time() - time.monotonic()
            yield f.getCvFrame(), ts + offset, f.getTransformation()


def factory_intrinsics(tf, size, info) -> tuple[np.ndarray, np.ndarray, dict]:
    """Intrinsics of the frames actually processed, from the factory EEPROM calibration.

    The ImgTransformation of each frame carries the ISP chain (sensor mode, scale, crop), so its
    matrix fits the image. Rescaling the EEPROM matrix to the output size instead is ~8 % off in
    focal length on the OAK-D-Lite at 1280x960 (TUTORIAL.md 3.15).
    """
    K = np.array(tf.getIntrinsicMatrix(), float)
    dist = np.array(tf.getDistortionCoefficients(), float)
    naive_fx = info["naive_K"][0][0]
    return K, dist, {
        "source": "factory",
        "source_text": "OAK EEPROM の工場校正値（CAM_A）。フレームの ImgTransformation で使用画像の条件に変換",
        "camera": info.get("camera"), "sensor": info.get("sensor"),
        "has_autofocus": info.get("has_autofocus"),
        "calibrated_at": info["calibrated_at"], "sensor_mode": list(tf.getSourceSize()),
        "image": list(tf.getSize()), "image_matches": list(tf.getSize()) == list(size),
        "K": K.tolist(), "dist_model": str(tf.getDistortionModel()).split(".")[-1],
        "dist_coeffs": dist.tolist(),
        "naive_fx": naive_fx, "naive_fx_error_pct": (naive_fx / K[0, 0] - 1) * 100,
    }


def file_intrinsics(path: Path, size, camera: str | None) -> tuple[np.ndarray, np.ndarray, dict]:
    d = json.loads(path.read_text())
    if list(d["image_size"]) != list(size):
        sys.exit(f"{path}: 校正画像サイズ {d['image_size']} が処理画像 {list(size)} と一致しない")
    if camera and d.get("camera_id") and d["camera_id"] not in camera:
        sys.exit(f"{path}: 別のカメラ {d['camera_id']} の校正値")
    K, dist = np.array(d["K"], float), np.array(d["dist"], float)
    return K, dist, {"source": "file", "source_text": f"ユーザー校正ファイル {path}（RMS {d.get('rms_px')} px）",
                     "camera": camera, "image": list(size), "image_matches": True, "K": K.tolist(),
                     "dist_model": d.get("model", "opencv"), "dist_coeffs": dist.tolist(),
                     "file": str(path), "calibrated_on": d.get("date")}


# --------------------------------------------------------------------------- drawing

def outline(img, corners: np.ndarray, colour, grow: float = 4.0) -> None:
    """Marker outline drawn `grow` px outside the detected corners so the real edge stays visible."""
    c = corners.mean(0)
    d = corners - c
    pts = corners + d / np.linalg.norm(d, axis=1, keepdims=True) * grow * math.sqrt(2)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
    p0 = corners[0] + d[0] / np.linalg.norm(d[0]) * (grow * math.sqrt(2) + 6)
    cv2.circle(img, tuple(np.round(p0).astype(int)), 5, (0, 255, 255), 2, cv2.LINE_AA)  # corner 0


def text(img, s: str, org, scale=0.55, colour=(255, 255, 255)) -> None:
    for c, w in (((0, 0, 0), 3), (colour, 1)):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, w, cv2.LINE_AA)


def draw_marker(img, o: pose.Observation, size: float | None, K, dist) -> None:
    col = STATUS_COLOURS.get(o.status, (200, 200, 200))
    outline(img, o.corners, col)
    if o.T_cam_marker is not None and o.status != "rejected":
        rvec, _ = cv2.Rodrigues(o.T_cam_marker[:3, :3])
        cv2.drawFrameAxes(img, K, dist, rvec, o.T_cam_marker[:3, 3], 0.7 * size, 1)
    x, y = int(o.corners[:, 0].min()), int(o.corners[:, 1].min()) - 12
    label = f"ID {o.id} {o.status}" + ("" if o.status == "ok" else "!")
    text(img, label, (x, max(y, 16)), colour=col)


def draw_block(img, T_cam_obj, size, K, dist, colour, dashed: bool) -> None:
    pts, edges = pose.block_edges(size)
    P = (T_cam_obj[:3, :3] @ pts.T).T + T_cam_obj[:3, 3]
    if (P[:, 2] <= 0.01).any():
        return
    uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    for i, j in edges:
        a, b = uv[i], uv[j]
        if dashed:
            n = 12
            for k in range(0, n, 2):
                p, q = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
                cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), colour, 1, cv2.LINE_AA)
        else:
            cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, 1, cv2.LINE_AA)


def draw_world(img, T_cam_world, K, dist, fixed: bool) -> None:
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    cv2.drawFrameAxes(img, K, dist, rvec, T_cam_world[:3, 3], 0.06, 2)
    uv, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, T_cam_world[:3, 3], K, dist)
    text(img, "world (ID 2) " + ("fixed" if fixed else "tracking"), tuple(np.round(uv.ravel() + [8, 22]).astype(int)))


# --------------------------------------------------------------------------- processing

def T_list(T):
    return None if T is None else np.asarray(T).tolist()


class Vision:
    def __init__(self, config_path: Path, world_path: Path) -> None:
        self.path = config_path
        self.cfg: pose.Config | None = None
        self.cfg_error = None
        self.detector = None
        self.observer = pose.Observer()
        self.world = world.WorldRef(world_path)
        self.lock = threading.Lock()      # guards world commands vs the processing thread
        self.capture: dict | None = None  # hand-eye sample being collected (frames of ID 1 corners)
        self.jig_mode = False             # grip-centre jig: the jig id is NOT a block marker while on
        self.reload()

    def reload(self) -> None:
        try:
            cfg = pose.load_config(self.path)
        except Exception as e:  # keep the last good config; show the problem on the page
            self.cfg_error = f"{self.path.name}: {e}"
            if self.cfg:
                self.cfg.mtime = self.path.stat().st_mtime
            return
        self.cfg, self.cfg_error = cfg, None
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Wider adaptive-threshold windows: a 20 mm marker seen obliquely (~25 px) was missed with
        # the default 3..23 px range. Measured on the saved stills: no extra false ids.
        params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeStep, params.adaptiveThreshWinSizeMax = 3, 4, 53
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
        self.detector = cv2.aruco.ArucoDetector(d, params)
        print(f"loaded {self.path.name}", flush=True)

    def maybe_reload(self) -> None:
        try:
            if self.path.stat().st_mtime != self.cfg.mtime:
                self.reload()
        except FileNotFoundError:
            pass

    def process(self, color, t: float, K, dist, meta: dict):
        cfg = self.cfg.with_jig() if self.jig_mode else self.cfg
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        obs = self.observer.observe(corners, ids, t, cfg, K, dist)
        with self.lock:
            ws = self.world.update(obs, t, cfg, meta)
        T_cam_world = np.array(ws["T_cam_world"]) if ws["valid"] else None
        T_world_cam = None if T_cam_world is None else pose.invert(T_cam_world)
        est = pose.block_estimates(obs, cfg)

        img = color.copy()
        if T_cam_world is not None:
            draw_world(img, T_cam_world, K, dist, ws["mode"] == "fixed")
        for mid, (T, st) in sorted(est.items()):
            draw_block(img, T, cfg.block_size.value, K, dist, BLOCK_COLOURS.get(mid, (200, 200, 200)), st not in ("ok", "pair"))
        for o in obs:
            spec = cfg.markers.get(o.id)
            draw_marker(img, o, spec.size_m if spec else None, K, dist)

        markers = []
        for o in obs:
            spec = cfg.markers.get(o.id)
            markers.append({
                "id": o.id, "role": spec.role if spec else "unknown", "status": o.status,
                "reasons": o.reasons, "t": o.t, "corners_px": o.corners.tolist(), "side_px": o.side_px,
                "size_m": spec.size_m if spec else None,
                "size_status": spec.size.status if spec else "unset",
                "T_cam_marker": T_list(o.T_cam_marker), "T_cam_marker_alt": T_list(o.T_alt),
                "reproj_px": o.reproj_px, "reproj_alt_px": o.reproj_alt_px, "ambiguity": o.ambiguity,
                "T_world_marker": (T_list(T_world_cam @ o.T_cam_marker)
                                   if T_world_cam is not None and o.T_cam_marker is not None else None),
            })
        seen = {o.id for o in obs}
        not_seen = {str(mid): (None if mid not in self.observer.last_seen else t - self.observer.last_seen[mid])
                    for mid in cfg.markers if mid not in seen}
        block = {}
        for mid, (T, st) in est.items():
            block[str(mid)] = {"status": st, "T_cam_obj": T.tolist(),
                               "T_world_obj": T_list(T_world_cam @ T) if T_world_cam is not None else None}
        cmp = None
        if 0 in est and 3 in est:
            cmp = pose.compare(est[0][0], est[3][0])
            cmp["statuses"] = {"0": est[0][1], "3": est[3][1]}
            # "pair" estimates were chosen *because* they agree, so they are not independent evidence.
            cmp["both_ok"] = est[0][1] == "ok" and est[3][1] == "ok"
        state = {
            "t": t, "frames": {"cam": "OpenCV カラーカメラ", "world": "ID 2 マーカー座標（暫定、Z=0 はカード表面）"},
            "config_error": self.cfg_error, "config": cfg.json(), "missing": pose.missing_report(cfg),
            "markers": markers, "not_seen_age_s": not_seen, "world": ws, "block": block,
            "compare_0_3": cmp,
            "jig_mode": {"on": self.jig_mode, "id": None if cfg.grip_jig is None else cfg.grip_jig.id,
                         "size_m": None if cfg.grip_jig is None else cfg.grip_jig.size_m},
            "thresholds": {"max_reproj_px": pose.MAX_REPROJ_PX, "ambiguous_ratio": pose.AMBIGUOUS_RATIO,
                           "flip_deg": pose.FLIP_DEG, "min_side_px": pose.MIN_SIDE_PX},
        }
        if self.capture is not None:
            o1 = next((o for o in obs if o.id == 1), None)
            cap = self.capture
            cap["frames"] += 1
            if o1 is not None and o1.status != "rejected":
                cap["corners"].append(o1.corners.copy())
                cap["status"].append(o1.status)
            for o in obs:
                if o.status != "rejected":
                    cap.setdefault("by_id", {}).setdefault(o.id, []).append(o.corners.copy())
            if T_cam_world is not None:
                cap["world"].append(T_cam_world)
            if cap["frames"] >= cap["n"]:
                cap["done"].set()
        return img, state


# --------------------------------------------------------------------------- serve

class Shared:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state_json = b"{}"
        self.state = {}
        self.raw = None
        self.annotated = None

    def put(self, state, raw, annotated) -> None:
        js = json.dumps(state, ensure_ascii=False, allow_nan=False, default=_nan_safe).encode()
        with self.lock:
            self.state_json, self.state, self.raw, self.annotated = js, state, raw, annotated

    def get(self):
        with self.lock:
            return self.state_json, self.state, self.raw, self.annotated


def _nan_safe(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def clean(o):
    """NaN / inf -> None so that state.json is strict JSON."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="18443", help="OAK device id or name substring")
    ap.add_argument("--config", type=Path, default=HERE / "markers.toml")
    ap.add_argument("--size", type=parse_size, default=(1280, 960), help="recognition image, WxH (4:3)")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--intrinsics", type=Path, help="user calibration JSON (default: factory EEPROM)")
    ap.add_argument("--replay", type=Path, help="recording directory to process instead of the camera")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve the LAN (then --allow-from applies)")
    ap.add_argument("--allow-from", default="127.0.0.1/32",
                    help="comma-separated CIDRs allowed to open the page, e.g. 127.0.0.1/32,192.168.32.0/24")
    ap.add_argument("--arm-control-from", default=None,
                    help="CIDRs allowed to POST arm commands (default: same as --allow-from)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--stream-width", type=int, default=960, help="browser stream width (px)")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--output", type=Path, default=HERE / "output")
    ap.add_argument("--world-file", type=Path, default=HERE / "calib" / "world_ref.json")
    ap.add_argument("--rec-max-mb", type=float, default=2000)
    ap.add_argument("--rec-max-s", type=float, default=120)
    ap.add_argument("--arm-url", default="http://127.0.0.1:8091")
    ap.add_argument("--sim-url", default="http://127.0.0.1:8092")
    ap.add_argument("--seconds", type=float, help="stop automatically after N seconds")
    args = ap.parse_args()

    vision = Vision(args.config, args.world_file if not args.replay else args.output / "replay_world_ref.json")
    shared, latest = Shared(), Latest()
    rec = recorder.Recorder(args.output)
    stop = threading.Event()
    intr: dict = {}
    meta: dict = {}
    notes: list[str] = []

    calib_root = HERE / "calib" / "handeye"

    def arm_state():
        try:
            with urllib.request.urlopen(args.arm_url + "/state.json", timeout=1) as r:
                return json.loads(r.read())
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handeye_api(cmd: str, body: dict) -> dict:
        """Sample capture / solve / adopt for the eye-to-hand calibration. Never moves the arm."""
        import handeye
        session = body.get("session") or time.strftime("%Y%m%d")
        d = calib_root / session
        if cmd == "sessions":
            out = []
            for sd in sorted(calib_root.iterdir()) if calib_root.exists() else []:
                n = len((sd / "samples.jsonl").read_text().splitlines()) if (sd / "samples.jsonl").exists() else 0
                r = json.loads((sd / "result.json").read_text()) if (sd / "result.json").exists() else None
                out.append({"session": sd.name, "samples": n, "result": None if r is None else
                            {k: r.get(k) for k in ("ok", "message", "n_calib", "n_eval", "diversity", "sd", "weighted_rms")}
                            | {"holdout": {k: v for k, v in r.get("holdout", {}).items() if k != "rows"},
                               "fit": {k: v for k, v in r.get("fit", {}).items() if k != "rows"}}})
            active = HERE / "calib" / "handeye_active.json"
            return {"sessions": out, "active": json.loads(active.read_text()) if active.exists() else None}
        if cmd == "capture":
            a0 = arm_state()
            if not a0.get("ok") or not a0.get("fk"):
                return {"ok": False, "message": "関節角が読めない: " + str(a0.get("error", "ok でない"))}
            if vision.capture is not None:
                return {"ok": False, "message": "記録中"}
            n = int(body.get("frames", 20))
            cap = {"n": n, "frames": 0, "corners": [], "status": [], "world": [], "done": threading.Event()}
            vision.capture = cap
            ok = cap["done"].wait(n / 10 + 5)
            vision.capture = None
            a1 = arm_state()
            if not ok:
                return {"ok": False, "message": "フレームが届かない"}
            q0 = np.array([j["pos"] for j in a0["joints"][:6]]); q1 = np.array([j["pos"] for j in a1["joints"][:6]])
            vel = max(abs(j["vel"] or 0) for j in a1["joints"][:6])
            if np.max(np.abs(q1 - q0)) > 0.002 or vel > 0.02:
                return {"ok": False, "message": f"記録中に関節が動いた（{np.max(np.abs(q1 - q0)):.4f} rad、速度 {vel:.3f}）。静止させて再記録"}
            if len(cap["corners"]) < 0.8 * n:
                return {"ok": False, "message": f"ID 1 の検出が {len(cap['corners'])}/{n} フレームしかない"}
            C = np.array(cap["corners"])
            cm = C.mean(0)
            cstd = float(np.sqrt(np.mean(np.sum((C - cm) ** 2, axis=2))))
            spec = vision.cfg.markers[1]
            sols = pose.solve_marker(cm, spec.size_m, np.array(intr["K"]), np.array(intr["dist_coeffs"]))
            d.mkdir(parents=True, exist_ok=True)
            f = d / "samples.jsonl"
            idx = len(f.read_text().splitlines()) if f.exists() else 0
            role = body.get("role") or ("eval" if idx % 4 == 3 else "calib")
            rec = {"id": idx, "role": role, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": body.get("label", ""),
                   "q": q1.tolist(), "T_base_ee": np.vstack([np.column_stack([np.array(a1["fk"]["R"]), a1["fk"]["p"]]), [0, 0, 0, 1]]).tolist(),
                   "fk_model": a1.get("fk_model"), "corners_px": cm.tolist(), "corner_std_px": cstd,
                   "n_frames": len(cap["corners"]), "status_counts": {k: cap["status"].count(k) for k in set(cap["status"])},
                   "status": max(set(cap["status"]), key=cap["status"].count),
                   "T_cam_m1": sols[0][1].tolist(), "T_cam_m1_alt": sols[1][1].tolist() if len(sols) > 1 else None,
                   "marker_size_m": spec.size_m, "K": intr["K"], "dist": intr["dist_coeffs"], "image_size": intr["image"],
                   "T_cam_world": None if not cap["world"] else np.mean(cap["world"], axis=0).tolist()}
            with f.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            return {"ok": True, "message": f"サンプル {idx}（{role}）を記録: ID 1 {len(cap['corners'])} フレーム、角点の揺れ {cstd:.3f} px",
                    "id": idx}
        if cmd == "jig_mode":
            vision.jig_mode = bool(body.get("on"))
            return {"ok": True, "message": ("治具モード ON: ID {} を治具（{:.0f} mm）として扱い、ブロック推定に使わない。ブロックは画角外へ"
                                            .format(vision.cfg.grip_jig.id, vision.cfg.grip_jig.size_m * 1000) if vision.jig_mode
                                            else "治具モード OFF: ID 3 はブロックのマーカーに戻った")}
        if cmd == "grip":
            # ID 1 and the jig marker (ID 4) seen together, static: T_m1_jig, and with the measured
            # jig->grip offset, a candidate T_m1_grip. Written to calib/grip/, not into markers.toml.
            spec1, spec4 = vision.cfg.markers.get(1), vision.cfg.grip_jig
            if spec4 is None or spec4.size_m is None:
                return {"ok": False, "message": "治具マーカーの設定がない（markers.toml [grip_jig]）"}
            if not vision.jig_mode:
                return {"ok": False, "message": f"治具モードを ON にしてから記録する（ID {spec4.id} を治具として扱う。ブロックは画角外へ）"}
            jid = spec4.id
            n = int(body.get("frames", 30))
            cap = {"n": n, "frames": 0, "corners": [], "status": [], "world": [], "done": threading.Event()}
            vision.capture = cap
            cap["done"].wait(n / 10 + 5)
            vision.capture = None
            by = cap.get("by_id", {})
            if len(by.get(1, [])) < 0.8 * n or len(by.get(jid, [])) < 0.8 * n:
                return {"ok": False, "message": f"ID 1 / 治具 ID {jid} の同時検出が不足（{len(by.get(1, []))}, {len(by.get(jid, []))} / {n}）。重複（ブロックの ID {jid} も写っている）も不採用になる"}
            K_, D_ = np.array(intr["K"]), np.array(intr["dist_coeffs"])
            res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": n}
            Ts = {}
            for mid, spec in ((1, spec1), (jid, spec4)):
                C = np.array(by[mid])
                cm = C.mean(0)
                sols = pose.solve_marker(cm, spec.size_m, K_, D_)
                Ts[mid] = sols[0][1]
                res[f"id{mid}"] = {"corner_std_px": float(np.sqrt(np.mean(np.sum((C - cm) ** 2, axis=2)))),
                                   "ambiguity": float(sols[0][0] / sols[1][0]) if len(sols) > 1 and sols[1][0] > 0 else 0.0,
                                   "side_px": float(np.min(np.linalg.norm(cm - np.roll(cm, 1, 0), axis=1)))}
            T_m1_jig = pose.invert(Ts[1]) @ Ts[jid]
            res["jig"] = {"id": jid, "size": spec4.size.json()}
            res["T_m1_jig"] = T_m1_jig.tolist()
            if spec4.grip is not None and spec4.grip.usable:
                res["T_m1_grip"] = (T_m1_jig @ spec4.grip.value).tolist()
                res["T_m1_grip_status"] = f"治具 + 実測オフセット（{spec4.grip.status}）。markers.toml [marker.1.grip] へは確認後に手で記入"
            else:
                res["T_m1_grip"] = None
                res["T_m1_grip_status"] = "治具→把持中心のオフセットが未入力のため T_m1_jig のみ"
            g = HERE / "calib" / "grip"
            g.mkdir(parents=True, exist_ok=True)
            out = g / (time.strftime("%Y%m%d-%H%M%S") + ".json")
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            t = T_m1_jig[:3, 3] * 1000
            return {"ok": True, "message": f"記録: ID1→治具 ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm。{res['T_m1_grip_status']}（{out.name}）"}
        if cmd == "solve":
            if not (d / "samples.jsonl").exists():
                return {"ok": False, "message": "サンプルがない"}
            res = handeye.solve(handeye.load_samples(d))
            (d / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
            return {"ok": res.get("ok", False), "message": res.get("message"), "session": session}
        if cmd == "adopt":
            r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
            if not r or not r.get("ok"):
                return {"ok": False, "message": "評価に合格した結果がないため採用しない"}
            act = dict(r, session=session, adopted_at=time.strftime("%Y-%m-%d %H:%M:%S"), status="calibrated",
                       note="eye-to-hand: T_cam_base と T_ee_m1。たわみ・関節零点誤差は残差に含まれる。把持中心は別校正。")
            (HERE / "calib" / "handeye_active.json").write_text(json.dumps(act, indent=1, ensure_ascii=False))
            return {"ok": True, "message": f"採用した（{session}）。シミュレーターは観測ブロックを校正済み配置で置く"}
        return {"ok": False, "message": f"unknown {cmd}"}

    if args.replay:
        src = recorder.ReplaySource(args.replay, loop=args.loop)
        m = src.meta
        K, dist = np.array(m["intrinsics"]["K"]), np.array(m["intrinsics"]["dist_coeffs"])
        intr.update(m["intrinsics"], replay=str(args.replay))
        size = tuple(m["image_size"])
        meta.update(intrinsics_source=intr["source"], image_size=list(size), camera=intr.get("camera"))
        source = None
    else:
        source = camera_source(args.device, args.size, args.fps)
        info = next(source)
        print(info["camera"], flush=True)
        size = args.size
        K = dist = None

    def capture() -> None:
        nonlocal K, dist
        fps = Ticker()
        try:
            if args.replay:
                frames = ((f, r["t"], r) for f, r in src)
            else:
                frames = ((f, t, tf) for f, t, tf in source)
            for color, t, extra in frames:
                if K is None:
                    if args.intrinsics:
                        K, dist, i = file_intrinsics(args.intrinsics, size, info["camera"])
                    else:
                        K, dist, i = factory_intrinsics(extra, size, info)
                        if not i["image_matches"]:
                            raise RuntimeError(f"ImgTransformation のサイズ {i['image']} が要求 {list(size)} と違う")
                    intr.update(i)
                    meta.update(intrinsics_source=i["source"], image_size=list(size), camera=i.get("camera"))
                    print(f"intrinsics: {i['source']} fx={K[0, 0]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}", flush=True)
                vision.maybe_reload()
                t_proc = time.time()
                img, st = vision.process(color, t, K, dist, meta)
                st["fps"] = fps.tick()
                st["latency_s"] = time.time() - t
                st["proc_ms"] = (time.time() - t_proc) * 1000
                st["intrinsics"] = intr
                st["source"] = "replay" if args.replay else "camera"
                if args.replay:
                    st["replay"] = {"dir": str(args.replay), "recorded_markers": extra.get("markers_brief")}
                st["recorder"] = rec.status()
                st["notes"] = notes[-5:]
                st = clean(st)
                h = round(img.shape[0] * args.stream_width / img.shape[1])
                small = cv2.resize(img, (args.stream_width, h), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    latest.put(buf.tobytes())
                shared.put(st, color, img)
                if not args.replay:
                    rec.push(color, {"t": t, "markers_brief": [
                        {"id": m["id"], "status": m["status"], "T_cam_marker": m["T_cam_marker"],
                         "corners_px": m["corners_px"]} for m in st["markers"]],
                        "world_valid": st["world"]["valid"]})
                if stop.is_set():
                    break
        except BaseException as exc:
            latest.fail(exc)
            raise
        finally:
            rec.stop("終了")
            if source is not None:
                source.close()   # releases the board (TUTORIAL.md 3.4)

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    static = HERE / "static"

    def proxy(handler, base: str, path: str, method: str, body: bytes | None) -> None:
        stream = path.split("?")[0].endswith(".mjpg")
        try:
            hdr = {"X-Origin-Page": "vision-web-ui", "X-Forwarded-For": handler.client_address[0]}
            if body:
                hdr["Content-Type"] = "application/json"
            req = urllib.request.Request(base + path, data=body, method=method, headers=hdr)
            with urllib.request.urlopen(req, timeout=10 if stream else 3) as r:
                ctype = r.headers.get("Content-Type", "application/octet-stream")
                if stream:
                    handler.send_response(200)
                    handler.send_header("Content-Type", ctype)
                    handler.send_header("Cache-Control", "no-store")
                    handler.end_headers()
                    while not stop.is_set():
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        handler.wfile.write(chunk)
                    return
                return handler._send(r.read(), ctype)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            body = json.dumps({"available": False, "error": f"{base} に接続できない: {e}"}, ensure_ascii=False)
            return handler._send(body.encode(), "application/json", 200)
        except (BrokenPipeError, ConnectionResetError):
            return

    import ipaddress
    allow = [ipaddress.ip_network(c.strip()) for c in args.allow_from.split(",") if c.strip()]
    arm_allow = ([ipaddress.ip_network(c.strip()) for c in args.arm_control_from.split(",") if c.strip()]
                 if args.arm_control_from else allow)

    def allowed(addr: str, nets) -> bool:
        a = ipaddress.ip_address(addr)
        return any(a in n for n in nets)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def handle_one_request(self):
            # Every request, page or API, must come from an allowed address.
            if not allowed(self.client_address[0], allow):
                self.raw_requestline = self.rfile.readline(65537)
                self.close_connection = True
                if self.raw_requestline and self.parse_request():
                    self.send_error(403, "not in --allow-from")
                return
            return super().handle_one_request()

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(clean(obj), ensure_ascii=False).encode(), "application/json", code)

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            if path.startswith("/arm/") and not allowed(self.client_address[0], arm_allow):
                return self._json({"ok": False, "message": f"{self.client_address[0]} からの実機操作は許可されていない（--arm-control-from）"}, 403)
            for prefix, base in (("/arm/", args.arm_url), ("/sim/", args.sim_url)):
                if path.startswith(prefix):
                    return proxy(self, base, path[len(prefix) - 1:], "POST", body or b"{}")
            if path.startswith("/api/handeye/"):
                return self._json(handeye_api(path.rsplit("/", 1)[-1], json.loads(body or b"{}")))
            if path == "/api/snapshot":
                _, st, raw, ann = shared.get()
                if raw is None:
                    return self._json({"ok": False, "message": "まだフレームがない"}, 503)
                d = recorder.save_snapshot(args.output, raw, ann, st)
                notes.append(f"静止画を保存: {d}")
                return self._json({"ok": True, "message": f"保存: {d}", "dir": str(d.relative_to(args.output))})
            if path == "/api/record/start":
                if args.replay:
                    return self._json({"ok": False, "message": "再生中は録画しない"})
                m = dict(meta, intrinsics=intr, image_size=list(size), config_text=args.config.read_text(),
                         started=time.strftime("%Y-%m-%d %H:%M:%S"))
                msg = rec.start(m, tuple(size), args.fps, args.rec_max_mb, args.rec_max_s)
                notes.append(msg)
                return self._json({"ok": True, "message": msg})
            if path == "/api/record/stop":
                rec.stop()
                return self._json({"ok": True, "message": "録画停止"})
            if path in ("/api/world/register", "/api/world/release", "/api/world/use_fixed"):
                with vision.lock:
                    {"/api/world/register": vision.world.start_register,
                     "/api/world/release": vision.world.release,
                     "/api/world/use_fixed": vision.world.use_fixed}[path]()
                    msg = vision.world.message
                return self._json({"ok": True, "message": msg})
            return self.send_error(404)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            for prefix, base in (("/arm/", args.arm_url), ("/sim/", args.sim_url)):
                if path.startswith(prefix):
                    return proxy(self, base, self.path[len(prefix) - 1:], "GET", None)
            if path == "/":
                return self._send((static / "vision.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                f = (static / path[len("/static/"):]).resolve()
                if static.resolve() not in f.parents or not f.is_file():
                    return self.send_error(404)
                ctype = {".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8"}.get(
                    f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            if path == "/state.json":
                return self._send(shared.get()[0], "application/json")
            if path == "/api/handeye/sessions":
                return self._json(handeye_api("sessions", {}))
            if path == "/api/files":
                out = []
                for kind in ("snapshots", "recordings"):
                    d = args.output / kind
                    for p in sorted(d.iterdir(), reverse=True)[:20] if d.exists() else []:
                        out.append({"kind": kind, "name": p.name,
                                    "files": sorted(x.name for x in p.iterdir())})
                return self._json(out)
            if path.startswith("/files/"):
                f = (args.output / path[len("/files/"):]).resolve()
                if args.output.resolve() not in f.parents or not f.is_file():
                    return self.send_error(404)
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
                         ".jsonl": "text/plain"}.get(f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            if path == "/raw.png":
                raw = shared.get()[2]
                if raw is None:
                    return self.send_error(503, "no frame yet")
                return self._send(cv2.imencode(".png", raw)[1].tobytes(), "image/png")
            if path == "/annotated.png":
                ann = shared.get()[3]
                if ann is None:
                    return self.send_error(503, "no frame yet")
                return self._send(cv2.imencode(".png", ann)[1].tobytes(), "image/png")
            if path == "/snapshot.jpg":
                jpeg, _ = latest.wait(-1)
                if jpeg is None:
                    return self.send_error(503, "no frame yet")
                return self._send(jpeg, "image/jpeg")
            if path != "/stream.mjpg":
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while not stop.is_set():
                    jpeg, seq = latest.wait(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    announce(args.host, server.server_port)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        if args.seconds:
            threading.Timer(args.seconds, server.shutdown).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
