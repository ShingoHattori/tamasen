#!/usr/bin/env python3
"""Live ArUco check for the block and gripper markers: annotated video + 3D view in a browser.

    python vision.py --device 18443
    python vision.py --device 18443 --host 0.0.0.0 --port 8090

Shows, in the colour camera's frame only (no table / arm-base frame — not calibrated yet):
  * marker corners, ids and axes on the live image, and the registered block projected onto it
  * every observed marker pose in 3D, the block (once its mounts are registered in markers.toml)
    and the block poses from ID 0 and ID 3 separately, with their disagreement when both are seen
  * the ID 1 (gripper) pose, and whether its grip-centre transform has been calibrated
  * where the intrinsics come from and whether they match the image actually being processed

Sends nothing to the arm. markers.toml is re-read whenever it changes.
"""
import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

import pose
from oak import Latest, Ticker, announce
from oak_common import describe, mono_size, pick_device

HERE = Path(__file__).resolve().parent
SIZE = (1280, 960)   # 4:3 like the IMX214, so the full field of view is kept
FPS = 30
WARMUP_FRAMES = 30

BLOCK_COLOURS = {0: (255, 255, 0), 3: (255, 0, 255)}   # BGR: estimate from ID 0 cyan, ID 3 magenta
MARKER_COLOUR = (0, 140, 255)


# --------------------------------------------------------------------------- capture

def frames(device: dai.Device):
    """Yield (colour BGR, depth uint16 mm aligned to colour, ImgTransformation of the colour frame).

    Close the generator when done (TUTORIAL.md 3.4).
    """
    mono = mono_size(device)
    with dai.Pipeline(device) as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left.requestOutput(mono, fps=FPS), right=right.requestOutput(mono, fps=FPS),
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*SIZE)  # mandatory alongside setDepthAlign (TUTORIAL.md 3.1)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        color_q = cam.requestOutput(SIZE, dai.ImgFrame.Type.BGR888i, fps=FPS).createOutputQueue(
            maxSize=4, blocking=False)
        depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()
        for _ in range(WARMUP_FRAMES):
            color_q.get(), depth_q.get()
        while True:
            c = color_q.get()
            yield c.getCvFrame(), depth_q.get().getFrame(), c.getTransformation()


def intrinsics_report(device: dai.Device, tf) -> tuple[np.ndarray, np.ndarray, dict]:
    """Intrinsics of the frames actually processed, plus what a naive rescale would have given.

    The factory calibration is stored for one resolution; the ISP output is a binned, scaled and
    cropped version of it. ImgTransformation carries that chain, so its matrix is the one that
    fits the image. Rescaling the calibration to the output size instead is ~10 % off in focal
    length on the OAK-D-Lite at 1280x720 (989 vs 903 px), i.e. ~10 % error in every distance.
    """
    cal = device.readCalibration()
    sock = dai.CameraBoardSocket.CAM_A
    K = np.array(tf.getIntrinsicMatrix(), float)
    dist = np.array(tf.getDistortionCoefficients(), float)
    K_naive = np.array(cal.getCameraIntrinsics(sock, *SIZE), float)
    _, cw, ch = cal.getDefaultIntrinsics(sock)
    info = {
        "source": "OAK EEPROM 工場校正 (CAM_A)、フレームの ImgTransformation で使用画像に変換",
        "device": describe(device),
        "calibrated_at": [cw, ch],
        "sensor_mode": list(tf.getSourceSize()),
        "image": list(tf.getSize()),
        "K": K.round(2).tolist(),
        "dist_model": str(tf.getDistortionModel()).split(".")[-1],
        "dist_coeffs": dist.round(5).tolist(),
        "naive_fx": round(float(K_naive[0, 0]), 2),
        "naive_fx_error_pct": round(float((K_naive[0, 0] / K[0, 0] - 1) * 100), 1),
        "image_matches": list(tf.getSize()) == list(SIZE),
    }
    return K, dist, info


# --------------------------------------------------------------------------- per frame

class Vision:
    def __init__(self, config_path: Path) -> None:
        self.path = config_path
        self.cfg = None
        self.cfg_error = None
        self.detector = None
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
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{cfg.dictionary}"))
        self.detector = cv2.aruco.ArucoDetector(d, params)
        print(f"loaded {self.path.name}", flush=True)

    def maybe_reload(self) -> None:
        try:
            if self.path.stat().st_mtime != self.cfg.mtime:
                self.reload()
        except FileNotFoundError:
            pass

    def process(self, color, depth, K, dist):
        cfg = self.cfg
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        obs = pose.observe(corners, ids.ravel(), cfg, K, dist, depth) if ids is not None else []
        est = pose.block_estimates(obs, cfg)
        img = color.copy()
        for mid, T in sorted(est.items()):
            draw_block(img, T, cfg.block_size, K, dist, BLOCK_COLOURS.get(mid, (200, 200, 200)))
        for o in obs:
            draw_marker(img, o, cfg, K, dist)
        return img, self.state(obs, est)

    def state(self, obs, est) -> dict:
        cfg = self.cfg
        markers = []
        for o in obs:
            spec = cfg.markers.get(o.id)
            T = o.T_cam_marker.copy()
            size, provisional = (spec.size_mm if spec else None), False
            if not o.size_known:
                # Rotation is exact without the size, translation is not. For the 3D view only,
                # scale by the stereo-depth size estimate and mark it provisional.
                if o.size_from_depth_mm:
                    size, provisional = o.size_from_depth_mm, True
                    T[:3, 3] *= size
                else:
                    size, T = None, None
            markers.append({
                "id": o.id, "role": spec.role if spec else "unknown",
                "registered": spec is not None,
                "size_mm": size, "size_registered": o.size_known, "provisional": provisional,
                "T": None if T is None else T.round(4).tolist(),
                "distance_mm": None if T is None else round(float(np.linalg.norm(T[:3, 3])), 1),
                "reproj_px": round(o.reproj_px, 3), "ambiguity": round(o.ambiguity, 3),
                "depth_mm": None if o.depth_mm is None else round(o.depth_mm, 1),
                "pnp_z_mm": round(float(o.T_cam_marker[2, 3]), 1) if o.size_known else None,
                "depth_over_pnp": (round(o.depth_mm / o.T_cam_marker[2, 3], 4)
                                   if o.size_known and o.depth_mm else None),
                "size_from_depth_mm": (None if o.size_from_depth_mm is None
                                       else round(o.size_from_depth_mm, 1)),
            })
        cmp = pose.compare(est[0], est[3]) if 0 in est and 3 in est else None
        mounts = {str(mid): s.T_obj_marker.round(4).tolist()
                  for mid, s in cfg.markers.items() if s.T_obj_marker is not None}
        grip = cfg.markers.get(1)
        return {
            "frame": "camera (OpenCV: x 右, y 下, z 前方, mm)",
            "config_error": self.cfg_error,
            "missing": pose.missing_report(cfg),
            "block_size": cfg.block_size.tolist(),
            "mounts": mounts,
            "sizes": {str(m): s.size_mm for m, s in cfg.markers.items()},
            "markers": markers,
            "block": {str(k): v.round(4).tolist() for k, v in est.items()},
            "compare_0_3": cmp,
            "grip_calibrated": bool(grip and grip.grip_T is not None),
            "grip_T": None if not grip or grip.grip_T is None else grip.grip_T.tolist(),
        }


def draw_marker(img, o: pose.Observation, cfg, K, dist) -> None:
    spec = cfg.markers.get(o.id)
    pts = o.corners.astype(np.int32)
    cv2.polylines(img, [pts], True, MARKER_COLOUR, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(int(v) for v in p), 6 if i == 0 else 3, (0, 255, 255) if i == 0 else MARKER_COLOUR, -1, cv2.LINE_AA)
    size = spec.size_mm if spec and spec.size_mm else 1.0
    rvec, _ = cv2.Rodrigues(o.T_cam_marker[:3, :3])
    # Projection is scale-invariant, so the axes are right even while the size is unregistered.
    cv2.drawFrameAxes(img, K, dist, rvec, o.T_cam_marker[:3, 3], size * 0.7, 2)
    note = "" if o.size_known else " size?"
    if spec is None:
        note += " unregistered"
    text = f"ID {o.id}{note}"
    x, y = int(o.corners[:, 0].min()), int(o.corners[:, 1].min()) - 10
    for colour, w in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (x, max(y, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, w, cv2.LINE_AA)


def draw_block(img, T_cam_obj, size, K, dist, colour) -> None:
    pts, edges = pose.block_edges(size)
    P = (T_cam_obj[:3, :3] @ pts.T).T + T_cam_obj[:3, 3]
    if (P[:, 2] <= 1).any():
        return
    uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2).astype(np.int32)
    for i, j in edges:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), colour, 2, cv2.LINE_AA)


def banner(img, text: str) -> None:
    for colour, w in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, w, cv2.LINE_AA)


# --------------------------------------------------------------------------- serve

class LatestState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.data = b"{}"
        self.raw = None

    def put(self, state: dict, raw: np.ndarray) -> None:
        with self._lock:
            self.data, self.raw = json.dumps(state, ensure_ascii=False).encode(), raw

    def get(self):
        with self._lock:
            return self.data, self.raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="id or product-name substring, e.g. '18443'")
    ap.add_argument("--config", type=Path, default=HERE / "markers.toml")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--seconds", type=float, help="stop automatically after N seconds")
    args = ap.parse_args()

    vision = Vision(args.config)
    device = pick_device(args.device)
    print(describe(device), flush=True)

    latest, state = Latest(), LatestState()
    stop = threading.Event()
    source = frames(device)
    intr = {}

    def capture() -> None:
        try:
            fps = Ticker()
            K = dist = None
            for color, depth, tf in source:
                if K is None:
                    K, dist, info = intrinsics_report(device, tf)
                    intr.update(info)
                    print(f"intrinsics fx={K[0, 0]:.2f} (naive rescale {info['naive_fx']}, "
                          f"{info['naive_fx_error_pct']:+.1f} %)", flush=True)
                vision.maybe_reload()
                img, st = vision.process(color, depth, K, dist)
                st["fps"] = fps.tick()
                st["intrinsics"] = intr
                st["time"] = time.time()
                banner(img, f"camera frame only | {st['fps']}")
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    latest.put(buf.tobytes())
                state.put(st, color)
                if stop.is_set():
                    break
        except BaseException as exc:
            latest.fail(exc)
            raise
        finally:
            source.close()

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    page = (HERE / "vision.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(page, "text/html; charset=utf-8")
            if path == "/state.json":
                return self._send(state.get()[0], "application/json")
            if path in ("/snapshot.jpg", "/raw.jpg"):
                if path == "/raw.jpg":
                    raw = state.get()[1]
                    if raw is None:
                        return self.send_error(503, "no frame yet")
                    return self._send(cv2.imencode(".png", raw)[1].tobytes(), "image/png")
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
