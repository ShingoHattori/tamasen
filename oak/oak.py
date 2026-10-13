#!/usr/bin/env python3
"""OAK camera CLI: list the connected boards, or open one and stream colour + depth to a browser.

    python oak.py list
    python oak.py serve --device lite
    python oak.py serve --device oak-d --port 8080 --layout stack --max-mm 4000
    python oak.py serve --device lite --codes          # overlay decoded QR codes / ArUco markers

There is no native window: everything is served over HTTP, which also works over SSH and
sidesteps macOS screen permissions. Ctrl-C stops the server and releases the board.
"""
import argparse
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import depthai as dai

from codes import CodeDetector, CodeTracker, aruco_dictionaries, draw
from oak_common import available, describe, mono_size, open_device, pick_device

SIZE = (1280, 720)
FPS = 30
WARMUP_FRAMES = 30  # auto-exposure / white balance need this long to settle


# --------------------------------------------------------------------------- list

def cmd_list(args: argparse.Namespace) -> int:
    infos = available()
    if not infos:
        print("No OAK device found. Check the USB3 cable and try replugging.")
        return 1

    print(f"{len(infos)} device(s):\n")
    for i, info in enumerate(infos, 1):
        # Product name and sensor layout are only readable once the board is booted, so the
        # listing has to open each one. A board still rebooting from a previous run is skipped.
        try:
            with open_device(info) as dev:
                print(f"{i}. {dev.getDeviceName()}")
                print(f"   id       {dev.getDeviceId()}")
                print(f"   platform {dev.getPlatformAsString()}   usb {dev.getUsbSpeed()}")
                for f in dev.getConnectedCameraFeatures():
                    kinds = "/".join(str(t).split(".")[-1] for t in f.supportedTypes)
                    print(f"   {str(f.socket).split('.')[-1]:6s} {f.sensorName:8s} "
                          f"{f.width}x{f.height}  {kinds}")
                print(f"   stream it:  python oak.py serve --device {dev.getDeviceId()[:5]}")
        except RuntimeError as e:
            print(f"{i}. {info.getDeviceId()}  [could not open: {e}]")
        print()
    return 0


# --------------------------------------------------------------------------- capture

def colorise(depth: np.ndarray, min_mm: int, max_mm: int) -> np.ndarray:
    norm = np.clip((depth.astype(np.float32) - min_mm) / max(max_mm - min_mm, 1), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth == 0] = 0  # no disparity match
    return vis


def label(img: np.ndarray, text: str) -> np.ndarray:
    for colour, width in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, width, cv2.LINE_AA)
    return img


def frames(device: dai.Device, warmup: int = WARMUP_FRAMES):
    """Yield (colour BGR, depth uint16 mm) pairs until the caller stops iterating.

    Close this generator when you are done — see TUTORIAL.md 3.4. Letting it be garbage
    collected instead leaves the board claimed and can wedge its firmware.
    """
    mono = mono_size(device)
    print(f"mono {mono[0]}x{mono[1]} -> depth {SIZE[0]}x{SIZE[1]}", flush=True)

    with dai.Pipeline(device) as pipeline:
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        mono_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=mono_left.requestOutput(mono, fps=FPS),
            right=mono_right.requestOutput(mono, fps=FPS),
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(*SIZE)  # mandatory alongside setDepthAlign, or nothing is emitted
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        color_q = cam_rgb.requestOutput(SIZE, dai.ImgFrame.Type.BGR888i, fps=FPS).createOutputQueue(
            maxSize=4, blocking=False)
        depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        for _ in range(warmup):
            color_q.get(), depth_q.get()

        while True:
            yield color_q.get().getCvFrame(), depth_q.get().getFrame()


def compose(color, depth, args, fps_txt="", hits=()):
    depth_vis = colorise(depth, args.min_mm, args.max_mm)
    valid = float((depth > 0).mean()) * 100
    left = label(draw(color.copy(), list(hits)), "colour")
    right = label(depth_vis, f"depth {args.min_mm}-{args.max_mm}mm  valid {valid:.0f}%  {fps_txt}")
    canvas = np.vstack([left, right]) if args.layout == "stack" else np.hstack([left, right])
    if args.scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=args.scale, fy=args.scale)
    return canvas


class Ticker:
    """Rolling frames-per-second, recomputed once a second."""

    def __init__(self) -> None:
        self.n, self.start, self.text = 0, time.monotonic(), ""

    def tick(self) -> str:
        self.n += 1
        elapsed = time.monotonic() - self.start
        if elapsed >= 1.0:
            self.text = f"{self.n / elapsed:.0f} fps"
            self.n, self.start = 0, time.monotonic()
        return self.text


# --------------------------------------------------------------------------- serve

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>OAK {name}</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; background:#111; color:#eee; font:14px/1.5 system-ui,sans-serif;
        display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }}
 img  {{ max-width:100%; height:auto; border-radius:6px; background:#000; }}
 header {{ display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }}
 code {{ color:#9cf; }}
 a {{ color:#9cf; }}
</style></head><body>
<header><strong>OAK {name}</strong><code>{id}</code>
<span>colour {layout} depth &middot; {min_mm}-{max_mm} mm</span></header>
<img src="/stream.mjpg" alt="OAK live stream">
<div><a href="/snapshot.jpg" download>snapshot.jpg</a></div>
</body></html>"""


class Latest:
    """Most recent encoded JPEG, shared between the capture thread and the HTTP handlers."""

    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._cond = threading.Condition()
        self._seq = 0
        self.error: BaseException | None = None

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg, self._seq = jpeg, self._seq + 1
            self._cond.notify_all()

    def wait(self, seq: int, timeout: float = 10.0):
        """Block until a frame newer than `seq` exists; returns (jpeg, seq) or (None, seq)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != seq or self.error, timeout):
                return None, seq
            if self.error:
                raise self.error
            return self._jpeg, self._seq

    def fail(self, exc: BaseException) -> None:
        with self._cond:
            self.error = exc
            self._cond.notify_all()


def lan_address() -> str | None:
    """Best-guess address other machines can reach, without sending anything."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 1))  # TEST-NET-1: routable-looking, never actually contacted
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def announce(host: str, port: int) -> None:
    urls = [f"http://localhost:{port}/"] if host in ("", "0.0.0.0", "127.0.0.1") else []
    if host in ("", "0.0.0.0"):
        ip = lan_address()
        if ip:
            urls.append(f"http://{ip}:{port}/   (from other machines)")
    elif host != "127.0.0.1":
        urls.append(f"http://{host}:{port}/")

    bar = "=" * 58
    print(f"\n{bar}\n  Open this in your browser:\n")
    for u in urls:
        print(f"      {u}")
    print(f"\n  Ctrl-C to stop.\n{bar}\n", flush=True)


def cmd_serve(args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    print(describe(device))
    name, dev_id = device.getDeviceName(), device.getDeviceId()

    latest = Latest()
    stop = threading.Event()
    source = frames(device)
    tracker = CodeTracker(CodeDetector(args.aruco_dict)) if args.codes else None
    if tracker:
        print(f"code overlay on: QR + ArUco DICT_{args.aruco_dict}", flush=True)

    def capture() -> None:
        try:
            fps = Ticker()
            for color, depth in source:
                hits = ()
                if tracker:
                    tracker.submit(color)
                    hits = tracker.current()
                canvas = compose(color, depth, args, fps.tick(), hits)
                ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if ok:
                    latest.put(buf.tobytes())
                if stop.is_set():
                    break
        except BaseException as exc:  # surfaced to every waiting HTTP client
            latest.fail(exc)
            raise
        finally:
            source.close()  # runs the pipeline's __exit__, releasing the board
            if tracker:
                tracker.close()

    worker = threading.Thread(target=capture, daemon=True)
    worker.start()

    page = PAGE.format(name=name, id=dev_id, min_mm=args.min_mm, max_mm=args.max_mm,
                       layout="above" if args.layout == "stack" else "|")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console readable
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
                return self._send(page.encode(), "text/html; charset=utf-8")
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
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode() +
                                     b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser navigated away

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    announce(args.host, server.server_port)

    # SIGTERM must shut down like Ctrl-C, or the board's firmware is left wedged.
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
        worker.join(timeout=10)  # let the capture thread close the device before we exit
    return 0


# --------------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list connected OAK devices and their sensors")

    sv = sub.add_parser("serve", help="open a device and stream colour + depth to a browser")
    sv.add_argument("--device", help="id or product-name substring, e.g. 'lite' or '14442'")
    sv.add_argument("--layout", choices=("side", "stack"), default="side")
    sv.add_argument("--scale", type=float, default=0.6, help="output scale (default 0.6)")
    sv.add_argument("--min-mm", type=int, default=300, dest="min_mm")
    sv.add_argument("--max-mm", type=int, default=6000, dest="max_mm")
    sv.add_argument("--seconds", type=float, help="stop automatically after N seconds")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines")
    sv.add_argument("--codes", action="store_true",
                    help="detect QR codes and ArUco markers and overlay them on the colour view")
    sv.add_argument("--aruco-dict", default="4X4_50", dest="aruco_dict", type=str.upper,
                    choices=aruco_dictionaries(), metavar="NAME",
                    help="ArUco dictionary for --codes, e.g. 4X4_50 (default), 6X6_250, "
                         "APRILTAG_36H11")
    sv.add_argument("--quality", type=int, default=80, help="JPEG quality (default 80)")

    args = ap.parse_args()
    return {"list": cmd_list, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
