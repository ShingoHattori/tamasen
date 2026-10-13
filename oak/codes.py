"""QR-code and ArUco-marker detection for `oak.py serve --codes`.

Detection runs on its own thread against the newest colour frame, so a slow decode lowers
how often the overlay refreshes but never the stream's frame rate. Each hit is kept on screen
for HOLD_SECONDS so a code that decodes on every other frame does not flicker.

QR codes are tried both as-is and inverted: OpenCV's QRCodeDetector only reads dark modules
on a light background, and some labels (e.g. white-on-black) are printed the other way round.
"""
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

HOLD_SECONDS = 0.5
ALPHA = 0.3  # fill opacity of the overlay

QR_COLOUR = (0, 200, 0)      # BGR
ARUCO_COLOUR = (255, 140, 0)


@dataclass
class Hit:
    kind: str           # "QR" or "ArUco"
    text: str
    corners: np.ndarray  # (4, 2) float32, in the frame's pixel coordinates
    seen: float          # time.monotonic() of the last detection


def aruco_dictionaries() -> list[str]:
    """Names accepted by --aruco-dict, e.g. '4X4_50' or 'APRILTAG_36H11'."""
    return sorted({n[len("DICT_"):].upper() for n in dir(cv2.aruco) if n.startswith("DICT_")})


class CodeDetector:
    def __init__(self, aruco_dict: str = "4X4_50") -> None:
        self.qr = cv2.QRCodeDetector()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{aruco_dict}"))
        self.aruco = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.aruco_name = aruco_dict

    def detect(self, bgr: np.ndarray) -> list[Hit]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        hits = []

        for img in (gray, 255 - gray):
            ok, texts, points, _ = self.qr.detectAndDecodeMulti(img)
            if not ok:
                continue
            for text, pts in zip(texts, points):
                # Located but not decoded comes back as "": only report what was actually read.
                if text and not any(h.text == text for h in hits):
                    hits.append(Hit("QR", text, pts.reshape(4, 2).astype(np.float32), now))

        corners, ids, _ = self.aruco.detectMarkers(gray)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                hits.append(Hit("ArUco", str(int(i)), c.reshape(4, 2), now))
        return hits


class CodeTracker:
    """Background detector: feed it frames with submit(), read results with current()."""

    def __init__(self, detector: CodeDetector) -> None:
        self.detector = detector
        self._frame: np.ndarray | None = None
        self._hits: list[Hit] = []
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, bgr: np.ndarray) -> None:
        with self._cond:
            self._frame = bgr  # overwrite: only the newest frame is worth decoding
            self._cond.notify()

    def current(self) -> list[Hit]:
        cutoff = time.monotonic() - HOLD_SECONDS
        with self._cond:
            self._hits = [h for h in self._hits if h.seen >= cutoff]
            return list(self._hits)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        last_printed: set[tuple[str, str]] = set()
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._frame is not None or self._stop)
                if self._stop:
                    return
                frame, self._frame = self._frame, None
            hits = self.detector.detect(frame)
            with self._cond:
                # Several markers can share an id, so a held hit is identified by what it says
                # AND where it is: a new hit replaces only the old one it overlaps.
                self._hits = [old for old in self._hits
                              if not any(_same(old, new) for new in hits)] + hits
            keys = {(h.kind, h.text) for h in hits}
            for kind, text in keys - last_printed:
                print(f"[{kind}] {text}", flush=True)
            last_printed = keys


def _same(a: Hit, b: Hit) -> bool:
    if (a.kind, a.text) != (b.kind, b.text):
        return False
    size = np.linalg.norm(b.corners[0] - b.corners[2])  # diagonal
    return np.linalg.norm(a.corners.mean(0) - b.corners.mean(0)) < size


def draw(img: np.ndarray, hits: list[Hit]) -> np.ndarray:
    """Translucent filled box + outline + label for each hit, drawn in place on `img`."""
    if not hits:
        return img
    fill = img.copy()
    for h in hits:
        colour = QR_COLOUR if h.kind == "QR" else ARUCO_COLOUR
        cv2.fillPoly(fill, [h.corners.astype(np.int32)], colour)
    cv2.addWeighted(fill, ALPHA, img, 1 - ALPHA, 0, dst=img)

    for h in hits:
        colour = QR_COLOUR if h.kind == "QR" else ARUCO_COLOUR
        pts = h.corners.astype(np.int32)
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)
        if h.kind == "ArUco":
            # Markers are often packed tightly, so a short tag centred on the marker
            # (the dictionary is fixed by --aruco-dict and printed once at startup).
            text = f"id {h.text}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cx, cy = h.corners.mean(0)
            x, y = int(cx - tw / 2 - 4), int(cy + th / 2)
        else:
            text = f"QR: {h.text}"
            if len(text) > 48:
                text = text[:45] + "..."
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x = int(np.clip(pts[:, 0].min(), 0, max(img.shape[1] - tw - 8, 0)))
            y = max(int(pts[:, 1].min()) - 6, th + base + 6)
        cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y + 2), colour, -1)
        cv2.putText(img, text, (x + 4, y - base), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img
