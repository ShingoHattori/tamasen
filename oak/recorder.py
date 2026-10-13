"""Stills, recordings and replay for `vision.py`. Nothing is recorded unless asked.

Still (snapshot):  output/snapshots/<time>/
    raw.png         the processed colour frame, untouched (lossless)
    annotated.png   same frame with the overlay (drawn outside the marker outlines)
    state.json      everything the page showed for that frame, incl. corners, both IPPE
                    solutions, intrinsics and the config in effect

Recording:         output/recordings/<time>/
    color.mkv       FFV1 (lossless) colour frames; written on a separate thread, frames it
                    could not keep up with are dropped and counted, never duplicated
    frames.jsonl    one line per written frame: index in color.mkv, capture time, observations
    meta.json       intrinsics, image size, config text, caps, drop count
    Stops at the size or duration cap, or when asked.

Replay:            ReplaySource yields the recorded frames with their recorded capture times and
                   intrinsics, so `vision.py --replay DIR` runs the same processing again.
"""
import json
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"


def save_snapshot(root: Path, raw: np.ndarray, annotated: np.ndarray, state: dict) -> Path:
    d = root / "snapshots" / stamp()
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "raw.png"), raw)
    cv2.imwrite(str(d / "annotated.png"), annotated)
    (d / "state.json").write_text(json.dumps(state, indent=1, ensure_ascii=False))
    return d


class Recorder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.active: dict | None = None
        self.last: dict | None = None

    def start(self, meta: dict, size: tuple[int, int], fps: float, max_mb: float, max_s: float) -> str:
        with self.lock:
            if self.active:
                return "既に録画中"
            free_mb = shutil.disk_usage(self.root.parent if not self.root.exists() else self.root).free / 1e6
            if free_mb < max_mb * 2:
                return f"空き容量 {free_mb:.0f} MB が上限 {max_mb:.0f} MB の 2 倍未満のため録画しない"
            d = self.root / "recordings" / stamp()
            d.mkdir(parents=True)
            w = cv2.VideoWriter(str(d / "color.mkv"), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
            if not w.isOpened():
                return "FFV1 の VideoWriter を開けない"
            (d / "meta.json").write_text(json.dumps(dict(meta, max_mb=max_mb, max_s=max_s),
                                                    indent=1, ensure_ascii=False))
            q = queue.Queue(maxsize=8)
            self.active = {"dir": d, "writer": w, "queue": q, "jsonl": open(d / "frames.jsonl", "w"),
                           "t0": time.time(), "written": 0, "dropped": 0, "bytes": 0,
                           "max_mb": max_mb, "max_s": max_s, "stop_reason": None}
            threading.Thread(target=self._writer, args=(self.active,), daemon=True).start()
            return f"録画開始: {d}"

    def stop(self, reason: str = "停止操作") -> None:
        with self.lock:
            a = self.active
            if not a:
                return
            a["stop_reason"] = reason
            self.active = None
        a["queue"].put(None)

    def push(self, frame: np.ndarray, record: dict) -> None:
        a = self.active
        if not a:
            return
        elapsed = time.time() - a["t0"]
        size_mb = (a["dir"] / "color.mkv").stat().st_size / 1e6 if a["written"] % 30 == 0 else a["bytes"]
        a["bytes"] = size_mb
        if elapsed > a["max_s"]:
            return self.stop(f"時間上限 {a['max_s']:.0f} s")
        if size_mb >= a["max_mb"]:
            return self.stop(f"容量上限 {a['max_mb']:.0f} MB")
        try:
            a["queue"].put_nowait((frame, record))
        except queue.Full:
            a["dropped"] += 1

    def _writer(self, a: dict) -> None:
        while True:
            item = a["queue"].get()
            if item is None:
                break
            frame, record = item
            a["writer"].write(frame)
            a["jsonl"].write(json.dumps(dict(record, video_index=a["written"]), ensure_ascii=False) + "\n")
            a["written"] += 1
        a["writer"].release()
        a["jsonl"].close()
        meta_path = a["dir"] / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(written_frames=a["written"], dropped_frames=a["dropped"],
                    duration_s=time.time() - a["t0"], stop_reason=a["stop_reason"],
                    size_mb=(a["dir"] / "color.mkv").stat().st_size / 1e6)
        meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        self.last = {"dir": str(a["dir"]), "written": a["written"], "dropped": a["dropped"],
                     "stop_reason": a["stop_reason"]}

    def status(self) -> dict:
        a = self.active
        if not a:
            return {"recording": False, "last": self.last}
        return {"recording": True, "dir": str(a["dir"]), "elapsed_s": time.time() - a["t0"],
                "written": a["written"], "dropped": a["dropped"], "size_mb": a["bytes"],
                "max_mb": a["max_mb"], "max_s": a["max_s"], "last": self.last}



class ReplaySource:
    """Frames of a recording with their original capture times; paced like the original."""

    def __init__(self, d: Path, realtime: bool = True, loop: bool = False) -> None:
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.records = [json.loads(l) for l in (d / "frames.jsonl").read_text().splitlines() if l]
        self.realtime, self.loop = realtime, loop

    def __iter__(self):
        while True:
            cap = cv2.VideoCapture(str(self.dir / "color.mkv"))
            t_wall0 = time.monotonic()
            t_rec0 = self.records[0]["t"] if self.records else 0.0
            for rec in self.records:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.realtime:
                    lag = (rec["t"] - t_rec0) - (time.monotonic() - t_wall0)
                    if lag > 0:
                        time.sleep(lag)
                yield frame, rec
            cap.release()
            if not self.loop:
                return
