"""プロジェクトの保存形式（DESIGN.md §2）。型の正本は web/src/types.ts、ここは同じ定義を dict で持つ。

長さはすべて mm、角度は度。変換 T_a_b は b → a、4×4 行優先の 16 要素。
project.json は保存（PUT）で確定、project.autosave.json は自動保存。どちらも一時ファイル + os.replace で原子的に書く。
サーバーが画像を追加・前処理したときは、両方のファイル（あるもの）に同じ変更を入れる（自動保存側の編集を失わないため）。
"""
from __future__ import annotations

import copy
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

VERSION = 1
FRAMES_DOC = ("T_a_b maps b to a, row-major 4x4; cam=OpenCV(x right,y down,z fwd); "
              "marker=center of black square, x right,y up,z out")
IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
LABEL = {"UNKNOWN": 0, "BACKGROUND": 1, "OBJECT": 2, "HOLE": 3, "MARKER": 4}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SERVER_ONLY_KEYS = ("autosave_available",)          # GET でだけ付ける。保存しない
SERVER_ONLY_IMAGE_KEYS = ("has_labels",)


class ProjectError(Exception):
    """利用者に返す日本語のエラー（HTTP の状態コード付き）。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- 既定値

def new_project(name: str) -> dict:
    return {
        "version": VERSION,
        "name": name,
        "units": {"length": "mm", "angle": "deg"},
        "frames_doc": FRAMES_DOC,
        "marker": {"dictionary": "DICT_4X4_50", "id": 7, "size_mm": 30.0},
        "cameras": {},
        "T_marker_obj": list(IDENTITY),
        "T_marker_obj_locked": False,
        "shape": {"parts": []},
        "measurements": [],
        "attributes": {"mass_g": None, "material": "", "notes": ""},
        "background_image": None,
        "images": [],
        "settings": {"autosave_s": 20},
    }


def new_image(iid: str, file: str, camera: str, width: int, height: int, role: str) -> dict:
    return {
        "id": iid, "file": file, "camera": camera, "width": int(width), "height": int(height),
        "orientation": 1, "role": role,
        "quality": {"blur_var": 0.0, "overexposed_pct": 0.0, "underexposed_pct": 0.0, "ok": True, "notes": []},
        "markers": [],
        "pose": {"T_cam_marker": None, "source": "none", "locked": False, "quality": {}},
        "contours": [],
        "correspondences": [],
        "depth": None,
    }


def no_camera(width: int, height: int) -> dict:
    """校正なしのカメラ。焦点距離は仮（OAK-D Lite 1280×960 の fx 913 ≒ 0.713·幅 に合わせる）。"""
    f = 0.713 * width
    K = [f, 0.0, (width - 1) / 2, 0.0, f, (height - 1) / 2, 0.0, 0.0, 1.0]
    return {"width": int(width), "height": int(height), "K": K, "dist": [], "K_und": list(K), "source": "none"}


# --------------------------------------------------------------------------- 検証

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_mat(m, what: str, allow_none: bool = False) -> None:
    if m is None and allow_none:
        return
    if not isinstance(m, list) or len(m) != 16 or not all(_is_num(v) for v in m):
        raise ProjectError(f"{what} は 16 個の数値（4×4 行優先）でなければならない")


def validate(p: dict) -> None:
    """保存前の形の検証（中身の妥当性までは見ない）。"""
    if not isinstance(p, dict):
        raise ProjectError("project はオブジェクトでなければならない")
    if p.get("version") != VERSION:
        raise ProjectError(f"version は {VERSION} でなければならない（受け取った値: {p.get('version')!r}）")
    for k, t in (("name", str), ("marker", dict), ("cameras", dict), ("shape", dict), ("images", list),
                 ("measurements", list), ("attributes", dict), ("settings", dict)):
        if not isinstance(p.get(k), t):
            raise ProjectError(f"{k} がない、または型が違う（{t.__name__} が必要）")
    if p.get("units", {}).get("length") != "mm":
        raise ProjectError("units.length は mm でなければならない")
    _check_mat(p.get("T_marker_obj"), "T_marker_obj")
    if not isinstance(p["shape"].get("parts"), list):
        raise ProjectError("shape.parts は配列でなければならない")
    ids = set()
    for im in p["images"]:
        if not isinstance(im, dict) or not ID_RE.match(str(im.get("id", ""))):
            raise ProjectError("images[] の id が不正")
        if im["id"] in ids:
            raise ProjectError(f"画像 id が重複している: {im['id']}")
        ids.add(im["id"])
        pose = im.get("pose")
        if not isinstance(pose, dict):
            raise ProjectError(f"{im['id']}: pose がない")
        _check_mat(pose.get("T_cam_marker"), f"{im['id']}.pose.T_cam_marker", allow_none=True)
    bg = p.get("background_image")
    if bg is not None and bg not in ids:
        raise ProjectError(f"background_image {bg} は images にない")


def same_mat(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(a, b))


def guard_locked(old: dict, new: dict) -> None:
    """ロック中の画像の姿勢を変える保存を拒否する。

    「ロック中」は保存済み（old）でも新しい内容（new）でも locked のもの。new で locked を外せば（明示的な解除）変更できる。
    """
    olds = {im["id"]: im for im in old.get("images", [])}
    for im in new.get("images", []):
        o = olds.get(im["id"])
        if o is None:
            continue
        op, np_ = o.get("pose") or {}, im.get("pose") or {}
        if op.get("locked") and np_.get("locked"):
            if not same_mat(op.get("T_cam_marker"), np_.get("T_cam_marker")) or op.get("source") != np_.get("source"):
                raise ProjectError(f"画像 {im['id']} の姿勢はロック（確定）されている。変えるにはロックを外す", 409)
    if old.get("T_marker_obj_locked") and new.get("T_marker_obj_locked") \
            and not same_mat(old.get("T_marker_obj"), new.get("T_marker_obj")):
        raise ProjectError("固定変換 T_marker_obj はロックされている。変えるにはロックを外す", 409)


# --------------------------------------------------------------------------- 保存

def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dump(p: dict) -> bytes:
    return json.dumps(p, ensure_ascii=False, indent=1, allow_nan=False).encode()


def strip_server_keys(p: dict) -> dict:
    p = copy.deepcopy(p)
    for k in SERVER_ONLY_KEYS + ("id",):
        p.pop(k, None)
    for im in p.get("images", []):
        for k in SERVER_ONLY_IMAGE_KEYS:
            im.pop(k, None)
    return p


class Store:
    """<projects_dir>/<pid>/ の読み書き。プロジェクトごとのロックで直列化する。"""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._glock = threading.Lock()

    def lock(self, pid: str) -> threading.RLock:
        with self._glock:
            return self._locks.setdefault(pid, threading.RLock())

    # ---- パス
    def dir(self, pid: str) -> Path:
        if not ID_RE.match(pid or ""):
            raise ProjectError("プロジェクト id が不正", 400)
        d = (self.root / pid).resolve()
        if d.parent != self.root:
            raise ProjectError("プロジェクト id が不正", 400)
        return d

    def image_dir(self, pid: str, iid: str) -> Path:
        if not ID_RE.match(iid or ""):
            raise ProjectError("画像 id が不正", 400)
        return self.dir(pid) / "images" / iid

    def main_file(self, pid: str) -> Path:
        return self.dir(pid) / "project.json"

    def autosave_file(self, pid: str) -> Path:
        return self.dir(pid) / "project.autosave.json"

    def exists(self, pid: str) -> bool:
        return self.main_file(pid).is_file()

    # ---- 一覧・作成
    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            f = d / "project.json"
            if not (d.is_dir() and ID_RE.match(d.name) and f.is_file()):
                continue
            try:
                p = json.loads(f.read_text())
            except (OSError, ValueError):
                out.append({"id": d.name, "name": "（読めない project.json）", "broken": True})
                continue
            a = d / "project.autosave.json"
            out.append({"id": d.name, "name": p.get("name", ""), "images": len(p.get("images", [])),
                        "parts": len(p.get("shape", {}).get("parts", [])),
                        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
                        "autosave_available": a.is_file() and a.stat().st_mtime > f.stat().st_mtime})
        out.sort(key=lambda x: x.get("updated", ""), reverse=True)
        return out

    def new_id(self) -> str:
        while True:
            pid = time.strftime("p%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            if not (self.root / pid).exists():
                return pid

    def create(self, name: str, template: dict | None = None) -> dict:
        name = (name or "").strip() or "無題"
        pid = self.new_id()
        p = copy.deepcopy(template) if template else new_project(name)
        p["name"] = name
        d = self.dir(pid)
        (d / "images").mkdir(parents=True)
        atomic_write(d / "project.json", dump(strip_server_keys(p)))
        return self.load(pid)

    def copy_from(self, src: Path, name: str, transform=None) -> dict:
        """既存のプロジェクト一式（サンプルなど）を新しい id で複製する。transform(project) で中身を変えられる。"""
        pid = self.new_id()
        d = self.dir(pid)
        shutil.copytree(src, d, ignore=shutil.ignore_patterns("project.autosave.json", "exports", ".*"))
        p = json.loads((d / "project.json").read_text())
        p["name"] = name
        if transform:
            transform(p)
        atomic_write(d / "project.json", dump(strip_server_keys(p)))
        return self.load(pid)

    # ---- 読み書き
    def _read(self, f: Path) -> dict:
        try:
            return json.loads(f.read_text())
        except FileNotFoundError:
            raise ProjectError("プロジェクトがない", 404) from None
        except ValueError as e:
            raise ProjectError(f"{f.name} が壊れている: {e}", 500) from None

    def autosave_newer(self, pid: str) -> bool:
        m, a = self.main_file(pid), self.autosave_file(pid)
        return a.is_file() and m.is_file() and a.stat().st_mtime > m.stat().st_mtime

    def load(self, pid: str, autosave: bool = False) -> dict:
        """GET 用。サーバーが計算する項目（autosave_available, has_labels, id）を付ける。"""
        if not self.exists(pid):
            raise ProjectError("プロジェクトがない", 404)
        if autosave:
            if not self.autosave_file(pid).is_file():
                raise ProjectError("自動保存がない", 404)
            p = self._read(self.autosave_file(pid))
        else:
            p = self._read(self.main_file(pid))
        return self.decorate(pid, p)

    def decorate(self, pid: str, p: dict) -> dict:
        p = copy.deepcopy(p)
        p["id"] = pid
        p["autosave_available"] = self.autosave_newer(pid)
        for im in p.get("images", []):
            im["has_labels"] = (self.image_dir(pid, im["id"]) / "labels.png").is_file()
        return p

    def working(self, pid: str) -> dict:
        """いちばん新しい内容（自動保存が新しければそちら）。ロック判定・書き出しに使う。"""
        if not self.exists(pid):
            raise ProjectError("プロジェクトがない", 404)
        f = self.autosave_file(pid) if self.autosave_newer(pid) else self.main_file(pid)
        return self._read(f)

    def save(self, pid: str, p: dict, autosave: bool = False) -> dict:
        with self.lock(pid):
            if not self.exists(pid):
                raise ProjectError("プロジェクトがない", 404)
            p = strip_server_keys(p)
            validate(p)
            old = self.working(pid)
            guard_locked(old, p)
            on_disk = {im["id"] for im in old.get("images", [])}
            for im in p["images"]:
                if im["id"] not in on_disk and not self.image_dir(pid, im["id"]).is_dir():
                    raise ProjectError(f"画像 {im['id']} はサーバーにない（取り込みは /images で行う）")
            if autosave:
                atomic_write(self.autosave_file(pid), dump(p))
            else:
                atomic_write(self.main_file(pid), dump(p))
                try:
                    self.autosave_file(pid).unlink()
                except FileNotFoundError:
                    pass
            return self.decorate(pid, p)

    def mutate(self, pid: str, fn) -> dict:
        """サーバー側の変更 fn(project) を project.json と（あれば）自動保存の両方に入れる。fn の戻り値を返す。

        fn は両方のファイルに対して呼ばれる（同じ結果になるよう決定的に書く）。最初の呼び出し（作業中の内容）の戻り値を返す。
        """
        with self.lock(pid):
            if not self.exists(pid):
                raise ProjectError("プロジェクトがない", 404)
            newer = self.autosave_newer(pid)
            # 古い自動保存（project.json より前）は使われないので触らない。新しい自動保存は作業中なので先に変える
            files = [self.autosave_file(pid), self.main_file(pid)] if newer else [self.main_file(pid)]
            result = None
            for i, f in enumerate(files):
                p = self._read(f)
                r = fn(p)
                if i == 0:
                    result = r
                validate(p)
                atomic_write(f, dump(p))
            if newer:                        # 自動保存のほうが新しい関係を保つ
                now = time.time()
                os.utime(self.main_file(pid), (now - 1, now - 1))
                os.utime(self.autosave_file(pid), (now, now))
            return result

    def mutate_image(self, pid: str, iid: str, fn) -> dict:
        def g(p):
            im = find_image(p, iid)
            return fn(p, im)
        return self.mutate(pid, g)


def find_image(p: dict, iid: str) -> dict:
    for im in p.get("images", []):
        if im["id"] == iid:
            return im
    raise ProjectError(f"画像 {iid} がない", 404)


def next_image_id(p: dict, taken_dirs: set[str] | None = None) -> str:
    n = 0
    for im in p.get("images", []):
        m = re.match(r"im_(\d+)$", im["id"])
        if m:
            n = max(n, int(m.group(1)))
    for d in taken_dirs or ():
        m = re.match(r"im_(\d+)$", d)
        if m:
            n = max(n, int(m.group(1)))
    return f"im_{n + 1:04d}"


def next_contour_id(im: dict) -> str:
    n = 0
    for c in im.get("contours", []):
        m = re.match(r"c(\d+)$", str(c.get("id", "")))
        if m:
            n = max(n, int(m.group(1)))
    return f"c{n + 1}"
