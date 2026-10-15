#!/usr/bin/env python3
"""modeler のサーバー（DESIGN.md §4）。標準ライブラリの ThreadingHTTPServer、既定 127.0.0.1:8096。

  modeler/.venv/bin/python modeler/server/app.py --port 8096

- プロジェクト一式は --projects-dir（既定 modeler/projects）。画面は --static（既定 modeler/web/dist）を配る。
- OAK の画像・内部パラメータは vision サービス（--vision-url、既定 http://127.0.0.1:8090）の GET /state.json と GET /raw.png だけを使う。
  **アームの API（/arm/...）は呼ばない**（CLAUDE.md の規則）。vision が動いていなくても他の機能はすべて使える。
- --allow-from（CIDR、カンマ区切り）以外からの要求は 403（oak/vision.py と同じ）。
エラーは {"ok": false, "error": "日本語の説明"}。長さは mm（GLB だけ m）。
"""
from __future__ import annotations

import argparse
import copy
import io
import ipaddress
import json
import math
import mimetypes
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import preprocess as pp  # noqa: E402
import project as pj  # noqa: E402
from preprocess import PreprocessError  # noqa: E402
from project import ProjectError  # noqa: E402

SAMPLES_DIR = ROOT / "samples"
MAX_BODY = 512 * 1024 * 1024
VISION_PATHS = ("/state.json", "/raw.png")          # vision サービスで使うのはこの 2 つだけ
COLLISION = {"status": "not_generated",
             "reason": "凸包は穴・空洞を失うため生成しない。元の穴付きメッシュ model.glb を使う"}


# --------------------------------------------------------------------------- 小道具

def clean(o):
    """NaN / inf → None、numpy → Python（厳密な JSON にする）。"""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def parse_multipart(ctype: str, body: bytes) -> dict[str, list[dict]]:
    """multipart/form-data → {name: [{filename, content_type, data}]}（cgi モジュールは使わない）。"""
    m = re.search(r'boundary="?([^";]+)"?', ctype or "")
    if not m:
        raise ProjectError("multipart の boundary がない")
    b = b"--" + m.group(1).encode("latin-1")
    out: dict[str, list[dict]] = {}
    for chunk in body.split(b)[1:]:
        if chunk.startswith(b"--"):
            break
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        head, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = {}
        for line in head.decode("utf-8", "replace").split("\r\n"):
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()
        cd = headers.get("content-disposition", "")
        nm = re.search(r'(?:^|;)\s*name="([^"]*)"', cd)
        fn = re.search(r'filename\*=UTF-8\'\'([^;]+)', cd)
        filename = urllib.parse.unquote(fn.group(1)) if fn else None
        if filename is None:
            fn = re.search(r'filename="([^"]*)"', cd)
            filename = fn.group(1) if fn else None
        if not nm:
            continue
        out.setdefault(nm.group(1), []).append(
            {"filename": filename, "content_type": headers.get("content-type", ""), "data": data})
    return out


def field_text(parts: dict, name: str, default=None):
    v = parts.get(name)
    return v[0]["data"].decode("utf-8", "replace") if v else default


def T_m(T_mm: list[float]) -> list[float]:
    a = np.asarray(T_mm, float).reshape(4, 4).copy()
    a[:3, 3] *= 0.001
    return a.ravel().tolist()


def decorate_image(store: pj.Store, pid: str, im: dict) -> dict:
    im = copy.deepcopy(im)
    im["has_labels"] = (store.image_dir(pid, im["id"]) / "labels.png").is_file()
    return im


# --------------------------------------------------------------------------- アプリ本体

class App:
    def __init__(self, projects_dir: Path, vision_url: str, static_dir: Path | None):
        self.store = pj.Store(projects_dir)
        self.vision_url = vision_url.rstrip("/")
        self.static = static_dir.resolve() if static_dir else None
        self._sample_lock = threading.Lock()

    # ---- vision サービス（GET の 2 か所だけ）
    def vision_get(self, path: str, timeout: float = 4.0) -> bytes:
        assert path in VISION_PATHS, path            # /arm/ などは呼ばない
        url = self.vision_url + path
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ProjectError(f"vision サービス {url} が {e.code} を返した（まだフレームがない可能性）", 502) from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise ProjectError(f"vision サービス（{self.vision_url}）に接続できない: {getattr(e, 'reason', e)}。"
                               "OAK を使わないときは画像ファイルを取り込む", 502) from None

    def vision_camera(self) -> dict:
        try:
            st = json.loads(self.vision_get("/state.json"))
        except ValueError:
            raise ProjectError("vision の state.json が JSON でない", 502) from None
        intr = st.get("intrinsics") or {}
        if not intr.get("K"):
            raise ProjectError("vision サービスにまだ内部パラメータがない（カメラの起動待ち）", 502)
        size = intr.get("image") or st.get("image_size") or [1280, 960]
        cam = pp.make_camera(intr["K"], intr.get("dist_coeffs") or intr.get("dist") or [], int(size[0]), int(size[1]),
                             "vision")
        return cam

    # ---- カメラの変更（全画像の歪み補正をやり直す）
    def apply_camera(self, pid: str, key: str, cam: dict) -> list[str]:
        """カメラ key を cam に置き換え、そのカメラの全画像を新しい歪み補正に合わせる。

        undistorted / labels_auto / edges / マーカーの und は作り直し、labels.png・深度（und 基準）・輪郭・対応点は
        原画像を経由して新しい und 座標へ写す。ロックしていない marker 由来の姿勢は解き直し、ロック中の姿勢は変えない。
        """
        notes = []
        st = self.store
        with st.lock(pid):
            p = st.working(pid)
            old = p.get("cameras", {}).get(key)
            ims = [im for im in p["images"] if im.get("camera") == key]
            bad = [im["id"] for im in ims if (im["width"], im["height"]) != (cam["width"], cam["height"])]
            if bad:
                raise ProjectError(f"校正の画像サイズ {cam['width']}×{cam['height']} と画像の大きさが違う: {', '.join(bad)}")
            p2 = copy.deepcopy(p)
            p2["cameras"][key] = cam
            updates = {}
            for im in ims:
                d = st.image_dir(pid, im["id"])
                if old is not None:
                    lf = d / "labels.png"
                    if lf.is_file():
                        lab = pp.read_labels(lf)
                        pp.write_png(lf, pp.remap_und_image(lab, old, cam))
                    if im.get("depth"):
                        df = d / im["depth"]["file"]
                        if df.is_file():
                            if df.suffix == ".npy":
                                a = np.load(df)
                                np.save(df, pp.remap_und_image(a, old, cam).astype(np.float32))
                            else:
                                a = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
                                pp.write_png(df, pp.remap_und_image(a, old, cam))
                new = pp.run_steps(p2, im, d, ["undistort", "markers", "quality", "mask", "edges"])
                u = {"markers": new["markers"], "quality": new["quality"], "preprocess_info": new["preprocess_info"]}
                if old is not None:
                    u["contours"] = [dict(c, points_und=pp.remap_und_points(c["points_und"], old, cam).round(3).tolist())
                                     if c.get("points_und") else c for c in im.get("contours", [])]
                    u["correspondences"] = [dict(c, uv_und=pp.remap_und_points([c["uv_und"]], old, cam)[0].round(3).tolist())
                                            for c in im.get("correspondences", [])]
                pose = im.get("pose") or {}
                if pose.get("locked"):
                    notes.append(f"{im['id']}: 姿勢はロック中なので解き直していない（新しい校正で確認する）")
                elif pose.get("source") == "marker":
                    mk = self._ref_marker(p2, new)
                    if mk:
                        r = pp.marker_pose(mk["corners_und"], pp.mat3(cam["K_und"]), p2["marker"]["size_mm"])
                        u["pose"] = {"T_cam_marker": r["solutions"][0]["T_cam_marker"], "source": "marker",
                                     "locked": False, "quality": {"reproj_px": r["reproj_px"], "ambiguity": r["ambiguity"],
                                                                   "note": r["note"]}}
                updates[im["id"]] = u

            def fn(pp_):
                pp_.setdefault("cameras", {})[key] = cam
                for im2 in pp_["images"]:
                    if im2["id"] in updates:
                        im2.update(copy.deepcopy(updates[im2["id"]]))
            st.mutate(pid, fn)
        notes.insert(0, f"カメラ {key} を設定し（{cam['source']}）、{len(ims)} 枚の歪み補正をやり直した")
        return notes

    def auto_marker_pose(self, p: dict, im: dict) -> dict | None:
        """基準マーカーが写っていて、姿勢がロックされておらず手動・対応点で決めたものでもなければ、マーカーから姿勢を求める。
        求めた pose を返す（im は変えない）。求めないときは None。"""
        pose = im.get("pose") or {}
        if im.get("role") != "object" or pose.get("locked") or pose.get("source") in ("manual", "pnp"):
            return None
        mk = self._ref_marker(p, im)
        cam = (p.get("cameras") or {}).get(im.get("camera"))
        if not mk or not cam:
            return None
        r = pp.marker_pose(mk["corners_und"], pp.mat3(cam["K_und"]), float(p["marker"]["size_mm"]))
        note = ("カメラ校正なし（焦点距離は仮）。" if cam.get("source") == "none" else "") + r["note"]
        return {"T_cam_marker": r["solutions"][0]["T_cam_marker"], "source": "marker", "locked": False,
                "quality": {"reproj_px": r["reproj_px"], "ambiguity": r["ambiguity"], "note": note}}

    @staticmethod
    def _ref_marker(p: dict, im: dict, marker_id: int | None = None) -> dict | None:
        mid = int(p["marker"]["id"] if marker_id is None else marker_id)
        ms = [m for m in im.get("markers", []) if m["id"] == mid]
        ms.sort(key=lambda m: 0 if m.get("source") == "manual" else 1)
        return ms[0] if ms else None

    # ---- 画像の取り込み
    def import_files(self, pid: str, files: list[tuple[str, bytes]], role: str) -> tuple[list[dict], list[str]]:
        if role not in ("object", "background"):
            raise ProjectError("role は object か background")
        st = self.store
        notes = []
        with st.lock(pid):                      # id の確保だけロックの中で
            p = st.working(pid)
            taken = {x.name for x in (st.dir(pid) / "images").iterdir()} if (st.dir(pid) / "images").exists() else set()
            ids = []
            q = copy.deepcopy(p)
            for _ in files:
                iid = pj.next_image_id(q, taken)
                taken.add(iid)
                st.image_dir(pid, iid).mkdir(parents=True)
                ids.append(iid)
        added, cams = [], {}
        try:
            for iid, (name, data) in zip(ids, files):
                im = pp.import_image(dict(p, cameras=dict(p.get("cameras", {}), **cams)), st.image_dir(pid, iid),
                                     iid, data, name, role, cams)
                added.append(im)
        except Exception:
            for iid in ids[len(added):]:
                shutil.rmtree(st.image_dir(pid, iid), ignore_errors=True)
            if not added:
                raise
            notes.append(f"{len(files) - len(added)} 枚は取り込めなかった")

        pc = dict(p, cameras=dict(p.get("cameras", {}), **cams))
        for im in added:
            ps = self.auto_marker_pose(pc, im)
            if ps:
                im["pose"] = ps

        def fn(q):
            for k, c in cams.items():
                q.setdefault("cameras", {}).setdefault(k, c)
            q["images"].extend(copy.deepcopy(added))
            if role == "background" and not q.get("background_image") and added:
                q["background_image"] = added[0]["id"]
        st.mutate(pid, fn)
        if role == "background":
            notes.append("背景のみの画像として取り込んだ。既存の画像は「前処理のやり直し」で背景差分を使う")
        if any(c["source"] == "none" for c in cams.values()):
            notes.append("カメラの校正がない（焦点距離は仮の値）。校正ファイルか vision から設定する")
        return [decorate_image(st, pid, im) for im in added], notes

    # ---- 書き出しの検証（trimesh）
    @staticmethod
    def verify_glb(glb: bytes) -> dict:
        import trimesh  # noqa: PLC0415  起動を軽くするため遅延 import
        try:
            mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"GLB を読めない: {e}"}
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            return {"ok": False, "error": "GLB に三角形がない"}
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.remove_unreferenced_vertices()
        bodies = mesh.split(only_watertight=False)
        res = []
        for b in bodies:
            wt = bool(b.is_watertight)
            e = int(b.euler_number)
            res.append({"faces": int(len(b.faces)), "watertight": wt, "euler_number": e,
                        "genus": (2 - e) // 2 if wt else None,
                        "volume_mm3": float(b.volume) * 1e9 if wt else None})
        lo, hi = mesh.bounds
        wt_all = all(r["watertight"] for r in res)
        return {
            "ok": True, "tool": f"trimesh {trimesh.__version__}", "glb_unit": "m",
            "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "bodies": len(res),
            "watertight": wt_all, "euler_number": int(mesh.euler_number),
            "through_holes": sum(r["genus"] for r in res) if wt_all else None,
            "through_holes_doc": "閉じた各物体の種数（genus = (2 − オイラー数) / 2）の合計 = 貫通穴の数",
            "volume_m3": float(mesh.volume) if wt_all else None,
            "volume_mm3": float(mesh.volume) * 1e9 if wt_all else None,
            "bbox_m": {"min": lo.tolist(), "max": hi.tolist()},
            "bbox_mm": {"min": (lo * 1000).tolist(), "max": (hi * 1000).tolist(), "size": ((hi - lo) * 1000).tolist()},
            "per_body": res,
        }

    def export(self, pid: str, glb: bytes, options: dict) -> dict:
        if glb[:4] != b"glTF":
            raise ProjectError("glb が GLB（バイナリ glTF）でない")
        coords = options.get("coords", "marker")
        if coords not in ("marker", "obj"):
            raise ProjectError("coords は marker か obj")
        p = options.get("project") or self.store.working(pid)
        ts = time.strftime("%Y%m%d-%H%M%S")
        d = self.store.dir(pid) / "exports" / ts
        k = 1
        while d.exists():
            d = self.store.dir(pid) / "exports" / f"{ts}-{k}"
            k += 1
        d.mkdir(parents=True)
        pj.atomic_write(d / "model.glb", glb)
        T = p.get("T_marker_obj", pj.IDENTITY)
        T_inv = np.linalg.inv(np.asarray(T, float).reshape(4, 4)).ravel().tolist()
        axes = {"x": "marker の x（印刷の右）", "y": "marker の y（印刷の上）", "z": "marker の z（マーカー面から外向き）"} \
            if coords == "marker" else {"x": "obj の x", "y": "obj の y", "z": "obj の z（部品の定義に使った物体座標）"}
        model = {
            "format": "tamasen-modeler export",
            "version": 1,
            "project": {"id": pid, "name": p.get("name", "")},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "units": {"glb_length": "m", "project_length": "mm", "angle": "deg",
                      "note": "GLB は glTF の規約どおり m（mm × 0.001）。この JSON の _mm の値と部品の寸法は mm"},
            "glb": {"file": "model.glb", "frame": coords, "axes": axes,
                    "note": "glTF の慣例（y 上）への軸の変換はしていない。上の軸定義のまま使う",
                    "options": {k: v for k, v in options.items() if k != "project"}},
            "frames_doc": pj.FRAMES_DOC,
            "marker": p.get("marker"),
            "T_marker_obj": {"mm": T, "m": T_m(T), "doc": "p_marker = T_marker_obj · p_obj（行優先 4×4）"},
            "T_obj_marker": {"mm": T_inv, "m": T_m(T_inv)},
            "parts": copy.deepcopy(p.get("shape", {}).get("parts", [])),
            "measurements": copy.deepcopy(p.get("measurements", [])),
            "attributes": copy.deepcopy(p.get("attributes", {})),
            "collision": dict(COLLISION),
            "verification": self.verify_glb(glb),
        }
        pj.atomic_write(d / "model.json", json.dumps(clean(model), ensure_ascii=False, indent=1).encode())
        rel = d.relative_to(self.store.dir(pid)).as_posix()
        return {"ok": True, "dir": rel, "files": {"glb": f"/files/{pid}/{rel}/model.glb",
                                                   "json": f"/files/{pid}/{rel}/model.json"}, "model": clean(model)}

    # ---- サンプル
    def samples(self) -> list[dict]:
        out = []
        for d in sorted(SAMPLES_DIR.iterdir()) if SAMPLES_DIR.exists() else []:
            t = d / "truth.json"
            if not t.is_file():
                continue
            try:
                tr = json.loads(t.read_text())
            except ValueError:
                continue
            out.append({"name": d.name, "title": tr.get("title", d.name), "synthetic": bool(tr.get("synthetic")),
                        "images": len(tr.get("images", [])) + (1 if tr.get("background_image") else 0),
                        "ready": (d / "project" / "project.json").is_file()})
        return out

    def load_sample(self, name: str) -> dict:
        if not pj.ID_RE.match(name or "") or not (SAMPLES_DIR / name / "truth.json").is_file():
            raise ProjectError(f"サンプル {name} がない", 404)
        src = SAMPLES_DIR / name / "project"
        with self._sample_lock:
            if not (src / "project.json").is_file():
                sys.path.insert(0, str(SAMPLES_DIR))
                import make_samples  # noqa: PLC0415
                make_samples.build_project(SAMPLES_DIR / name)
        title = json.loads((SAMPLES_DIR / name / "truth.json").read_text()).get("title", name)

        def reset(p):                # 形状・固定変換・姿勢は空から（人が作る）。真値は入れない
            p["shape"] = {"parts": []}
            p["measurements"] = []
            p["T_marker_obj"] = list(pj.IDENTITY)
            p["T_marker_obj_locked"] = False
            for im in p["images"]:
                im["pose"] = {"T_cam_marker": None, "source": "none", "locked": False, "quality": {}}
                im["correspondences"] = []
        return self.store.copy_from(src, title, reset)


# --------------------------------------------------------------------------- HTTP

ROUTES: list[tuple[str, re.Pattern, str]] = []


def route(method: str, pattern: str):
    def deco(fn):
        if not any(m == method and rx.pattern == "^" + pattern + "$" for m, rx, _ in ROUTES):
            ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__))
        return fn
    return deco


P = r"(?P<pid>[A-Za-z0-9_-]{1,64})"
I = r"(?P<iid>[A-Za-z0-9_-]{1,64})"


def make_handler(app: App, allow: list):
    st = app.store

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "tamasen-modeler/1"

        def log_message(self, fmt, *a):
            if getattr(self.server, "verbose", False):
                sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

        def handle_one_request(self):
            # 画面も API も、許可したアドレスからだけ
            try:
                a = ipaddress.ip_address(self.client_address[0])
                ok = any(a in n for n in allow)
            except ValueError:
                ok = False
            if not ok:
                self.raw_requestline = self.rfile.readline(65537)
                self.close_connection = True
                if self.raw_requestline and self.parse_request():
                    self._json({"ok": False, "error": f"{self.client_address[0]} からの接続は許可されていない（--allow-from）"}, 403)
                return
            return super().handle_one_request()

        # ---- 応答
        def _send(self, body: bytes, ctype: str, code: int = 200, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(clean(obj), ensure_ascii=False).encode(), "application/json; charset=utf-8", code)

        def _err(self, msg: str, code: int = 400) -> None:
            self._json({"ok": False, "error": msg}, code)

        # ---- 要求
        def body(self) -> bytes:
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                raise ProjectError("chunked 転送には対応していない（Content-Length を付ける）", 411)
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ProjectError(f"送信が大きすぎる（{n / 1e6:.0f} MB > {MAX_BODY / 1e6:.0f} MB）", 413)
            if not hasattr(self, "_body"):
                self._body = self.rfile.read(n) if n else b""
            return self._body

        def json_body(self, default=None):
            b = self.body()
            if not b.strip():
                return {} if default is None else default
            try:
                return json.loads(b)
            except ValueError as e:
                raise ProjectError(f"JSON として読めない: {e}") from None

        def query(self) -> dict:
            return {k: v[-1] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items()}

        def dispatch(self, method: str):
            # HTTP/1.1 の keep-alive では同じ Handler が次の要求も受けるので、前の要求の本文を必ず捨てる
            self.__dict__.pop("_body", None)
            try:
                return self._dispatch(method)
            finally:
                try:
                    self.body()   # 経路が読まなかった本文も読み切る（残りが次の要求行として読まれないように）
                except Exception:
                    self.close_connection = True

        def _dispatch(self, method: str):
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            try:
                for m, rx, name in ROUTES:
                    mm = rx.match(path)
                    if mm and m == method:
                        return getattr(self, name)(**mm.groupdict())
                if any(rx.match(path) for _, rx, _ in ROUTES):
                    self.body()
                    return self._err(f"{method} は {path} に使えない", 405)
                if method in ("GET", "HEAD") and not path.startswith("/api/"):
                    return self.static(path)
                self.body()
                return self._err(f"{path} はない", 404)
            except ProjectError as e:
                return self._err(str(e), e.status)
            except PreprocessError as e:
                return self._err(str(e), 422)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._err(f"サーバー内部のエラー: {type(e).__name__}: {e}", 500)

        def do_GET(self):
            self.dispatch("GET")

        def do_HEAD(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def do_PUT(self):
            self.dispatch("PUT")

        def do_DELETE(self):
            self.dispatch("DELETE")

        # ---- 静的ファイル
        def static(self, path: str):
            root = app.static
            if root is None or not (root / "index.html").is_file():
                html = ("<!doctype html><meta charset=utf-8><title>modeler</title>"
                        "<p>画面のビルドがない。<code>cd modeler/web &amp;&amp; npm install &amp;&amp; npm run build</code> "
                        "を実行するか、開発時は <code>npm run dev</code>（5173 番）を開く。API は /api/ で動いている。</p>")
                return self._send(html.encode(), "text/html; charset=utf-8", 200 if path == "/" else 404)
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            f = (root / rel).resolve()
            if (f != root and root not in f.parents) or not f.is_file():
                return self._send(b"not found", "text/plain; charset=utf-8", 404)
            return self._send(f.read_bytes(), ctype_of(f))

        # ---- プロジェクト
        @route("GET", r"/api/health")
        def health(self):
            return self._json({"ok": True, "projects_dir": str(st.root), "vision_url": app.vision_url})

        @route("GET", r"/api/projects")
        def list_projects(self):
            return self._json({"ok": True, "projects": st.list()})

        @route("POST", r"/api/projects")
        def create_project(self):
            b = self.json_body()
            return self._json({"ok": True, "project": st.create(str(b.get("name") or ""))})

        @route("GET", r"/api/projects/" + P)
        def get_project(self, pid):
            return self._json({"ok": True, "project": st.load(pid, autosave=self.query().get("autosave") == "1")})

        @route("PUT", r"/api/projects/" + P)
        def put_project(self, pid):
            b = self.json_body()
            if isinstance(b, dict) and "project" in b and "version" not in b:
                b = b["project"]
            p = st.save(pid, b, autosave=self.query().get("autosave") == "1")
            return self._json({"ok": True, "project": p})

        # ---- 画像
        @route("POST", r"/api/projects/" + P + r"/images")
        def upload_images(self, pid):
            st.dir(pid)
            if not st.exists(pid):
                raise ProjectError("プロジェクトがない", 404)
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                raise ProjectError("multipart/form-data で files[] を送る")
            parts = parse_multipart(ctype, self.body())
            files = [(f["filename"] or "image", f["data"]) for k in ("files[]", "files", "file") for f in parts.get(k, [])]
            if not files:
                raise ProjectError("画像がない（フィールド名 files[]）")
            role = field_text(parts, "role", "object") or "object"
            added, notes = app.import_files(pid, files, role)
            return self._json({"ok": True, "images": added, "notes": notes})

        def _image(self, pid, iid) -> tuple[dict, dict]:
            p = st.working(pid)
            return p, pj.find_image(p, iid)

        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/preprocess")
        def preprocess(self, pid, iid):
            b = self.json_body()
            p, im = self._image(pid, iid)
            steps = b.get("steps") or pp.ALL_STEPS
            new = pp.run_steps(p, im, st.image_dir(pid, iid), steps, b.get("options") or {})
            keys = ["quality", "preprocess_info"] + (["markers"] if {"markers", "undistort"} & set(steps) else []) \
                + (["contours"] if "contours" in steps else [])

            def fn(q, im2):
                for k in keys:
                    im2[k] = copy.deepcopy(new[k])
                if "markers" in keys:   # マーカーを検出し直したら、ロックされていないマーカー由来の姿勢も解き直す
                    ps = app.auto_marker_pose(q, im2)
                    if ps:
                        im2["pose"] = ps
                    elif (im2.get("pose") or {}).get("source") == "marker" and not im2["pose"].get("locked") \
                            and not app._ref_marker(q, im2):
                        im2["pose"] = {"T_cam_marker": None, "source": "none", "locked": False, "quality": {}}
                return im2
            out = st.mutate_image(pid, iid, fn)
            return self._json({"ok": True, "image": decorate_image(st, pid, out)})

        @route("GET", r"/files/" + P + r"/(?P<rest>(?:images|exports)/.+)")
        def files(self, pid, rest):
            base = st.dir(pid)
            f = (base / rest).resolve()
            if base not in f.parents or not f.is_file() or any(s.startswith(".") for s in Path(rest).parts):
                raise ProjectError("ファイルがない", 404)
            return self._send(f.read_bytes(), ctype_of(f))

        @route("PUT", r"/api/projects/" + P + r"/images/" + I + r"/labels")
        def put_labels(self, pid, iid):
            _, im = self._image(pid, iid)
            a = cv2.imdecode(np.frombuffer(self.body(), np.uint8), cv2.IMREAD_UNCHANGED)
            if a is None:
                raise ProjectError("ラベルが PNG として読めない")
            if a.ndim == 3:
                a = a[..., 2]                   # RGB(A) のときは R にラベル値
            if a.dtype != np.uint8:
                raise ProjectError("ラベルは 8 bit でなければならない")
            if a.shape != (im["height"], im["width"]):
                raise ProjectError(f"ラベルの大きさ {a.shape[1]}×{a.shape[0]} が画像 {im['width']}×{im['height']} と違う")
            if a.max() > 4:
                raise ProjectError(f"ラベル値は 0〜4（受け取った最大値 {int(a.max())}）")
            pp.write_png(st.image_dir(pid, iid) / "labels.png", a)
            return self._json({"ok": True, "image": decorate_image(st, pid, im),
                               "counts": np.bincount(a.ravel(), minlength=5).tolist()})

        @route("DELETE", r"/api/projects/" + P + r"/images/" + I + r"/labels")
        def delete_labels(self, pid, iid):
            _, im = self._image(pid, iid)
            try:
                (st.image_dir(pid, iid) / "labels.png").unlink()
            except FileNotFoundError:
                pass
            return self._json({"ok": True, "image": decorate_image(st, pid, im)})

        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/grabcut")
        def grabcut(self, pid, iid):
            b = self.json_body()
            _, im = self._image(pid, iid)
            d = st.image_dir(pid, iid)
            und = pp.read_bgr(d / "undistorted.png")
            seed = None if b.get("seed") == "none" else pp.read_labels(pp.labels_file(d))
            lab = pp.grabcut_with_seeds(und, seed, b.get("fg_strokes") or [], b.get("bg_strokes") or [],
                                        int(b.get("iters", 5)))
            pp.write_png(d / "labels.png", lab)
            return self._json({"ok": True, "image": decorate_image(st, pid, im),
                               "counts": np.bincount(lab.ravel(), minlength=5).tolist()})

        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/contours_from_labels")
        def contours(self, pid, iid):
            b = self.json_body()
            p, im = self._image(pid, iid)
            new = pp.run_steps(p, im, st.image_dir(pid, iid), ["contours"], b.get("options") or b)

            def fn(q, im2):
                im2["contours"] = copy.deepcopy(new["contours"])
                return im2
            out = st.mutate_image(pid, iid, fn)
            return self._json({"ok": True, "image": decorate_image(st, pid, out)})

        # ---- 姿勢
        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/marker_pose")
        def marker_pose(self, pid, iid):
            b = self.json_body()
            p, im = self._image(pid, iid)
            if (im.get("pose") or {}).get("locked"):
                raise ProjectError(f"画像 {iid} の姿勢はロック（確定）されている。変えるにはロックを外す", 409)
            cam = p["cameras"][im["camera"]]
            if b.get("corners_und"):
                corners = b["corners_und"]
            elif b.get("corners_raw"):
                corners = pp.raw_to_und(b["corners_raw"], cam).tolist()
            else:
                mk = app._ref_marker(p, im, b.get("marker_id"))
                if not mk:
                    raise ProjectError(f"基準マーカー ID {b.get('marker_id', p['marker']['id'])} の四隅がない"
                                       "（見えない画像は手動で合わせるか、四隅を手で指定する）")
                corners = mk["corners_und"]
            if np.asarray(corners, float).shape != (4, 2):
                raise ProjectError("四隅は [[u,v]×4]（0..3 = 左上・右上・右下・左下）")
            r = pp.marker_pose(corners, pp.mat3(cam["K_und"]), float(b.get("size_mm") or p["marker"]["size_mm"]))
            note = r["note"]
            if cam.get("source") == "none":
                note = "カメラ校正なし（焦点距離は仮）。" + note
            pose = {"T_cam_marker": r["solutions"][0]["T_cam_marker"], "source": "marker", "locked": False,
                    "quality": {"reproj_px": r["reproj_px"], "ambiguity": r["ambiguity"], "note": note}}
            out = im
            if b.get("write", True):
                def fn(q, im2):
                    if (im2.get("pose") or {}).get("locked"):
                        raise ProjectError(f"画像 {iid} の姿勢はロック（確定）されている", 409)
                    im2["pose"] = copy.deepcopy(pose)
                    return im2
                out = st.mutate_image(pid, iid, fn)
            return self._json({"ok": True, "pose": pose, "solutions": r["solutions"], "side_px": r["side_px"],
                               "written": bool(b.get("write", True)), "image": decorate_image(st, pid, out)})

        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/pnp")
        def pnp(self, pid, iid):
            b = self.json_body()
            p, im = self._image(pid, iid)
            cs = b.get("correspondences")
            if cs is None:
                cs = im.get("correspondences", [])
            try:
                uv = [c["uv_und"] for c in cs]
                po = np.asarray([c["p_obj"] for c in cs], float).reshape(-1, 3)
            except (KeyError, TypeError, ValueError):
                raise ProjectError("correspondences は [{uv_und: [u,v], p_obj: [x,y,z]}]") from None
            T = np.asarray(b.get("T_marker_obj") or p.get("T_marker_obj") or pj.IDENTITY, float).reshape(4, 4)
            pm = po @ T[:3, :3].T + T[:3, 3]
            cam = p["cameras"][im["camera"]]
            r = pp.pnp(uv, pm, pp.mat3(cam["K_und"]))
            if cam.get("source") == "none":
                r["notes"].append("カメラ校正なし（焦点距離は仮）")
            locked = bool((im.get("pose") or {}).get("locked"))
            if locked:
                r["notes"].append("この画像の姿勢はロック中（候補は計算したが適用できない）")
            return self._json({"ok": True, "locked": locked, "written": False, **r})

        # ---- 深度
        @route("POST", r"/api/projects/" + P + r"/images/" + I + r"/depth")
        def depth(self, pid, iid):
            q = self.query()
            p, im = self._image(pid, iid)
            data = self.body()
            name = q.get("filename", "")
            ctype = self.headers.get("Content-Type", "")
            if ctype.startswith("multipart/form-data"):
                parts = parse_multipart(ctype, data)
                f = (parts.get("file") or parts.get("depth") or [None])[0]
                if f is None:
                    raise ProjectError("深度ファイルがない（フィールド名 file）")
                data, name = f["data"], f["filename"] or ""
                q.update({k: field_text(parts, k) for k in ("registered_to", "invalid") if parts.get(k)})
            a, unit, fname = pp.load_depth(data, name)
            if a.shape != (im["height"], im["width"]):
                raise ProjectError(f"深度の大きさ {a.shape[1]}×{a.shape[0]} が画像 {im['width']}×{im['height']} と違う"
                                   "（RGB に位置合わせ済みの深度を送る）")
            reg = q.get("registered_to", "rgb_raw")
            if reg not in ("rgb_raw", "rgb_und"):
                raise ProjectError("registered_to は rgb_raw か rgb_und")
            invalid = float(q.get("invalid", 0))
            if invalid != 0:
                a[a == invalid] = 0
            cam = p["cameras"][im["camera"]]
            if reg == "rgb_raw":
                a = pp.undistort(a, cam, cv2.INTER_NEAREST)
            d = st.image_dir(pid, iid)
            for old in ("depth.png", "depth.npy"):
                (d / old).unlink(missing_ok=True)
            if fname.endswith(".npy"):
                buf = io.BytesIO()
                np.save(buf, a.astype(np.float32))
                pj.atomic_write(d / fname, buf.getvalue())
            else:
                pp.write_png(d / fname, a.astype(np.uint16))
            info = {"file": fname, "unit": unit, "invalid": 0, "registered_to": "rgb_und"}
            valid = a[a > 0]

            def fn(q2, im2):
                im2["depth"] = dict(info)
                return im2
            out = st.mutate_image(pid, iid, fn)
            return self._json({"ok": True, "image": decorate_image(st, pid, out), "depth": info,
                               "valid_pct": float(valid.size / a.size * 100),
                               "note": "原画像に合わせた深度を歪み補正画像へ写した" if reg == "rgb_raw" else ""})

        @route("GET", r"/api/projects/" + P + r"/images/" + I + r"/depth_points")
        def depth_points(self, pid, iid):
            q = self.query()
            p, im = self._image(pid, iid)
            info = im.get("depth")
            if not info:
                raise ProjectError("この画像に深度がない", 404)
            d = st.image_dir(pid, iid)
            f = d / info["file"]
            a = np.load(f) if f.suffix == ".npy" else cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            und = pp.read_bgr(d / "undistorted.png") if q.get("colors", "1") == "1" else None
            cam = p["cameras"][im["camera"]]
            r = pp.depth_points(a, info["unit"], info.get("invalid", 0), pp.mat3(cam["K_und"]),
                                int(q.get("step", 4)), und, int(q.get("max", 300000)))
            return self._json({"ok": True, **r})

        # ---- OAK・カメラ
        @route("POST", r"/api/projects/" + P + r"/capture")
        def capture(self, pid):
            b = self.json_body()
            p = st.working(pid)
            vcam = app.vision_camera()
            png = app.vision_get("/raw.png", timeout=8)
            notes = []
            cur = p.get("cameras", {}).get("oak")
            if cur is None or cur.get("source") == "none":
                notes += app.apply_camera(pid, "oak", vcam)
            elif cur["width"] == vcam["width"] and cur["height"] == vcam["height"]:
                if abs(cur["K"][0] / vcam["K"][0] - 1) > 0.005:
                    notes.append(f"プロジェクトの校正（fx {cur['K'][0]:.1f}）と vision（fx {vcam['K'][0]:.1f}）が違う。"
                                 "プロジェクトの校正を使った")
            added, n2 = app.import_files(pid, [(time.strftime("oak_%Y%m%d-%H%M%S.png"), png)], b.get("role", "object"))
            return self._json({"ok": True, "images": added, "notes": notes + n2})

        @route("POST", r"/api/projects/" + P + r"/camera")
        def camera(self, pid):
            st.working(pid)
            ctype = self.headers.get("Content-Type", "")
            key, source = "oak", "manual"
            if ctype.startswith("multipart/form-data"):
                parts = parse_multipart(ctype, self.body())
                f = (parts.get("file") or parts.get("calibration") or [None])[0]
                key = field_text(parts, "camera", "oak") or "oak"
                if field_text(parts, "from") == "vision":
                    b = {"from": "vision"}
                elif f is None:
                    raise ProjectError("校正ファイルがない（フィールド名 file）")
                else:
                    try:
                        b = json.loads(f["data"])
                    except ValueError as e:
                        raise ProjectError(f"校正ファイルが JSON でない: {e}") from None
                    source = "file"
            else:
                b = self.json_body()
                key = str(b.get("camera") or "oak")
                if b.get("source") == "file":
                    source = "file"
            if not pj.ID_RE.match(key):
                raise ProjectError("カメラ名が不正")
            if b.get("from") == "vision":
                cam = app.vision_camera()
            else:
                K = b.get("K")
                dist = b.get("dist", b.get("dist_coeffs", []))
                w, h = b.get("width"), b.get("height")
                if (w is None or h is None) and b.get("image_size"):
                    w, h = b["image_size"]
                if K is None or w is None or h is None:
                    raise ProjectError("校正には K（3×3）・dist・width・height が要る")
                try:
                    cam = pp.make_camera(K, dist, int(w), int(h), source)
                except (ValueError, TypeError) as e:
                    raise ProjectError(f"校正の値が不正: {e}") from None
            notes = app.apply_camera(pid, key, cam)
            return self._json({"ok": True, "camera": cam, "notes": notes, "project": st.load(pid)})

        # ---- 書き出し
        @route("POST", r"/api/projects/" + P + r"/export")
        def export(self, pid):
            st.working(pid)
            ctype = self.headers.get("Content-Type", "")
            if ctype.startswith("multipart/form-data"):
                parts = parse_multipart(ctype, self.body())
                g = (parts.get("glb") or [None])[0]
                if g is None:
                    raise ProjectError("glb がない（フィールド名 glb）")
                glb = g["data"]
                opt_text = field_text(parts, "options")
                try:
                    options = json.loads(opt_text) if opt_text else {}
                except ValueError as e:
                    raise ProjectError(f"options が JSON でない: {e}") from None
                if field_text(parts, "coords"):
                    options["coords"] = field_text(parts, "coords")
            elif ctype.startswith("model/gltf-binary") or ctype.startswith("application/octet-stream"):
                glb, options = self.body(), {"coords": self.query().get("coords", "marker")}
            else:
                raise ProjectError("multipart/form-data（glb と options）で送る")
            return self._json(app.export(pid, glb, options))

        # ---- サンプル
        @route("GET", r"/api/samples")
        def samples(self):
            return self._json({"ok": True, "samples": app.samples()})

        @route("POST", r"/api/samples/(?P<name>[A-Za-z0-9_-]{1,64})/load")
        def load_sample(self, name):
            self.body()
            return self._json({"ok": True, "project": app.load_sample(name)})

    return Handler


def ctype_of(f: Path) -> str:
    ext = f.suffix.lower()
    t = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".wasm": "application/wasm",
         ".json": "application/json; charset=utf-8", ".glb": "model/gltf-binary", ".png": "image/png",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml", ".npy": "application/octet-stream",
         ".map": "application/json"}.get(ext)
    return t or mimetypes.guess_type(f.name)[0] or "application/octet-stream"


def make_server(host: str, port: int, projects_dir: Path, vision_url: str, allow_from: str,
                static_dir: Path | None) -> ThreadingHTTPServer:
    allow = [ipaddress.ip_network(c.strip(), strict=False) for c in allow_from.split(",") if c.strip()]
    app = App(projects_dir, vision_url, static_dir)
    server = ThreadingHTTPServer((host, port), make_handler(app, allow))
    server.daemon_threads = True
    server.app = app
    return server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="modeler のサーバー（API と画面）")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 で LAN に出す（そのときは --allow-from が効く）")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--projects-dir", type=Path, default=ROOT / "projects")
    ap.add_argument("--vision-url", default="http://127.0.0.1:8090", help="OAK の vision サービス（/state.json と /raw.png だけ使う）")
    ap.add_argument("--allow-from", default="127.0.0.1/32,::1/128", help="接続を許す CIDR（カンマ区切り）")
    ap.add_argument("--static", type=Path, default=ROOT / "web" / "dist", help="画面（npm run build の出力）")
    ap.add_argument("--verbose", action="store_true", help="要求を 1 行ずつ表示")
    a = ap.parse_args(argv)
    server = make_server(a.host, a.port, a.projects_dir, a.vision_url, a.allow_from, a.static)
    server.verbose = a.verbose
    print(f"modeler: http://{a.host}:{server.server_port}/  （プロジェクト {server.app.store.root}）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
