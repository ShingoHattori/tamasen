"""modeler サーバーの試験（合成データ。実機・vision サービスは使わない）。

  modeler/.venv/bin/python -m pytest modeler/tests -q
"""
from __future__ import annotations

import io
import json
import math
import socket
import struct
import sys
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "samples"))
import app as server_app  # noqa: E402
import make_samples  # noqa: E402
import preprocess as pp  # noqa: E402

PLATE = ROOT / "samples" / "plate_synthetic"


# --------------------------------------------------------------------------- 準備

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start(tmp: Path, allow="127.0.0.1/32"):
    srv = server_app.make_server("127.0.0.1", 0, tmp, f"http://127.0.0.1:{free_port()}", allow, None)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


class Client:
    def __init__(self, base):
        self.base = base

    def req(self, method, path, body=None, ctype=None, raw=False):
        data = None
        headers = {}
        if body is not None:
            if isinstance(body, (bytes, bytearray)):
                data = bytes(body)
            else:
                data = json.dumps(body).encode()
                ctype = ctype or "application/json"
        if ctype:
            headers["Content-Type"] = ctype
        r = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                b, code = resp.read(), resp.status
        except urllib.error.HTTPError as e:
            b, code = e.read(), e.code
        return (code, b) if raw else (code, json.loads(b))

    def multipart(self, path, fields: dict, files: list[tuple[str, str, bytes]]):
        bd = uuid.uuid4().hex
        out = io.BytesIO()
        for k, v in fields.items():
            out.write(f'--{bd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for name, fn, data in files:
            out.write(f'--{bd}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'
                      f'Content-Type: application/octet-stream\r\n\r\n'.encode())
            out.write(data + b"\r\n")
        out.write(f"--{bd}--\r\n".encode())
        return self.req("POST", path, out.getvalue(), f"multipart/form-data; boundary={bd}")


@pytest.fixture(scope="session")
def truth():
    if not (PLATE / "truth.json").is_file():
        make_samples.generate(make_samples.plate_spec(), PLATE, seed=100)
    return json.loads((PLATE / "truth.json").read_text())


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    srv, base = start(tmp_path_factory.mktemp("projects"))
    yield Client(base)
    srv.shutdown()
    srv.server_close()


def rot_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return math.degrees(math.acos(max(-1, min(1, c))))


@pytest.fixture(scope="session")
def plate(server, truth):
    """校正を入れ、背景 1 枚・上面 2 枚・裏面 1 枚を取り込んだプロジェクト。"""
    c, r = server.req("POST", "/api/projects", {"name": "板（合成データ）"})
    assert c == 200 and r["ok"]
    pid = r["project"]["id"]
    p = r["project"]
    p["marker"] = dict(truth["marker"])
    c, r = server.req("PUT", f"/api/projects/{pid}", p)
    assert c == 200, r
    cam = truth["camera"]
    c, r = server.req("POST", f"/api/projects/{pid}/camera",
                      {"K": cam["K"], "dist": cam["dist"], "width": cam["width"], "height": cam["height"], "source": "file"})
    assert c == 200, r
    bgf = truth["background_image"]
    c, r = server.multipart(f"/api/projects/{pid}/images", {"role": "background"},
                            [("files[]", bgf, (PLATE / "images" / bgf).read_bytes())])
    assert c == 200, r
    picks = [truth["images"][0], truth["images"][2], truth["images"][5]]   # 上面・上面（手あり）・裏面
    c, r = server.multipart(f"/api/projects/{pid}/images", {"role": "object"},
                            [("files[]", t["file"], (PLATE / "images" / t["file"]).read_bytes()) for t in picks])
    assert c == 200, r
    return {"pid": pid, "images": r["images"], "truth": picks}


# --------------------------------------------------------------------------- 試験

def test_create_list_and_errors(server):
    c, r = server.req("POST", "/api/projects", {"name": "一覧テスト"})
    assert c == 200
    c, r2 = server.req("GET", "/api/projects")
    assert c == 200 and any(x["id"] == r["project"]["id"] for x in r2["projects"])
    c, r = server.req("GET", "/api/projects/nope")
    assert c == 404 and r["ok"] is False and "ない" in r["error"]
    c, r = server.req("POST", "/api/projects", b"{broken", "application/json")
    assert c == 400 and r["ok"] is False


def test_marker_detection_and_pose(server, plate, truth):
    K = np.asarray(truth["camera"]["K"]).reshape(3, 3)
    n = 0
    for im, t in zip(plate["images"], plate["truth"]):
        mk = [m for m in im["markers"] if m["id"] == truth["marker"]["id"]]
        if not t["marker_visible"]:
            assert not mk
            assert im["pose"]["source"] == "none" and im["pose"]["T_cam_marker"] is None
            continue
        # 取り込んだだけでマーカーから姿勢が付いている
        assert im["pose"]["source"] == "marker" and im["pose"]["T_cam_marker"] is not None
        Ta = np.asarray(im["pose"]["T_cam_marker"]).reshape(4, 4)
        assert rot_deg(Ta, np.asarray(t["T_cam_marker"]).reshape(4, 4)) < 1.0
        assert mk, f"{t['file']}: マーカー未検出"
        err = np.abs(np.asarray(mk[0]["corners_und"]) - np.asarray(t["corners_und"])).max()
        assert err < 1.5, f"{t['file']}: 四隅の誤差 {err:.2f} px"
        err_raw = np.abs(np.asarray(mk[0]["corners_raw"]) - np.asarray(t["corners_raw"])).max()
        assert err_raw < 1.5
        c, r = server.req("POST", f"/api/projects/{plate['pid']}/images/{im['id']}/marker_pose", {})
        assert c == 200, r
        T = np.asarray(r["pose"]["T_cam_marker"]).reshape(4, 4)
        Tt = np.asarray(t["T_cam_marker"]).reshape(4, 4)
        assert rot_deg(T, Tt) < 1.0
        assert np.linalg.norm(T[:3, 3] - Tt[:3, 3]) < 2.0
        assert len(r["solutions"]) == 2 and r["image"]["pose"]["source"] == "marker"
        n += 1
    assert n >= 2
    _ = K


def test_locked_pose_rejects_marker_pose(server, plate):
    pid = plate["pid"]
    iid = plate["images"][1]["id"]
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/marker_pose", {})
    assert c == 200
    c, r = server.req("GET", f"/api/projects/{pid}")
    p = r["project"]
    im = next(i for i in p["images"] if i["id"] == iid)
    im["pose"]["locked"] = True
    c, r = server.req("PUT", f"/api/projects/{pid}", p)
    assert c == 200, r
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/marker_pose", {})
    assert c == 409 and r["ok"] is False and "ロック" in r["error"]
    # ロックしたまま姿勢を変える保存も拒否
    im["pose"]["T_cam_marker"] = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 500, 0, 0, 0, 1]
    c, r = server.req("PUT", f"/api/projects/{pid}", p)
    assert c == 409
    # 解除すれば通る
    im["pose"]["locked"] = False
    c, r = server.req("PUT", f"/api/projects/{pid}", p)
    assert c == 200
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/marker_pose", {})
    assert c == 200


def _plate_points():
    pts = [[x, y, z] for x in (-60, 60) for y in (-30, 30) for z in (-1.5, 1.5)]
    pts += [[-40, 6, 1.5], [-40, -6, -1.5], [50, 0, 1.5], [30, 0, -1.5]]
    return np.asarray(pts, float)


def test_pnp(server, plate, truth):
    pid, iid = plate["pid"], plate["images"][0]["id"]
    K = np.asarray(truth["camera"]["K"]).reshape(3, 3)
    T_cm = np.asarray(plate["truth"][0]["T_cam_marker"]).reshape(4, 4)
    T_mo = np.asarray(truth["T_marker_obj"]).reshape(4, 4)
    P = _plate_points()
    uv = pp.project(T_cm @ T_mo, P, K) + np.random.default_rng(1).normal(0, 0.2, (len(P), 2))
    cs = [{"uv_und": u.tolist(), "p_obj": p.tolist()} for u, p in zip(uv, P)]
    body = {"correspondences": cs, "T_marker_obj": truth["T_marker_obj"]}
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/pnp", body)
    assert c == 200, r
    assert not r["degenerate"] and not r["ambiguous"] and r["written"] is False
    T = np.asarray(r["solutions"][0]["T_cam_marker"]).reshape(4, 4)
    assert rot_deg(T, T_cm) < 0.5 and np.linalg.norm(T[:3, 3] - T_cm[:3, 3]) < 2.0
    assert len(r["solutions"][0]["residuals_px"]) == len(P)
    # 3 点は縮退
    body3 = {"correspondences": cs[:3], "T_marker_obj": truth["T_marker_obj"]}
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/pnp", body3)
    assert c == 200 and r["degenerate"] is True
    # 遠くの小さな正方形（同一平面・正面向き）は 2 解が拮抗する → ambiguous
    sq = np.array([[-10, 10, 0], [10, 10, 0], [10, -10, 0], [-10, -10, 0]], float)
    Tf = np.eye(4)
    Tf[:3, :3] = cv2.Rodrigues(np.array([0.05, -0.04, 0.0]))[0]
    Tf[:3, 3] = [5, -3, 900]
    uv = pp.project(Tf, sq, K) + np.random.default_rng(2).normal(0, 0.3, (4, 2))
    body4 = {"correspondences": [{"uv_und": u.tolist(), "p_obj": p.tolist()} for u, p in zip(uv, sq)],
             "T_marker_obj": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/pnp", body4)
    assert c == 200 and r["coplanar"] and len(r["solutions"]) >= 2 and r["ambiguous"] is True
    # 姿勢はプロジェクトに書かれない
    c, p = server.req("GET", f"/api/projects/{pid}")
    im = next(i for i in p["project"]["images"] if i["id"] == iid)
    assert im["pose"]["source"] != "pnp"


def _poly_mask(shape, T_cam_obj, K, poly_obj):
    uv = pp.project(T_cam_obj, np.asarray(poly_obj, float), K)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(uv * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    return m.astype(bool)


def test_hole_candidates_and_dark_disc(server, plate, truth):
    """穴 2 つはラベル 3、黒い塗装の円（穴ではない）はラベル 3 にならない。"""
    K = np.asarray(truth["camera"]["K"]).reshape(3, 3)
    pid = plate["pid"]
    sh = truth["shape"]
    for im, t in zip(plate["images"], plate["truth"]):
        c, raw = server.req("GET", f"/files/{pid}/images/{im['id']}/labels_auto.png", raw=True)
        assert c == 200
        lab = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        T = np.asarray(t["T_cam_obj"]).reshape(4, 4)
        z = 1.5 if t["marker_visible"] else -1.5          # 見えている面
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        circ = [[-40 + 4.5 * math.cos(a), 4.5 * math.sin(a), z] for a in ang]          # 穴（半径 6）の内側
        slot = [[40 + 6 * math.cos(a) * 1.0, 2.5 * math.sin(a), z] for a in ang]        # 長穴の内側
        for poly, name in ((circ, "円穴"), (slot, "長穴")):
            m = _poly_mask(lab.shape, T, K, poly)
            frac = float((lab[m] == 3).mean())
            assert frac > 0.6, f"{t['file']} {name}: ラベル 3 の割合 {frac:.2f}"
        cnt = [c for c in im["contours"] if c["kind"] == "hole"]
        assert len(cnt) == 2, f"{t['file']}: 穴の輪郭候補 {len(cnt)}"
        if t["marker_visible"]:
            d = sh["dark_disc_not_hole"]
            disc = [[d["center_obj"][0] + 4 * math.cos(a), d["center_obj"][1] + 4 * math.sin(a), 1.5] for a in ang]
            m = _poly_mask(lab.shape, T, K, disc)
            assert float((lab[m] == 3).mean()) < 0.05
            assert float((lab[m] == 2).mean()) > 0.9


def test_dark_region_is_not_a_hole_grabcut():
    """背景画像なし（GrabCut）: 物体上の暗い円は穴にしない。背景の見える本物の穴は穴にする。"""
    rng = np.random.default_rng(3)
    h, w = 480, 640
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (90, 130, 165)
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    for _ in range(25):
        cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(5, 25)),
                   rng.uniform(80, 200, 3).tolist(), -1)
    obj = np.zeros((h, w), np.uint8)
    cv2.rectangle(obj, (180, 140), (460, 340), 1, -1)
    img[obj.astype(bool)] = (215, 205, 190)
    hole = np.zeros((h, w), np.uint8)
    cv2.circle(hole, (250, 240), 22, 1, -1)
    bgcopy = np.zeros_like(img)
    bgcopy[:] = (90, 130, 165)
    bgcopy += rng.normal(0, 6, img.shape).astype(np.float32)
    img[hole.astype(bool)] = bgcopy[hole.astype(bool)]
    dark = np.zeros((h, w), np.uint8)
    cv2.circle(dark, (390, 240), 22, 1, -1)
    img[dark.astype(bool)] = (25, 25, 28)
    img = np.clip(img, 0, 255).astype(np.uint8)
    lab, notes, method = pp.auto_labels(img, np.ones((h, w), bool), [], None)
    assert method == "grabcut"
    inner = lambda m: cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)  # noqa: E731
    assert (lab[inner(dark)] == 3).mean() < 0.05
    assert (lab[inner(hole)] == 3).mean() > 0.8
    assert (lab[inner(obj) & ~hole.astype(bool) & ~dark.astype(bool)] == 2).mean() > 0.95


def test_save_load_autosave(server):
    c, r = server.req("POST", "/api/projects", {"name": "保存テスト"})
    p = r["project"]
    pid = p["id"]
    p["name"] = "保存テスト 2"
    p["shape"]["parts"].append({"id": "p1", "name": "板", "op": "add", "visible": True,
                                "transform": {"t_mm": [0, 0, 0], "r_deg": [0, 0, 0]},
                                "kind": "box", "params": {"size": [10, 20, 3]}})
    p["measurements"].append({"id": "m1", "label": "板厚", "part": "p1", "param": "size.2", "value_mm": 3.0,
                              "locked": True})
    c, r = server.req("PUT", f"/api/projects/{pid}", p)
    assert c == 200
    c, r = server.req("GET", f"/api/projects/{pid}")
    q = r["project"]
    assert q["name"] == "保存テスト 2" and q["shape"]["parts"][0]["params"]["size"] == [10, 20, 3]
    assert q["measurements"][0]["locked"] is True and q["autosave_available"] is False
    # 自動保存
    q["attributes"]["notes"] = "自動保存の内容"
    c, r = server.req("PUT", f"/api/projects/{pid}?autosave=1", q)
    assert c == 200
    c, r = server.req("GET", f"/api/projects/{pid}")
    assert r["project"]["autosave_available"] is True and r["project"]["attributes"]["notes"] == ""
    c, r = server.req("GET", f"/api/projects/{pid}?autosave=1")
    assert r["project"]["attributes"]["notes"] == "自動保存の内容"
    c, r = server.req("PUT", f"/api/projects/{pid}", r["project"])
    assert c == 200 and r["project"]["autosave_available"] is False
    c, r = server.req("GET", f"/api/projects/{pid}")
    assert r["project"]["attributes"]["notes"] == "自動保存の内容"
    # 形の検証
    bad = dict(r["project"], version=2)
    c, r = server.req("PUT", f"/api/projects/{pid}", bad)
    assert c == 400 and "version" in r["error"]


def _plate_glb() -> bytes:
    import trimesh
    V, F = make_samples.mesh_of(make_samples.plate_manifold())
    return trimesh.Trimesh(V * 0.001, F, process=False).export(file_type="glb")


def test_export(server, plate):
    pid = plate["pid"]
    glb = _plate_glb()
    c, r = server.multipart(f"/api/projects/{pid}/export", {"options": json.dumps({"coords": "obj"})},
                            [("glb", "model.glb", glb)])
    assert c == 200, r
    m = r["model"]
    assert m["collision"]["status"] == "not_generated" and "凸包" in m["collision"]["reason"]
    v = m["verification"]
    assert v["watertight"] is True and v["through_holes"] == 2 and v["bodies"] == 1
    assert abs(v["bbox_mm"]["size"][0] - 120) < 0.01 and abs(v["bbox_mm"]["size"][2] - 3) < 0.01
    expect = 120 * 60 * 3 - math.pi * 36 * 3 - (12 * 8 + math.pi * 16) * 3
    assert abs(v["volume_mm3"] - expect) / expect < 0.01
    assert m["units"]["glb_length"] == "m" and m["glb"]["frame"] == "obj"
    assert len(m["T_marker_obj"]["mm"]) == 16
    c, raw = server.req("GET", r["files"]["json"], raw=True)
    assert c == 200 and json.loads(raw)["collision"]["status"] == "not_generated"
    c, raw = server.req("GET", r["files"]["glb"], raw=True)
    assert c == 200 and raw[:4] == b"glTF"


def test_labels_grabcut_contours(server, plate):
    pid, im = plate["pid"], plate["images"][0]
    iid = im["id"]
    c, raw = server.req("GET", f"/files/{pid}/images/{iid}/labels_auto.png", raw=True)
    lab = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    lab2 = lab.copy()
    lab2[:10, :10] = 2
    ok, png = cv2.imencode(".png", lab2)
    c, r = server.req("PUT", f"/api/projects/{pid}/images/{iid}/labels", png.tobytes(), "image/png")
    assert c == 200 and r["image"]["has_labels"] is True
    c, r = server.req("PUT", f"/api/projects/{pid}/images/{iid}/labels", cv2.imencode(".png", lab2 * 60)[1].tobytes(),
                      "image/png")
    assert c == 400
    ys, xs = np.nonzero(lab == 2)
    fg = [{"points": [[float(xs.mean()), float(ys.mean())]], "radius": 6}]
    bg = [{"points": [[20, 900], [300, 900]], "radius": 8}]
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/grabcut", {"fg_strokes": fg, "bg_strokes": bg})
    assert c == 200, r
    c, raw = server.req("GET", f"/files/{pid}/images/{iid}/labels.png", raw=True)
    new = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(new == 3, lab2 == 3) and np.array_equal(new == 4, lab2 == 4)   # 3・4 は保持
    c, r = server.req("POST", f"/api/projects/{pid}/images/{iid}/contours_from_labels", {})
    assert c == 200 and any(cn["kind"] == "outer" for cn in r["image"]["contours"])
    c, r = server.req("DELETE", f"/api/projects/{pid}/images/{iid}/labels")
    assert c == 200 and r["image"]["has_labels"] is False


def test_paths_capture_and_samples(server, plate, tmp_path):
    pid, iid = plate["pid"], plate["images"][0]["id"]
    for bad in (f"/files/{pid}/images/../project.json", f"/files/{pid}/images/%2e%2e/project.json",
                f"/files/{pid}/images/{iid}/../../../../etc/passwd", "/files/..%2f/x/images/a"):
        c, r = server.req("GET", bad, raw=True)
        assert c == 404, bad
    c, r = server.req("POST", f"/api/projects/{pid}/capture", {})
    assert c == 502 and r["ok"] is False and "vision" in r["error"]
    c, r = server.req("GET", "/api/samples")
    assert any(s["name"] == "plate_synthetic" and s["synthetic"] for s in r["samples"])


def test_sample_load(server, truth):
    if not (PLATE / "project" / "project.json").is_file():
        pytest.skip("取り込み済みのサンプルがない（make_samples.py を実行する）")
    c, r = server.req("POST", "/api/samples/plate_synthetic/load", {})
    assert c == 200, r
    p = r["project"]
    assert p["shape"]["parts"] == [] and "合成データ" in p["name"]
    assert all(im["pose"]["T_cam_marker"] is None for im in p["images"])
    assert len(p["images"]) == len(truth["images"]) + 1 and p["background_image"]


def test_allow_from(tmp_path):
    srv, base = start(tmp_path, allow="10.0.0.0/8")
    try:
        c, r = Client(base).req("GET", "/api/projects")
        assert c == 403 and r["ok"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def _jpeg_with_orientation(img, o: int) -> bytes:
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    ifd = struct.pack("<H", 1) + struct.pack("<HHIHH", 0x0112, 3, 1, o, 0) + struct.pack("<I", 0)
    tiff = b"II*\x00" + struct.pack("<I", 8) + ifd
    seg = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(seg) + 2) + seg
    return jpg[:2] + app1 + jpg[2:]


def test_exif_orientation_and_depth(server):
    c, r = server.req("POST", "/api/projects", {"name": "向き"})
    pid = r["project"]["id"]
    img = np.full((120, 200, 3), 200, np.uint8)
    cv2.rectangle(img, (10, 10), (60, 40), (0, 0, 255), -1)        # 左上に赤
    data = _jpeg_with_orientation(img, 6)
    assert pp.exif_orientation(data) == 6
    c, r = server.multipart(f"/api/projects/{pid}/images", {"role": "object"}, [("files[]", "rot.jpg", data)])
    assert c == 200, r
    im = r["images"][0]
    assert (im["width"], im["height"]) == (120, 200) and im["orientation"] == 1
    assert im["import_info"]["exif_orientation"] == 6 and im["file"] == "original.png"
    c, raw = server.req("GET", f"/files/{pid}/images/{im['id']}/undistorted.png", raw=True)
    u = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert u[20, 100, 2] > 200 and u[20, 100, 0] < 60                # 90° 時計回り → 右上に赤
    # 深度（16 bit mm、0 は無効）
    d = np.full((200, 120), 400, np.uint16)
    d[:50] = 0
    c, r = server.req("POST", f"/api/projects/{pid}/images/{im['id']}/depth?registered_to=rgb_und",
                      cv2.imencode(".png", d)[1].tobytes(), "image/png")
    assert c == 200, r
    c, r = server.req("GET", f"/api/projects/{pid}/images/{im['id']}/depth_points?step=2")
    assert c == 200 and r["count"] == 75 * 60
    pts = np.asarray(r["points"])
    assert np.allclose(pts[:, 2], 400) and r["unit"] == "mm"
    c, r = server.req("POST", f"/api/projects/{pid}/images/{im['id']}/depth", np.zeros((5, 5), np.uint16).tobytes(),
                      "application/octet-stream")
    assert c == 422 and r["ok"] is False


def test_camera_change_redoes_undistortion(server, truth):
    """校正なしで取り込み → 校正を設定すると、und の四隅・人のラベル・輪郭が新しい歪み補正へ写り、姿勢が解き直される。"""
    c, r = server.req("POST", "/api/projects", {"name": "校正あと"})
    p = r["project"]
    pid = p["id"]
    p["marker"] = dict(truth["marker"])
    server.req("PUT", f"/api/projects/{pid}", p)
    t = truth["images"][0]
    c, r = server.multipart(f"/api/projects/{pid}/images", {}, [("files[]", t["file"], (PLATE / "images" / t["file"]).read_bytes())])
    assert c == 200 and any("校正がない" in n for n in r["notes"])
    im = r["images"][0]
    mk = im["markers"][0]
    assert np.allclose(mk["corners_raw"], mk["corners_und"])          # 校正なし: raw = und
    c, r = server.req("POST", f"/api/projects/{pid}/images/{im['id']}/marker_pose", {})
    assert c == 200 and "校正なし" in r["pose"]["quality"]["note"]
    lab = np.zeros((im["height"], im["width"]), np.uint8)
    lab[900:950, 1200:1270] = 2                                         # 右下の隅（歪みで動く）
    server.req("PUT", f"/api/projects/{pid}/images/{im['id']}/labels", cv2.imencode(".png", lab)[1].tobytes(), "image/png")
    cam = truth["camera"]
    c, r = server.req("POST", f"/api/projects/{pid}/camera",
                      {"K": cam["K"], "dist": cam["dist"], "width": cam["width"], "height": cam["height"]})
    assert c == 200, r
    im2 = next(i for i in r["project"]["images"] if i["id"] == im["id"])
    assert np.abs(np.asarray(im2["markers"][0]["corners_und"]) - np.asarray(t["corners_und"])).max() < 1.5
    T = np.asarray(im2["pose"]["T_cam_marker"]).reshape(4, 4)
    assert np.linalg.norm(T[:3, 3] - np.asarray(t["T_cam_marker"]).reshape(4, 4)[:3, 3]) < 2.0
    c, raw = server.req("GET", f"/files/{pid}/images/{im['id']}/labels.png", raw=True)
    l2 = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    ys, xs = np.nonzero(l2 == 2)
    assert len(xs) > 0 and xs.mean() > 1235 - 0 and ys.mean() > 925      # 樽型の補正で外側へ動く
    c, r = server.req("POST", f"/api/projects/{pid}/camera", {"K": cam["K"], "dist": [], "width": 640, "height": 480})
    assert c == 400 and "サイズ" in r["error"]


def test_capture_from_fake_vision(tmp_path, truth):
    """vision サービスの代わり（/state.json と /raw.png だけ）から 1 枚取り込む。他のパスは要求しない。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []
    img = cv2.imread(str(PLATE / "images" / truth["images"][0]["file"]))
    png = cv2.imencode(".png", img)[1].tobytes()
    cam = truth["camera"]
    state = json.dumps({"intrinsics": {"K": np.asarray(cam["K"]).reshape(3, 3).tolist(), "dist_coeffs": cam["dist"],
                                       "image": [cam["width"], cam["height"]], "source": "factory"}}).encode()

    class V(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(("GET", self.path))
            body, ct = {"/state.json": (state, "application/json"), "/raw.png": (png, "image/png")}.get(self.path, (b"", ""))
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", ct or "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen.append(("POST", self.path))
            self.send_response(500)
            self.end_headers()

    vs = ThreadingHTTPServer(("127.0.0.1", 0), V)
    threading.Thread(target=vs.serve_forever, daemon=True).start()
    srv = server_app.make_server("127.0.0.1", 0, tmp_path, f"http://127.0.0.1:{vs.server_port}", "127.0.0.1/32", None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cl = Client(f"http://127.0.0.1:{srv.server_port}")
        c, r = cl.req("POST", "/api/projects", {"name": "OAK"})
        pid = r["project"]["id"]
        c, r = cl.req("POST", f"/api/projects/{pid}/capture", {})
        assert c == 200, r
        assert r["images"][0]["width"] == 1280 and r["images"][0]["markers"]
        c, r = cl.req("GET", f"/api/projects/{pid}")
        assert r["project"]["cameras"]["oak"]["source"] == "vision"
        assert set(seen) <= {("GET", "/state.json"), ("GET", "/raw.png")}
    finally:
        srv.shutdown()
        srv.server_close()
        vs.shutdown()
        vs.server_close()


def test_keepalive_bodies_not_reused(server):
    """同じ接続で続けて送った要求が、それぞれ自分の本文を受け取る（以前は最初の本文を使い回していた）。"""
    import http.client
    import urllib.parse
    u = urllib.parse.urlsplit(server.base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    names = []
    for name in ("ka_one", "ka_two"):
        body = json.dumps({"name": name}).encode()
        c.request("POST", "/api/projects", body, {"Content-Type": "application/json", "Content-Length": str(len(body))})
        r = c.getresponse()
        names.append(json.loads(r.read())["project"]["name"])
    c.close()
    assert names == ["ka_one", "ka_two"]
