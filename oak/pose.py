"""ArUco marker pose and the block model, for `vision.py`.

Everything here is in the colour camera's frame (OpenCV convention: x right, y down, z forward,
millimetres). There is deliberately no table / arm-base frame: that needs a camera-to-base
calibration that has not been done, and faking one would make the 3D view look more trustworthy
than it is.

Marker mounts are built only from values measured by hand and written to `markers.toml`. A mount
with any field missing is reported as unregistered rather than guessed from the face size.
"""
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Block faces that carry markers, as seen in the face drawings documented in markers.toml:
# (outward normal, drawing-right, drawing-up) in object coordinates, and which block axes span it.
FACES = {
    "+Z": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (0, 1)),
    "-Y": (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), (0, 2)),
}
CORNERS = {"top-left": (-1, 1), "top-right": (1, 1), "bottom-left": (-1, -1), "bottom-right": (1, -1)}
AXIS_DIRS = ("right", "left", "up", "down")


def _missing(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v))


def rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def marker_object_points(size: float) -> np.ndarray:
    """Corner order returned by cv2.aruco (TL, TR, BR, BL), marker frame: x right, y up, z out."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


# --------------------------------------------------------------------------- config

@dataclass
class MarkerSpec:
    id: int
    role: str                      # "block" | "gripper"
    size_mm: float | None
    T_obj_marker: np.ndarray | None = None   # block markers: marker -> object frame
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    grip_T: np.ndarray | None = None         # gripper: marker -> grip centre, if calibrated


@dataclass
class Config:
    dictionary: str
    block_size: np.ndarray
    markers: dict[int, MarkerSpec]
    mtime: float


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    block_size = np.array(raw["block"]["size_mm"], float)
    markers = {}
    for key, m in raw.get("marker", {}).items():
        mid = int(key)
        size = None if _missing(m.get("size_mm")) else float(m["size_mm"])
        spec = MarkerSpec(mid, m["role"], size)
        if size is None:
            spec.missing.append("size_mm（黒枠外寸）")
        if spec.role == "block":
            _block_mount(spec, m, block_size)
        elif spec.role == "gripper":
            g = m.get("grip", {})
            vals = list(g.get("translation_mm", [])) + list(g.get("rotation_rpy_deg", []))
            if g.get("calibrated") and len(vals) == 6 and not any(_missing(v) for v in vals):
                r = np.radians(vals[3:])
                R = (_rot("z", r[2]) @ _rot("y", r[1]) @ _rot("x", r[0]))
                spec.grip_T = rigid(R, vals[:3])
        markers[mid] = spec
    return Config(raw.get("dictionary", "4X4_50"), block_size, markers, path.stat().st_mtime)


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _block_mount(spec: MarkerSpec, m: dict, block_size: np.ndarray) -> None:
    face = m.get("face", "")
    if face not in FACES:
        spec.errors.append(f"face={face!r} は未対応（{'/'.join(FACES)}）")
        return
    for name, label in (("corner", "corner（寄っている面の角）"),
                        ("offset_x_mm", "offset_x_mm（左右の辺からの距離）"),
                        ("offset_y_mm", "offset_y_mm（上下の辺からの距離）"),
                        ("x_axis", "x_axis（赤軸が面図で向く方向）")):
        if _missing(m.get(name)):
            spec.missing.append(label)
    if spec.missing:
        return
    if m["corner"] not in CORNERS:
        spec.errors.append(f"corner={m['corner']!r} は {'/'.join(CORNERS)} のどれか")
    if m["x_axis"] not in AXIS_DIRS:
        spec.errors.append(f"x_axis={m['x_axis']!r} は {'/'.join(AXIS_DIRS)} のどれか")
    if spec.errors:
        return

    normal, right, up, (ia, ib) = FACES[face]
    W, H = block_size[ia], block_size[ib]          # face extent along drawing-right / drawing-up
    s, dx, dy = spec.size_mm, float(m["offset_x_mm"]), float(m["offset_y_mm"])
    if dx < 0 or dy < 0 or dx + s > W or dy + s > H:
        spec.errors.append(f"マーカー {s:g} mm + オフセット ({dx:g}, {dy:g}) が面 {W:g} x {H:g} に収まらない")
        return

    su, sv = CORNERS[m["corner"]]
    u = su * (W / 2 - dx - s / 2)                  # marker centre in drawing coordinates
    v = sv * (H / 2 - dy - s / 2)
    centre = normal * (block_size @ np.abs(normal)) / 2 + u * right + v * up

    x = {"right": right, "left": -right, "up": up, "down": -up}[m["x_axis"]]
    z = normal
    spec.T_obj_marker = rigid(np.column_stack([x, np.cross(z, x), z]), centre)


def missing_report(cfg: Config) -> list[str]:
    lines = []
    for mid, spec in sorted(cfg.markers.items()):
        for item in spec.missing:
            lines.append(f"ID {mid}: {item}")
        for e in spec.errors:
            lines.append(f"ID {mid}: エラー: {e}")
        if spec.role == "gripper" and spec.grip_T is None:
            lines.append(f"ID {mid}: ID 1 → 把持中心の変換は未校正")
    return lines


# --------------------------------------------------------------------------- observation

@dataclass
class Observation:
    id: int
    corners: np.ndarray            # (4, 2) px
    T_cam_marker: np.ndarray       # translation in mm when size is known, else for a 1 mm marker
    size_known: bool
    reproj_px: float
    ambiguity: float               # reproj error ratio of best / second IPPE solution (low = clear)
    depth_mm: float | None = None  # median stereo depth inside the marker
    size_from_depth_mm: float | None = None


def solve_marker(corners: np.ndarray, size: float | None, K: np.ndarray,
                 dist: np.ndarray) -> tuple[np.ndarray, float, float]:
    obj = marker_object_points(size or 1.0)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners.astype(np.float32), K, dist,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    errs = np.asarray(errs).ravel()
    order = np.argsort(errs)
    best = order[0]
    R, _ = cv2.Rodrigues(rvecs[best])
    amb = float(errs[order[0]] / errs[order[1]]) if n > 1 and errs[order[1]] > 0 else 0.0
    return rigid(R, tvecs[best].ravel()), float(errs[best]), amb


def depth_inside(depth: np.ndarray, corners: np.ndarray, shrink: float = 0.6) -> float | None:
    """Median valid depth over the central part of the marker polygon (0 = no stereo match)."""
    c = corners.mean(0)
    poly = ((corners - c) * shrink + c).astype(np.int32)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    vals = depth[(mask > 0) & (depth > 0)]
    return float(np.median(vals)) if vals.size >= 20 else None


def observe(corners_list, ids, cfg: Config, K, dist, depth) -> list[Observation]:
    out = []
    for c, mid in zip(corners_list, ids):
        c = c.reshape(4, 2)
        spec = cfg.markers.get(int(mid))
        size = spec.size_mm if spec else None
        T, err, amb = solve_marker(c, size, K, dist)
        o = Observation(int(mid), c, T, size is not None, err, amb)
        if depth is not None:
            o.depth_mm = depth_inside(depth, c)
            if o.depth_mm:
                # PnP translation scales linearly with the marker size, so stereo depth at the
                # marker centre gives an independent estimate of the printed size.
                z_unit = T[2, 3] / (size or 1.0)
                o.size_from_depth_mm = o.depth_mm / z_unit if z_unit > 0 else None
        out.append(o)
    return out


def block_estimates(obs: list[Observation], cfg: Config) -> dict[int, np.ndarray]:
    """Block pose in the camera frame, one per visible block marker with a registered mount."""
    est = {}
    for o in obs:
        spec = cfg.markers.get(o.id)
        if spec and spec.role == "block" and spec.T_obj_marker is not None and o.size_known:
            est[o.id] = o.T_cam_marker @ invert(spec.T_obj_marker)
    return est


def compare(Ta: np.ndarray, Tb: np.ndarray) -> dict:
    d = invert(Ta) @ Tb
    return {"dt_mm": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
            "dt_obj_mm": d[:3, 3].round(2).tolist(),
            "drot_deg": rotation_angle_deg(d[:3, :3])}


def block_edges(size: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = size / 2
    pts = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(pts[i] != pts[j]) == 1]
    return pts, edges
