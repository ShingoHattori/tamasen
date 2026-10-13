"""ArUco marker pose, configuration and the block model, for `vision.py`.

Units are metres and radians everywhere in this module (markers.toml is written in mm / deg and
converted on load). Transforms are 4x4 and named T_a_b: they map points in frame b into frame a,
p_a = T_a_b @ p_b. Frames: cam (OpenCV colour camera), m<ID> (marker), world (= m2, provisional),
obj (block centre), base (arm base, not calibrated against the camera).

Nothing here guesses: a value whose *_status is "unset" (or that is nan / "") is reported as
missing and is not replaced by 0, an identity transform or a size estimated from the image.
"""
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MM = 1e-3
STATUSES = ("unset", "sample", "measured", "calibrated")
USABLE = ("sample", "measured", "calibrated")   # "sample" is usable but always shown as provisional

# Block faces that carry markers, as seen in the face drawings documented in markers.toml:
# (outward normal, drawing-right, drawing-up) in object coordinates, and which block axes span it.
FACES = {
    "+Z": (np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), (1, 0)),
    "-X": (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0]), (1, 2)),
}
CORNERS = {"top-left": (-1, 1), "top-right": (1, 1), "bottom-left": (-1, -1), "bottom-right": (1, -1)}
AXIS_DIRS = ("right", "left", "up", "down")

# Observation quality thresholds.
MAX_REPROJ_PX = 1.5        # mean corner reprojection error of the chosen solution
AMBIGUOUS_RATIO = 0.6      # IPPE: err(best) / err(second) above this = the two tilts are near-equal
MIN_SIDE_PX = 14           # shorter than this, the corners are too coarse for orientation
FLIP_DEG = 25.0            # normal jump vs the last accepted pose that counts as a flip
FLIP_WINDOW_S = 0.5        # how recent "last accepted" must be to judge a flip


def _missing(v) -> bool:
    if isinstance(v, np.ndarray):
        return bool(np.isnan(v).any())
    return v is None or v == "" or (isinstance(v, float) and math.isnan(v))


def rigid(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return rigid(R.T, -R.T @ t)


def rotation_angle(R: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def rpy_matrix(rpy) -> np.ndarray:
    """Fixed-axis roll-pitch-yaw (x, then y, then z), the URDF convention."""
    return rot("z", rpy[2]) @ rot("y", rpy[1]) @ rot("x", rpy[0])


def marker_object_points(size: float) -> np.ndarray:
    """Corner order returned by cv2.aruco (TL, TR, BR, BL), marker frame: x right, y up, z out."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


# --------------------------------------------------------------------------- config

@dataclass
class Value:
    """A configured quantity together with how far it can be trusted."""
    value: object
    status: str

    @property
    def usable(self) -> bool:
        return self.status in USABLE and not _missing(self.value)

    def json(self):
        v = self.value
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, float) and math.isnan(v):
            v = None
        return {"value": v, "status": self.status}


@dataclass
class MarkerSpec:
    id: int
    role: str                       # "block" | "gripper" | "world_reference"
    size: Value                     # m
    T_obj_marker: np.ndarray | None = None    # block markers: marker -> object frame
    mount_status: str = "unset"
    mount_evidence: str = ""
    grip: Value | None = None       # gripper: T_m1_grip (marker -> grip centre)
    card_thickness: Value | None = None
    physically_fixed: bool = False
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def size_m(self) -> float | None:
        return float(self.size.value) if self.size.usable else None


@dataclass
class Config:
    dictionary: str
    block_size: Value               # (3,) m
    markers: dict[int, MarkerSpec]
    mtime: float
    raw: dict
    grip_jig: "MarkerSpec | None" = None

    def with_jig(self) -> "Config":
        """Copy in which the jig id is the grip jig (not a block marker): used in jig mode only."""
        import copy
        c = copy.copy(self)
        c.markers = dict(self.markers)
        if self.grip_jig is not None:
            c.markers[self.grip_jig.id] = self.grip_jig
        return c

    def json(self) -> dict:
        out = {"dictionary": self.dictionary, "block_size": self.block_size.json(), "markers": {}}
        for mid, s in self.markers.items():
            d = {"role": s.role, "size": s.size.json(), "mount_status": s.mount_status,
                 "mount_evidence": s.mount_evidence,
                 "T_obj_marker": None if s.T_obj_marker is None else s.T_obj_marker.tolist()}
            if s.grip is not None:
                d["grip"] = {"status": s.grip.status,
                             "T_m1_grip": s.grip.value.tolist() if s.grip.usable else None}
            if s.card_thickness is not None:
                d["card_thickness"] = s.card_thickness.json()
                d["physically_fixed"] = s.physically_fixed
            out["markers"][str(mid)] = d
        return out


def _status(m: dict, key: str, label: str, spec: MarkerSpec | None = None) -> str:
    s = m.get(key, "unset")
    if s not in STATUSES:
        if spec is not None:
            spec.errors.append(f"{key}={s!r} は {'/'.join(STATUSES)} のどれか")
        return "unset"
    return s


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    b = raw["block"]
    bsize = np.array(b["size_mm"], float) * MM
    block_size = Value(bsize, b.get("size_status", "unset"))
    markers = {}
    for key, m in raw.get("marker", {}).items():
        mid = int(key)
        spec = MarkerSpec(mid, m.get("role", "unknown"), Value(None, "unset"))
        st = _status(m, "size_status", "size", spec)
        size = m.get("size_mm")
        spec.size = Value(None if _missing(size) else float(size) * MM, st)
        if not spec.size.usable:
            spec.missing.append("size_mm（黒枠の一辺）が未入力")
        if spec.role == "block":
            _block_mount(spec, m, block_size)
        elif spec.role == "gripper":
            g = m.get("grip", {})
            gst = _status(g, "status", "grip", spec)
            vals = list(g.get("translation_mm", [])) + list(g.get("rotation_rpy_deg", []))
            T = None
            if len(vals) == 6 and not any(_missing(v) for v in vals):
                T = rigid(rpy_matrix(np.radians(vals[3:])), np.array(vals[:3]) * MM)
            spec.grip = Value(T, gst if T is not None else "unset")
            if not spec.grip.usable:
                spec.missing.append("ID 1 → 把持中心の変換 T_m1_grip は未校正")
        elif spec.role == "world_reference":
            th = m.get("card_thickness_mm")
            spec.card_thickness = Value(None if _missing(th) else float(th) * MM,
                                        _status(m, "card_thickness_status", "thickness", spec))
            spec.physically_fixed = bool(m.get("physically_fixed", False))
            if not spec.card_thickness.usable:
                spec.missing.append("カード厚み card_thickness_mm が未測定（床面の高さは未校正）")
            if not spec.physically_fixed:
                spec.missing.append("基準カードは物理的に未固定（physically_fixed = false）")
        markers[mid] = spec
    dname = raw.get("dictionary", "DICT_4X4_50")
    cfg = Config(dname, block_size, markers, path.stat().st_mtime, raw)
    j = raw.get("grip_jig")
    if j:
        spec = MarkerSpec(int(j["id"]), "grip_jig", Value(None if _missing(j.get("size_mm")) else float(j["size_mm"]) * MM,
                                                         _status(j, "size_status", "size")))
        v = j.get("jig_to_grip_mm", [])
        ok = len(v) == 3 and not any(_missing(x) for x in v)
        spec.grip = Value(rigid(np.eye(3), np.array(v) * MM) if ok else None,
                          _status(j, "jig_to_grip_status", "jig") if ok else "unset")
        cfg.grip_jig = spec
    return cfg


def block_mount(face: str, corner: str, x_axis: str, size: float, dx: float, dy: float,
                block_size: np.ndarray) -> np.ndarray:
    """T_obj_marker for a marker of side `size` sitting `dx`, `dy` in from `corner` of `face`."""
    normal, right, up, (ia, ib) = FACES[face]
    W, H = block_size[ia], block_size[ib]          # face extent along drawing-right / drawing-up
    if dx < 0 or dy < 0 or dx + size > W or dy + size > H:
        raise ValueError(f"マーカー {size / MM:g} mm + オフセット ({dx / MM:g}, {dy / MM:g}) mm が "
                         f"面 {W / MM:g} x {H / MM:g} mm に収まらない")
    su, sv = CORNERS[corner]
    u = su * (W / 2 - dx - size / 2)               # marker centre in drawing coordinates
    v = sv * (H / 2 - dy - size / 2)
    centre = normal * (block_size @ np.abs(normal)) / 2 + u * right + v * up
    x = {"right": right, "left": -right, "up": up, "down": -up}[x_axis]
    z = normal
    return rigid(np.column_stack([x, np.cross(z, x), z]), centre)


def _block_mount(spec: MarkerSpec, m: dict, block_size: Value) -> None:
    face = m.get("face", "")
    spec.mount_status = _status(m, "mount_status", "mount", spec)
    spec.mount_evidence = m.get("mount_evidence", "")
    if face not in FACES:
        spec.errors.append(f"face={face!r} は未対応（{'/'.join(FACES)}）")
        return
    if spec.mount_status == "unset":
        spec.missing.append("取り付け（corner・x_axis）が未確定（mount_status = unset）")
        return
    for name, label in (("corner", "corner（寄っている面の角）"), ("x_axis", "x_axis（赤軸の向き）"),
                        ("offset_x_mm", "offset_x_mm"), ("offset_y_mm", "offset_y_mm")):
        if _missing(m.get(name)):
            spec.missing.append(f"{label} が未入力")
    if _status(m, "offset_status", "offset", spec) == "unset":
        spec.missing.append("offset_status が unset")
    if spec.missing or not spec.size.usable or not block_size.usable:
        return
    if m["corner"] not in CORNERS:
        spec.errors.append(f"corner={m['corner']!r} は {'/'.join(CORNERS)} のどれか")
    if m["x_axis"] not in AXIS_DIRS:
        spec.errors.append(f"x_axis={m['x_axis']!r} は {'/'.join(AXIS_DIRS)} のどれか")
    if spec.errors:
        return
    try:
        spec.T_obj_marker = block_mount(face, m["corner"], m["x_axis"], spec.size_m,
                                        float(m["offset_x_mm"]) * MM, float(m["offset_y_mm"]) * MM,
                                        block_size.value)
    except ValueError as e:
        spec.errors.append(str(e))


def missing_report(cfg: Config) -> list[str]:
    lines = []
    if not cfg.block_size.usable:
        lines.append("ブロック寸法が未入力")
    for mid, spec in sorted(cfg.markers.items()):
        lines += [f"ID {mid}: {item}" for item in spec.missing]
        lines += [f"ID {mid}: エラー: {e}" for e in spec.errors]
    return lines


# --------------------------------------------------------------------------- observation

@dataclass
class Observation:
    id: int
    t: float                       # capture time (host clock, s)
    corners: np.ndarray            # (4, 2) px, cv2.aruco order
    side_px: float
    size_known: bool
    T_cam_marker: np.ndarray | None          # chosen solution; None if the size is unknown
    T_alt: np.ndarray | None                 # the other IPPE solution
    reproj_px: float
    reproj_alt_px: float
    ambiguity: float               # err(best) / err(second): near 1 = tilt sign undecidable
    status: str = "ok"             # ok | ambiguous | flip | rejected
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Usable as the current pose (shown normally, may feed comparisons)."""
        return self.status == "ok"

    @property
    def shown(self) -> bool:
        """Has a pose at all (ambiguous / flip poses are shown, but flagged)."""
        return self.T_cam_marker is not None and self.status != "rejected"


def reprojection_error(T: np.ndarray, obj: np.ndarray, corners: np.ndarray, K, dist) -> float:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    uv, _ = cv2.projectPoints(obj, rvec, T[:3, 3], K, dist)
    return float(np.linalg.norm(uv.reshape(-1, 2) - corners, axis=1).mean())


def solve_marker(corners: np.ndarray, size: float, K, dist):
    """Both IPPE solutions, best first, with mean corner reprojection errors in px."""
    obj = marker_object_points(size)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, corners.astype(np.float64), K, dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
    sols = []
    for i in range(n):
        R, _ = cv2.Rodrigues(rvecs[i])
        T = rigid(R, tvecs[i].ravel())
        sols.append((reprojection_error(T, obj, corners, K, dist), T))
    sols.sort(key=lambda s: s[0])
    return sols


class Observer:
    """Per-frame marker poses with quality flags; remembers the last accepted pose per id."""

    def __init__(self) -> None:
        self.last_ok: dict[int, tuple[float, np.ndarray]] = {}
        self.last_seen: dict[int, float] = {}

    def observe(self, corners_list, ids, t: float, cfg: Config, K, dist) -> list[Observation]:
        ids = [] if ids is None else [int(i) for i in np.ravel(ids)]
        counts = {i: ids.count(i) for i in ids}
        out = []
        for c, mid in zip(corners_list, ids):
            c = np.asarray(c, float).reshape(4, 2)
            side = float(min(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)))
            spec = cfg.markers.get(mid)
            size = spec.size_m if spec else None
            o = Observation(mid, t, c, side, size is not None, None, None, math.nan, math.nan, math.nan)
            self.last_seen[mid] = t
            if spec is None:
                o.status = "rejected"
                o.reasons.append("設定にない ID")
            elif size is None:
                o.status = "rejected"
                o.reasons.append("実寸が未入力のため位置を算出しない")
            if counts[mid] > 1:
                o.status = "rejected"
                o.reasons.append(f"同じ ID が {counts[mid]} 個検出された（重複は採用しない）")
            if size is not None:
                sols = solve_marker(c, size, K, dist)
                o.reproj_px, o.T_cam_marker = sols[0]
                if len(sols) > 1:
                    o.reproj_alt_px, o.T_alt = sols[1]
                    o.ambiguity = (o.reproj_px / o.reproj_alt_px) if o.reproj_alt_px > 0 else 1.0
                else:
                    o.ambiguity = 0.0
                if o.T_cam_marker[2, 3] <= 0:
                    o.status = "rejected"
                    o.reasons.append("カメラの後ろに解が出た")
            if o.status != "rejected":
                self._grade(o)
            out.append(o)
        return out

    def _grade(self, o: Observation) -> None:
        if o.reproj_px > MAX_REPROJ_PX:
            o.status = "rejected"
            o.reasons.append(f"再投影誤差 {o.reproj_px:.2f} px > {MAX_REPROJ_PX}")
            return
        if o.side_px < MIN_SIDE_PX:
            o.status = "ambiguous"
            o.reasons.append(f"マーカーが小さい（{o.side_px:.0f} px）")
        if o.ambiguity > AMBIGUOUS_RATIO:
            o.status = "ambiguous"
            o.reasons.append(f"IPPE 2 解の誤差比 {o.ambiguity:.2f}（面の傾きの向きが決めにくい）")
        prev = self.last_ok.get(o.id)
        if prev and o.t - prev[0] < FLIP_WINDOW_S:
            jump = math.degrees(math.acos(np.clip(prev[1][:3, 2] @ o.T_cam_marker[:3, 2], -1, 1)))
            if jump > FLIP_DEG:
                o.status = "flip"
                alt = ""
                if o.T_alt is not None:
                    a = math.degrees(math.acos(np.clip(prev[1][:3, 2] @ o.T_alt[:3, 2], -1, 1)))
                    alt = f"、もう一方の解なら {a:.0f}°"
                o.reasons.append(f"直前の姿勢から法線が {jump:.0f}° 反転{alt}（採用しない）")
        if o.status == "ok":
            self.last_ok[o.id] = (o.t, o.T_cam_marker)


PAIR_MAX_ROT = math.radians(10.0)


def block_estimates(obs: list[Observation], cfg: Config) -> dict[int, tuple[np.ndarray, str]]:
    """T_cam_obj per visible block marker with a registered mount, plus a status.

    Each estimate comes from one marker only; they are never averaged. When two block markers are
    visible, the rigid mount relation can decide a marker's IPPE ambiguity: if exactly one choice
    of solutions makes the two object rotations agree within PAIR_MAX_ROT, those solutions are used
    and the status becomes "pair" (decided by the other marker, not by this marker alone).
    """
    block = [(o, cfg.markers[o.id]) for o in obs
             if o.id in cfg.markers and cfg.markers[o.id].role == "block"
             and cfg.markers[o.id].T_obj_marker is not None and o.shown]
    est = {o.id: (o.T_cam_marker @ invert(s.T_obj_marker), o.status) for o, s in block}
    if len(block) == 2:
        (oa, sa), (ob, sb) = block
        options = []
        for Ta in (oa.T_cam_marker, oa.T_alt):
            for Tb in (ob.T_cam_marker, ob.T_alt):
                if Ta is None or Tb is None:
                    continue
                A, B = Ta @ invert(sa.T_obj_marker), Tb @ invert(sb.T_obj_marker)
                options.append((rotation_angle(A[:3, :3].T @ B[:3, :3]), A, B, Ta is oa.T_cam_marker, Tb is ob.T_cam_marker))
        agree = [x for x in options if x[0] < PAIR_MAX_ROT]
        if len(agree) == 1:
            _, A, B, a_best, b_best = agree[0]
            if oa.status != "ok" or not a_best:
                est[oa.id] = (A, "pair")
            if ob.status != "ok" or not b_best:
                est[ob.id] = (B, "pair")
    return est


def compare(Ta: np.ndarray, Tb: np.ndarray) -> dict:
    d = invert(Ta) @ Tb
    return {"dt_m": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
            "dt_obj_m": d[:3, 3].tolist(),
            "drot_rad": rotation_angle(d[:3, :3])}


def block_edges(size: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h = size / 2
    pts = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(pts[i] != pts[j]) == 1]
    return pts, edges
