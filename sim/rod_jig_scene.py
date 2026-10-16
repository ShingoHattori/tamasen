"""Rod jig (cad/tools/rod_jig.py) in the MuJoCo scene — for sim/check_rod_jig.py only.

The running sim service (sim_server.py) does not import this module; scene.py is not changed.

Jig frame (mm, as in the CAD): x = insertion axis (entry face +x, fingers pushed in along -x), y = closing axis,
z = finger width; origin = jig centre. The marker area is on the -z face, so the jig rests on its +z face with the
marker up (jig z = world down).

Collision geometry: exact convex pieces derived from the generator's parameters (rod_jig.py defaults, the same
formulas). Along x the pocket is piecewise linear (wedge, its kink, the 45 deg transition, the entry rectangle,
the entry chamfer); in each x interval the cavity is a rectangle [yi, yo] x [-h, h] linear in x, so the solid
band y in [15, 40.67] minus the cavity is four convex lofts (inner wall, outer wall, top, bottom). Centre band
(rod hole) = top and bottom slabs; the marker rim = 4 boxes. Screw pilot holes are ignored (on the face that
rests on the support). check_rod_jig.py geom checks the pieces against rod_jig.stl.

Fingers: the model's front / mid / rear convex hulls are coarser than the real finger where it enters the
pocket (one hull over the 4.6 mm tip and the taper behind it). With finger_mode="cad" they are replaced by
convex slices of the CAD finger (01_Finger, from fingers_closed_in_jig_frame.stl), 2.5 mm long in x.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import scene

HERE = Path(__file__).resolve().parent
JIG_DIR = HERE / "models" / "rod_jig"

# --- generator parameters (cad/tools/rod_jig.py defaults; rod_jig.json values) ---------------------------------
D, INSERT, RECT, CHAMFER = 44.0, 45.0, 20.0, 2.5
TAPER, TIP_HALF = (15.0, 5.80, 0.3155), 4.60
FINGER_T, FIT, FIT_ENTRY, SLACK = 12.74, 1.0, 1.0, 1.0
ROD_W, ROD_H, INNER, OUTER = 30.0, 20.0, 6.0, 4.0
WALL_Z = (5.0, 8.0)
SHOULDER_HALF_Z, SHOULDER_T, RECT_FIT = 16.56, 13.67, 1.0
MARKER, MARKER_MARGIN, RIM_W, RIM_H = 30.0, 5.5, 1.2, 0.8
PW = FINGER_T + FIT + SLACK                    # 14.74 wedge width (y)
PW_R = SHOULDER_T + FIT + SLACK                # 15.67 rectangle width (y)
PH2 = SHOULDER_HALF_Z + RECT_FIT               # 17.56 rectangle half height (z)
Y = ROD_W / 2 + INNER + PW_R + OUTER           # 40.67
ZL, ZH = -max(ROD_H, 2 * PH2) / 2 - WALL_Z[0], max(ROD_H, 2 * PH2) / 2 + WALL_Z[1]   # -22.56, 25.56
YS = ROD_W / 2 + INNER                         # 21: inner wall face (jaws press here)
GRIP_W = 2 * YS                                # 42
ENTRY_GAP = 43.2                               # rod_jig.json entry_gap_mm.suggested
MARKER_Z = ZL - RIM_H                          # -23.36: the marker face (sticker on the -z face inside the rim)
# placement in end_link at full insertion (CAD: jig centre at end_link x = D/2 - INSERT)
JIG_X_IN_END = D / 2 - INSERT                  # -23 mm


def set_variant(slack=None, chamfer=None, entry_gap=None, fit_z=None):
    """Design variant for the simulation: pocket slack along y (outer side, mm), entry chamfer (mm), jaw gap while
    entering (mm). Recomputes the derived sizes. The visual mesh stays the CAD one (rod_jig.stl)."""
    global SLACK, CHAMFER, ENTRY_GAP, PW, PW_R, Y, FIT, FIT_ENTRY
    if fit_z is not None:                       # z clearance per side of the wedge (tip end and entry end); y keeps FIT=1
        FIT_ENTRY = fit_z
        FIT = fit_z
    if slack is not None:
        SLACK = slack
    if chamfer is not None:
        CHAMFER = chamfer
    PW = FINGER_T + 1.0 + SLACK
    PW_R = SHOULDER_T + 1.0 + SLACK
    Y = ROD_W / 2 + INNER + PW_R + OUTER
    if entry_gap is not None:
        ENTRY_GAP = entry_gap
    return {"slack": SLACK, "chamfer": CHAMFER, "fit_z": FIT, "entry_gap": ENTRY_GAP, "pocket_y_wedge": PW, "jig_width_y": 2 * Y,
            "inner_clear_wedge": ENTRY_GAP / 2 - YS, "outer_clear_wedge": YS + PW - ENTRY_GAP / 2 - FINGER_T}


def lerp_half(dx):
    return max(TAPER[1] + TAPER[2] * (dx - TAPER[0]), TIP_HALF)


def pocket_sections():
    """x breakpoints with (yi, yo, h) of the cavity (one side, y >= 0), linear between them."""
    tip = D / 2 - INSERT
    kink = tip + TAPER[0] + (TIP_HALF - TAPER[1]) / TAPER[2]
    xb = D / 2 - RECT
    hw0 = lerp_half(INSERT - RECT) + FIT_ENTRY
    t = max((PH2 - hw0) / (1 - TAPER[2]), PW_R - PW)
    xt = xb - t
    hw = lerp_half(INSERT - RECT - t) + FIT_ENTRY
    xc = D / 2 - CHAMFER
    h = lambda x: lerp_half(x - tip) + FIT
    return [(-D / 2, YS, YS + PW, h(-D / 2)), (kink, YS, YS + PW, h(kink)), (xt, YS, YS + PW, hw),
            (xb, YS, YS + PW_R, PH2), (xc, YS, YS + PW_R, PH2), (D / 2, YS - CHAMFER, YS + PW_R + CHAMFER, PH2 + CHAMFER)]


def _loft(x0, r0, x1, r1):
    """Convex hull vertices of rectangles r = (y0, y1, z0, z1) at x0 and x1."""
    return np.array([[x, y, z] for x, r in ((x0, r0), (x1, r1)) for y in r[:2] for z in r[2:]])


def jig_pieces() -> list[tuple[str, np.ndarray]]:
    """Convex pieces (name, vertices mm, jig frame) whose union is the jig body (no screw holes)."""
    out = [("centre_top", _loft(-D / 2, (-ROD_W / 2, ROD_W / 2, ROD_H / 2, ZH), D / 2, (-ROD_W / 2, ROD_W / 2, ROD_H / 2, ZH))),
           ("centre_bot", _loft(-D / 2, (-ROD_W / 2, ROD_W / 2, ZL, -ROD_H / 2), D / 2, (-ROD_W / 2, ROD_W / 2, ZL, -ROD_H / 2)))]
    secs = pocket_sections()
    for s, sn in ((1, "p"), (-1, "n")):
        for i in range(len(secs) - 1):
            (xa, yia, yoa, ha), (xb, yib, yob, hb) = secs[i], secs[i + 1]
            parts = {"inner": ((ROD_W / 2, yia, ZL, ZH), (ROD_W / 2, yib, ZL, ZH)),
                     "outer": ((yoa, Y, ZL, ZH), (yob, Y, ZL, ZH)),
                     "top": ((yia, yoa, ha, ZH), (yib, yob, hb, ZH)),
                     "bot": ((yia, yoa, ZL, -ha), (yib, yob, ZL, -hb))}
            for k, (ra, rb) in parts.items():
                v = _loft(xa, ra, xb, rb)
                v[:, 1] *= s
                out.append((f"{sn}{i}_{k}", v))
    o, n = MARKER / 2 + MARKER_MARGIN + RIM_W, MARKER / 2 + MARKER_MARGIN
    z0, z1 = ZL - RIM_H, ZL
    for k, r in (("rim_a", (-o, o, n, o)), ("rim_b", (-o, o, -o, -n)), ("rim_c", (n, o, -n, n)), ("rim_d", (-o, -n, -n, n))):
        out.append((k, np.array([[x, y, z] for x in r[:2] for y in r[2:] for z in (z0, z1)])))
    return out


def T_end_jig(depth_mm: float = INSERT) -> np.ndarray:
    """Jig pose in end_link with ID 1 up (end_link z = -jig z, x = -jig x, y = jig y) at insertion depth
    depth_mm (45 = full; smaller = gripper backed out along -end x)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    T[:3, 3] = [(JIG_X_IN_END + (INSERT - depth_mm)) * 1e-3, 0, 0]
    return T


def T_world_jig(px, py, yaw, support_h=0.0, tilt=0.0) -> np.ndarray:
    """Jig resting on its +z face (marker up); yaw = direction of jig +x (the entry face normal) in the world;
    tilt = rotation about jig y that raises the entry face (rad); support_h = height of the centre of the +z
    face (the support's top) above the table."""
    c, s_ = math.cos(yaw), math.sin(yaw)
    R = np.column_stack([[c, s_, 0], [s_, -c, 0], [0, 0, -1]])
    if tilt:
        R = R @ _rot([0, 1, 0], tilt)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([px, py, support_h]) - R @ np.array([0, 0, ZH * 1e-3])
    return T


def support_for(T_jig: np.ndarray, support_h: float) -> dict | None:
    """Static box under the jig, aligned with it, top = the jig's +z face, reaching down into the table."""
    if support_h <= 0:
        return None
    R = T_jig[:3, :3]
    hs = support_h / max(abs(R[2, 2]), 0.3) + 0.03
    half = np.array([D / 2 + 1.0, Y + 1.0, 0]) * 1e-3
    half[2] = hs / 2
    pos = T_jig[:3, 3] + R @ np.array([0, 0, ZH * 1e-3 + hs / 2])
    return {"h": support_h, "pos": pos, "R": R, "half": half, "lip": abs(R[2, 2]) < 0.999}


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def cad_finger_slices(m, d, ids, slice_mm=2.5):
    """Convex slices of the CAD fingers in each finger body's frame: {'left': [verts...], 'right': [...]}.
    Needs trimesh (analysis venv)."""
    import trimesh
    fm = trimesh.load(JIG_DIR / "fingers_closed_in_jig_frame.stl")
    gap_cad = 42.4                                               # rod_jig.py: the STL is the closed fingers at 42.4 mm
    # jig (CAD orientation: jig = Rz(pi) end_link, centre at end x = -23) -> end_link
    Rcad = np.diag([-1.0, -1.0, 1.0])
    out = {}
    q = (gap_cad * 1e-3 - scene._gap_model(m).gap0) / 2
    d.qpos[ids.fl_q], d.qpos[ids.fr_q] = q, -q
    mujoco.mj_kinematics(m, d)
    Re, pe = d.xmat[ids.end_body].reshape(3, 3), d.xpos[ids.end_body]
    for part in fm.split():
        Pe = (Rcad @ part.vertices.T).T + [JIG_X_IN_END, 0, 0]        # mm, end_link
        side = "left" if np.median(Pe[:, 1]) > 0 else "right"
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"finger_{side}_link")
        Rb, pb = d.xmat[body].reshape(3, 3), d.xpos[body]
        Pw = (Re @ (Pe * 1e-3).T).T + pe
        Pb = (Rb.T @ (Pw - pb).T).T                                    # m, finger body frame
        # end_link x is the slicing axis; express it in the body frame (bodies are aligned with end_link)
        ax = Rb.T @ Re[:, 0]
        mesh_b = trimesh.Trimesh(Pb, part.faces, process=True)
        xs = Pb @ ax
        lo, hi = xs.min(), xs.max()
        n = max(1, int(math.ceil((hi - lo) / (slice_mm * 1e-3))))
        pieces = []
        for i in range(n):
            a, b = lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n
            sl = trimesh.intersections.slice_mesh_plane(mesh_b, ax, ax * a, cap=False)
            sl = trimesh.intersections.slice_mesh_plane(sl, -ax, ax * b, cap=False) if len(sl.vertices) else sl
            if len(sl.vertices) >= 4:
                pieces.append(np.asarray(sl.convex_hull.vertices))
        out[side] = {"pieces": pieces, "mesh_body": mesh_b, "body": f"finger_{side}_link"}
    return out


SIM_DATA = JIG_DIR / "sim_data.json"        # precomputed by check_rod_jig.py export: CAD finger slices, unit inertia
PARK = (3.0, -3.0)                            # where the jig and its support wait when not in use (world x, y)


def load_sim_data() -> dict:
    """CAD finger slices (finger body frames, m) and the jig's inertia per kg, precomputed so that sim/.venv
    (no trimesh) can build the scene."""
    import json
    return json.loads(SIM_DATA.read_text())


def add_jig(root: ET.Element, opts: dict) -> None:
    """Add the jig (free body: 46 convex collision pieces, visual mesh, marker, a bounding box for distance checks),
    its support (static box + stop lip, geoms named obst_* so the path checks treat them as obstacles), an optional
    OAK camera and optional CAD finger slices, to a scene built by scene.build_xml.

    opts: jig_T (4x4 world), support (support_for() dict or None = parked), mass_kg, friction, inertia
    ({com_mm, full} for mass_kg), finger_mode ('model' | 'cad'), finger_friction, finger_slices
    ({'left': [verts], 'right': [verts]} in the finger body frames, m), cam (T_base_cam, fovy), solref."""
    asset, wb = root.find("asset"), root.find("worldbody")
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    fl = opt.find("flag")
    if fl is None:
        fl = ET.SubElement(opt, "flag")
    if opts.get("multiccd", True):
        fl.set("multiccd", "enable")         # several contacts per convex pair (flat faces sliding on each other)
    sz = root.find("size")
    if sz is None:
        sz = ET.SubElement(root, "size")
    sz.set("memory", "64M")
    if opts.get("noslip"):
        opt.set("noslip_iterations", str(opts["noslip"]))
    fr = opts.get("friction", 0.35)
    # --- jig body
    T = opts["jig_T"]
    jb = ET.SubElement(wb, "body", name="jig", pos=_fmt(T[:3, 3]), quat=_fmt(_quat(T[:3, :3])))
    ET.SubElement(jb, "freejoint", name="jig_free")
    I = opts["inertia"]
    ET.SubElement(jb, "inertial", pos=_fmt(np.array(I["com_mm"]) * 1e-3), mass=str(opts["mass_kg"]),
                  fullinertia=_fmt(I["full"]))
    ET.SubElement(asset, "mesh", name="jig_visual_mesh", file=str((JIG_DIR / "rod_jig.stl").resolve()), scale="0.001 0.001 0.001")
    ET.SubElement(jb, "geom", name="jig_visual", type="mesh", mesh="jig_visual_mesh", group="2", contype="0",
                  conaffinity="0", mass="0", rgba="0.85 0.55 0.20 1")
    for name, v in jig_pieces():
        ET.SubElement(asset, "mesh", name=f"jigc_{name}", vertex=_fmt(v * 1e-3))
        ET.SubElement(jb, "geom", name=f"jig_{name}", type="mesh", mesh=f"jigc_{name}", group="0", contype="1",
                      conaffinity="1", condim="3", priority="1", friction=f"{fr} 0.005 0.0001", mass="0",
                      solref=opts.get("solref", "0.002 1"), solimp="0.95 0.99 0.0005", rgba="0.85 0.55 0.2 0.0")
    # outer box for distance checks of a held jig (never collides, not drawn)
    zc = (ZH + MARKER_Z) / 2 * 1e-3
    ET.SubElement(jb, "geom", name="jig_bbox", type="box", group="6", contype="0", conaffinity="0", mass="0",
                  pos=f"0 0 {zc:.5f}", size=_fmt(np.array([D / 2, Y, (ZH - MARKER_Z) / 2]) * 1e-3), rgba="1 1 1 0")
    # marker: white area + black square (visual), site at the centre (x = jig x, y = -jig y, z out = -jig z)
    h = MARKER / 2 * 1e-3
    zf = MARKER_Z * 1e-3
    ET.SubElement(jb, "geom", name="jig_marker_white", type="box", group="2", contype="0", conaffinity="0", mass="0",
                  pos=f"0 0 {zf + 0.00005}", size=f"{(MARKER / 2 + MARKER_MARGIN) * 1e-3:.5f} {(MARKER / 2 + MARKER_MARGIN) * 1e-3:.5f} 0.0001",
                  rgba="0.97 0.97 0.97 1")
    ET.SubElement(jb, "geom", name="jig_marker_black", type="box", group="2", contype="0", conaffinity="0", mass="0",
                  pos=f"0 0 {zf - 0.0001}", size=f"{h:.5f} {h:.5f} 0.0001", rgba="0.02 0.02 0.02 1")
    ET.SubElement(jb, "site", name="jig_marker", pos=f"0 0 {zf}", quat=_fmt(_quat(np.diag([1.0, -1.0, -1.0]))),
                  size="0.004", rgba="0 1 1 1", group="1")
    # --- support (static box under the jig, always present; parked when there is none)
    sp = opts.get("support") or {"h": 0.0, "pos": np.array([PARK[0], PARK[1], -0.5]), "R": np.eye(3),
                                  "half": np.array([D / 2 + 1.0, Y + 1.0, 10.0]) * 1e-3, "lip": False}
    b = ET.SubElement(wb, "body", name="jig_support", pos=_fmt(sp["pos"]), quat=_fmt(_quat(sp.get("R", np.eye(3)))))
    ET.SubElement(b, "geom", name="obst_jig_support", type="box", size=_fmt(np.array(sp["half"])), contype="1",
                  conaffinity="1", group="0", rgba="0.55 0.6 0.65 1", condim="3")
    # stop at the low (far-face, jig -x) end of a tilted support: 3 mm thick, up to 8 mm above the support top
    # (jig z 25.56 -> 17.56), clear of the pocket exits (|z| <= 5.6 at the far face); hidden under the support if flat
    hz = sp["half"][2]
    lp = np.array([-(D / 2 + 1.5) * 1e-3, 0, -hz - 4e-3]) if sp.get("lip") else np.array([0, 0, 0])
    ET.SubElement(b, "geom", name="obst_jig_support_lip", type="box", pos=_fmt(lp),
                  size=_fmt([1.5e-3, sp["half"][1], 4e-3]), contype="1", conaffinity="1", group="0",
                  rgba="0.45 0.5 0.55 1", condim="3")
    # --- OAK camera (MuJoCo camera looks along -z, y up = OpenCV frame rotated 180 deg about x)
    cam = opts.get("cam")
    if cam is not None:
        Tb = np.asarray(cam["T_base_cam"])
        xa, ya = Tb[:3, 0], -Tb[:3, 1]
        ET.SubElement(wb, "camera", name="oak", pos=_fmt(Tb[:3, 3]), xyaxes=_fmt(np.r_[xa, ya]), fovy=f"{cam['fovy']:.3f}")
    # --- fingers
    ff = opts.get("finger_friction")
    for g in root.iter("geom"):
        n = g.get("name", "")
        if n.startswith("finger_") and n.endswith("_collision") and ff is not None:
            g.set("friction", f"{ff} 0.005 0.0001")
    if opts.get("finger_mode") == "cad":
        sl = opts["finger_slices"]
        for body in root.iter("body"):
            nm = body.get("name", "")
            for side in ("left", "right"):
                if nm != f"finger_{side}_link":
                    continue
                for g in list(body.findall("geom")):
                    if any(k in g.get("name", "") for k in ("_front_collision", "_mid_collision", "_rear_collision")):
                        tmpl = dict(g.attrib)
                        body.remove(g)
                pieces = sl[side]["pieces"] if isinstance(sl[side], dict) else sl[side]
                for i, v in enumerate(pieces):
                    ET.SubElement(asset, "mesh", name=f"cadf_{side}_{i}", vertex=_fmt(np.asarray(v)))
                    a = {k: tmpl[k] for k in ("contype", "conaffinity", "condim", "friction", "solref", "solimp", "margin") if k in tmpl}
                    ET.SubElement(body, "geom", name=f"finger_{side}_cad{i}_collision", type="mesh", mesh=f"cadf_{side}_{i}",
                                  group="3", mass="0.0005", rgba="0.95 0.45 0.10 0.55", **a)


def build_xml(s: dict, opts: dict) -> str:
    """check_rod_jig.py: scene.build_xml (block parked, without the scene.toml jig) + add_jig(opts)."""
    s2 = dict(s, jig=dict(s.get("jig", {}), enabled=False))
    root = ET.fromstring(scene.build_xml(s2))
    for b in root.find("worldbody").findall("body"):
        if b.get("name") == "block":
            b.set("pos", "3 3 0.0155")
    add_jig(root, opts)
    return ET.tostring(root, encoding="unicode")


def place_support(m, sp: dict | None) -> None:
    """Move / resize the (static) support at run time; None parks it. Kinematics must be recomputed after."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jig_support")
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obst_jig_support")
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obst_jig_support_lip")
    if bid < 0:
        return
    if sp is None:
        sp = {"pos": np.array([PARK[0], PARK[1], -0.5]), "R": np.eye(3), "half": np.array([D / 2 + 1.0, Y + 1.0, 10.0]) * 1e-3, "lip": False}
    m.body_pos[bid] = sp["pos"]
    m.body_quat[bid] = _quat(sp["R"])
    m.geom_size[g] = sp["half"]
    m.geom_rbound[g] = float(np.linalg.norm(sp["half"]))
    hz = sp["half"][2]
    m.geom_pos[gl] = np.array([-(D / 2 + 1.5) * 1e-3, 0, -hz - 4e-3]) if sp.get("lip") else np.zeros(3)


def jig_inertia(mass_kg: float) -> dict:
    """COM and inertia of rod_jig.stl at uniform density scaled to mass_kg (jig frame)."""
    import trimesh
    mm = trimesh.load(JIG_DIR / "rod_jig.stl")
    mm.density = mass_kg / mm.volume                    # kg per mm^3: trimesh mass = mass_kg
    com = mm.center_mass
    I = mm.moment_inertia * 1e-6                        # kg mm^2 -> kg m^2
    return {"com_mm": com.tolist(), "full": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], "volume_cm3": mm.volume / 1000}


class JigIds(scene.Ids):
    """scene.Ids (which finds the jig lookups itself when the jig is in the model) + the OAK camera."""

    def __init__(self, m):
        super().__init__(m)
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "oak")


def scene_opts(s: dict) -> dict:
    """add_jig options for the production scene (scene.toml [jig]); the jig starts parked."""
    jg = s["jig"]
    data = load_sim_data()
    mass = float(jg["mass_kg"])
    I = data["inertia_per_kg"]
    return {"jig_T": T_world_jig(PARK[0], PARK[1], 0.0), "support": None, "mass_kg": mass,
            "friction": float(jg["friction"]), "inertia": {"com_mm": I["com_mm"], "full": (np.array(I["full"]) * mass).tolist()},
            "finger_mode": jg.get("finger_collision", "model"), "finger_slices": data["finger_slices"],
            "finger_friction": None, "cam": None, "multiccd": bool(jg.get("multiccd", False))}
