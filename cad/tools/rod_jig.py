"""Gripper jig for rod-shaped objects (mm). Three through holes side by side:

    [wall][finger pocket][inner wall][ rod hole ][inner wall][finger pocket][wall]      <- y (closing axis)

  - rod hole (centre): rod_w x rod_h, through x; tapping-screw pilot holes from the +z face press the rod;
  - finger pockets (left / right): shaped like the B601 finger (01_Finger) so it sits snugly -- along z the
    pocket follows the finger's straight taper (11.6 mm wide 15 mm from the tip, 26.8 mm at 39 mm, from the CAD)
    plus `fit` per side, and keeps the tip width (9.2 mm) from there to the far face: one slope, no steps; along y it is the finger thickness (12.7) + `fit` + `slack`, so the jaws enter
    opened `slack` beyond the grip and then close onto the inner walls (the jig is gripped on the inner walls,
    width grip_w = rod_w + 2 inner); requester 2026-10-15: "爪がピッタリハマるように、三角形のような形状";
  - entry chamfers on the +x face guide the fingers in;
  - marker area on the -z face: a raised rim around a square for the printed ArUco sticker.

Frame: x = through axis (= -end_link x when gripped: the fingers enter through the +x face),
y = closing axis, z = finger width direction. Origin at the jig centre. Print with x vertical, entry face up:
every hole is vertical, no supports.

usage: rod_jig.py outdir [--finger-step wrist_gripper_in_end_link.step] [options]
Writes rod_jig.step / .stl / .json and, with the finger STEP, checks the fingers in the pockets
(open for entry, closed on the inner walls).
"""
import argparse, json, math
from pathlib import Path
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--finger-step", help="wrist_gripper_in_end_link.step (end_link frame) for the fit check")
ap.add_argument("--depth", type=float, default=44.0, help="jig thickness along the fingers (x); 44 so the 41 mm marker area + rim fits the -z face")
ap.add_argument("--insert", type=float, default=None,
                help="finger depth into the jig from the entry face (default depth + 1: the tip 1 mm out of the far face; "
                     "requester 2026-10-15: the tip should reach the far end or stick out a little)")
ap.add_argument("--rect", type=float, default=20.0,
                help="entry part of each pocket: a plain rectangle this deep, big enough for the finger's shoulder "
                     "(33.1 wide, 13.7 thick, from 44.3 mm behind the tip); the wedge follows behind it")
ap.add_argument("--rect-fit", type=float, default=1.0, help="clearance of the entry rectangle around the shoulder (z per side)")
ap.add_argument("--rod", type=float, nargs=2, default=[30.0, 20.0], metavar=("W", "H"), help="rod hole, y x z (30 along the closing axis; requester 2026-10-14)")
# finger half width along z vs distance from the tip (01_Finger in wrist_gripper_in_end_link.step): a straight taper,
# 5.80 mm at 15 mm from the tip growing 0.3155 mm per mm (13.38 at 39 mm). Only this taper along the insertion is
# followed (requester 2026-10-15); nearer the tip the finger is at most 4.60 mm, so the pocket keeps that width
# from there to the far face -- one sloped face, no steps.
TAPER = (15.0, 5.80, 0.3155)            # dx0, half width at dx0, slope
TIP_HALF = 4.60
ap.add_argument("--finger-t", type=float, default=12.74, help="finger thickness along y")
ap.add_argument("--fit", type=float, default=1.0, help="clearance around the finger at the tip end (z both sides, y outer side)")
ap.add_argument("--fit-entry", type=float, default=1.0, help="z clearance per side of the wedge at its entry end (grows linearly from --fit)")
ap.add_argument("--slack", type=float, default=1.0, help="extra room along y on the outer side: jaw opening for entry")
ap.add_argument("--inner", type=float, default=6.0, help="wall between pocket and rod hole (the jaws press it)")
ap.add_argument("--outer", type=float, default=4.0, help="outer wall of each pocket")
ap.add_argument("--wall-z", type=float, nargs=2, default=[5.0, 8.0], metavar=("MINUS", "PLUS"),
                help="walls below / above the holes along z (+z carries the screws)")
ap.add_argument("--chamfer", type=float, default=2.5, help="45 deg entry chamfer of the pockets (+x face)")
ap.add_argument("--screw-pilot", type=float, nargs=2, default=[2.7, 2.4], metavar=("ENTRY", "END"),
                help="tapered pilot hole for 3 mm tapping screws: diameter at the entry and at the rod hole")
ap.add_argument("--screw-csk", type=float, default=5.0, help="90 deg entry countersink diameter on the +z face")
ap.add_argument("--screws", type=int, default=2)
ap.add_argument("--marker", type=float, default=30.0, help="ArUco black square, mm")
ap.add_argument("--marker-margin", type=float, default=5.5,
                help="white margin inside the rim: area 30 + 2 x 5.5 = 41 mm holds the 40 mm sticker (requester 2026-10-15: 40 mm incl. margin)")
ap.add_argument("--rim", type=float, nargs=2, default=[1.2, 0.8], metavar=("WIDTH", "HEIGHT"))
a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)


def box(x0, y0, z0, x1, y1, z1):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()


def cut(s, t):
    return BRepAlgoAPI_Cut(s, t).Shape()


def fuse(s, t):
    return BRepAlgoAPI_Fuse(s, t).Shape()


def volume(s):
    p = GProp_GProps(); BRepGProp.VolumeProperties_s(s, p); return p.Mass()


def bbox(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b)
    if b.IsVoid():
        return None
    p, q = b.CornerMin(), b.CornerMax()
    return [p.X(), p.Y(), p.Z(), q.X(), q.Y(), q.Z()]


def lerp_half(dx):
    return max(TAPER[1] + TAPER[2] * (dx - TAPER[0]), TIP_HALF)


def finger_pocket(D, ins, yc, pw, fit, fit_entry, out_far=1.5):
    """Through-pocket along x following the finger taper: rectangles pw (y) x 2 (half_z(dx) + fit) (z), dx = distance
    from the finger tip, which sits at x = D/2 - ins when the jig is fully on (beyond it: the tip section, a slot)."""
    tip = D / 2 - ins
    kink = TAPER[0] + (TIP_HALF - TAPER[1]) / TAPER[2]           # where the taper meets the tip width
    xs = [-D / 2 - out_far] + [tip + kink] * (-D / 2 < tip + kink < D / 2) + [D / 2 + 0.01]
    xs = sorted(set(round(x, 3) for x in xs))
    ts = BRepOffsetAPI_ThruSections(True, True)
    for x in xs:
        h = lerp_half(x - tip) + fit + (fit_entry - fit) * min(max((x + D / 2) / D, 0.0), 1.0)
        poly = BRepBuilderAPI_MakePolygon()
        for y, z in ((-pw / 2, -h), (pw / 2, -h), (pw / 2, h), (-pw / 2, h)):
            poly.Add(gp_Pnt(x, yc + y, z))
        poly.Close()
        ts.AddWire(poly.Wire())
    ts.Build()
    return ts.Shape()


def frustum(x_top, w0, h0, c, yc):
    """Pocket entry funnel: rectangle w0 x h0 at x_top - c widening by c on every side at x_top."""
    def rect(x, w, h):
        poly = BRepBuilderAPI_MakePolygon()
        for y, z in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            poly.Add(gp_Pnt(x, yc + y, z))
        poly.Close()
        return poly.Wire()
    ts = BRepOffsetAPI_ThruSections(True, True)
    ts.AddWire(rect(x_top - c, w0, h0)); ts.AddWire(rect(x_top + 0.01, w0 + 2 * c, h0 + 2 * c))
    ts.Build()
    return ts.Shape()


D = a.depth
if a.insert is None:
    a.insert = D + 1.0
SHOULDER_DX, SHOULDER_HALF_Z, SHOULDER_T = 44.3, 16.56, 13.67      # finger shoulder (CAD)
if a.insert - SHOULDER_DX > a.rect - 1.0:
    raise SystemExit(f"--insert {a.insert}: the shoulder would go beyond the entry rectangle ({a.rect} mm)")
rw, rh = a.rod
pw = a.finger_t + a.fit + a.slack                         # along y, wedge part
pw_r = SHOULDER_T + a.fit + a.slack                       # along y, entry rectangle
ph = 2 * (SHOULDER_HALF_Z + a.rect_fit)                   # along z, entry rectangle (widest)
Y = rw / 2 + a.inner + pw_r + a.outer                     # half width along y
zl, zh = -max(rh, ph) / 2 - a.wall_z[0], max(rh, ph) / 2 + a.wall_z[1]
body = box(-D / 2, -Y, zl, D / 2, Y, zh)
body = cut(body, box(-D / 2 - 1, -rw / 2, -rh / 2, D / 2 + 1, rw / 2, rh / 2))       # rod hole
y_pocket = rw / 2 + a.inner + pw_r / 2                                                # entry rectangle centre |y|
for s in (-1, 1):
    ys = s * (rw / 2 + a.inner)                                   # inner wall face (the jaws close onto it)
    # wedge (tip side) + entry rectangle, joined by a 45 deg transition; y measured outwards from the inner wall
    body = cut(body, finger_pocket(D, a.insert, ys + s * pw / 2, pw, a.fit, a.fit_entry))
    xb = D / 2 - a.rect                                           # start of the rectangle
    hw0 = lerp_half(a.insert - a.rect) + a.fit_entry              # wedge half width where the rectangle starts
    t = max((ph / 2 - hw0) / (1 - TAPER[2]), pw_r - pw)           # ~45 deg transition, starting on the wedge's slope
    hw = lerp_half(a.insert - a.rect - t) + a.fit_entry           # wedge half width where the transition starts

    def sec(x, w, h):
        poly = BRepBuilderAPI_MakePolygon()
        for y, z in ((0, -h), (w, -h), (w, h), (0, h)):
            poly.Add(gp_Pnt(x, ys - s * 0.01 + s * y, z))
        poly.Close()
        return poly.Wire()
    ts = BRepOffsetAPI_ThruSections(True, True)
    for x, w, h in ((xb - t, pw, hw), (xb, pw_r, ph / 2), (D / 2 + 0.01, pw_r, ph / 2)):
        ts.AddWire(sec(x, w + 0.01, h))
    ts.Build()
    body = cut(body, ts.Shape())
    if a.chamfer > 0:
        body = cut(body, frustum(D / 2, pw_r, ph, a.chamfer, ys + s * pw_r / 2))
# screws: pilot holes along -z from the +z face into the rod hole, spread along x
xs = [-D / 2 + D * (i + 1) / (a.screws + 1) for i in range(a.screws)]
# tapered pilot (the tapping screw starts easily and bites harder towards the rod) + 90 deg entry countersink
for x in xs:
    Lp = zh - rh / 2
    pil = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, 0, zh + 0.01), gp_Dir(0, 0, -1)), a.screw_pilot[0] / 2, a.screw_pilot[1] / 2, Lp + 0.02).Shape()
    body = cut(body, pil)
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, 0, rh / 2 + 0.5), gp_Dir(0, 0, -1)), a.screw_pilot[1] / 2, 1.0).Shape())
    hc = (a.screw_csk - a.screw_pilot[0]) / 2                     # 90 deg: depth = radial step
    body = cut(body, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, 0, zh + 0.01), gp_Dir(0, 0, -1)), a.screw_csk / 2 + 0.01, a.screw_pilot[0] / 2, hc + 0.01).Shape())
# marker area on the -z face: raised rim around a square of marker + 2 margins
inner = a.marker + 2 * a.marker_margin
rwid, rhgt = a.rim
if inner + 2 * rwid > min(D, 2 * Y):
    raise SystemExit(f"marker area {inner + 2 * rwid:.1f} mm does not fit the -z face ({D:.0f} x {2 * Y:.0f})")
outer_sq = inner + 2 * rwid
rim = cut(box(-outer_sq / 2, -outer_sq / 2, zl - rhgt, outer_sq / 2, outer_sq / 2, zl + 0.01),
          box(-inner / 2, -inner / 2, zl - rhgt - 1, inner / 2, inner / 2, zl + 1))
body = fuse(body, rim)

grip_w = rw + 2 * a.inner
info = {
    "units": "mm", "frame": "x through (entry face +x), y closing axis, z finger width; origin = jig centre",
    "size_mm": [D, 2 * Y, zh - zl], "x_range": [-D / 2, D / 2], "y_range": [-Y, Y], "z_range": [zl - rhgt, zh],
    "rod_hole_mm": [rw, rh], "pocket_centres_y_mm": [-y_pocket, y_pocket],
    "finger_pockets_mm": {"entry_rectangle": {"depth": a.rect, "y": round(pw_r, 2), "z": round(ph, 2)},
                          "wedge": {"y": round(pw, 2), "z_at_tip_end": round(2 * (lerp_half(a.insert - D) + a.fit), 2)},
                          "finger_insert_mm": a.insert, "tip_beyond_far_face_mm": round(a.insert - D, 2),
                          "shape": "entry: plain rectangle (room for the finger's shoulder), 45 deg transition, then a wedge following the finger taper to the far face",
                          "fit_tip_end": a.fit, "fit_entry": a.fit_entry, "slack": a.slack},
    "grip_width_mm": grip_w,
    "entry_gap_mm": {"min": grip_w, "max": round(grip_w + 2 * (pw - a.finger_t), 2),  # the wedge part (narrower than the rectangle)
                     "suggested": round(grip_w + 2 * a.slack * 0.6, 1),
                     "note": "jaw gap (inner faces) while entering: between the inner walls and the pocket's outer walls"},
    "screws": {"pilot_mm": {"entry": a.screw_pilot[0], "end": a.screw_pilot[1], "taper": "linear over the wall"},
               "entry_countersink_mm": {"diameter": a.screw_csk, "angle_deg": 90}, "x_mm": xs, "axis": "-z from the +z face", "length_into_wall_mm": zh - rh / 2,
               "screw": "countersunk 3 mm tapping screw, 3 x 16 to 3 x 20 (wall 14 mm): the tip presses the rod"},
    "marker": {"face": "-z", "centre_mm": [0, 0, zl - rhgt], "black_square_mm": a.marker, "sticker_area_mm": inner,
               "rim_mm": {"width": rwid, "height": rhgt}, "normal": [0, 0, -1]},
    "volume_cm3": round(volume(body) / 1000, 1),
}

w = STEPControl_Writer(); w.Transfer(body, STEPControl_AsIs); w.Write(str(out / "rod_jig.step"))
BRepMesh_IncrementalMesh(body, 0.05, False, 0.2, True)
sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(body, str(out / "rod_jig.stl"))

if a.finger_step:
    # fingers from the gripper STEP (end_link frame, closed: inner faces at y = 0). Put the jig on them, entry
    # face (jig +x) towards the palm: jig x = -end_link x (180 deg about z; the jig is symmetric in y), inserted
    # to depth `insert` (entry face at end_link x = -insert, the finger tips near x = 0).
    r = STEPControl_Reader(); r.ReadFile(a.finger_step); r.TransferRoots(); sh = r.OneShape()
    ex, fingers = TopExp_Explorer(sh, TopAbs_SOLID), []
    while ex.More():
        b = bbox(ex.Current())
        if b and b[3] > -3 and -70 < b[0] < -40:
            fingers.append((ex.Current(), 1 if b[4] > 1 else -1))
        ex.Next()
    jig_x = D / 2 - a.insert                   # jig centre in end_link x (entry face at x = -insert)
    T = gp_Trsf(); T.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.pi)
    T2 = gp_Trsf(); T2.SetTranslation(gp_Vec(jig_x, 0, 0))
    jig_e = BRepBuilderAPI_Transform(BRepBuilderAPI_Transform(body, T, True).Shape(), T2, True).Shape()
    res = {}
    for name, gap in (("entry", info["entry_gap_mm"]["suggested"]), ("closed_on_walls", grip_w), ("walls_plus_0.4", grip_w + 0.4)):
        pen, dist = 0.0, []
        for f, s in fingers:
            Tf = gp_Trsf(); Tf.SetTranslation(gp_Vec(0, s * gap / 2, 0))
            ff = BRepBuilderAPI_Transform(f, Tf, True).Shape()
            com = BRepAlgoAPI_Common(ff, jig_e).Shape()
            v_ = volume(com)
            pen += v_
            if v_ > 0.01:
                res.setdefault("overlap_where_" + name, []).append([round(c, 1) for c in bbox(com)])
            dd = BRepExtrema_DistShapeShape(ff, jig_e); dd.Perform(); dist.append(dd.Value())
        res[name] = {"gap_mm": gap, "overlap_mm3": round(pen, 2), "min_distance_mm": [round(v, 2) for v in dist]}
        if name == "walls_plus_0.4":             # fingers in the jig frame, for the preview
            both = None
            for f, s in fingers:                 # end_link -> jig: translate by -jig_x, then 180 deg about z
                Tf = gp_Trsf(); Tf.SetTranslation(gp_Vec(-jig_x, s * gap / 2, 0))
                ff = BRepBuilderAPI_Transform(BRepBuilderAPI_Transform(f, Tf, True).Shape(), T, True).Shape()
                both = ff if both is None else fuse(both, ff)
            BRepMesh_IncrementalMesh(both, 1.0, False, 0.5, True)
            sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(both, str(out / "fingers_closed_in_jig_frame.stl"))
    # the way in: fingers at the entry gap, slid in 2 mm at a time up to the full depth (the jig stays put)
    path = []
    for depth in [d_ for d_ in range(2, int(a.insert), 2)] + [a.insert]:
        T3 = gp_Trsf(); T3.SetTranslation(gp_Vec(depth - a.insert, 0, 0))     # finger backed out (towards the palm, -x) by insert - depth
        pen, dmin = 0.0, 1e9
        for f, s_ in fingers:
            Tf = gp_Trsf(); Tf.SetTranslation(gp_Vec(0, s_ * info["entry_gap_mm"]["suggested"] / 2, 0))
            ff = BRepBuilderAPI_Transform(BRepBuilderAPI_Transform(f, Tf, True).Shape(), T3, True).Shape()
            pen += volume(BRepAlgoAPI_Common(ff, jig_e).Shape())
            dd = BRepExtrema_DistShapeShape(ff, jig_e); dd.Perform(); dmin = min(dmin, dd.Value())
        path.append({"depth_mm": depth, "overlap_mm3": round(pen, 3), "min_distance_mm": round(dmin, 2)})
    res["insertion_path"] = {"gap_mm": info["entry_gap_mm"]["suggested"], "steps": path,
                             "worst_min_distance_mm": min(p_["min_distance_mm"] for p_ in path),
                             "any_overlap": any(p_["overlap_mm3"] > 0.01 for p_ in path)}
    info["finger_check"] = dict(res, insert_depth_mm=a.insert, note="fingers of 01_Finger (CAD), jig entry face towards the palm")

(out / "rod_jig.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
print(json.dumps(info, indent=1, ensure_ascii=False))
