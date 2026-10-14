"""Gripper jig for rod-shaped objects (mm). Three through holes side by side:

    [wall][finger pocket][inner wall][ rod hole ][inner wall][finger pocket][wall]      <- y (closing axis)

  - rod hole (centre): rod_w x rod_h, through x; tapping-screw pilot holes from the +z face press the rod;
  - finger pockets (left / right): the B601 fingers (01_Finger, 12.7 mm thick, 3 -> 30 mm wide over the first
    45 mm) go in up to the full jig depth; the pocket is wider than the finger along y, so the jaws enter with a
    few mm of position error and then close onto the inner walls (the jig is gripped on the inner walls, width
    grip_w = rod_w + 2 inner);
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
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
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
ap.add_argument("--depth", type=float, default=40.0, help="jig thickness along the fingers (x)")
ap.add_argument("--rod", type=float, nargs=2, default=[20.0, 30.0], metavar=("W", "H"), help="rod hole, y x z")
ap.add_argument("--pocket", type=float, nargs=2, default=[20.0, 32.0], metavar=("W", "H"),
                help="finger pocket, y x z (finger: 12.7 thick; 27 wide at 40 mm, 30 at 45 mm from the tip)")
ap.add_argument("--inner", type=float, default=6.0, help="wall between pocket and rod hole (the jaws press it)")
ap.add_argument("--outer", type=float, default=4.0, help="outer wall of each pocket")
ap.add_argument("--wall-z", type=float, nargs=2, default=[5.0, 8.0], metavar=("MINUS", "PLUS"),
                help="walls below / above the holes along z (+z carries the screws)")
ap.add_argument("--chamfer", type=float, default=2.5, help="45 deg entry chamfer of the pockets (+x face)")
ap.add_argument("--screw-pilot", type=float, default=2.5, help="pilot hole for 3 mm tapping screws")
ap.add_argument("--screws", type=int, default=2)
ap.add_argument("--marker", type=float, default=30.0, help="ArUco black square, mm")
ap.add_argument("--marker-margin", type=float, default=2.0, help="white margin of the sticker inside the rim")
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
rw, rh = a.rod
pw, ph = a.pocket
Y = rw / 2 + a.inner + pw + a.outer                       # half width along y
zl, zh = -max(rh, ph) / 2 - a.wall_z[0], max(rh, ph) / 2 + a.wall_z[1]
body = box(-D / 2, -Y, zl, D / 2, Y, zh)
body = cut(body, box(-D / 2 - 1, -rw / 2, -rh / 2, D / 2 + 1, rw / 2, rh / 2))       # rod hole
y_pocket = rw / 2 + a.inner + pw / 2                                                  # pocket centre |y|
for s in (-1, 1):
    yc = s * y_pocket
    body = cut(body, box(-D / 2 - 1, yc - pw / 2, -ph / 2, D / 2 + 1, yc + pw / 2, ph / 2))
    if a.chamfer > 0:
        body = cut(body, frustum(D / 2, pw, ph, a.chamfer, yc))
# screws: pilot holes along -z from the +z face into the rod hole, spread along x
xs = [-D / 2 + D * (i + 1) / (a.screws + 1) for i in range(a.screws)]
for x in xs:
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, 0, zh + 1), gp_Dir(0, 0, -1)), a.screw_pilot / 2, zh - rh / 2 + 2).Shape()
    body = cut(body, cyl)
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
    "rod_hole_mm": [rw, rh], "finger_pockets_mm": [pw, ph], "pocket_centres_y_mm": [-y_pocket, y_pocket],
    "grip_width_mm": grip_w,
    "entry_gap_mm": {"min": grip_w, "max": 2 * (y_pocket + pw / 2) - 2 * 12.7,
                     "suggested": round(2 * (rw / 2 + a.inner) + (pw - 12.7), 1),
                     "note": "jaw gap (inner faces) while entering: inner faces between the inner walls and outer walls minus the 12.7 mm finger"},
    "screws": {"pilot_mm": a.screw_pilot, "x_mm": xs, "axis": "-z from the +z face", "length_into_wall_mm": zh - rh / 2,
               "screw": "3 mm tapping screw (e.g. 3 x 12), tip presses the rod"},
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
    # to depth D - 1 (tips 1 mm out of the far face).
    r = STEPControl_Reader(); r.ReadFile(a.finger_step); r.TransferRoots(); sh = r.OneShape()
    ex, fingers = TopExp_Explorer(sh, TopAbs_SOLID), []
    while ex.More():
        b = bbox(ex.Current())
        if b and b[3] > -3 and -70 < b[0] < -40:
            fingers.append((ex.Current(), 1 if b[4] > 1 else -1))
        ex.Next()
    jig_x = -D / 2 + 1.0                       # jig centre in end_link x (far face at x = +1, entry face at 1 - D)
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
    info["finger_check"] = dict(res, insert_depth_mm=D - 1.0, note="fingers of 01_Finger (CAD), jig entry face towards the palm")

(out / "rod_jig.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
print(json.dumps(info, indent=1, ensure_ascii=False))
