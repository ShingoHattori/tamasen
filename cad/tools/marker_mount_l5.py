"""Marker mount 2 (separate from the ID 1 mount on the gripper): two inclined 30 mm marker faces on the back
plate of link5 -- the 3 mm metal plate of 03_Link5 that covers the joint 6 motor's cable restraint
(01_Joint6_7_Cable Restraint_A) -- requester 2026-10-14.

Coordinates: the wrist STEP frame = end_link at joint 6 = 0, mm (cad/out/wrist_gripper_in_end_link.step). The plate's
outer face is x = X_PLATE (normal -x); link5 carries it, so the markers move with link5 (joint 6 turns the gripper
relative to them). Fixed like the ID 1 mount: the two upper countersunk screws (KM3-9, y = +-13.4, z = +13.4)
are replaced by countersunk M3 machine screws longer by the base thickness t (9 + t; t = 5 -> M3 x 14), whose tip
then ends where it ends today; cone bosses on the back of the base sit in the emptied 90 deg countersinks of the plate
(dia 6.0, 1.4 deep) and locate it, the new screw heads sit in countersinks on the front of the base.

Faces: tilt,azimuth (deg): the plate normal -x tilted by `tilt` towards `azimuth` in the plate plane (0 = +z, up the
plate; 90 = +y; -90 = -y; 180 = -z). Each face is a plate with a raised rim around a square for the sticker, on a
wedge extruded straight back onto the base. With the two faces tilted about z (azimuth +-90) the ridge runs along
z: print standing on the base's bottom edge (z down), every face is then a vertical wall, no supports.

usage: marker_mount_l5.py wrist_gripper_in_end_link.step outdir --face 45,90 --face 45,-90 [--t 5]
"""
import argparse, json, math
from pathlib import Path
import numpy as np
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakePrism
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID

ap = argparse.ArgumentParser()
ap.add_argument("step"); ap.add_argument("outdir")
ap.add_argument("--face", action="append", required=True, help="tilt,azimuth (deg), see above")
ap.add_argument("--t", type=float, default=5.0, help="base thickness, mm (screws become M3 x (9 + t))")
ap.add_argument("--plate", type=float, default=3.0, help="marker plate thickness, mm")
ap.add_argument("--marker", type=float, default=30.0)
ap.add_argument("--margin", type=float, default=2.0, help="white margin of the sticker inside the rim")
ap.add_argument("--rim", type=float, nargs=2, default=[1.2, 0.8], metavar=("WIDTH", "HEIGHT"))
ap.add_argument("--z-range", type=float, nargs=2, default=[-1.0, 34.5], help="base extent along z (Lower_Arm_Limit top is -2.1, plate top 34.5)")
ap.add_argument("--y-half", type=float, default=25.0, help="base half width along y (plate is +-25)")
ap.add_argument("--spin", type=float, default=None,
                help="turn the marker square in its own plane, deg (default: 45 for a diagonal azimuth, so the footprint fits the plate)")
ap.add_argument("--boss-clearance", type=float, default=0.1)
ap.add_argument("--boss-height", type=float, default=0.8)
a = ap.parse_args()

X_PLATE = -211.39                      # outer face of the 03_Link5 back plate (BRep plane -211.37) less 0.02 so contact does not read as overlap
SCREWS = [(-13.4, 13.4), (13.4, 13.4)]  # upper KM3-9 (y, z); countersunk flush in the plate
CSK_D, CSK_DEPTH = 6.0, 1.4             # plate countersink (90 deg)
HOLE_D, HEAD_CSK_D = 3.4, 6.4           # through hole and head countersink in the new base


def box(x0, y0, z0, x1, y1, z1): return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()
def cut(s, u): return BRepAlgoAPI_Cut(s, u).Shape()
def fuse(s, u): return BRepAlgoAPI_Fuse(s, u).Shape()
def vol(s):
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); return g.Mass()
def bbox(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b); p, q = b.CornerMin(), b.CornerMax(); return [p.X(), p.Y(), p.Z(), q.X(), q.Y(), q.Z()]
def poly_face(pts):
    pl = BRepBuilderAPI_MakePolygon()
    for p in pts: pl.Add(gp_Pnt(*map(float, p)))
    pl.Close()
    return BRepBuilderAPI_MakeFace(pl.Wire()).Face()


t = a.t
z0, z1 = a.z_range
Yh = a.y_half
xb = X_PLATE - t                         # outer face of the base
body = box(xb, -Yh, z0, X_PLATE, Yh, z1)
out_n = np.array([-1.0, 0, 0])

# ---- faces: footprints share the base; place them side by side along the azimuth direction ----
S = a.marker + 2 * a.margin + 2 * a.rim[0]
specs = [tuple(float(v) for v in f.split(",")) for f in a.face]
faces = []
for k, (tilt, az) in enumerate(specs):
    T_, A_ = math.radians(tilt), math.radians(az)
    w = np.array([0.0, math.sin(A_), math.cos(A_)])              # in-plate direction of the tilt
    n = math.cos(T_) * out_n + math.sin(T_) * w                  # face normal
    e_up = np.cross(n, np.cross(w, n)); e_up /= np.linalg.norm(e_up) if np.linalg.norm(e_up) > 1e-9 else 1
    if tilt < 1e-6:
        e_up = np.array([0, 0, 1.0])
    # e_up points along w tilted out of the plate (towards the low edge for a positive tilt): use -e_up as
    # "towards the high edge" so that the marker's +y points to the high edge
    e_y = -e_up if tilt > 1e-6 else e_up
    e_x = np.cross(e_y, n)
    spin = a.spin if a.spin is not None else (45.0 if abs(math.remainder(az, 90.0)) > 1e-6 else 0.0)
    if spin:
        c_, s_ = math.cos(math.radians(spin)), math.sin(math.radians(spin))
        e_x, e_y = c_ * e_x + s_ * e_y, -s_ * e_x + c_ * e_y
    # footprint centre in the plate: half the faces' projected width from the centre line, on the tilt side
    proj = S * math.cos(T_)
    side = np.array([0.0, math.sin(A_), math.cos(A_)])
    centre_plate = np.array([xb, 0.0, (z0 + z1) / 2]) + side * (proj / 2 if len(specs) > 1 else 0.0)
    # lowest edge (towards -w... the edge nearest the plate) touches the base; plate underside centre:
    c_bottom = centre_plate + out_n * (S / 2 * math.sin(T_) + 0.3)
    corners = [c_bottom + sx * S / 2 * e_x + sy * S / 2 * e_y for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # keep the face above the base's lower edge (the Lower_Arm_Limit block stands below it) -- shift up if needed
    lift = max(0.0, z0 + 0.3 - min(p[2] for p in corners) + max(0.0, -n[2]) * (a.plate + a.rim[1]))
    if lift:
        c_bottom = c_bottom + np.array([0, 0, lift]); corners = [p + np.array([0, 0, lift]) for p in corners]
        print(f"face {k + 1}: moved up {lift:.1f} mm to stay above z = {z0}")
    sq = poly_face(corners)
    plate = BRepPrimAPI_MakePrism(sq, gp_Vec(*(n * a.plate))).Shape()
    depth = max(float((p - np.array([xb, 0, 0])) @ out_n) for p in corners) + 1.0
    wedge = BRepPrimAPI_MakePrism(sq, gp_Vec(*(-out_n * depth))).Shape()
    top = c_bottom + n * a.plate
    ring = lambda h: [top + sx * h * e_x + sy * h * e_y for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    rim = cut(BRepPrimAPI_MakePrism(poly_face(ring(S / 2)), gp_Vec(*(n * a.rim[1]))).Shape(),
              BRepPrimAPI_MakePrism(poly_face([p - n * 0.5 for p in ring(S / 2 - a.rim[0])]), gp_Vec(*(n * (a.rim[1] + 1)))).Shape())
    part = fuse(fuse(plate, wedge), rim)
    faces.append({"tilt_deg": tilt, "azimuth_deg": az, "normal": n, "centre": top, "x_axis": e_x, "y_axis": e_y})
    body = fuse(body, part)
# nothing behind the plate face (the wedges were extruded into it)
body = cut(body, box(X_PLATE, -100, -100, X_PLATE + 50, 100, 100))

# ---- screws: through holes, head countersinks on the front, cone bosses on the back ----
for y, z in SCREWS:
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_PLATE + 1, y, z), gp_Dir(-1, 0, 0)), HOLE_D / 2, 200).Shape())
    hc = (HEAD_CSK_D - HOLE_D) / 2
    body = cut(body, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(xb + 0.01, y, z), gp_Dir(1, 0, 0)), HEAD_CSK_D / 2 + 0.01, HOLE_D / 2, hc + 0.01).Shape())
    # clear a driver path from the head straight out along -x through anything above it
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(xb + 0.01, y, z), gp_Dir(-1, 0, 0)), HEAD_CSK_D / 2 + 0.3, 200).Shape())
rb = CSK_D / 2 - a.boss_clearance
assert a.boss_height < CSK_DEPTH
for y, z in SCREWS:
    boss = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(X_PLATE - 0.01, y, z), gp_Dir(1, 0, 0)), rb, rb - a.boss_height - 0.01, a.boss_height + 0.01).Shape()
    boss = cut(boss, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_PLATE - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, a.boss_height + 2).Shape())
    body = fuse(body, boss)

ex, nsol = TopExp_Explorer(body, TopAbs_SOLID), 0
while ex.More(): nsol += 1; ex.Next()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
w = STEPControl_Writer(); w.Transfer(body, STEPControl_AsIs); assert w.Write(str(out / "marker_mount_l5.step")) == IFSelect_RetDone
BRepMesh_IncrementalMesh(body, 0.02, False, 0.1, True); sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(body, str(out / "marker_mount_l5.stl"))
coarse = BRepBuilderAPI_Transform(body, gp_Trsf(), True).Shape()
BRepMesh_IncrementalMesh(coarse, 0.2, False, 0.5, True); sw.Write(coarse, str(out / "marker_mount_l5_coarse.stl"))
mk = []
for k, fc in enumerate(faces):
    R = np.array([fc["x_axis"], fc["y_axis"], fc["normal"]]).T
    mk.append({"face": k + 1, "tilt_deg": fc["tilt_deg"], "azimuth_deg": fc["azimuth_deg"], "centre_mm": fc["centre"].round(3).tolist(),
               "normal": fc["normal"].round(5).tolist(),
               "T_wrist_marker": np.vstack([np.c_[R, fc["centre"]], [0, 0, 0, 1]]).round(5).tolist()})
spec = {"frame": "wrist STEP = end_link at joint 6 = 0 (mm); the part is carried by link5", "link": "link5",
        "markers": mk, "marker_size_mm": a.marker, "sticker_area_mm": a.marker + 2 * a.margin, "rim_mm": a.rim,
        "base": {"x_mm": [xb, X_PLATE], "y_mm": [-Yh, Yh], "z_mm": [z0, z1], "t_mm": t},
        "screws": f"replace the two upper KM3-9 (y = +-13.4, z = +13.4) by countersunk M3 x {9 + t:g} machine screws (same tip depth)",
        "bosses": {"fits": "90 deg countersink dia 6.0 x 1.4 in the 03_Link5 plate", "base_dia_mm": 2 * rb, "height_mm": a.boss_height},
        "solids": nsol, "bbox_mm": [round(v, 2) for v in bbox(body)], "volume_mm3": round(vol(body), 1),
        "note": "design value from CAD; sticker plane = rim base (sticker thickness not included)"}
json.dump(spec, open(out / "marker_mount_l5.json", "w"), indent=1)
print(json.dumps({k: spec[k] for k in ("markers", "solids", "bbox_mm", "volume_mm3", "screws")}, indent=1))

# ---- interference with the wrist / gripper parts: link5 parts static; link6 + gripper swept over joint 6 ----
doc = TDocStd_Document(TCollection_ExtendedString("d"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); assert r.ReadFile(a.step) == IFSelect_RetDone; r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
labs = Sequence_TDF_Label(); st.GetFreeShapes(labs)
rows = []
for i in range(1, labs.Length() + 1):
    lab = labs.Value(i); nm = TDataStd_Name(); lab.FindAttribute(TDataStd_Name.GetID_s(), nm)
    name = nm.Get().ToExtString(); shp = st.GetShape_s(lab)
    b = Bnd_Box(); BRepBndLib.Add_s(shp, b, False)
    if "KM3-9" in name and b.CornerMin().X() < X_PLATE + 0.1 and any(abs(b.CornerMin().Y() + 3 - y) < 1 for y, _ in SCREWS):
        continue                                                     # the two screws being replaced
    angles = [0] if name.startswith("Link5/") else range(-180, 180, 10)
    dmin, ov = 1e9, 0.0
    for ang in angles:
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(ang))
        s_ = BRepBuilderAPI_Transform(shp, tr, True).Shape() if ang else shp
        dd = BRepExtrema_DistShapeShape(body, s_); dmin = min(dmin, dd.Value())
        if dd.Value() < 1e-6:
            ov = max(ov, vol(BRepAlgoAPI_Common(body, s_).Shape()))
    rows.append((dmin, name, ov))
rows.sort()
print("closest parts (mm; link6 / gripper swept over joint 6):")
for dmin, name, ov in rows[:10]:
    print(f"  {dmin:7.3f}  {name}" + (f"   OVERLAP {ov:.3f} mm3" if ov > 1e-6 else ""))
