"""Marker mount 2 (separate from the ID 1 mount on the gripper): inclined 30 mm marker faces on the back plate of
link5 -- the 3 mm metal plate of 03_Link5 over the joint 6 motor's cable restraint (01_Joint6_7_Cable Restraint_A).
Requester 2026-10-14: in gripper-down poses ID 1 falls into the motor's shadow.

Two printed parts, because the faces have to cover the two fixing screws:
  A  base plate on the link5 plate, held by the two upper countersunk screws (KM3-9 at y = +-13.4, z = +13.4,
     replaced by countersunk M3 x (9 + tA) so the tip ends where it ends today); cone bosses on its back sit in the
     emptied 90 deg countersinks of the metal plate (dia 6.0, 1.4 deep) and locate it, the screw heads sit in
     countersinks on its front -- as on the ID 1 mount. A dovetail rail runs along z on its front (between the
     screw heads).
  B  marker block: a slab with a dovetail groove (closed at the top) on its back, slid down onto the rail after
     A is screwed on, and locked by one tapping screw from the top into the rail; the faces stand on solid wedges
     (each face square extruded straight back onto the slab, so nothing is left unfilled).

Print: both parts standing on their bottom (z = z0) face, z up. A: the rail and its dovetail flanks are vertical
walls, the bosses and countersinks horizontal 45 deg cones. B: the groove is vertical (its closed top is a short
bridge), faces tilted about z are vertical walls, a face tilted towards -z overhangs by its tilt (45 deg max).

Coordinates: wrist STEP frame = end_link at joint 6 = 0, mm (cad/out/wrist_gripper_in_end_link.step); the parts are
carried by link5. Faces: --face tilt,azimuth[,y,z]: the plate normal (-x) tilted by `tilt` towards `azimuth` in the
plate plane (0 = +z, 90 = +y, -90 = -y, 180 = -z); y,z = centre of the face's footprint on the plate (default: side
by side); a 5th value turns the square in its own plane (default 45 for a diagonal azimuth, else 0).
Marker axes: +y towards the face's high edge (before the turn), +x = y x normal.
One-marker version: --face 45,-135,0,16.75

usage: marker_mount_l5.py wrist_gripper_in_end_link.step outdir --face 45,-90 --face 45,180
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
ap.add_argument("--face", action="append", required=True, help="tilt,azimuth[,y,z] (deg, deg, mm, mm), see above")
ap.add_argument("--tA", type=float, default=5.0, help="base plate thickness (screws become M3 x (9 + tA))")
ap.add_argument("--tB", type=float, default=9.0, help="marker block slab thickness (holds the groove)")
ap.add_argument("--plate", type=float, default=3.0, help="marker plate thickness")
ap.add_argument("--marker", type=float, default=30.0)
ap.add_argument("--margin", type=float, default=2.0, help="white margin of the sticker inside the rim")
ap.add_argument("--rim", type=float, nargs=2, default=[1.2, 0.8], metavar=("WIDTH", "HEIGHT"))
ap.add_argument("--z-range", type=float, nargs=2, default=[-1.0, 34.5], help="base plate along z (Lower_Arm_Limit top -2.1, plate top 34.5)")
ap.add_argument("--y-half", type=float, default=25.0, help="base plate half width (link5 plate is +-25)")
ap.add_argument("--rail", type=float, nargs=3, default=[10.0, 14.0, 4.0], metavar=("W_ROOT", "W_TOP", "H"),
                help="dovetail rail: width at the plate, width at the top, height")
ap.add_argument("--fit", type=float, default=0.2, help="clearance of the groove around the rail")
ap.add_argument("--boss-clearance", type=float, default=0.1)
ap.add_argument("--boss-height", type=float, default=0.8)
a = ap.parse_args()

X_PLATE = -211.39                       # outer face of the 03_Link5 plate (BRep -211.37) less 0.02
SCREWS = [(-13.4, 13.4), (13.4, 13.4)]  # upper KM3-9 (y, z)
CSK_D, CSK_DEPTH = 6.0, 1.4             # plate countersinks (90 deg)
HOLE_D, HEAD_CSK_D = 3.4, 6.4           # through hole / head countersink in A
LOCK_PILOT, LOCK_HOLE, LOCK_CSK = 2.5, 3.4, 6.4   # lock screw: pilot in the rail, hole + countersink in B's top


def box(x0, y0, z0, x1, y1, z1): return BRepPrimAPI_MakeBox(gp_Pnt(min(x0, x1), min(y0, y1), min(z0, z1)), gp_Pnt(max(x0, x1), max(y0, y1), max(z0, z1))).Shape()
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
def nsolids(s):
    e, k = TopExp_Explorer(s, TopAbs_SOLID), 0
    while e.More(): k += 1; e.Next()
    return k
def dovetail(x_root, w_root, w_top, h, z0, z1, grow=0.0):
    """Prism along z of the dovetail section (root at x_root, top at x_root - h), grown by `grow` on every side."""
    pts = [(x_root + grow, -(w_root / 2 + grow)), (x_root + grow, w_root / 2 + grow),
           (x_root - h - grow, w_top / 2 + grow), (x_root - h - grow, -(w_top / 2 + grow))]
    return BRepPrimAPI_MakePrism(poly_face([(x, y, z0) for x, y in pts]), gp_Vec(0, 0, z1 - z0)).Shape()


z0, z1 = a.z_range
Yh = a.y_half
xA = X_PLATE - a.tA                      # front of A
w_root, w_top, h_rail = a.rail
# ---------------- A: base plate ----------------
A = box(xA, -Yh, z0, X_PLATE, Yh, z1)
A = fuse(A, dovetail(xA + 0.01, w_root, w_top, h_rail + 0.01, z0, z1))
for y, z in SCREWS:
    assert abs(y) - HEAD_CSK_D / 2 > w_top / 2 + 1, "screw heads under the rail"
    A = cut(A, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_PLATE + 1, y, z), gp_Dir(-1, 0, 0)), HOLE_D / 2, a.tA + 2).Shape())
    hc = (HEAD_CSK_D - HOLE_D) / 2
    A = cut(A, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(xA - 0.01, y, z), gp_Dir(1, 0, 0)), HEAD_CSK_D / 2 + 0.01, HOLE_D / 2, hc + 0.02).Shape())
rb = CSK_D / 2 - a.boss_clearance
for y, z in SCREWS:
    boss = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(X_PLATE - 0.01, y, z), gp_Dir(1, 0, 0)), rb, rb - a.boss_height - 0.01, a.boss_height + 0.01).Shape()
    boss = cut(boss, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_PLATE - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, a.boss_height + 2).Shape())
    A = fuse(A, boss)
x_rail_mid = xA - h_rail / 2
A = cut(A, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x_rail_mid, 0, z1 + 0.01), gp_Dir(0, 0, -1)), LOCK_PILOT / 2, 10.0).Shape())

# ---------------- B: marker block ----------------
xB = xA - a.fit                          # back of B (against A's front, fit gap)
xBf = xB - a.tB                          # front of B's slab (wedges stand on it)
out_n = np.array([-1.0, 0, 0])
S = a.marker + 2 * a.margin + 2 * a.rim[0]
specs = []
for f in a.face:
    v = [float(x) for x in f.split(",")]
    specs.append(v + [None] * (5 - len(v)))
faces, foot = [], []
cursor_y = -Yh
for k, (tilt, az, fy, fz, spin) in enumerate(specs):
    T_, A_ = math.radians(tilt), math.radians(az)
    w = np.array([0.0, math.sin(A_), math.cos(A_)])
    n = math.cos(T_) * out_n + math.sin(T_) * w
    e_y = -np.cross(n, np.cross(w, n)) if tilt > 1e-6 else np.array([0, 0, 1.0])
    e_y /= np.linalg.norm(e_y)
    e_x = np.cross(e_y, n)
    if spin is None:                     # a diagonal tilt: turn the square 45 deg in its plane, so it stays compact on the plate
        spin = 45.0 if abs(math.remainder(az, 90.0)) > 1e-6 else 0.0
    if spin:
        c_, s_ = math.cos(math.radians(spin)), math.sin(math.radians(spin))
        e_x, e_y = c_ * e_x + s_ * e_y, -s_ * e_x + c_ * e_y
    # footprint of the square on the slab plane (projection along x)
    half = np.abs(np.array([e_x, e_y])[:, 1:]).sum(axis=0) * S / 2           # half extents in (y, z)
    if fy is None:
        fy = cursor_y + half[0]; cursor_y = fy + half[0] + 1.0
    if fz is None:
        fz = (z0 + z1) / 2
    centre_plate = np.array([xBf, fy, fz])
    c_bottom = centre_plate + out_n * (S / 2 * math.sin(T_) + 0.3)
    corners = [c_bottom + sx * S / 2 * e_x + sy * S / 2 * e_y for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    sq = poly_face(corners)
    plate = BRepPrimAPI_MakePrism(sq, gp_Vec(*(n * a.plate))).Shape()
    depth = max(float(xBf - p[0]) for p in corners) + 1.0
    wedge = BRepPrimAPI_MakePrism(sq, gp_Vec(*(-out_n * (depth + 2.0)))).Shape()     # straight back, into the slab
    top = c_bottom + n * a.plate
    ring = lambda hh: [top + sx * hh * e_x + sy * hh * e_y for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    rim = cut(BRepPrimAPI_MakePrism(poly_face(ring(S / 2)), gp_Vec(*(n * a.rim[1]))).Shape(),
              BRepPrimAPI_MakePrism(poly_face([p - n * 0.5 for p in ring(S / 2 - a.rim[0])]), gp_Vec(*(n * (a.rim[1] + 1)))).Shape())
    faces.append({"tilt_deg": tilt, "azimuth_deg": az, "normal": n, "centre": top, "x_axis": e_x, "y_axis": e_y,
                  "solid": fuse(fuse(plate, wedge), rim)})
    foot.append((fy - half[0], fy + half[0], fz - half[1], fz + half[1]))
# slab covers the base plate and every footprint; its top is above the rail top (closed groove + lock screw)
ys = [Yh, -Yh] + [v for f_ in foot for v in f_[:2]]
zs = [z0, z1] + [v for f_ in foot for v in f_[2:]]
lock_z_top = z1 + 6.0
B = box(xB, min(ys), z0, xBf, max(ys), max(max(zs), lock_z_top))
for fc in faces:
    B = fuse(B, fc["solid"])
B = cut(B, box(xB, -200, -200, xB + 50, 200, 200))                      # nothing behind B's back plane
B = cut(B, box(-400, -200, z0 - 100, 0, 200, z0))                         # flat bottom for printing
B = cut(B, dovetail(xA, w_root, w_top, h_rail, z0 - 1, z1, grow=a.fit))   # groove, closed above the rail top
B = cut(B, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x_rail_mid, 0, 200), gp_Dir(0, 0, -1)), LOCK_HOLE / 2, 200 - z1 + 0.5).Shape())
hc = (LOCK_CSK - LOCK_HOLE) / 2
zt = bbox(B)[5]
B = cut(B, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x_rail_mid, 0, zt + 0.01), gp_Dir(0, 0, -1)), LOCK_CSK / 2 + 0.01, LOCK_HOLE / 2, hc + 0.02).Shape())
lock_len = zt - z1 + 8.0

out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
for name, part in (("A_base", A), ("B_markers", B)):
    w_ = STEPControl_Writer(); w_.Transfer(part, STEPControl_AsIs); assert w_.Write(str(out / f"marker_mount_l5_{name}.step")) == IFSelect_RetDone
    BRepMesh_IncrementalMesh(part, 0.02, False, 0.1, True); sw = StlAPI_Writer(); sw.ASCIIMode = False
    sw.Write(part, str(out / f"marker_mount_l5_{name}.stl"))
both = fuse(A, B)
c_ = BRepBuilderAPI_Transform(both, gp_Trsf(), True).Shape()
BRepMesh_IncrementalMesh(c_, 0.2, False, 0.5, True); sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(c_, str(out / "marker_mount_l5_coarse.stl"))
mk = []
for k, fc in enumerate(faces):
    R = np.array([fc["x_axis"], fc["y_axis"], fc["normal"]]).T
    mk.append({"face": k + 1, "tilt_deg": fc["tilt_deg"], "azimuth_deg": fc["azimuth_deg"], "centre_mm": fc["centre"].round(3).tolist(),
               "normal": fc["normal"].round(5).tolist(), "T_wrist_marker": np.vstack([np.c_[R, fc["centre"]], [0, 0, 0, 1]]).round(5).tolist()})
spec = {"frame": "wrist STEP = end_link at joint 6 = 0 (mm); the parts are carried by link5", "link": "link5",
        "markers": mk, "marker_size_mm": a.marker, "sticker_area_mm": a.marker + 2 * a.margin, "rim_mm": a.rim,
        "parts": {"A_base": {"solids": nsolids(A), "bbox_mm": [round(v, 2) for v in bbox(A)], "volume_mm3": round(vol(A))},
                  "B_markers": {"solids": nsolids(B), "bbox_mm": [round(v, 2) for v in bbox(B)], "volume_mm3": round(vol(B))}},
        "screws": {"fixing": f"replace the two upper KM3-9 (y = +-13.4, z = +13.4) by countersunk M3 x {9 + a.tA:g} machine screws (same tip depth)",
                   "lock": f"one countersunk 3 mm tapping screw, about 3 x {math.floor(lock_len):d}, from B's top into the rail (pilot {LOCK_PILOT} mm)"},
        "bosses": {"fits": "90 deg countersink dia 6.0 x 1.4 in the 03_Link5 plate", "base_dia_mm": 2 * rb, "height_mm": a.boss_height},
        "assembly": "screw A on; slide B down onto A's rail from above until the closed groove rests on the rail top; lock screw from the top",
        "print": "both parts standing on their bottom face (z = z0), z up; no supports expected (faces tilted towards -z overhang by their tilt)",
        "note": "design value from CAD; sticker plane = rim base (sticker thickness not included)"}
json.dump(spec, open(out / "marker_mount_l5.json", "w"), indent=1)
print(json.dumps({k: spec[k] for k in ("markers", "parts", "screws")}, indent=1))

# ---------------- interference: wrist parts (link6 + gripper swept over joint 6), and A vs B ----------------
d_ab = BRepExtrema_DistShapeShape(A, B); d_ab.Perform()
print(f"A-B min distance {d_ab.Value():.3f} mm, overlap {vol(BRepAlgoAPI_Common(A, B).Shape()):.3f} mm3")
doc = TDocStd_Document(TCollection_ExtendedString("d"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); assert r.ReadFile(a.step) == IFSelect_RetDone; r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
labs = Sequence_TDF_Label(); st.GetFreeShapes(labs)
rows = []
for i in range(1, labs.Length() + 1):
    lab = labs.Value(i); nm = TDataStd_Name(); lab.FindAttribute(TDataStd_Name.GetID_s(), nm)
    name = nm.Get().ToExtString(); shp = st.GetShape_s(lab)
    bb_ = Bnd_Box(); BRepBndLib.Add_s(shp, bb_, False)
    if "KM3-9" in name and bb_.CornerMin().X() < X_PLATE + 0.1 and any(abs(bb_.CornerMin().Y() + 3 - y) < 1 for y, _ in SCREWS):
        continue
    angles = [0] if name.startswith("Link5/") else range(-180, 180, 10)
    dmin, ov = 1e9, 0.0
    for ang in angles:
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(ang))
        s_ = BRepBuilderAPI_Transform(shp, tr, True).Shape() if ang else shp
        dd = BRepExtrema_DistShapeShape(both, s_); dmin = min(dmin, dd.Value())
        if dd.Value() < 1e-6:
            ov = max(ov, vol(BRepAlgoAPI_Common(both, s_).Shape()))
    rows.append((dmin, name, ov))
rows.sort()
print("closest parts (mm; link6 / gripper swept over joint 6):")
for dmin, name, ov in rows[:8]:
    print(f"  {dmin:7.3f}  {name}" + (f"   OVERLAP {ov:.3f} mm3" if ov > 1e-6 else ""))
