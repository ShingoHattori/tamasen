"""Marker mount v2: the ID 1 L-bracket of marker_mount.py, extended backwards over the top of the gripper
motor (DM-J4310:3, end_link x -148.4..-102.4, radius 28.5) with two inclined faces for two more 30 mm
markers (requester 2026-10-14: ID 1 disappears behind the motor top at some angles).

Same fixing as v1 (two countersunk tapping screws 3 x 16 through the tab, cone bosses in the emptied plate
countersinks). The wing is a slab above the motor (z >= SLAB_Z0) joined to the tab, and each marker face is
a plate standing on a wedge that is extruded straight down (-z) onto the slab, so it prints without
supports with the tab's outer face on the bed... (see cad/README.md for the print orientation).

Faces are given as  tilt,azimuth,x,y  (deg, deg, mm, mm): the face normal is end_link +z tilted by `tilt`
towards the direction `azimuth` (0 = -x, i.e. backwards / towards the wrist; 90 = +y; -90 = -y), and (x, y)
is the face centre's footprint. The face centre height follows from the slab and the plate size.

usage: marker_mount2.py wrist_gripper_in_end_link.step outdir --face 35,0,-128,0 --face ...
Writes marker_mount2.step / .stl / _coarse.stl / .json (T_end_link_marker for each marker) and prints the
interference check against every part of the wrist/gripper STEP (Link5 swept over joint 6).
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
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ap = argparse.ArgumentParser()
ap.add_argument("step"); ap.add_argument("outdir")
ap.add_argument("--face", action="append", required=True, help="tilt,azimuth,x,y (deg, deg, mm, mm), see above")
ap.add_argument("--t", type=float, default=4.0, help="tab / platform / slab thickness, mm")
ap.add_argument("--plate", type=float, default=3.0, help="marker plate thickness, mm")
ap.add_argument("--marker", type=float, default=30.0, help="black square side, mm")
ap.add_argument("--margin", type=float, default=2.0, help="white margin of the sticker inside the rim, mm")
ap.add_argument("--rim", type=float, nargs=2, default=[1.2, 0.8], metavar=("WIDTH", "HEIGHT"))
ap.add_argument("--slab-z0", type=float, default=39.5, help="underside of the wing slab: 1 mm above the cable restraint rings on the motor (z <= 38.5, x -152..-122), so also clear of the screw heads (z = 30) and the driver path to them")
ap.add_argument("--x-min", type=float, default=-148.0, help="rearmost x of the wing (Link5 sweeps x < -151.4)")
ap.add_argument("--boss-clearance", type=float, default=0.1)
ap.add_argument("--boss-height", type=float, default=0.8)
ap.add_argument("--relief", type=float, default=1.0)
a = ap.parse_args()

# --- v1 reference geometry (end_link, mm; cad/README.md) ---
X_BACK, Z_TOP = -103.374, 34.24
SCREW_YZ = [(-11.0, 30.0), (11.0, 30.0)]
PLATE_CSK_D, PLATE_CSK_DEPTH = 6.0, 1.4
R_CLEAR = 29.0
HOLE_D, CSK_D = 3.4, 6.4


def box(x0, y0, z0, dx, dy, dz): return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()
def cut(s, u): return BRepAlgoAPI_Cut(s, u).Shape()
def fuse(s, u): return BRepAlgoAPI_Fuse(s, u).Shape()
def vol(s):
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); return g.Mass()
def bbox(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b); p, q = b.CornerMin(), b.CornerMax(); return [p.X(), p.Y(), p.Z(), q.X(), q.Y(), q.Z()]


t = a.t
W = max(40.0, a.marker + 10.0)
L = W
x0 = X_BACK - t
# ---- v1 body: platform (ID 1) + tab + motor arc + chamfer + screw holes + bosses ----
body = fuse(box(x0, -W / 2, Z_TOP, L, W, t), box(x0, -W / 2, 20.0, t, W, Z_TOP + t - 20.0))

# ---- wing: slab over the motor + one plate (with rim) on a wedge per face ----
S = a.marker + 2 * a.margin + 2 * a.rim[0]          # plate side
faces = []
for k, f in enumerate(a.face):
    tilt, az, fx, fy = (float(v) for v in f.split(","))
    T_, A_ = math.radians(tilt), math.radians(az)
    n = np.array([-math.sin(T_) * math.cos(A_), math.sin(T_) * math.sin(A_), math.cos(T_)])
    # plate frame: e_up = direction of steepest ascent on the plate (marker "up"), e_side = n x e_up
    horiz = np.array([-math.cos(A_), math.sin(A_), 0.0])             # azimuth direction in the xy plane
    e_up = np.cross(np.cross(n, -horiz), n); e_up /= np.linalg.norm(e_up)   # towards the high edge
    if tilt < 1e-6:
        e_up = np.array([-1.0, 0, 0])
    e_side = np.cross(n, e_up)
    # lowest plate corner sits on the slab top: centre height so that the low edge is at the slab top + plate
    zs = a.slab_z0 + t
    half_drop = (S / 2) * abs(e_up[2])
    c_bottom = np.array([fx, fy, zs + half_drop + 0.5])                # plate underside centre
    corners = [c_bottom + sx * S / 2 * e_side + sy * S / 2 * e_up for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    # plate = prism of the square along n; wedge = prism of the same square straight down to below the slab top
    poly = BRepBuilderAPI_MakePolygon()
    for p in corners: poly.Add(gp_Pnt(*p))
    poly.Close()
    sq = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    plate = BRepPrimAPI_MakePrism(sq, gp_Vec(*(n * a.plate))).Shape()
    depth = max(p[2] for p in corners) - zs + 1.0
    wedge = BRepPrimAPI_MakePrism(sq, gp_Vec(0, 0, -depth)).Shape()
    # rim: ring on the plate face
    top_c = c_bottom + n * a.plate
    ring_o = [top_c + sx * S / 2 * e_side + sy * S / 2 * e_up for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    si = S / 2 - a.rim[0]
    ring_i = [top_c + sx * si * e_side + sy * si * e_up for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    def prism(pts, h):
        pl = BRepBuilderAPI_MakePolygon()
        for p in pts: pl.Add(gp_Pnt(*p))
        pl.Close()
        return BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(pl.Wire()).Face(), gp_Vec(*(n * h))).Shape()
    rim = cut(prism(ring_o, a.rim[1]), prism([p - n * 0.5 for p in ring_i], a.rim[1] + 1.0))
    part = fuse(fuse(plate, wedge), rim)
    faces.append({"tilt_deg": tilt, "azimuth_deg": az, "normal": n.tolist(), "centre_mm": top_c.tolist(),
                  "x_axis": e_side.tolist(), "y_axis": e_up.tolist(), "part": part})
    body = fuse(body, part)
# slab: covers every wedge footprint, joined to the tab
xs, ys = [], []
for fc in faces:
    b = bbox(fc["part"]); xs += [b[0], b[3]]; ys += [b[1], b[4]]
sx0, sx1 = max(min(xs) - 1, a.x_min), x0 + t
sy0, sy1 = min(min(ys) - 1, -W / 2), max(max(ys) + 1, W / 2)
slab = box(sx0, sy0, a.slab_z0, sx1 - sx0, sy1 - sy0, t)
body = fuse(body, slab)
# riser joining the slab to the top of the tab / back edge of the ID 1 platform (the slab sits higher)
if a.slab_z0 > Z_TOP + t - 0.01:
    body = fuse(body, box(x0, -W / 2, Z_TOP + t - 0.5, t, W, a.slab_z0 - (Z_TOP + t) + 1.0))
# trim everything to the wing envelope: nothing below the slab (except the v1 tab), nothing behind x_min
below = box(a.x_min - 50, -200, -100, x0 - (a.x_min - 50), 400, 100 + a.slab_z0)     # x < tab, z < slab underside
body = cut(body, below)
body = cut(body, box(a.x_min - 100, -200, -100, 100, 400, 300))                          # x < x_min

# ---- v1 details: motor arc, inner chamfer, screw holes, bosses ----
body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0 - 1, 0, 0), gp_Dir(1, 0, 0)), R_CLEAR, t + 2).Shape())
c = a.relief
tri = BRepBuilderAPI_MakePolygon(gp_Pnt(X_BACK + 0.01, -W / 2 - 1, Z_TOP - 0.01), gp_Pnt(X_BACK - c, -W / 2 - 1, Z_TOP - 0.01),
                                 gp_Pnt(X_BACK + 0.01, -W / 2 - 1, Z_TOP + c), True).Wire()
body = cut(body, BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(tri).Face(), gp_Vec(0, W + 2, 0)).Shape())
for y, z in SCREW_YZ:
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0 - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, t + 2).Shape())
    h = (CSK_D - HOLE_D) / 2
    body = cut(body, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x0 - 0.01, y, z), gp_Dir(1, 0, 0)), CSK_D / 2 + 0.01, HOLE_D / 2, h + 0.01).Shape())
rb = PLATE_CSK_D / 2 - a.boss_clearance
for y, z in SCREW_YZ:
    boss = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(X_BACK - 0.01, y, z), gp_Dir(1, 0, 0)), rb, rb - a.boss_height - 0.01, a.boss_height + 0.01).Shape()
    boss = cut(boss, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_BACK - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, a.boss_height + 2).Shape())
    body = fuse(body, boss)
# screw access: the screw heads sit on the tab's back face, under the wing slab -- keep a drill/driver channel
for y, z in SCREW_YZ:
    if a.slab_z0 < z + CSK_D / 2 + 1:
        print(f"note: screw at z={z} is under the slab (z0={a.slab_z0}); driver reaches it along x through the channel")
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0 - 0.01, y, z), gp_Dir(-1, 0, 0)), CSK_D / 2 + 0.5, x0 - a.x_min + 1).Shape())

out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
w = STEPControl_Writer(); w.Transfer(body, STEPControl_AsIs); assert w.Write(str(out / "marker_mount2.step")) == IFSelect_RetDone
BRepMesh_IncrementalMesh(body, 0.02, False, 0.1, True); sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(body, str(out / "marker_mount2.stl"))
coarse = BRepBuilderAPI_Transform(body, gp_Trsf(), True).Shape()
BRepMesh_IncrementalMesh(coarse, 0.2, False, 0.5, True); sw.Write(coarse, str(out / "marker_mount2_coarse.stl"))

mk = [{"id": "1 (platform, as v1)", "centre_mm": [x0 + L / 2, 0.0, Z_TOP + t], "x_axis": [0, 1, 0], "y_axis": [-1, 0, 0], "normal": [0, 0, 1]}]
for k, fc in enumerate(faces):
    mk.append({"id": f"rear face {k + 1}", "tilt_deg": fc["tilt_deg"], "azimuth_deg": fc["azimuth_deg"], "centre_mm": [round(v, 3) for v in fc["centre_mm"]],
               "x_axis": [round(v, 5) for v in fc["x_axis"]], "y_axis": [round(v, 5) for v in fc["y_axis"]], "normal": [round(v, 5) for v in fc["normal"]]})
for m_ in mk:
    R = np.array([m_["x_axis"], m_["y_axis"], m_["normal"]]).T
    m_["T_end_link_marker"] = np.vstack([np.c_[R, m_["centre_mm"]], [0, 0, 0, 1]]).round(5).tolist()
bb = bbox(body)
spec = {"frame": "end_link (URDF), mm", "link": "end_link", "markers": mk, "marker_size_mm": a.marker,
        "sticker_area_mm": a.marker + 2 * a.margin, "rim_mm": a.rim, "slab_z_mm": [a.slab_z0, a.slab_z0 + t],
        "bbox_mm": [round(v, 2) for v in bb], "volume_mm3": vol(body),
        "screws": "same as v1: countersunk tapping screws 3 x 16 through the tab (y = +-11, z = +30)",
        "note": "design value from CAD; marker centre = black square centre on the sticker plane (sticker thickness not included)"}
json.dump(spec, open(out / "marker_mount2.json", "w"), indent=1)
print(json.dumps({k: spec[k] for k in ("markers", "bbox_mm", "volume_mm3")}, indent=1))

# ---- interference with every part of the wrist/gripper STEP (Link5 swept over joint 6) ----
doc = TDocStd_Document(TCollection_ExtendedString("d"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); assert r.ReadFile(a.step) == IFSelect_RetDone; r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
labs = TDF_LabelSequence(); st.GetFreeShapes(labs)
rows = []
for i in range(1, labs.Length() + 1):
    lab = labs.Value(i); nm = TDataStd_Name(); lab.FindAttribute(TDataStd_Name.GetID_s(), nm)
    name = nm.Get().ToExtString(); shp = st.GetShape_s(lab)
    b = Bnd_Box(); BRepBndLib.Add_s(shp, b, False)
    if "KA3-12" in name and any(abs(b.CornerMin().Y() + 3 - y) < 1 and abs(b.CornerMin().Z() + 3 - z) < 1 for y, z in SCREW_YZ) \
            and b.CornerMin().X() < X_BACK + 0.1:
        continue
    angles = range(-180, 180, 10) if name.startswith("Link5/") else [0]
    dmin, ov = 1e9, 0.0
    for ang in angles:
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(ang))
        s = BRepBuilderAPI_Transform(shp, tr, True).Shape() if ang else shp
        dd = BRepExtrema_DistShapeShape(body, s); dmin = min(dmin, dd.Value())
        if dd.Value() < 1e-6:
            ov = max(ov, vol(BRepAlgoAPI_Common(body, s).Shape()))
    rows.append((dmin, name, ov))
rows.sort()
print("closest parts (mm; Link5 swept over joint 6 = -180..170 deg):")
for dmin, name, ov in rows[:12]:
    print(f"  {dmin:7.3f}  {name}" + (f"   OVERLAP {ov:.3f} mm3" if ov > 1e-6 else ""))
