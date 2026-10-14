"""ID 1 marker mount for the B601-DM gripper, built in end_link coordinates (mm).

An L-bracket hooked over the top edge of the metal 02_Slider_Bracket (gripper back plate):
  - tab on the back face, held by the two existing upper countersunk screws (KA3-12 at y = +-11, z = +30),
    replaced by countersunk tapping screws of the same kind, longer by the tab thickness (KA3x12 -> 3 x 16 mm
    for t = 4). KA3 is a self-tapping screw (upstream BOM) cutting its own thread in the 2.7 mm pilot hole of
    the printed 01_Rail_Bracket; the longer screw ends at the same depth, so it follows the thread already cut
    (the pilot hole is 3 mm deeper than the tip);
  - on the inner face of the tab, a 45 deg cone boss under each screw that sits in the 90 deg countersink the
    old screw head leaves empty in the metal plate (dia 6.0, 1.4 deep). The boss is lower than the recess,
    so the flat face seats on the plate (x) and the cones locate y and z; the new screw head sits in a
    countersink on the outer face of the tab;
  - platform on top (normal = end_link +z), the marker is stuck on it, cut to the platform size.
Located by the back face (x), the top edge (z) and the two screws (y).

usage: marker_mount.py wrist_gripper_in_end_link.step outdir [--t 4] [--marker 30]
Writes marker_mount.step / .stl (end_link frame, mm), marker_mount.json (T_end_link_marker) and prints
the interference check against every part of the wrist/gripper STEP (link5 swept over joint 6).
"""
import argparse, json, math
from pathlib import Path
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Trsf, gp_Ax1, gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakePrism
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs, STEPControl_Reader
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.IFSelect import IFSelect_RetDone
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

ap = argparse.ArgumentParser()
ap.add_argument("step"); ap.add_argument("outdir")
ap.add_argument("--t", type=float, default=4.0, help="tab and platform thickness, mm")
ap.add_argument("--marker", type=float, default=30.0, help="black square side, mm")
ap.add_argument("--boss-clearance", type=float, default=0.1, help="radial gap between boss cone and plate countersink, mm")
ap.add_argument("--boss-height", type=float, default=0.8, help="boss height (< 1.4 mm recess depth), mm")
ap.add_argument("--relief", type=float, default=1.0, help="45 deg chamfer along the inner corner of the L, mm")
a = ap.parse_args()

# --- reference geometry from the CAD (end_link frame, mm; see cad/README.md) ---
X_BACK = -103.374      # back face of 02_Slider_Bracket (BRep plane)
Z_TOP = 34.24          # top edge of 02_Slider_Bracket (01_Rail_Bracket top is 34.04, 0.2 mm lower)
SCREW_YZ = [(-11.0, 30.0), (11.0, 30.0)]   # KA3-12:4 / :5, axis along x, countersunk heads flush with X_BACK
PLATE_CSK_D, PLATE_CSK_DEPTH = 6.0, 1.4    # 90 deg countersink in 02_Slider_Bracket (BRep cone, semi-angle 45)
R_MOTOR = 28.5         # DM-J4310 gripper motor directly behind the back plate (axis = end_link x)
R_CLEAR = 29.0         # tab keeps outside this radius
HOLE_D, CSK_D = 3.4, 6.4

t = a.t
W = max(40.0, a.marker + 10.0)   # platform width (y) and depth (x): marker + 5 mm white margin each side
L = W
x0 = X_BACK - t

def box(x0, y0, z0, dx, dy, dz): return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()
def cut(a, b): return BRepAlgoAPI_Cut(a, b).Shape()
def fuse(a, b): return BRepAlgoAPI_Fuse(a, b).Shape()

platform = box(x0, -W / 2, Z_TOP, L, W, t)
TAB_Z0 = 20.0
tab = box(x0, -W / 2, TAB_Z0, t, W, Z_TOP + t - TAB_Z0)
body = fuse(platform, tab)
# clearance around the motor (cylinder along end_link x through y = z = 0)
body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0 - 1, 0, 0), gp_Dir(1, 0, 0)), R_CLEAR, t + 2).Shape())
# 45 deg chamfer along the inner corner, so the L seats on both the back face and the top edge without
# riding on the plate's edge (a square groove would leave a horizontal overhang when printed marker face down)
c = a.relief
tri = BRepBuilderAPI_MakePolygon(gp_Pnt(X_BACK + 0.01, -W / 2 - 1, Z_TOP - 0.01), gp_Pnt(X_BACK - c, -W / 2 - 1, Z_TOP - 0.01),
                                 gp_Pnt(X_BACK + 0.01, -W / 2 - 1, Z_TOP + c), True).Wire()
body = cut(body, BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(tri).Face(), gp_Vec(0, W + 2, 0)).Shape())
for y, z in SCREW_YZ:
    body = cut(body, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x0 - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, t + 2).Shape())
    h = (CSK_D - HOLE_D) / 2   # 90 deg countersink depth
    body = cut(body, BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x0 - 0.01, y, z), gp_Dir(1, 0, 0)), CSK_D / 2 + 0.01, HOLE_D / 2, h + 0.01).Shape())
# cone bosses that fit the emptied countersinks of the metal plate
assert a.boss_height < PLATE_CSK_DEPTH
rb = PLATE_CSK_D / 2 - a.boss_clearance
for y, z in SCREW_YZ:
    boss = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(X_BACK - 0.01, y, z), gp_Dir(1, 0, 0)), rb, rb - a.boss_height - 0.01, a.boss_height + 0.01).Shape()
    boss = cut(boss, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(X_BACK - 1, y, z), gp_Dir(1, 0, 0)), HOLE_D / 2, a.boss_height + 2).Shape())
    body = fuse(body, boss)

out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
w = STEPControl_Writer(); w.Transfer(body, STEPControl_AsIs); assert w.Write(str(out / "marker_mount.step")) == IFSelect_RetDone
BRepMesh_IncrementalMesh(body, 0.02, False, 0.1, True); sw = StlAPI_Writer(); sw.ASCIIMode = False; sw.Write(body, str(out / "marker_mount.stl"))
# coarse copy for the simulator's visual mesh (sim/models/marker_mount/)
coarse = BRepBuilderAPI_Transform(body, gp_Trsf(), True).Shape()
BRepMesh_IncrementalMesh(coarse, 0.2, False, 0.5, True); sw.Write(coarse, str(out / "marker_mount_coarse.stl"))
g = GProp_GProps(); BRepGProp.VolumeProperties_s(body, g)

marker_c = [x0 + L / 2, 0.0, Z_TOP + t]
spec = {
    "frame": "end_link (URDF), mm",
    "link": "end_link",
    "thickness_mm": t, "platform_mm": [L, W], "marker_size_mm": a.marker, "inner_corner_chamfer_mm": a.relief,
    "screws": f"replace KA3-12:4 and KA3-12:5 (upper back, y = +-11, z = +30) by countersunk self-tapping screws 3 x {12 + t:g} mm, "
              "the same kind as KA3 (upstream BOM: Phillips tapping screw) so that they follow the existing thread in the printed rail bracket",
    "bosses": {"cone_semi_angle_deg": 45, "base_dia_mm": 2 * rb, "height_mm": a.boss_height,
               "fits": "90 deg countersink dia 6.0 x 1.4 deep in 02_Slider_Bracket", "radial_clearance_mm": a.boss_clearance},
    "marker_centre_mm": marker_c,
    "marker_axes_in_end_link": {"x": [0, 1, 0], "y": [-1, 0, 0], "z": [0, 0, 1]},
    "T_end_link_marker": [[0, -1, 0, marker_c[0]], [1, 0, 0, 0.0], [0, 0, 1, marker_c[2]], [0, 0, 0, 1]],
    "note": "design value from CAD; sticker thickness (~0.1 mm) and print tolerance not included; not built, not measured",
    "volume_mm3": g.Mass(),
    # boxes for collision checks (centre, half sizes; end_link, mm). The tab box also covers the motor
    # cut-out; that region belongs to link6 (the parent of end_link), whose contacts MuJoCo skips anyway.
    "collision_boxes_mm": [
        {"name": "platform", "centre": [x0 + L / 2, 0.0, Z_TOP + t / 2], "half": [L / 2, W / 2, t / 2]},
        {"name": "tab", "centre": [x0 + t / 2, 0.0, (TAB_Z0 + Z_TOP + t) / 2], "half": [t / 2, W / 2, (Z_TOP + t - TAB_Z0) / 2]},
    ],
}
json.dump(spec, open(out / "marker_mount.json", "w"), indent=1)
print(json.dumps({k: spec[k] for k in ("screws", "marker_centre_mm", "marker_axes_in_end_link", "volume_mm3")}, indent=1))

# --- interference check against every part of the wrist/gripper STEP ---
doc = TDocStd_Document(TCollection_ExtendedString("d"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); assert r.ReadFile(a.step) == IFSelect_RetDone; r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
labs = Sequence_TDF_Label(); st.GetFreeShapes(labs)
rows = []
for i in range(1, labs.Length() + 1):
    lab = labs.Value(i); n = TDataStd_Name(); lab.FindAttribute(TDataStd_Name.GetID_s(), n)
    name = n.Get().ToExtString(); shp = st.GetShape_s(lab)
    bb = Bnd_Box(); BRepBndLib.Add_s(shp, bb, False)
    if "KA3-12" in name and any(abs(bb.CornerMin().Y() + 3 - y) < 1 and abs(bb.CornerMin().Z() + 3 - z) < 1 for y, z in SCREW_YZ) \
            and bb.CornerMin().X() < X_BACK + 0.1:
        continue      # the two screws being replaced
    angles = range(-180, 180, 10) if name.startswith("Link5/") else [0]   # link5 turns about end_link x (joint 6)
    dmin = 1e9; vol = 0.0
    for ang in angles:
        tr = gp_Trsf(); tr.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), math.radians(ang))
        s = BRepBuilderAPI_Transform(shp, tr, True).Shape() if ang else shp
        d = BRepExtrema_DistShapeShape(body, s); dmin = min(dmin, d.Value())
        if d.Value() < 1e-6:
            gp_ = GProp_GProps(); BRepGProp.VolumeProperties_s(BRepAlgoAPI_Common(body, s).Shape(), gp_); vol = max(vol, gp_.Mass())
    rows.append((dmin, name, vol))
rows.sort()
print("closest parts (mm; Link5 swept over joint 6 = -180..170 deg):")
for dmin, name, vol in rows[:10]:
    print(f"  {dmin:7.3f}  {name}" + (f"   OVERLAP {vol:.3f} mm3" if vol > 1e-6 else ""))
