"""Re-express selected STEP assembly nodes in a link frame and write a flat, named STEP.
usage: step_reframe.py in.step out.step 'r11 r12 r13 tx r21 ... tz' Node1 Node2 ...
The 3x4 matrix (mm) is T_link_cad, applied to every leaf's world placement."""
import sys
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Trsf
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

src, dst, mat, nodes = sys.argv[1], sys.argv[2], [float(v) for v in sys.argv[3].split()], sys.argv[4:]
tr = gp_Trsf(); tr.SetValues(*mat)
doc = TDocStd_Document(TCollection_ExtendedString("in"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); r.SetColorMode(True)
assert r.ReadFile(src) == IFSelect_RetDone
r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
out = TDocStd_Document(TCollection_ExtendedString("out"))
ost = XCAFDoc_DocumentTool.ShapeTool_s(out.Main())

def name(lab):
    n = TDataStd_Name()
    return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

count = 0
def walk(lab, loc, p, take):
    global count
    nm = name(lab)
    if st.IsReference_s(lab):
        ref = TDF_Label(); st.GetReferredShape_s(lab, ref); loc = loc.Multiplied(st.GetLocation_s(lab)); proto = ref
    else:
        proto = lab
    p = p + [nm]
    take = take or (len(p) == 2 and nm in nodes)
    if st.IsAssembly_s(proto):
        kids = Sequence_TDF_Label(); st.GetComponents_s(proto, kids, False)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, p, take)
    elif take:
        shp = BRepBuilderAPI_Transform(st.GetShape_s(proto).Moved(loc), tr, True).Shape()
        l = ost.AddShape(shp, False)
        TDataStd_Name.Set_s(l, TCollection_ExtendedString("/".join(x.split(":")[0] if i < len(p) - 1 else x for i, x in enumerate(p[1:]))))
        count += 1

roots = Sequence_TDF_Label(); st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), [], False)
w = STEPCAFControl_Writer(); w.SetNameMode(True)
w.Transfer(out, STEPControl_AsIs)
assert w.Write(dst) == IFSelect_RetDone
print(count, "parts ->", dst)
