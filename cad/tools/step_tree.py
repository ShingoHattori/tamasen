"""List the product tree of a STEP assembly: name, depth, world placement, bbox (mm)."""
import sys, json
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopLoc import TopLoc_Location
from OCP.IFSelect import IFSelect_RetDone

path = sys.argv[1]
doc = TDocStd_Document(TCollection_ExtendedString("doc"))
r = STEPCAFControl_Reader(); r.SetNameMode(True)
assert r.ReadFile(path) == IFSelect_RetDone
r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

def name(lab):
    n = TDataStd_Name()
    return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

out = []
def walk(lab, loc, depth, path):
    nm = name(lab)
    ref = TDF_Label()
    if st.IsReference_s(lab):
        st.GetReferredShape_s(lab, ref)
        loc = loc.Multiplied(st.GetLocation_s(lab))
        proto = ref
    else:
        proto = lab
    pnm = name(proto)
    shape = st.GetShape_s(proto).Moved(loc)
    b = Bnd_Box(); BRepBndLib.Add_s(shape, b, False)
    bb = None if b.IsVoid() else (*[b.CornerMin().Coord(i) for i in (1,2,3)], *[b.CornerMax().Coord(i) for i in (1,2,3)])
    T = loc.Transformation()
    M = [[T.Value(i, j) for j in range(1, 5)] for i in range(1, 4)]
    kids = TDF_LabelSequence()
    is_asm = st.IsAssembly_s(proto)
    out.append({"depth": depth, "inst": nm, "proto": pnm, "asm": bool(is_asm), "M": M, "bbox": bb, "path": path + [nm]})
    if is_asm:
        st.GetComponents_s(proto, kids, False)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, depth + 1, path + [nm])

roots = TDF_LabelSequence(); st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), 0, [])
json.dump(out, open(sys.argv[2], "w"))
for o in out:
    if o["depth"] <= (int(sys.argv[3]) if len(sys.argv) > 3 else 2):
        bb = o["bbox"]; s = "" if bb is None else " bbox=[%.1f %.1f %.1f]..[%.1f %.1f %.1f]" % bb
        print("  " * o["depth"] + f"{o['inst']} <{o['proto']}>{' ASM' if o['asm'] else ''}{s}")
