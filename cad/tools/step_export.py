"""Tessellate STEP assembly nodes (world placement, mm) to STL.
usage: step_export.py file.step outdir 'regex-of-instance-path' ...
Each depth-1 node -> <name>.stl; nodes whose path matches a regex are exported individually too."""
import sys, re, json
from pathlib import Path
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_Label
from OCP.collections import Sequence_TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.TopLoc import TopLoc_Location
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.IFSelect import IFSelect_RetDone

path, outdir = sys.argv[1], Path(sys.argv[2]); pats = [re.compile(p) for p in sys.argv[3:]]
outdir.mkdir(parents=True, exist_ok=True)
doc = TDocStd_Document(TCollection_ExtendedString("doc"))
r = STEPCAFControl_Reader(); r.SetNameMode(True)
assert r.ReadFile(path) == IFSelect_RetDone
r.Transfer(doc)
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())

def name(lab):
    n = TDataStd_Name()
    return n.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), n) else "?"

def export(shape, fn):
    BRepMesh_IncrementalMesh(shape, 0.05, False, 0.2, True)
    w = StlAPI_Writer(); w.ASCIIMode = False
    w.Write(shape, str(outdir / fn))

index = []
def walk(lab, loc, depth, p):
    nm = name(lab)
    if st.IsReference_s(lab):
        ref = TDF_Label(); st.GetReferredShape_s(lab, ref)
        loc = loc.Multiplied(st.GetLocation_s(lab)); proto = ref
    else:
        proto = lab
    p = p + [nm]
    key = "/".join(p[1:])
    want = depth == 1 or any(x.search(key) for x in pats)
    if want:
        fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".stl"
        export(st.GetShape_s(proto).Moved(loc), fn)
        index.append({"path": key, "proto": name(proto), "file": fn, "depth": depth})
    if st.IsAssembly_s(proto):
        kids = Sequence_TDF_Label(); st.GetComponents_s(proto, kids, False)
        for i in range(1, kids.Length() + 1):
            walk(kids.Value(i), loc, depth + 1, p)

roots = Sequence_TDF_Label(); st.GetFreeShapes(roots)
for i in range(1, roots.Length() + 1):
    walk(roots.Value(i), TopLoc_Location(), 0, [])
json.dump(index, open(outdir / "index.json", "w"), indent=1)
print(len(index), "files")
