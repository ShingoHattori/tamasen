"""Bboxes of exported CAD parts in end_link frame; nearest part to a query point."""
import sys, json, importlib.util
import numpy as np, trimesh
spec = importlib.util.spec_from_file_location("fl", sys.argv[0].replace("parts_in_link.py", "fklib.py"))
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
model = fl.Model(sys.argv[1]); cad_dir = sys.argv[2]
T = fl.model_T = model.fk([-np.pi, 0, 0, 0, 0, 0])
Tcad_base = np.eye(4); Tcad_base[2, 3] = 0.003295
for ln in ("link5", "link6", "end_link"):
    Tl = Tcad_base @ T[ln]; Tinv = np.linalg.inv(Tl)
    print(f"T_cad_{ln} (mm):\n", np.round(np.hstack([Tl[:3, :3], Tl[:3, 3:] * 1e3]), 5))
Tinv = np.linalg.inv(Tcad_base @ T["end_link"])
q = np.array([float(v) for v in sys.argv[3].split(",")]) * 1e-3
best = []
for e in json.load(open(f"{cad_dir}/index.json")):
    if e["depth"] < 3: continue
    m = trimesh.load(f"{cad_dir}/{e['file']}"); m.apply_scale(1e-3); m.apply_transform(Tinv)
    lo, hi = m.bounds * 1e3
    _, d, _ = trimesh.proximity.closest_point(m, [q])
    best.append((d[0] * 1e3, e["path"]))
    if not any(k in e["path"] for k in ("KM3", "KA3", "HM3", "S-M4")):
        print(f"{e['path']:70s} [{lo[0]:7.1f} {lo[1]:6.1f} {lo[2]:6.1f}]..[{hi[0]:7.1f} {hi[1]:6.1f} {hi[2]:6.1f}]")
print("nearest to", sys.argv[3], "mm:", sorted(best)[:4])
