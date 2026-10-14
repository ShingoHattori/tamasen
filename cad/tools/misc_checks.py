import sys, json, importlib.util
import numpy as np, trimesh
spec = importlib.util.spec_from_file_location("fl", sys.argv[0].replace("misc_checks.py", "fklib.py"))
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
pub, sdk, cad_dir = sys.argv[1:4]
mp, ms = fl.Model(pub), fl.Model(sdk)
T = mp.fk([-np.pi, 0, 0, 0, 0, 0]); Tc = np.eye(4); Tc[2, 3] = 0.003295
Tinv = np.linalg.inv(Tc @ T["end_link"])
# 1) SDK meshes vs public meshes per link (link frame)
for ln in ms.links:
    a, b = ms.links[ln], mp.links.get(ln)
    if b is None: print(ln, "no public visual"); continue
    P = trimesh.sample.sample_surface(a, 4000, seed=4)[0]
    _, d, _ = trimesh.proximity.closest_point(b, P)
    print(f"SDK {ln:9s} vs public: med {np.median(d)*1e3:.2f} mm p90 {np.percentile(d,90)*1e3:.2f} p99 {np.percentile(d,99)*1e3:.2f}  bbox sdk {np.round(a.bounds*1e3,1).tolist()}")
# 2) top faces of the gripper body in end_link frame
idx = {e["path"]: e["file"] for e in json.load(open(f"{cad_dir}/index.json"))}
for part in ("Gripper:1/Gripper_3D_Printed_Parts:1/01_Rail_Bracket:1", "Gripper:1/Gripper_Metal_Parts:1/02_Slider_Bracket:1"):
    m = trimesh.load(f"{cad_dir}/{idx[part]}"); m.apply_scale(1e-3); m.apply_transform(Tinv)
    for sgn in (+1, -1):
        sel = (m.face_normals[:, 2] * sgn > 0.999)
        zc = m.triangles_center[sel][:, 2]
        if not sel.any(): continue
        zt = zc.max() if sgn > 0 else zc.min()
        top = sel & (np.abs(m.triangles_center[:, 2] - zt) < 1e-5)
        v = m.triangles[top].reshape(-1, 3)
        print(f"{part.split('/')[-1]:22s} face n={'+' if sgn>0 else '-'}z at z={zt*1e3:.2f} mm, area {m.area_faces[top].sum()*1e6:.0f} mm2, x {v[:,0].min()*1e3:.1f}..{v[:,0].max()*1e3:.1f}, y {v[:,1].min()*1e3:.1f}..{v[:,1].max()*1e3:.1f}")
