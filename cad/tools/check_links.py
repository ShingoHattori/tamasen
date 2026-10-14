"""Per-link check: URDF link mesh at FK(q) vs CAD group mesh (exact point-to-mesh), and rigid ICP offset."""
import sys, json, importlib.util
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
spec = importlib.util.spec_from_file_location("reg", sys.argv[0].replace("check_links.py", "fklib.py"))
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
urdf, cad_dir = sys.argv[1], sys.argv[2]
model = fl.Model(urdf)
groups = {"base_link": ["Base:1"], "link1": ["Link1:1"], "link2": ["Link2:1"], "link3": ["Link3:1", "01_Lower_Arm_Filler_M:1"],
          "link4": ["Link4:1"], "link5": ["Link5:1"], "link6": ["Link6:1"], "end_link": ["Gripper:1"],
          "finger_left_link": ["Gripper:1"], "finger_right_link": ["Gripper:1"]}
idx = {e["path"]: e["file"] for e in json.load(open(f"{cad_dir}/index.json"))}
cad = {}
def cadmesh(names):
    k = tuple(names)
    if k not in cad:
        m = trimesh.util.concatenate([trimesh.load(f"{cad_dir}/{idx[n]}") for n in names]); m.apply_scale(1e-3); cad[k] = m
    return cad[k]
fit = json.load(open(f"{cad_dir}/register.json"))
for label, q in (("q = (-pi,0,0,0,0,0)", [-np.pi, 0, 0, 0, 0, 0]), ("q fitted", fit["q_cad"])):
    T = model.fk(q, 0.0)
    print("==", label)
    for ln, gn in groups.items():
        P = trimesh.sample.sample_surface(model.links[ln], 3000, seed=2)[0]
        Pw = P @ T[ln][:3, :3].T + T[ln][:3, 3]
        M = cadmesh(gn)
        _, d, _ = trimesh.proximity.closest_point(M, Pw)
        # rigid ICP on the points within 3 mm (outliers = parts absent in one model)
        A = Pw.copy(); Tacc = np.eye(4)
        for _ in range(30):
            c, d2, _ = trimesh.proximity.closest_point(M, A)
            w = d2 < 0.003
            mA, mC = A[w].mean(0), c[w].mean(0)
            U, _, Vt = np.linalg.svd((A[w] - mA).T @ (c[w] - mC))
            R = (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T
            t = mC - R @ mA
            A = A @ R.T + t; St = np.eye(4); St[:3, :3] = R; St[:3, 3] = t; Tacc = St @ Tacc
        # express the correction at the link origin, in the link frame
        Tl = T[ln]; D = np.linalg.inv(Tl) @ Tacc @ Tl
        print(f"  {ln:18s} med {np.median(d)*1e3:5.2f} p90 {np.percentile(d,90)*1e3:6.2f} mm in<0.5mm {np.mean(d<5e-4):.2f} | "
              f"ICP residual med {np.median(d2[w])*1e3:5.3f} mm  corr t(link) {np.round(D[:3,3]*1e3,2)} mm rot {np.degrees(np.linalg.norm(Rot.from_matrix(D[:3,:3]).as_rotvec())):.3f} deg")
