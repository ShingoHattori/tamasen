"""Fit one rigid T_cad_base (6 dof) with q = (-pi,0,0,0,0,0) exactly, using links 2..end (exact point-to-mesh)."""
import sys, json, importlib.util
import numpy as np, trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
spec = importlib.util.spec_from_file_location("fl", sys.argv[0].replace("fit_offset.py", "fklib.py"))
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
model = fl.Model(sys.argv[1]); cad_dir = sys.argv[2]
idx = {e["path"]: e["file"] for e in json.load(open(f"{cad_dir}/index.json"))}
G = {"link2": ["Link2:1"], "link3": ["Link3:1", "01_Lower_Arm_Filler_M:1"], "link4": ["Link4:1"], "link5": ["Link5:1"], "link6": ["Link6:1"], "end_link": ["Gripper:1"]}
T = model.fk([-np.pi, 0, 0, 0, 0, 0])
P, Ms = [], []
for ln, gn in G.items():
    m = trimesh.util.concatenate([trimesh.load(f"{cad_dir}/{idx[n]}") for n in gn]); m.apply_scale(1e-3)
    p = trimesh.sample.sample_surface(model.links[ln], 1500, seed=3)[0]
    P.append((p @ T[ln][:3, :3].T + T[ln][:3, 3], m))
def res(x):
    R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:]
    out = []
    for p, m in P:
        q = p @ R.T + t; _, d, _ = trimesh.proximity.closest_point(m, q); out.append(np.minimum(d, 0.006))
    return np.concatenate(out)
x = least_squares(res, np.array([0,0,0,0,0,float(sys.argv[3])]), diff_step=1e-3).x
print("T_cad_base: t (mm) =", np.round(x[3:] * 1e3, 3), " rotvec (deg) =", np.round(np.degrees(x[:3]), 3))
for (p, m), ln in zip(P, G):
    R = Rot.from_rotvec(x[:3]).as_matrix(); _, d, _ = trimesh.proximity.closest_point(m, p @ R.T + x[3:])
    print(f"  {ln:9s} med {np.median(d)*1e3:.3f} mm  p90 {np.percentile(d,90)*1e3:.3f} mm  in<0.5mm {np.mean(d<5e-4):.2f}")
