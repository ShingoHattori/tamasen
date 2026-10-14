"""Fit URDF (link frames, metres) to a STEP assembly export (world, mm).
Assumes CAD world == URDF base_link frame up to mm->m, checks that, then solves joint angles q_cad."""
import sys, json, math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

urdf, cad_dir = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(0)

def rpy(r, p, y): return Rot.from_euler("xyz", [r, p, y]).as_matrix()
def H(R, t): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M

root = ET.parse(urdf).getroot()
joints = []
for j in root.findall("joint"):
    o = j.find("origin"); a = j.find("axis")
    joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
        child=j.find("child").get("link"),
        T=H(rpy(*map(float, o.get("rpy").split())), np.array(list(map(float, o.get("xyz").split())))),
        axis=np.array(list(map(float, a.get("xyz").split()))) if a is not None else None))
links = {}
for l in root.findall("link"):
    fs = [m.get("filename") for m in l.findall("visual/geometry/mesh")]
    if fs:
        ms = [trimesh.load(urdf.parent / f) for f in fs]
        links[l.get("name")] = trimesh.util.concatenate(ms)
ARM = [f"joint{i}" for i in range(1, 7)]

def fk(q, fingers=0.0):
    T = {"base_link": np.eye(4)}
    for j in joints:
        M = T[j["parent"]] @ j["T"]
        if j["type"] == "revolute":
            M = M @ H(Rot.from_rotvec(j["axis"] * q[ARM.index(j["name"])]).as_matrix(), np.zeros(3))
        elif j["type"] == "prismatic":
            s = fingers if j["name"] == "finger_left" else -fingers
            M = M @ H(np.eye(3), j["axis"] * s)
        T[j["child"]] = M
    return T

idx = json.load(open(cad_dir / "index.json"))
groups = {}
for e in idx:
    if e["depth"] == 1:
        m = trimesh.load(cad_dir / e["file"]); m.apply_scale(1e-3)
        groups[e["path"]] = m
def samp(m, n): return trimesh.sample.sample_surface(m, n, seed=1)[0]
cad_pts, cad_lab = [], []
for k, m in groups.items():
    p = samp(m, 15000); cad_pts.append(p); cad_lab += [k] * len(p)
cad_pts = np.vstack(cad_pts); cad_lab = np.array(cad_lab)
tree = cKDTree(cad_pts)
lpts = {k: samp(m, 4000) for k, m in links.items()}

def place(q, fingers=0.0):
    T = fk(q, fingers); out = {}
    for k, p in lpts.items():
        out[k] = p @ T[k][:3, :3].T + T[k][:3, 3]
    return out

def resid(q, fingers=0.0, trunc=0.01):
    P = np.vstack(list(place(q, fingers).values()))
    d, _ = tree.query(P)
    return np.minimum(d, trunc)

# coarse: base only check
d, i = tree.query(lpts["base_link"])
print("base_link (identity CAD world): median %.2f mm, p90 %.2f mm, nearest CAD group %s"
      % (np.median(d) * 1e3, np.percentile(d, 90) * 1e3, dict(zip(*np.unique(cad_lab[i], return_counts=True)))))
# sequential grid per joint, then least squares
q = np.zeros(6)
grid = np.linspace(-3.14, 3.14, 73)
for k in range(6):
    best = None
    for v in grid:
        qq = q.copy(); qq[k] = v
        c = resid(qq).mean()
        if best is None or c < best[0]: best = (c, v)
    q[k] = best[1]
for trunc in (0.01, 0.003):
    q = least_squares(lambda x: resid(x, 0.0, trunc), q, diff_step=1e-3).x
fing = least_squares(lambda f: resid(q, f[0], 0.003), [0.0], bounds=([-0.01], [0.06])).x[0]
q = least_squares(lambda x: resid(x, fing, 0.003), q, diff_step=1e-3).x
print("q_cad (rad):", np.round(q, 5).tolist(), " deg:", np.round(np.degrees(q), 2).tolist(), " finger slide m:", round(fing, 5))
pl = place(q, fing)
rep = {"q_cad": q.tolist(), "finger": fing, "links": {}}
for k, P in pl.items():
    d, i = tree.query(P)
    labs, cnt = np.unique(cad_lab[i[d < 0.002]], return_counts=True)
    share = {str(a): round(int(c) / len(P), 3) for a, c in sorted(zip(labs, cnt), key=lambda x: -x[1])}
    print(f"{k:18s} median {np.median(d)*1e3:6.2f} mm  p90 {np.percentile(d,90)*1e3:6.2f} mm  within2mm->{share}")
    rep["links"][k] = {"median_mm": float(np.median(d) * 1e3), "p90_mm": float(np.percentile(d, 90) * 1e3), "cad_groups": share}
T = fk(q, fing)
rep["T_base_link"] = {k: v.tolist() for k, v in T.items()}
json.dump(rep, open(cad_dir / "register.json", "w"), indent=1)
