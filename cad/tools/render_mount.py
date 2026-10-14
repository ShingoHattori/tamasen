"""Preview of the marker mount on the gripper (end_link frame). usage: render_mount.py dm0425_dir mount.stl out.png"""
import sys, json, importlib.util
import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
spec = importlib.util.spec_from_file_location("fl", sys.argv[0].replace("render_mount.py", "fklib.py"))
fl = importlib.util.module_from_spec(spec); spec.loader.exec_module(fl)
cad_dir, mount, png = sys.argv[1:4]
urdf = sys.argv[4]
T = fl.Model(urdf).fk([-np.pi, 0, 0, 0, 0, 0]); Tc = np.eye(4); Tc[2, 3] = 0.003295
Ti = np.linalg.inv(Tc @ T["end_link"])
meshes = []
for e in json.load(open(f"{cad_dir}/index.json")):
    if e["depth"] != 3 or not (e["path"].startswith("Gripper") or e["path"].startswith("Link6")): continue
    if "KA3-12" in e["path"] and e["path"].startswith("Gripper"):
        continue
    m = trimesh.load(f"{cad_dir}/{e['file']}"); m.apply_scale(1e-3); m.apply_transform(Ti); m.apply_scale(1e3)
    col = (0.95, 0.75, 0.1, 0.35) if "Rail_Bracket" in e["path"] else (0.6, 0.6, 0.65, 0.35) if "Metal" in e["path"] else (0.2, 0.2, 0.2, 0.25)
    meshes.append((m, col))
mm = trimesh.load(mount); meshes.append((mm, (0.1, 0.6, 1.0, 0.9)))
fig = plt.figure(figsize=(14, 7))
for k, (el, az) in enumerate([(25, -135), (20, 150)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for m, col in meshes:
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolor=col, edgecolor="none"))
    # marker square
    js = json.load(open(mount.replace(".stl", ".json")))
    c = np.array(js["marker_centre_mm"]); h = js["marker_size_mm"] / 2
    sq = np.array([[c[0] - h, -h, c[2] + 0.05], [c[0] + h, -h, c[2] + 0.05], [c[0] + h, h, c[2] + 0.05], [c[0] - h, h, c[2] + 0.05]])
    ax.add_collection3d(Poly3DCollection([sq], facecolor="k"))
    for v, cl in ((js["marker_axes_in_end_link"]["x"], "r"), (js["marker_axes_in_end_link"]["y"], "g"), (js["marker_axes_in_end_link"]["z"], "b")):
        ax.quiver(*c, *(np.array(v) * 25), color=cl, linewidth=2)
    ax.quiver(0, 0, 0, 30, 0, 0, color="r", linestyle=":"); ax.quiver(0, 0, 0, 0, 30, 0, color="g", linestyle=":"); ax.quiver(0, 0, 0, 0, 0, 30, color="b", linestyle=":")
    ax.set_xlim(-170, -20); ax.set_ylim(-75, 75); ax.set_zlim(-75, 75); ax.set_box_aspect((1, 1, 1))
    ax.view_init(el, az); ax.set_xlabel("end_link x"); ax.set_ylabel("y"); ax.set_zlabel("z")
fig.suptitle("ID 1 marker mount (blue) on 02_Slider_Bracket / 01_Rail_Bracket, end_link frame [mm]; marker axes r/g/b, end_link axes dotted at origin")
plt.tight_layout(); plt.savefig(png, dpi=110)
