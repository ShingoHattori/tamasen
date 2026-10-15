"""模様付き OBJ を GPU（MuJoCo EGL）で 6 方向から描いて 1 枚にまとめる。使い方: render_views.py <obj> <tex.png> <out.jpg>"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np
import trimesh

obj, tex, out = sys.argv[1:4]
m = trimesh.load(obj, force="mesh", process=False)
c, ext = m.bounds.mean(0), float(np.linalg.norm(m.extents))
dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 1), (-1, -1, 1)]
cams = ""
for i, d in enumerate(dirs):
    z = np.array(d, float); z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1] if abs(z[2]) < 0.9 else [0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x); p = c + z * 3 * ext
    cams += f'<camera name="c{i}" pos="{p[0]} {p[1]} {p[2]}" xyaxes="{x[0]} {x[1]} {x[2]} {y[0]} {y[1]} {y[2]}" fovy="{2 * np.degrees(np.arctan(0.55 / 3))}"/>'
xml = f"""<mujoco><visual><global offwidth="500" offheight="500"/><headlight ambient="0.75 0.75 0.75" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
<map znear="0.001"/></visual><asset><texture name="t" type="2d" file="{os.path.abspath(tex)}"/><material name="mt" texture="t"/>
<mesh name="o" file="{os.path.abspath(obj)}"/></asset><worldbody><geom type="mesh" mesh="o" material="mt"/>{cams}</worldbody></mujoco>"""
mdl = mujoco.MjModel.from_xml_string(xml); dat = mujoco.MjData(mdl); mujoco.mj_forward(mdl, dat)
r = mujoco.Renderer(mdl, 500, 500)
ims = []
for i in range(len(dirs)):
    r.update_scene(dat, camera=i); im = r.render()[:, :, ::-1].copy()
    im[(im == 0).all(-1)] = 255
    ims.append(im)
cv2.imwrite(out, np.vstack([np.hstack(ims[:4]), np.hstack(ims[4:])]))
