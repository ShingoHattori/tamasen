"""Preview of the rod jig with MuJoCo (EGL). usage: render_rod_jig.py outdir   (sim/.venv; reads rod_jig.stl/.json,
fingers_closed_in_jig_frame.stl) -> preview.png: three views of the jig, and the fingers closed in the pockets."""
import sys, json
from pathlib import Path
import numpy as np, mujoco

d = Path(sys.argv[1]).resolve()
js = json.load(open(d / "rod_jig.json"))
m = js["marker"]; c = m["centre_mm"]; h = m["black_square_mm"] / 2000
xml = f"""<mujoco><compiler meshdir="{d}"/>
<visual><global offwidth="900" offheight="700"/><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
<asset><mesh name="jig" file="rod_jig.stl" scale="0.001 0.001 0.001"/>
<mesh name="fing" file="fingers_closed_in_jig_frame.stl" scale="0.001 0.001 0.001"/></asset>
<worldbody>
<geom type="mesh" mesh="jig" rgba="0.35 0.62 0.95 1"/>
<geom name="fing" type="mesh" mesh="fing" rgba="0.15 0.15 0.15 1" group="1"/>
<geom type="box" size="{h} {h} 0.0002" pos="{c[0] / 1000} {c[1] / 1000} {c[2] / 1000 - 0.0002}" rgba="0 0 0 1"/>
<geom type="box" size="{h * 0.45} {h * 0.45} 0.0003" pos="{c[0] / 1000} {c[1] / 1000} {c[2] / 1000 - 0.0003}" rgba="1 1 1 1"/>
</worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 700, 900)
tiles = []
for az, el, fingers, dist in ((-135, -35, False, 0.22), (-60, 35, False, 0.22), (90, -89, False, 0.2), (-135, -30, True, 0.3)):
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0.0 if not fingers else -0.005, 0, 0]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 1 if fingers else 0
    r.update_scene(data, cam, opt); tiles.append(r.render())
img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
np.save(d / "preview.npy", img)   # converted to PNG by the caller (no PIL in sim/.venv)
print(d / "preview.png")
