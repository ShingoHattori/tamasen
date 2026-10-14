"""Arm (sim scene meshes) with marker mount 2 on link5, gripper-down pose; close-ups of the wrist + camera view."""
import sys, json, math, urllib.request
import numpy as np, mujoco
import scene
OUT = sys.argv[1]; png = sys.argv[2]
s = scene.load_settings()
spec = mujoco.MjSpec.from_string(scene.build_xml(s))
m0 = spec.compile(); d0 = mujoco.MjData(m0); mujoco.mj_kinematics(m0, d0)
ids0 = scene.Ids(m0); b5 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "link5")
T = lambda b: np.vstack([np.c_[d0.xmat[b].reshape(3, 3), d0.xpos[b]], [0, 0, 0, 1]])
T_l5_end = np.linalg.inv(T(b5)) @ T(ids0.end_body)
info = json.load(open(f"{OUT}/marker_mount_l5.json"))
bd = spec.body("link5")
q = np.zeros(4); mujoco.mju_mat2Quat(q, T_l5_end[:3, :3].ravel())
for part, rgba in (("A_base", [0.95, 0.55, 0.15, 1]), ("B_markers", [0.25, 0.6, 1.0, 1])):
    spec.add_mesh(name=part, file=f"{OUT}/marker_mount_l5_{part}.stl", scale=[1e-3] * 3)
    bd.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=part, pos=T_l5_end[:3, 3], quat=q, rgba=rgba, group=2, contype=0, conaffinity=0)
for mk in info["markers"]:
    P = T_l5_end @ np.array(mk["T_wrist_marker"]) @ np.diag([1, 1, 1, 1]); P = P.copy()
    Tm = np.array(mk["T_wrist_marker"]); Tm[:3, 3] /= 1000; P = T_l5_end @ Tm
    qm = np.zeros(4); mujoco.mju_mat2Quat(qm, P[:3, :3].ravel())
    bd.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.0004], pos=P[:3, 3] + P[:3, 2] * 0.0005, quat=qm, rgba=[0.05, 0.05, 0.05, 1], group=2, contype=0, conaffinity=0)
    bd.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.006, 0.0004], pos=P[:3, 3] + P[:3, 2] * 0.0008 + P[:3, 0] * 0.004, quat=qm, rgba=[1, 1, 1, 1], group=2, contype=0, conaffinity=0)
he = json.load(open("../oak/calib/handeye_active.json")); Xi = np.linalg.inv(np.array(he["T_cam_base"]))
K = np.array(json.load(urllib.request.urlopen("http://127.0.0.1:8090/state.json"))["intrinsics"]["K"])
cam_rot = Xi[:3, :3] @ np.diag([1, -1, -1]); qc = np.zeros(4); mujoco.mju_mat2Quat(qc, cam_rot.ravel())
spec.worldbody.add_camera(name="oak", pos=Xi[:3, 3], quat=qc, fovy=2 * math.degrees(math.atan(480 / K[1, 1])))
spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
m = spec.compile(); d = mujoco.MjData(m); ids = scene.Ids(m)
qpose = np.radians([-21, -94, -70, 66, 0, -21])
d.qpos[ids.arm_qadr] = qpose; mujoco.mj_forward(m, d)
b5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link5")
c = d.xpos[b5] + d.xmat[b5].reshape(3, 3) @ (T_l5_end[:3, :3] @ np.array([-0.225, 0, 0.015]) + T_l5_end[:3, 3])
opt = mujoco.MjvOption(); opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
r = mujoco.Renderer(m, 480, 640); tiles = []
for az, el, dist in ((-160, -25, 0.32), (-100, -35, 0.32), (20, -30, 0.35), (-130, -60, 0.6)):
    cam = mujoco.MjvCamera(); cam.lookat[:] = c; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, cam, opt); tiles.append(r.render())
r2 = mujoco.Renderer(m, 960, 1280); r2.update_scene(d, camera="oak", scene_option=opt); camimg = r2.render()
# crop the camera view around the wrist
P = np.array(he["T_cam_base"]) @ np.r_[c, 1]; u = K[0, 0] * P[0] / P[2] + K[0, 2]; v = K[1, 1] * P[1] / P[2] + K[1, 2]
u0, v0 = int(np.clip(u - 320, 0, 640)), int(np.clip(v - 240, 0, 480)); crop = camimg[v0:v0 + 480, u0:u0 + 640]
img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:3] + [crop])])
np.save(png + ".npy", img); print("ok", u, v)
