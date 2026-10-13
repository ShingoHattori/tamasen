"""Build the MuJoCo scene: the public B601-DM model + floor + the 67 x 78 x 31 mm block.

The upstream MJCF (Seeed-Projects/Borot-Arm_Mujoco, vendored unmodified in models/borot/) is edited
in memory only:

  * its demo table, cubes and IK target are removed; its floor becomes collidable (z = 0 = the
    plane the arm base stands on)
  * geometry groups, so visual and physical shapes can be shown separately:
        group 2  visual meshes (never collide)
        group 3  jaws / palm collision meshes from the model — the jaws that do the physical grasp
        group 4  arm-link collision, added here as the convex hulls of each link's visual parts
                 (the upstream model has no link collision at all)
  * finger_right is tied to finger_left (finger_right = -finger_left), as the upstream code does
  * position actuators for joint1..6 and the gripper, used only inside the simulator
  * the block: free body, box of the measured size from oak/markers.toml
  * a "grip" site midway along the finger pads, the point the planner puts on the block centre

Contact bits: floor / block / jaws (contype 1, conaffinity 1); link hulls (contype 2, conaffinity 3),
so hulls touch the floor, the block and each other (self-collision) but MuJoCo still skips
parent-child pairs.
"""
import math
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
LINK_BODIES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
EXCLUDED_SELF_PAIRS = [("base_link", "link1"), ("link4", "link6")]
REMOVE_BODIES = ["task_table", "red_cube", "blue_block", "yellow_cylinder", "ik_target"]


@dataclass
class Param:
    value: object
    status: str

    def json(self):
        v = self.value.tolist() if isinstance(self.value, np.ndarray) else self.value
        return {"value": v, "status": self.status}


def load_settings(path: Path = HERE / "scene.toml") -> dict:
    s = tomllib.loads(path.read_text())
    oak = tomllib.loads((ROOT / "oak" / "markers.toml").read_text())
    s["block"]["size_m"] = (np.array(oak["block"]["size_mm"], float) * 1e-3).tolist()
    s["block"]["size_status"] = oak["block"].get("size_status", "unset")
    return s


def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def usable_obstacles(s: dict) -> list[dict]:
    """Obstacles that enter the scene, each with the margin it is inflated by."""
    out = []
    for ob in s.get("obstacle", []):
        if ob.get("status") not in ("sample", "measured", "calibrated"):
            continue
        if any(_bad(v) for v in list(ob["position_m"]) + list(ob["size_m"])):
            continue
        inflate = ob.get("margin_m", 0.0) + (ob.get("uncertainty_m", 0.0) if ob["status"] == "sample" else 0.0)
        out.append(dict(ob, inflate_m=inflate))
    return out


def obstacle_report(s: dict) -> list[dict]:
    rows = []
    used = {o["name"]: o for o in usable_obstacles(s)}
    for ob in s.get("obstacle", []):
        u = used.get(ob["name"])
        rows.append({"name": ob["name"], "note": ob.get("note", ""), "status": ob.get("status", "unset"),
                     "in_scene": u is not None, "inflate_m": None if u is None else u["inflate_m"],
                     "label": ("未登録（経路チェックに入っていない）" if u is None else
                               f"登録済み（{ob['status']}、各方向 +{u['inflate_m'] * 1000:.0f} mm で確認）"
                               + ("。概略値で校正済みではない" if ob["status"] == "sample" else ""))})
    return rows


def physics_params(s: dict) -> dict[str, Param]:
    b, j, a = s["block"], s["jaw"], s["actuators"]
    return {
        "ブロック寸法 m": Param(b["size_m"], b["size_status"]),
        "ブロック質量 kg": Param(b["mass_kg"], b["mass_status"]),
        "ブロック摩擦": Param(b["friction"], b["friction_status"]),
        "顎の摩擦（接触特性）": Param(j["friction"], j["friction_status"]),
        "顎の押し付け力上限 N": Param(j["force_limit_n"], j["force_limit_status"]),
        "顎の最大開口 m": Param(j["max_gap_m"], j["max_gap_status"]),
        "グリッパー角→開口の換算": Param(f"gap = motor/{j['gripper_rad_full_open']} * max_gap", j["gripper_map_status"]),
        "関節の位置制御ゲイン（sim 内）": Param({"kp": a["kp"], "kv": a["kv"], "force": a["force_limit_nm"]}, a["status"]),
    }


def build_xml(s: dict) -> str:
    src = HERE / s["model"]["mjcf"]
    tree = ET.parse(src)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str((src.parent / "meshes").resolve()))
    wb = root.find("worldbody")
    for b in list(wb):
        if b.tag == "body" and b.get("name") in REMOVE_BODIES:
            wb.remove(b)
    for c in list(wb.findall("camera")):
        wb.remove(c)
    floor = next(g for g in wb.findall("geom") if g.get("name") == "floor")
    floor.attrib.update(contype="1", conaffinity="1", condim="3", friction="0.8 0.02 0.002", group="0")
    asset = root.find("asset")
    # Upstream applies colours from its URDF at runtime (ROS code); give the materials plain colours.
    colours = {"anodized_grey": "0.55 0.57 0.60 1", "hardware_black": "0.12 0.12 0.13 1", "matte_black": "0.18 0.18 0.19 1",
               "seeed_yellow": "0.72 0.80 0.12 1", "silver_trim": "0.78 0.78 0.80 1", "gripper_finger_black": "0.15 0.15 0.16 1",
               "gripper_carriage_grey": "0.45 0.47 0.48 1", "gripper_rack_metal": "0.70 0.70 0.72 1",
               "gripper_seeed_yellow": "0.72 0.80 0.12 1", "gripper_hardware_black": "0.12 0.12 0.13 1",
               "gripper_base_metal": "0.70 0.70 0.72 1"}
    for mat in asset.findall("material"):
        key = mat.get("name", "").removesuffix("_mat")
        if key in colours and "rgba" not in mat.attrib:
            mat.set("rgba", colours[key])

    # group the existing geoms
    for g in root.iter("geom"):
        n = g.get("name", "")
        if n.endswith("_visual"):
            g.set("group", "2")
        elif n.endswith("_collision"):
            g.set("group", "3")
            g.set("rgba", "0.95 0.45 0.10 0.55")
    # link collision = convex hulls of each link's visual parts
    for body in root.iter("body"):
        if body.get("name") not in LINK_BODIES:
            continue
        for g in list(body.findall("geom")):
            if g.get("name", "").endswith("_visual"):
                h = ET.SubElement(body, "geom")
                h.attrib.update(name=g.get("name").replace("_visual", "_hull"), type="mesh", mesh=g.get("mesh"),
                                group="4", contype="2", conaffinity="3", condim="3", rgba="0.2 0.6 1.0 0.35",
                                mass="0")
    # grip site midway along the pads (end_link x: pads span about -0.065..0, fingertips at 0)
    end = next(b for b in root.iter("body") if b.get("name") == "end_link")
    ET.SubElement(end, "site", name="grip", pos="-0.025 0 0", size="0.006", rgba="0 1 0 1", group="1")

    # static obstacles (group 5): registered boxes only; unset ones are listed, never guessed
    for ob in usable_obstacles(s):
        half = np.array(ob["size_m"]) / 2 + ob["inflate_m"]
        b = ET.SubElement(wb, "body", name=f"obst_{ob['name']}", pos=" ".join(map(str, ob["position_m"])),
                          euler=f"0 0 {ob.get('yaw_rad', 0.0)}")
        ET.SubElement(b, "geom", name=f"obst_{ob['name']}", type="box", size=" ".join(f"{v:.4f}" for v in half),
                      contype="1", conaffinity="1", group="5",
                      rgba="0.9 0.2 0.2 0.35" if ob["status"] == "sample" else "0.9 0.5 0.2 0.35")

    # block
    bs = np.array(s["block"]["size_m"]) / 2
    bd = ET.SubElement(wb, "body", name="block", pos="0.3 0 0.0155")
    ET.SubElement(bd, "freejoint", name="block_free")
    ET.SubElement(bd, "geom", name="block_geom", type="box", size=" ".join(f"{v:.5f}" for v in bs),
                  mass=str(s["block"]["mass_kg"]), friction=" ".join(map(str, s["block"]["friction"])),
                  contype="1", conaffinity="1", condim="4", solref="0.002 1", solimp="0.98 0.995 0.0005",
                  rgba="0.95 0.85 0.15 1", group="0")
    ET.SubElement(bd, "site", name="block_centre", size="0.004", rgba="1 0 0 1", group="1")

    # jaw friction from settings
    for g in root.iter("geom"):
        if g.get("name", "").startswith("finger_") and g.get("name", "").endswith("_collision"):
            g.set("friction", " ".join(map(str, s["jaw"]["friction"])))

    # Hull pairs that touch in (almost) every configuration because the hulls are coarser than the
    # parts: base_link-link1 in 100 % of 400 random poses (base_link is welded to the world, so the
    # parent-child filter does not apply), link4-link6 across the wrist in 63 %. Excluded, and
    # reported as "not checked" (EXCLUDED_SELF_PAIRS).
    con = root.find("contact")
    for b1, b2 in EXCLUDED_SELF_PAIRS:
        ET.SubElement(con, "exclude", body1=b1, body2=b2)
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", name="finger_coupling", joint1="finger_right", joint2="finger_left",
                  polycoef="0 -1 0 0 0")
    act = ET.SubElement(root, "actuator")
    a = s["actuators"]
    for i, jn in enumerate(ARM_JOINTS):
        f = a["force_limit_nm"][i]
        ET.SubElement(act, "position", name=f"act_{jn}", joint=jn, kp=str(a["kp"][i]), kv=str(a["kv"][i]),
                      forcerange=f"{-f} {f}", ctrlrange="-3.2 3.2")
    fl = s["jaw"]["force_limit_n"]
    ET.SubElement(act, "position", name="act_gripper", joint="finger_left", kp="400", kv="10",
                  forcerange=f"{-fl} {fl}", ctrlrange="0 0.05")
    vis = root.find("visual")
    g = vis.find("global")
    if g is None:
        g = ET.SubElement(vis, "global")
    g.attrib.update(offwidth="1280", offheight="960")
    sc = vis.find("scale")
    if sc is None:
        sc = ET.SubElement(vis, "scale")
    sc.attrib.update(contactwidth="0.008", contactheight="0.002", forcewidth="0.004")
    return ET.tostring(root, encoding="unicode")


def load_model(s: dict | None = None) -> tuple[mujoco.MjModel, dict]:
    s = s or load_settings()
    m = mujoco.MjModel.from_xml_string(build_xml(s))
    return m, s


class Ids:
    """Name → id lookups used everywhere."""

    def __init__(self, m: mujoco.MjModel) -> None:
        j = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_qadr = [m.jnt_qposadr[j(n)] for n in ARM_JOINTS]
        self.arm_dadr = [m.jnt_dofadr[j(n)] for n in ARM_JOINTS]
        self.arm_range = np.array([m.jnt_range[j(n)] for n in ARM_JOINTS])
        self.fl_q = m.jnt_qposadr[j("finger_left")]
        self.fr_q = m.jnt_qposadr[j("finger_right")]
        self.fl_range = m.jnt_range[j("finger_left")].copy()
        self.block_q = m.jnt_qposadr[j("block_free")]
        self.block_d = m.jnt_dofadr[j("block_free")]
        self.block_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.block_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.grip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self.end_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "end_link")
        self.act_arm = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in ARM_JOINTS]
        self.act_grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper")
        bodies = lambda prefix: {i for i in range(m.nbody) if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i).startswith(prefix)}
        self.finger_left = bodies("finger_left")
        self.finger_right = bodies("finger_right")
        self.obstacle_geoms = [g for g in range(m.ngeom)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("obst_")]
        self.arm_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINK_BODIES + ["end_link"]} \
            | self.finger_left | self.finger_right


def gap_from_finger(q_left: float, m: mujoco.MjModel, d: mujoco.MjData, ids: Ids) -> float:
    """Inner gap between the jaw pads at a given finger_left position (both fingers symmetric)."""
    return _gap_model(m)(q_left)


_gap_cache: dict = {}


def _gap_model(m: mujoco.MjModel):
    """gap(q) = gap0 + 2 q, with gap0 measured once from the pad collision meshes at q = 0."""
    if id(m) in _gap_cache:
        return _gap_cache[id(m)]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ids = Ids(m)
    R = d.xmat[ids.end_body].reshape(3, 3)
    p = d.xpos[ids.end_body]
    ys = {"left": [], "right": []}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.startswith("finger_") and any(k in n for k in ("_front_", "_mid_")) and n.endswith("_collision"):
            mid = m.geom_dataid[g]
            vadr, vnum = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[vadr:vadr + vnum]
            Wg = (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]
            local = (R.T @ (Wg - p).T).T
            ys["left" if "left" in n else "right"].append(local[:, 1])
    left_inner = min(np.min(v) for v in ys["left"]) if np.median(np.concatenate(ys["left"])) > 0 else max(np.max(v) for v in ys["left"])
    right_inner = max(np.max(v) for v in ys["right"]) if np.median(np.concatenate(ys["right"])) < 0 else min(np.min(v) for v in ys["right"])
    gap0 = float(abs(left_inner - right_inner))
    f = lambda q: gap0 + 2.0 * q
    f.gap0 = gap0
    _gap_cache[id(m)] = f
    return f
