"""Forward kinematics of end_link straight from the SDK's URDF (numpy only, no Pinocchio).

Used by handeye.py to re-evaluate the end_link pose with estimated joint zero offsets. Checked
against the Pinocchio FK stored with each calibration sample (same URDF).
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = (Path(__file__).resolve().parent.parent / "rebot" / "reBotArm_control_py" / "urdf"
        / "reBot-DevArm_fixend_description" / "urdf" / "reBot-DevArm_fixend.urdf")
ARM = [f"joint{i}" for i in range(1, 7)]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a, t):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


class Chain:
    def __init__(self, urdf: Path = URDF, tip: str = "end_link") -> None:
        joints = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "T": np.vstack([np.column_stack([_rpy(*rpy), xyz]), [0, 0, 0, 1]]),
                "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]}
        chain, link = [], tip
        while link in joints:
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]

    def __call__(self, q) -> np.ndarray:
        qd = dict(zip(ARM, q))
        T = np.eye(4)
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], qd.get(j["name"], 0.0))
                T = T @ R
        return T
