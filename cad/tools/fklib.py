import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
ARM = [f"joint{i}" for i in range(1, 7)]
def H(R, t): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M
class Model:
    def __init__(self, urdf):
        urdf = Path(urdf); root = ET.parse(urdf).getroot(); self.joints = []
        for j in root.findall("joint"):
            o = j.find("origin"); a = j.find("axis")
            self.joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                T=H(Rot.from_euler("xyz", list(map(float, o.get("rpy").split()))).as_matrix(), np.array(list(map(float, o.get("xyz").split())))),
                axis=np.array(list(map(float, a.get("xyz").split()))) if a is not None else None))
        self.links = {}
        for l in root.findall("link"):
            fs = [m.get("filename") for m in l.findall("visual/geometry/mesh")]
            if fs: self.links[l.get("name")] = trimesh.util.concatenate([trimesh.load(urdf.parent / f if (urdf.parent / f).exists() else urdf.parent.parent / f) for f in fs])
    def fk(self, q, fingers=0.0):
        T = {"base_link": np.eye(4)}
        for j in self.joints:
            M = T[j["parent"]] @ j["T"]
            if j["type"] == "revolute": M = M @ H(Rot.from_rotvec(j["axis"] * q[ARM.index(j["name"])]).as_matrix(), np.zeros(3))
            elif j["type"] == "prismatic": M = M @ H(np.eye(3), j["axis"] * (fingers if j["name"] == "finger_left" else -fingers))
            T[j["child"]] = M
        return T
