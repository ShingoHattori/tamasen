"""箱型メッシュの「向かい合う面の距離」を測る（面の法線で点を分け、中央値どうしの差）。使い方: measure_box.py <mesh.stl>"""
import sys

import numpy as np
import trimesh

m = trimesh.load(sys.argv[1])
o = m.bounding_box_oriented
p, fi = trimesh.sample.sample_surface(m, 300000)
n = m.face_normals[fi]
R = o.primitive.transform[:3, :3]
q = (p - o.primitive.transform[:3, 3]) @ R
nn = n @ R
d = []
for k in range(3):
    pos, neg = q[nn[:, k] > 0.9, k], q[nn[:, k] < -0.9, k]
    d.append(round(float(np.median(pos) - np.median(neg)) * 1000, 1))
print("face_to_face_mm", sorted(d), "obb_mm", np.round(np.sort(o.primitive.extents * 1000), 1).tolist(), "watertight", m.is_watertight)
