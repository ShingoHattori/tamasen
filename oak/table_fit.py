"""Block pose constrained to rest on the table, from the block markers' corners.

With the camera <-> base calibration adopted, the table plane is known (base z = 0: the arm base stands on
it). A block lying on the table has one of its six faces down, so its height and tilt are fixed and only
(x, y, yaw) in the base frame are unknown. Those three are fitted to the image corners of every visible
block marker at once (8 measurements per marker), which removes the tilt / depth noise and the IPPE
two-solution ambiguity of a small marker seen nearly head-on (2026-10-14: ID 3, 20 mm at 0.6 m, tilt read
as 12 +- 5 deg, height +- 4 mm, 40 % of frames ambiguous).

Not used when the block is not resting flat (raw tilt > MAX_TILT: held, leaning) or the fit is poor.
A static block is smoothed over time; a move larger than the reset thresholds is followed at once.
"""
import math

import cv2
import numpy as np

import pose

MAX_TILT = math.radians(35)      # raw estimate must be this close to some face-down pose
MAX_RMS_PX = 1.5                 # corner reprojection of the constrained fit
RESET_M, RESET_RAD = 0.008, math.radians(4)    # filter resets on a jump larger than this
ALPHA = 0.25                     # exponential smoothing while static


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _face_down(R_base_obj):
    """Rotation that puts the block axis closest to vertical exactly vertical (same sign), and its yaw."""
    k = int(np.argmax(np.abs(R_base_obj[2, :])))         # block axis k is (anti)parallel to base z
    sgn = 1.0 if R_base_obj[2, k] > 0 else -1.0
    # canonical R_face: block axis k -> sgn * z, the next axis j -> base x (then yawed)
    j, l = (k + 1) % 3, (k + 2) % 3
    R_face = np.zeros((3, 3))
    R_face[:, k] = [0, 0, sgn]
    R_face[:, j] = [1, 0, 0]
    R_face[:, l] = np.cross(R_face[:, k], R_face[:, j])  # right-handed: e_k x e_(k+1) = e_(k+2)
    tilt = math.acos(min(1.0, abs(R_base_obj[2, k])))
    # yaw: rotation about z that best maps R_face onto the raw rotation
    M = R_base_obj @ R_face.T
    yaw = math.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    return k, R_face, yaw, tilt


def fit(obs_specs, T_cam_base, size, K, dist):
    """obs_specs: [(Observation, MarkerSpec)] of visible block markers with registered mounts.

    Returns dict(T_base_obj, T_cam_obj, rms_px, face, tilt_raw_deg, markers) or None (with reason)."""
    T_base_cam = pose.invert(T_cam_base)
    pts_obj, pts_img = [], []
    for o, s in obs_specs:
        pts_obj.append((s.T_obj_marker @ np.c_[pose.marker_object_points(s.size_m), np.ones(4)].T).T[:, :3])
        pts_img.append(o.corners)
    P_obj, P_img = np.vstack(pts_obj), np.vstack(pts_img).astype(float)

    def project(x, y, yaw, R_face, z):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _rz(yaw) @ R_face, [x, y, z]
        Tc = T_cam_base @ T
        uv, _ = cv2.projectPoints(P_obj, cv2.Rodrigues(Tc[:3, :3])[0], Tc[:3, 3], K, dist)
        return uv.reshape(-1, 2), T

    best, reason = None, "初期値がない"
    for o, s in obs_specs:
        for Tm in (o.T_cam_marker, o.T_alt):
            if Tm is None:
                continue
            Tb = T_base_cam @ Tm @ pose.invert(s.T_obj_marker)
            k, R_face, yaw, tilt = _face_down(Tb[:3, :3])
            if tilt > MAX_TILT:
                reason = f"机に置かれていない（傾き {math.degrees(tilt):.0f}°）"
                continue
            z = size[k] / 2
            p = np.array([Tb[0, 3], Tb[1, 3], yaw])
            for _ in range(15):                          # Gauss-Newton, numeric Jacobian
                r0 = (project(*p, R_face, z)[0] - P_img).ravel()
                J = np.zeros((r0.size, 3))
                for i, h in enumerate((1e-5, 1e-5, 1e-5)):
                    dp = p.copy(); dp[i] += h
                    J[:, i] = ((project(*dp, R_face, z)[0] - P_img).ravel() - r0) / h
                step = np.linalg.lstsq(J, -r0, rcond=None)[0]
                p += step
                if np.max(np.abs(step)) < 1e-7:
                    break
            uv, T = project(*p, R_face, z)
            rms = float(np.sqrt(np.mean(np.sum((uv - P_img) ** 2, axis=1))))
            if best is None or rms < best["rms_px"]:
                best = {"T_base_obj": T, "T_cam_obj": T_cam_base @ T, "rms_px": rms, "face": f"{'+-'[0 if R_face[2, k] > 0 else 1]}{'XYZ'[k]} 上",
                        "tilt_raw_deg": math.degrees(tilt), "markers": [ob.id for ob, _ in obs_specs], "p": p.copy()}
    if best is None:
        return None, reason
    if best["rms_px"] > MAX_RMS_PX:
        return None, f"当てはめの誤差が大きい（{best['rms_px']:.2f} px）"
    return best, None


class Smoother:
    """Exponential smoothing of (x, y, yaw) while the block stays put; resets on a real move."""

    def __init__(self) -> None:
        self.p = None
        self.face = None
        self.n = 0

    def update(self, res: dict) -> tuple[np.ndarray, int]:
        p = res["p"]
        if (self.p is None or res["face"] != self.face or np.hypot(*(p[:2] - self.p[:2])) > RESET_M
                or abs(math.remainder(p[2] - self.p[2], 2 * math.pi)) > RESET_RAD):
            self.p, self.face, self.n = p.copy(), res["face"], 1
        else:
            dyaw = math.remainder(p[2] - self.p[2], 2 * math.pi)
            self.p[:2] += ALPHA * (p[:2] - self.p[:2])
            self.p[2] += ALPHA * dyaw
            self.n += 1
        T = res["T_base_obj"].copy()
        R_face = _rz(-res["p"][2]) @ T[:3, :3]
        T[:3, :3] = _rz(self.p[2]) @ R_face
        T[0, 3], T[1, 3] = self.p[0], self.p[1]
        return T, self.n
