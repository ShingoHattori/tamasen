// 4×4 同次変換（DESIGN.md §1）。Mat4 は **行優先** の 16 要素: m[r*4 + c]。
// T_a_b は b 座標の点を a 座標へ移す（p_a = T_a_b · p_b）。長さは mm、角度は度。
// THREE.Matrix4 の .elements は **列優先** なので、変換は mat4ToThree / mat4FromThree だけを使うこと。
import * as THREE from "three";
import type { Mat4, Transform, Vec2, Vec3 } from "../types";

const DEG = Math.PI / 180;

export function mat4Identity(): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** a·b（行優先）。T_a_c = mat4Mul(T_a_b, T_b_c)。 */
export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const r = new Array<number>(16).fill(0);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
      r[i * 4 + j] = s;
    }
  return r;
}

/** 左から順に掛ける: mat4Compose(A, B, C) = A·B·C。 */
export function mat4Compose(...ms: Mat4[]): Mat4 {
  return ms.reduce((acc, m) => mat4Mul(acc, m), mat4Identity());
}

/** 剛体変換（回転＋並進）の逆: [Rᵀ, −Rᵀt]。 */
export function mat4InvRigid(m: Mat4): Mat4 {
  const r = mat4Identity();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 4 + j] = m[j * 4 + i];
  for (let i = 0; i < 3; i++) r[i * 4 + 3] = -(r[i * 4] * m[3] + r[i * 4 + 1] * m[7] + r[i * 4 + 2] * m[11]);
  return r;
}

/** 一般の 4×4 逆行列（剛体でない可能性があるときに使う）。特異なら例外。 */
export function mat4Invert(m: Mat4): Mat4 {
  const e = new THREE.Matrix4();
  e.set(...(m as Parameters<THREE.Matrix4["set"]>));
  if (Math.abs(e.determinant()) < 1e-15) throw new Error("変換行列が特異で逆行列を作れません");
  return mat4FromThree(e.invert());
}

/** 点（同次座標 w=1）に作用させる。 */
export function mat4Apply(m: Mat4, p: Vec3): Vec3 {
  return [
    m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
    m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
    m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
  ];
}

/** 方向ベクトル（並進なし）に作用させる。 */
export function mat4ApplyDir(m: Mat4, d: Vec3): Vec3 {
  return [
    m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
    m[4] * d[0] + m[5] * d[1] + m[6] * d[2],
    m[8] * d[0] + m[9] * d[1] + m[10] * d[2],
  ];
}

/** Trans(t)·Rz(rz)·Ry(ry)·Rx(rx)。回転は度（固定軸 X→Y→Z の順に回す）。 */
export function mat4FromRt(rvecDeg: Vec3, t_mm: Vec3): Mat4 {
  const [cx, sx] = [Math.cos(rvecDeg[0] * DEG), Math.sin(rvecDeg[0] * DEG)];
  const [cy, sy] = [Math.cos(rvecDeg[1] * DEG), Math.sin(rvecDeg[1] * DEG)];
  const [cz, sz] = [Math.cos(rvecDeg[2] * DEG), Math.sin(rvecDeg[2] * DEG)];
  return [
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, t_mm[0],
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, t_mm[1],
    -sy, cy * sx, cy * cx, t_mm[2],
    0, 0, 0, 1,
  ];
}

export function mat4FromTransform(t: Transform): Mat4 {
  return mat4FromRt(t.r_deg, t.t_mm);
}

/** mat4FromTransform の逆分解。ry = ±90°（ジンバルロック）のときは rx = 0 とする。 */
export function transformFromMat4(m: Mat4): Transform {
  const r20 = Math.max(-1, Math.min(1, m[8]));
  const ry = Math.asin(-r20);
  let rx: number;
  let rz: number;
  if (Math.abs(r20) < 1 - 1e-12) {
    rx = Math.atan2(m[9], m[10]);
    rz = Math.atan2(m[4], m[0]);
  } else {
    rx = 0;
    rz = Math.atan2(-m[1], m[5]);
  }
  return { t_mm: [m[3], m[7], m[11]], r_deg: [rx / DEG, ry / DEG, rz / DEG] };
}

/** 行優先 Mat4 → THREE.Matrix4（Matrix4.set は行優先で受け取り、内部 .elements は列優先）。 */
export function mat4ToThree(m: Mat4): THREE.Matrix4 {
  return new THREE.Matrix4().set(...(m as Parameters<THREE.Matrix4["set"]>));
}

/** THREE.Matrix4（.elements 列優先）→ 行優先 Mat4。 */
export function mat4FromThree(m: THREE.Matrix4): Mat4 {
  const e = m.elements;
  const r = new Array<number>(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) r[i * 4 + j] = e[j * 4 + i];
  return r;
}

/** manifold-3d の Mat4（列優先）へ。 */
export function mat4ToColumnMajor(m: Mat4): number[] {
  const r = new Array<number>(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) r[j * 4 + i] = m[i * 4 + j];
  return r;
}

// ---- 投影・逆投影 -------------------------------------------------------------

/** obj の点 → 歪み補正画像の画素。カメラの後ろ（z ≤ 0）なら null。K_und は 3×3 行優先。 */
export function projectPoint(K_und: number[], T_cam_marker: Mat4, T_marker_obj: Mat4, p_obj: Vec3): Vec2 | null {
  const pc = mat4Apply(mat4Mul(T_cam_marker, T_marker_obj), p_obj);
  if (!(pc[2] > 1e-9)) return null;
  const x = pc[0] / pc[2];
  const y = pc[1] / pc[2];
  return [K_und[0] * x + K_und[1] * y + K_und[2], K_und[3] * x + K_und[4] * y + K_und[5]];
}

/** obj 座標の平面。origin・法線・面内 x 軸（法線と直交しなくてよい。直交化して使う）。 */
export interface PlaneDef {
  origin: Vec3;
  normal: Vec3;
  xAxis: Vec3;
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]);
  if (!(l > 1e-12)) throw new Error("平面の向きベクトルの長さが 0 です");
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** 平面の正規直交基底（x, y = n × x, n）。 */
export function planeBasis(plane: PlaneDef): { x: Vec3; y: Vec3; n: Vec3 } {
  const n = norm(plane.normal);
  const xd = dot(plane.xAxis, n);
  const x = norm(sub(plane.xAxis, [n[0] * xd, n[1] * xd, n[2] * xd]));
  return { x, y: cross(n, x), n };
}

/** T_obj_plane: 平面座標（x, y, 法線 = z）→ obj。extrude 部品をこの平面に置く変換にそのまま使える。 */
export function planeToMat4(plane: PlaneDef): Mat4 {
  const { x, y, n } = planeBasis(plane);
  const o = plane.origin;
  return [x[0], y[0], n[0], o[0], x[1], y[1], n[1], o[1], x[2], y[2], n[2], o[2], 0, 0, 0, 1];
}

/** 平面座標 (u, v) → obj の点。 */
export function planePoint(plane: PlaneDef, uv: Vec2): Vec3 {
  return mat4Apply(planeToMat4(plane), [uv[0], uv[1], 0]);
}

/**
 * 画素（歪み補正画像）→ その画像の姿勢で作った光線と、obj 座標の平面との交点。
 * 正投影とはみなさない。光線が平面と平行、または交点がカメラの後ろなら null。
 */
export function backProjectToPlane(
  uv_und: Vec2,
  K_und: number[],
  T_cam_marker: Mat4,
  T_marker_obj: Mat4,
  plane: PlaneDef,
): { p_obj: Vec3; uv_plane: Vec2 } | null {
  const T_obj_cam = mat4Invert(mat4Mul(T_cam_marker, T_marker_obj));
  // K⁻¹ [u, v, 1]（K は上三角: fx, s, cx / 0, fy, cy / 0, 0, 1 を仮定せず一般に解く）
  const K: Mat4 = [K_und[0], K_und[1], K_und[2], 0, K_und[3], K_und[4], K_und[5], 0, K_und[6], K_und[7], K_und[8], 0, 0, 0, 0, 1];
  const dCam = mat4Apply(mat4Invert(K), [uv_und[0], uv_und[1], 1]);
  const o = mat4Apply(T_obj_cam, [0, 0, 0]);
  const d = mat4ApplyDir(T_obj_cam, dCam);
  const { x, y, n } = planeBasis(plane);
  const nd = dot(n, d);
  if (Math.abs(nd) < 1e-12 * Math.hypot(d[0], d[1], d[2])) return null;
  const s = dot(n, sub(plane.origin, o)) / nd;
  if (!(s > 0)) return null;
  const p: Vec3 = [o[0] + s * d[0], o[1] + s * d[1], o[2] + s * d[2]];
  const r = sub(p, plane.origin);
  return { p_obj: p, uv_plane: [dot(r, x), dot(r, y)] };
}
