// OpenCV のピンホール（K_und、画素中心が整数座標）→ Three.js（WebGL）のカメラ行列。
//
// cam（OpenCV）: x 右、y 下、z 前方。GL: x 右、y 上、-z 前方。p_gl = CV_TO_GL · p_cv（diag(1,-1,-1)）。
// 画素: OpenCV の u はビューポート座標（左端 0、画素 i は [i, i+1]）では u + 0.5。
import * as THREE from "three";

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export function intrinsicsFromK(K: number[]): Intrinsics {
  return { fx: K[0], fy: K[4], cx: K[2], cy: K[5] };
}

/** 幅 W・高さ H のビューポートに対する投影行列。near/far は mm。 */
export function projectionFromK(k: Intrinsics, W: number, H: number, near = 1, far = 100000): THREE.Matrix4 {
  const a = (2 * k.fx) / W;
  const b = (2 * k.fy) / H;
  const c = 1 - (2 * (k.cx + 0.5)) / W;
  const d = (2 * (k.cy + 0.5)) / H - 1;
  const A = -(far + near) / (far - near);
  const B = (-2 * far * near) / (far - near);
  return new THREE.Matrix4().set(a, 0, c, 0, 0, b, d, 0, 0, 0, A, B, 0, 0, -1, 0);
}

/** 画像ビューの拡大・移動（screen = (u + 0.5)·zoom + pan）を内部パラメータに畳み込む。 */
export function viewIntrinsics(k: Intrinsics, zoom: number, panX: number, panY: number): Intrinsics {
  return {
    fx: k.fx * zoom,
    fy: k.fy * zoom,
    cx: (k.cx + 0.5) * zoom + panX - 0.5,
    cy: (k.cy + 0.5) * zoom + panY - 0.5,
  };
}

export const CV_TO_GL = new THREE.Matrix4().set(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1);

/** 行優先 16 要素 → THREE.Matrix4。 */
export function m4(rowMajor: number[]): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(...(rowMajor.slice(0, 16) as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  return m;
}

/** obj 座標の点を GL のカメラ座標へ移す行列（カメラは原点・単位行列に置く）。 */
export function glModelMatrix(T_cam_obj: number[]): THREE.Matrix4 {
  return CV_TO_GL.clone().multiply(m4(T_cam_obj));
}

/** NDC → OpenCV 画素（テスト・ピック用）。 */
export function ndcToPixel(ndc: THREE.Vector3, W: number, H: number): [number, number] {
  return [((ndc.x + 1) / 2) * W - 0.5, ((1 - ndc.y) / 2) * H - 0.5];
}

export function pixelToNdc(u: number, v: number, W: number, H: number): THREE.Vector2 {
  return new THREE.Vector2(((u + 0.5) / W) * 2 - 1, 1 - ((v + 0.5) / H) * 2);
}

/** 原点・単位姿勢の PerspectiveCamera に投影行列を直接与える（updateProjectionMatrix は呼ばないこと）。 */
export function makeCvCamera(): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera();
  cam.matrixAutoUpdate = false;
  cam.matrixWorldAutoUpdate = false;
  cam.matrix.identity();
  cam.matrixWorld.identity();
  cam.matrixWorldInverse.identity();
  return cam;
}

export function setCvProjection(cam: THREE.PerspectiveCamera, k: Intrinsics, W: number, H: number, near = 1, far = 100000): void {
  cam.projectionMatrix.copy(projectionFromK(k, W, H, near, far));
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}
