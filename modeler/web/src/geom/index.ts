// 形状生成の公開 API（UI はここだけを使う）。長さは mm、Mat4 は行優先（DESIGN.md §1）。
import * as THREE from "three";
import type { Mat4, Part, Vec3 } from "../types";
import { loadManifold } from "./manifold";
import { manifoldStats, manifoldToGeometry, type ModelStats } from "./mesh";
import { assemble, freeAssemble } from "./parts";

export type { ModelStats } from "./mesh";
export type { PlaneDef } from "./transform";
export type { MeasurementCheck } from "./constraints";
export type { ExportFrame } from "./export";
export {
  mat4Identity,
  mat4Mul,
  mat4Compose,
  mat4InvRigid,
  mat4Invert,
  mat4Apply,
  mat4ApplyDir,
  mat4FromTransform,
  transformFromMat4,
  mat4ToThree,
  mat4FromThree,
  mat4FromRt,
  projectPoint,
  backProjectToPlane,
  planeBasis,
  planeToMat4,
  planePoint,
} from "./transform";
export { isLocked, applyParamEdit, checkMeasurements, getParam } from "./constraints";
export { defaultPart, sheetGeometry } from "./parts";
export { exportGLB } from "./export";
export { validatePolygon, signedArea } from "./polygon";

/** WASM を一度だけ読み込む。buildModel の前に await すること。 */
export async function initGeometry(): Promise<void> {
  await loadManifold();
}

export interface PartGeometry {
  id: string;
  op: "add" | "subtract";
  geometry: THREE.BufferGeometry | null; // その部品だけ（obj 座標・mm、transform 適用済み）
  error?: string;
}

export interface BuildResult {
  geometry: THREE.BufferGeometry | null; // 合成結果（obj 座標・mm）
  parts: PartGeometry[];
  stats: ModelStats | null;
  errors: { partId: string; message: string }[]; // partId "" は合成全体のエラー
}

/** 部品から合成モデルと部品ごとの形を作る（同期。initGeometry の後）。geometry の dispose は呼び出し側。 */
export function buildModel(parts: Part[]): BuildResult {
  const r = assemble(parts);
  try {
    return {
      geometry: r.model ? manifoldToGeometry(r.model) : null,
      parts: r.parts.map((p) => ({
        id: p.id,
        op: p.op,
        geometry: p.manifold ? manifoldToGeometry(p.manifold) : null,
        ...(p.error !== undefined ? { error: p.error } : {}),
      })),
      stats: r.model ? manifoldStats(r.model) : null,
      errors: r.errors,
    };
  } finally {
    freeAssemble(r);
  }
}

export type { Vec3, Mat4 };
