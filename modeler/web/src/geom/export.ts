// 合成モデルを GLB（バイナリ glTF）に書き出す（DESIGN.md §1・§5）。
// 単位は m（mm × 0.001）。座標は obj または marker（T_marker_obj を適用）。glTF の y 上への軸変換はしない。
// 軸の定義は asset.extras に書く。node（vitest）でも動くよう FileReader がなければ最小限の代用を入れる。
import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import type { Mat4, Part } from "../types";
import { manifoldToGeometry } from "./mesh";
import { assemble, freeAssemble } from "./parts";
import { mat4Identity } from "./transform";

export type ExportFrame = "marker" | "obj";

export const AXES_DOC: Record<ExportFrame, string> = {
  marker:
    "marker: origin = center of the black square, x right, y up, z out of the marker plane (OpenCV ArUco corner order TL,TR,BR,BL). No glTF y-up conversion.",
  obj: "obj: object model frame of the modeler project (parts defined here; p_marker = T_marker_obj * p_obj). No glTF y-up conversion.",
};

/** node には FileReader がないので、GLTFExporter が使う readAsArrayBuffer / readAsDataURL だけ代用する。 */
function ensureFileReader(): void {
  const g = globalThis as unknown as { FileReader?: unknown };
  if (typeof g.FileReader !== "undefined") return;
  class MiniFileReader {
    result: ArrayBuffer | string | null = null;
    onloadend: (() => void) | null = null;
    readAsArrayBuffer(blob: Blob) {
      void blob.arrayBuffer().then((b) => {
        this.result = b;
        this.onloadend?.();
      });
    }
    readAsDataURL(blob: Blob) {
      void blob.arrayBuffer().then((b) => {
        let bin = "";
        const u = new Uint8Array(b);
        for (let i = 0; i < u.length; i++) bin += String.fromCharCode(u[i]);
        this.result = `data:${blob.type || "application/octet-stream"};base64,${btoa(bin)}`;
        this.onloadend?.();
      });
    }
  }
  g.FileReader = MiniFileReader;
}

/**
 * 部品から合成したモデルを GLB にする。合成できなければ（部品エラーで空など）日本語の例外。
 * frame = 'marker' なら p_marker = T_marker_obj · p_obj を書き出す。
 */
export async function exportGLB(parts: Part[], T_marker_obj: Mat4, frame: ExportFrame = "marker"): Promise<ArrayBuffer> {
  const r = assemble(parts);
  let geom: THREE.BufferGeometry;
  try {
    if (!r.model) {
      const msg = r.errors.map((e) => (e.partId ? `${e.partId}: ${e.message}` : e.message)).join(" / ");
      throw new Error(`書き出す形がありません${msg ? `（${msg}）` : ""}`);
    }
    geom = manifoldToGeometry(r.model, { T: frame === "marker" ? T_marker_obj : mat4Identity(), scale: 0.001 });
  } finally {
    freeAssemble(r);
  }
  return geometryToGLB(geom, frame);
}

/** BufferGeometry（すでに m・出力座標）→ GLB。 */
export async function geometryToGLB(geom: THREE.BufferGeometry, frame: ExportFrame): Promise<ArrayBuffer> {
  ensureFileReader();
  const scene = new THREE.Scene();
  const mat = new THREE.MeshStandardMaterial({ color: 0xb0b0b0, metalness: 0, roughness: 0.8 });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.name = "model";
  scene.add(mesh);
  const exporter = new GLTFExporter();
  const extras = { units: "m", frame, axes: AXES_DOC[frame], generator: "tamasen modeler" };
  exporter.register((writer) => ({
    afterParse() {
      const json = (writer as unknown as { json: { asset: Record<string, unknown> } }).json;
      json.asset.extras = extras;
    },
  }));
  try {
    const out = await new Promise<ArrayBuffer>((resolve, reject) => {
      exporter.parse(
        scene,
        (res) => {
          if (res instanceof ArrayBuffer) resolve(res);
          else reject(new Error("GLB ではなく JSON が返りました"));
        },
        (err) => reject(err instanceof Error ? err : new Error(String(err))),
        { binary: true },
      );
    });
    return out;
  } finally {
    geom.dispose();
    mat.dispose();
  }
}
