// 組み立てた形状のキャッシュ（部品 JSON が変わったときだけ作り直す）。
import { buildModel, initGeometry } from "./geo";
import { store } from "./store";

export type ModelResult = ReturnType<typeof buildModel>;

let ready = false;
let initError: string | null = null;
let key = "";
let cur: ModelResult | null = null;
let version = 0;

export function startGeometry(): void {
  initGeometry().then(
    () => {
      ready = true;
      store.emit("model");
    },
    (e: unknown) => {
      initError = `形状エンジンを読み込めません: ${(e as Error).message ?? e}`;
      store.msg(initError, "error");
      store.emit("model");
    },
  );
}

export function geometryReady(): boolean {
  return ready;
}
export function geometryError(): string | null {
  return initError;
}

/** 現在の部品から組み立てた形状（obj 座標、mm）。未初期化なら null。 */
export function currentModel(): ModelResult | null {
  if (!ready || !store.project) return null;
  const parts = store.project.shape.parts;
  const k = JSON.stringify(parts);
  if (k !== key || !cur) {
    const old = cur;
    try {
      cur = buildModel(parts);
    } catch (e) {
      cur = { geometry: null, parts: [], stats: null, errors: [{ partId: "", message: String((e as Error).message ?? e) }] } as unknown as ModelResult;
    }
    key = k;
    version++;
    if (old) {
      old.geometry?.dispose();
      for (const p of old.parts) p.geometry?.dispose();
    }
  }
  return cur;
}

export function modelVersion(): number {
  currentModel();
  return version;
}

/** 形状の中心（obj 座標）。形状がなければ原点。 */
export function modelCenterObj(): [number, number, number] {
  const m = currentModel();
  const s = m?.stats;
  if (!s) return [0, 0, 0];
  return [(s.bbox_min[0] + s.bbox_max[0]) / 2, (s.bbox_min[1] + s.bbox_max[1]) / 2, (s.bbox_min[2] + s.bbox_max[2]) / 2];
}
