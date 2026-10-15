// 画像ごとのラベル（歪み補正画像の解像度、値は LABEL）。塗り・表示用の色付き canvas・画像ごとの Undo。
import { api, fileUrl } from "./api";
import { decodeGrayPng, encodeGrayPng } from "./png";
import { store } from "./store";

export interface Stroke {
  label: number;
  r: number;
  points: [number, number][];
}

export interface LabelLayer {
  iid: string;
  w: number;
  h: number;
  data: Uint8Array<ArrayBufferLike>;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  dirty: boolean;
  undo: Uint8Array<ArrayBufferLike>[];
  redo: Uint8Array<ArrayBufferLike>[];
  strokes: Stroke[];
  source: "labels" | "labels_auto" | "empty";
}

const COLORS: Record<number, [number, number, number, number]> = {
  0: [0, 0, 0, 0],
  1: [40, 110, 255, 45],
  2: [20, 220, 90, 100],
  3: [255, 40, 210, 140],
  4: [255, 200, 0, 120],
};

const layers = new Map<string, LabelLayer>();
const loading = new Map<string, Promise<LabelLayer | null>>();

export function getLabels(iid: string): LabelLayer | null {
  return layers.get(iid) ?? null;
}

export function dropLabels(iid?: string): void {
  if (iid) {
    layers.delete(iid);
    loading.delete(iid);
  } else {
    layers.clear();
    loading.clear();
  }
}

async function fetchPng(url: string): Promise<Blob | null> {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.blob();
  } catch {
    return null;
  }
}

export function loadLabels(iid: string, force = false): Promise<LabelLayer | null> {
  if (!force && layers.has(iid)) return Promise.resolve(layers.get(iid)!);
  if (!force && loading.has(iid)) return loading.get(iid)!;
  const pr = (async () => {
    const pid = store.pid;
    const img = store.image(iid);
    if (!pid || !img) return null;
    const v = store.fileVersion[iid] ?? 0;
    let blob: Blob | null = null;
    let source: LabelLayer["source"] = "empty";
    if (img.has_labels !== false) {
      blob = await fetchPng(fileUrl(pid, iid, "labels.png", v));
      if (blob) source = "labels";
    }
    if (!blob) {
      blob = await fetchPng(fileUrl(pid, iid, "labels_auto.png", v));
      if (blob) source = "labels_auto";
    }
    let w = img.width;
    let h = img.height;
    let data: Uint8Array<ArrayBufferLike> = new Uint8Array(w * h);
    if (blob) {
      try {
        const d = await decodeGrayPng(blob);
        w = d.w;
        h = d.h;
        data = d.data;
      } catch {
        source = "empty";
      }
    }
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    const layer: LabelLayer = { iid, w, h, data, canvas, ctx, dirty: false, undo: [], redo: [], strokes: [], source };
    repaint(layer, 0, 0, w, h);
    layers.set(iid, layer);
    loading.delete(iid);
    store.emit("labels");
    return layer;
  })();
  loading.set(iid, pr);
  return pr;
}

export function repaint(L: LabelLayer, x0: number, y0: number, x1: number, y1: number): void {
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(L.w, Math.ceil(x1));
  y1 = Math.min(L.h, Math.ceil(y1));
  if (x1 <= x0 || y1 <= y0) return;
  const id = L.ctx.createImageData(x1 - x0, y1 - y0);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const c = COLORS[L.data[y * L.w + x]] ?? COLORS[0];
      const o = ((y - y0) * (x1 - x0) + (x - x0)) * 4;
      id.data[o] = c[0];
      id.data[o + 1] = c[1];
      id.data[o + 2] = c[2];
      id.data[o + 3] = c[3];
    }
  }
  L.ctx.putImageData(id, x0, y0);
}

export function strokeBegin(L: LabelLayer, label: number, r: number): void {
  L.undo.push(L.data.slice());
  if (L.undo.length > 30) L.undo.shift();
  L.redo = [];
  L.strokes.push({ label, r, points: [] });
}

/** 円を塗る（u, v は OpenCV 画素座標）。前の点との間も埋める。 */
export function strokePoint(L: LabelLayer, u: number, v: number): void {
  const s = L.strokes[L.strokes.length - 1];
  if (!s) return;
  const prev = s.points[s.points.length - 1];
  s.points.push([u, v]);
  const pts: [number, number][] = [];
  if (prev) {
    const d = Math.hypot(u - prev[0], v - prev[1]);
    const n = Math.max(1, Math.ceil(d / Math.max(1, s.r / 2)));
    for (let i = 1; i <= n; i++) pts.push([prev[0] + ((u - prev[0]) * i) / n, prev[1] + ((v - prev[1]) * i) / n]);
  } else pts.push([u, v]);
  const r = s.r;
  let bx0 = Infinity,
    by0 = Infinity,
    bx1 = -Infinity,
    by1 = -Infinity;
  for (const [cu, cv] of pts) {
    const x0 = Math.max(0, Math.floor(cu - r)),
      x1 = Math.min(L.w - 1, Math.ceil(cu + r));
    const y0 = Math.max(0, Math.floor(cv - r)),
      y1 = Math.min(L.h - 1, Math.ceil(cv + r));
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) if ((x - cu) ** 2 + (y - cv) ** 2 <= r * r) L.data[y * L.w + x] = s.label;
    bx0 = Math.min(bx0, x0);
    by0 = Math.min(by0, y0);
    bx1 = Math.max(bx1, x1 + 1);
    by1 = Math.max(by1, y1 + 1);
  }
  L.dirty = true;
  repaint(L, bx0, by0, bx1, by1);
}

export function labelsUndo(L: LabelLayer): void {
  const d = L.undo.pop();
  if (!d) return;
  L.redo.push(L.data);
  L.data = d;
  L.strokes.pop();
  L.dirty = true;
  repaint(L, 0, 0, L.w, L.h);
  store.emit("labels");
}
export function labelsRedo(L: LabelLayer): void {
  const d = L.redo.pop();
  if (!d) return;
  L.undo.push(L.data);
  L.data = d;
  L.dirty = true;
  repaint(L, 0, 0, L.w, L.h);
  store.emit("labels");
}

export async function saveLabels(L: LabelLayer): Promise<boolean> {
  if (!store.pid) return false;
  const png = await encodeGrayPng(L.w, L.h, L.data);
  await api.putLabels(store.pid, L.iid, png);
  L.dirty = false;
  L.source = "labels";
  const img = store.image(L.iid);
  if (img) img.has_labels = true;
  store.emit("labels");
  return true;
}
