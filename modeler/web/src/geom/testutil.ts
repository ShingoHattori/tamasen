// テスト用の補助（円・長穴の多角形、GLB の読み出し）。
import type { Vec2 } from "../types";

export function circlePts(cx: number, cy: number, r: number, n = 64): Vec2[] {
  return Array.from({ length: n }, (_, i) => [cx + r * Math.cos((2 * Math.PI * i) / n), cy + r * Math.sin((2 * Math.PI * i) / n)] as Vec2);
}

/** 長穴（全長 l、幅 w、長手 = x）。 */
export function slotPts(cx: number, cy: number, l: number, w: number, nHalf = 32): Vec2[] {
  const r = w / 2;
  const a = (l - w) / 2;
  const pts: Vec2[] = [];
  for (let i = 0; i <= nHalf; i++) {
    const t = -Math.PI / 2 + (Math.PI * i) / nHalf;
    pts.push([cx + a + r * Math.cos(t), cy + r * Math.sin(t)]);
  }
  for (let i = 0; i <= nHalf; i++) {
    const t = Math.PI / 2 + (Math.PI * i) / nHalf;
    pts.push([cx - a + r * Math.cos(t), cy + r * Math.sin(t)]);
  }
  return pts;
}

export function polyArea(p: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const q = p[i], r = p[(i + 1) % p.length];
    s += q[0] * r[1] - r[0] * q[1];
  }
  return Math.abs(s) / 2;
}

/** GLB を JSON と BIN に分け、最初のメッシュの POSITION を返す。 */
export function readGLB(buf: ArrayBuffer): { json: any; positions: Float32Array } {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("GLB ではありません");
  const jsonLen = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, jsonLen)));
  const binStart = 20 + jsonLen + 8;
  const acc = json.accessors[json.meshes[0].primitives[0].attributes.POSITION];
  const bv = json.bufferViews[acc.bufferView];
  const off = binStart + (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const positions = new Float32Array(buf.slice(off, off + acc.count * 12));
  return { json, positions };
}
