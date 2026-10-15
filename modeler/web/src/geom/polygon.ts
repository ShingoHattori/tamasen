// 2D 多角形の検査（向き・自己交差）。輪郭は閉じた折れ線で、最後の点から最初の点へ戻る辺を含む。
import type { Vec2 } from "../types";

/** 符号付き面積（反時計回りで正）。 */
export function signedArea(p: Vec2[]): number {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const q = p[i];
    const r = p[(i + 1) % p.length];
    a += q[0] * r[1] - r[0] * q[1];
  }
  return a / 2;
}

/** 連続する重複点と、終点＝始点の重複を除く。 */
export function cleanPolygon(p: Vec2[], eps = 1e-9): Vec2[] {
  const out: Vec2[] = [];
  for (const q of p) {
    const last = out[out.length - 1];
    if (!last || Math.hypot(q[0] - last[0], q[1] - last[1]) > eps) out.push([q[0], q[1]]);
  }
  while (out.length > 1 && Math.hypot(out[0][0] - out[out.length - 1][0], out[0][1] - out[out.length - 1][1]) <= eps) out.pop();
  return out;
}

/** 反時計回りにそろえる。 */
export function toCCW(p: Vec2[]): Vec2[] {
  return signedArea(p) < 0 ? [...p].reverse() : p;
}

function orient(a: Vec2, b: Vec2, c: Vec2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSeg(a: Vec2, b: Vec2, c: Vec2, eps: number): boolean {
  return (
    Math.min(a[0], b[0]) - eps <= c[0] && c[0] <= Math.max(a[0], b[0]) + eps &&
    Math.min(a[1], b[1]) - eps <= c[1] && c[1] <= Math.max(a[1], b[1]) + eps
  );
}

/** 線分 ab と cd が交わる（接触を含む）か。 */
export function segmentsIntersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2, eps = 1e-9): boolean {
  const scale = Math.max(1, Math.abs(a[0]), Math.abs(a[1]), Math.abs(b[0]), Math.abs(b[1]));
  const e = eps * scale * scale;
  const o1 = orient(a, b, c);
  const o2 = orient(a, b, d);
  const o3 = orient(c, d, a);
  const o4 = orient(c, d, b);
  if (((o1 > e && o2 < -e) || (o1 < -e && o2 > e)) && ((o3 > e && o4 < -e) || (o3 < -e && o4 > e))) return true;
  if (Math.abs(o1) <= e && onSeg(a, b, c, eps * scale)) return true;
  if (Math.abs(o2) <= e && onSeg(a, b, d, eps * scale)) return true;
  if (Math.abs(o3) <= e && onSeg(c, d, a, eps * scale)) return true;
  if (Math.abs(o4) <= e && onSeg(c, d, b, eps * scale)) return true;
  return false;
}

/**
 * 閉じた多角形の自己交差（隣り合わない辺どうしの交差・接触）を探す。
 * 見つかれば交差する辺の番号の組、なければ null。O(n²)。
 */
export function findSelfIntersection(p: Vec2[]): [number, number] | null {
  const n = p.length;
  for (let i = 0; i < n; i++) {
    const a = p[i];
    const b = p[(i + 1) % n];
    for (let j = i + 1; j < n; j++) {
      // 隣接辺（共有頂点）は除外
      if (j === i + 1 || (i === 0 && j === n - 1)) continue;
      if (segmentsIntersect(a, b, p[j], p[(j + 1) % n])) return [i, j];
    }
  }
  return null;
}

/**
 * 輪郭の検査と正規化。問題があれば日本語の説明を投げる。
 * 戻り値は重複点を除いた反時計回りの輪郭。
 */
export function validatePolygon(raw: Vec2[] | undefined, what: string): Vec2[] {
  if (!raw || raw.length < 3) throw new Error(`${what}の点が 3 つ未満です`);
  for (const q of raw)
    if (!Array.isArray(q) || q.length < 2 || !Number.isFinite(q[0]) || !Number.isFinite(q[1]))
      throw new Error(`${what}に数値でない座標があります`);
  const p = cleanPolygon(raw);
  if (p.length < 3) throw new Error(`${what}の異なる点が 3 つ未満です`);
  const hit = findSelfIntersection(p);
  if (hit) throw new Error(`${what}が自分自身と交差しています（辺 ${hit[0] + 1} と辺 ${hit[1] + 1}）。点の順序を見直してください`);
  if (Math.abs(signedArea(p)) < 1e-9) throw new Error(`${what}の面積が 0 です`);
  return toCCW(p);
}
