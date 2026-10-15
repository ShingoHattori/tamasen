// Manifold → THREE.BufferGeometry と統計。位置は mm（obj 座標）、法線は面ごと（外向き）。
import * as THREE from "three";
import type { Manifold } from "manifold-3d";
import type { Mat4, Vec3 } from "../types";

export interface ModelStats {
  volume_mm3: number;
  area_mm2: number;
  genus: number; // 貫通穴の数（穴 2 つの板 → 2）
  watertight: boolean; // 全辺がちょうど 2 枚の三角形に逆向きで共有される
  triangles: number;
  bbox_min: Vec3;
  bbox_max: Vec3;
}

/**
 * Manifold → インデックスなしの BufferGeometry（三角形ごとに 3 頂点、面法線）。
 * scale・T（行優先、mm → 出力座標）を指定すると位置に適用する: p' = scale · (T · p)。
 */
export function manifoldToGeometry(m: Manifold, opts: { T?: Mat4; scale?: number } = {}): THREE.BufferGeometry {
  const mesh = m.getMesh();
  const np = mesh.numProp;
  const vp = mesh.vertProperties;
  const tv = mesh.triVerts;
  const nt = tv.length / 3;
  const pos = new Float32Array(nt * 9);
  const nor = new Float32Array(nt * 9);
  const T = opts.T;
  const k = opts.scale ?? 1;
  const getP = (vi: number, out: number[]) => {
    const x = vp[vi * np];
    const y = vp[vi * np + 1];
    const z = vp[vi * np + 2];
    if (T) {
      out[0] = k * (T[0] * x + T[1] * y + T[2] * z + T[3]);
      out[1] = k * (T[4] * x + T[5] * y + T[6] * z + T[7]);
      out[2] = k * (T[8] * x + T[9] * y + T[10] * z + T[11]);
    } else {
      out[0] = k * x;
      out[1] = k * y;
      out[2] = k * z;
    }
  };
  const a = [0, 0, 0];
  const b = [0, 0, 0];
  const c = [0, 0, 0];
  for (let t = 0; t < nt; t++) {
    getP(tv[3 * t], a);
    getP(tv[3 * t + 1], b);
    getP(tv[3 * t + 2], c);
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (let j = 0; j < 3; j++) {
      const src = j === 0 ? a : j === 1 ? b : c;
      pos[t * 9 + j * 3] = src[0];
      pos[t * 9 + j * 3 + 1] = src[1];
      pos[t * 9 + j * 3 + 2] = src[2];
      nor[t * 9 + j * 3] = nx;
      nor[t * 9 + j * 3 + 1] = ny;
      nor[t * 9 + j * 3 + 2] = nz;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** 辺の共有を数えて水密（閉じた向き付き 2 次元多様体）かを調べる。 */
export function isWatertight(m: Manifold): boolean {
  const mesh = m.getMesh();
  const tv = mesh.triVerts;
  // 位置で頂点を統合（プロパティ境界で分かれた頂点を戻す）
  const np = mesh.numProp;
  const vp = mesh.vertProperties;
  const nv = vp.length / np;
  const key = new Map<string, number>();
  const id = new Uint32Array(nv);
  for (let i = 0; i < nv; i++) {
    const s = `${vp[i * np]},${vp[i * np + 1]},${vp[i * np + 2]}`;
    let v = key.get(s);
    if (v === undefined) {
      v = key.size;
      key.set(s, v);
    }
    id[i] = v;
  }
  const edges = new Map<string, number>();
  for (let t = 0; t < tv.length / 3; t++) {
    for (let j = 0; j < 3; j++) {
      const a = id[tv[3 * t + j]];
      const b = id[tv[3 * t + ((j + 1) % 3)]];
      if (a === b) return false;
      const k = `${a},${b}`;
      edges.set(k, (edges.get(k) ?? 0) + 1);
    }
  }
  for (const [k, cnt] of edges) {
    if (cnt !== 1) return false;
    const [a, b] = k.split(",");
    if (edges.get(`${b},${a}`) !== 1) return false;
  }
  return edges.size > 0;
}

export function manifoldStats(m: Manifold): ModelStats {
  const bb = m.boundingBox();
  return {
    volume_mm3: m.volume(),
    area_mm2: m.surfaceArea(),
    genus: m.genus(),
    watertight: m.status() === "NoError" && !m.isEmpty() && isWatertight(m),
    triangles: m.numTri(),
    bbox_min: [bb.min[0], bb.min[1], bb.min[2]],
    bbox_max: [bb.max[0], bb.max[1], bb.max[2]],
  };
}
