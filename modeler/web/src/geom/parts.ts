// 部品ごとの立体（manifold-3d の Manifold）を作る（DESIGN.md §3）。寸法は mm、角度は度。
// 作った Manifold は WASM のメモリなので、呼び出し側が delete() すること。
import type { CrossSection, Manifold } from "manifold-3d";
import type { Part, PartKind, SheetHole, Vec2 } from "../types";
import { manifold } from "./manifold";
import { cleanPolygon, findSelfIntersection, toCCW, validatePolygon } from "./polygon";
import { mat4FromTransform, mat4ToColumnMajor } from "./transform";

type MMat4 = Parameters<Manifold["transform"]>[0];

const DEFAULT_SEGMENTS = 64;

function pos(v: number | undefined, what: string): number {
  if (v === undefined || !Number.isFinite(v) || !(v > 0)) throw new Error(`${what}は正の数にしてください（現在 ${v}）`);
  return v;
}

function segs(v: number | undefined): number {
  if (v === undefined) return DEFAULT_SEGMENTS;
  if (!Number.isFinite(v) || v < 3) throw new Error(`分割数は 3 以上にしてください（現在 ${v}）`);
  return Math.round(v);
}

/** 一時オブジェクトをまとめて解放する。 */
class Scope {
  private items: { delete(): void }[] = [];
  add<T extends { delete(): void }>(x: T): T {
    this.items.push(x);
    return x;
  }
  release() {
    for (const x of this.items) x.delete();
    this.items = [];
  }
}

function checkResult(m: Manifold, what: string): Manifold {
  const st = m.status();
  if (st !== "NoError") {
    m.delete();
    throw new Error(`${what}の立体が作れませんでした（${st}）`);
  }
  if (m.isEmpty()) {
    m.delete();
    throw new Error(`${what}の立体が空です（寸法・輪郭を確認してください）`);
  }
  return m;
}

// ---- 各形状（部品座標） ----------------------------------------------------------

function buildBox(p: Part): Manifold {
  const s = p.params.size;
  if (!s || s.length !== 3) throw new Error("箱の寸法 size [x, y, z] がありません");
  return manifold().Manifold.cube([pos(s[0], "箱の x 寸法"), pos(s[1], "箱の y 寸法"), pos(s[2], "箱の z 寸法")], true);
}

function buildCylinder(p: Part): Manifold {
  const r = pos(p.params.radius, "円柱の半径");
  const h = pos(p.params.height, "円柱の高さ");
  return manifold().Manifold.cylinder(h, r, r, segs(p.params.segments), true);
}

function buildFrustum(p: Part): Manifold {
  const rb = p.params.radius_bottom ?? 0;
  const rt = p.params.radius_top ?? 0;
  const h = pos(p.params.height, "円錐台の高さ");
  if (!Number.isFinite(rb) || !Number.isFinite(rt) || rb < 0 || rt < 0) throw new Error("円錐台の半径は 0 以上にしてください");
  if (rb <= 0 && rt <= 0) throw new Error("円錐台の上下の半径がどちらも 0 です");
  const M = manifold().Manifold;
  if (rb > 0) return M.cylinder(h, rb, rt, segs(p.params.segments), true);
  // 底が尖っている: 逆さに作って z を反転（mirror は向きを保ったまま裏返す）
  const c = M.cylinder(h, rt, 0, segs(p.params.segments), true);
  const r = c.mirror([0, 0, 1]);
  c.delete();
  return r;
}

/** 外形＋穴から CrossSection を作る（穴は外形からの差、向きは自動でそろえる）。 */
function profileSection(outerRaw: Vec2[] | undefined, holesRaw: Vec2[][] | undefined, what: string, sc: Scope): CrossSection {
  const CS = manifold().CrossSection;
  const outer = validatePolygon(outerRaw, `${what}の外形`);
  const outerCS = sc.add(new CS([outer], "Positive"));
  const holes = (holesRaw ?? []).map((h, i) => validatePolygon(h, `${what}の穴 ${i + 1}`));
  if (holes.length === 0) return outerCS.translate([0, 0]); // outerCS は Scope が解放するので複製を返す
  const outerArea = outerCS.area();
  for (let i = 0; i < holes.length; i++) {
    const hcs = sc.add(new CS([holes[i]], "Positive"));
    const inside = sc.add(hcs.intersect(outerCS));
    if (inside.area() < hcs.area() * (1 - 1e-6)) throw new Error(`${what}の穴 ${i + 1} が外形からはみ出しています`);
  }
  const holesCS = sc.add(new CS(holes, "Positive")); // 重なる穴は合併
  const r = outerCS.subtract(holesCS);
  if (r.area() < outerArea * 1e-6) {
    r.delete();
    throw new Error(`${what}が穴で全部なくなりました`);
  }
  return r;
}

function buildExtrude(p: Part): Manifold {
  const t = pos(p.params.thickness, "押し出しの厚さ");
  const sc = new Scope();
  try {
    const cs = sc.add(profileSection(p.params.outer, p.params.holes, "押し出し", sc));
    return cs.extrude(t, 0, 0, [1, 1], !!p.params.symmetric);
  } finally {
    sc.release();
  }
}

function buildRevolve(p: Part): Manifold {
  const raw = p.params.profile;
  if (raw) for (const q of raw) if (q[0] < -1e-9) throw new Error(`回転体の断面に r < 0 の点があります（r = ${q[0]}）。断面は軸の片側に描いてください`);
  const prof = validatePolygon(
    raw?.map((q) => [Math.max(0, q[0]), q[1]] as Vec2),
    "回転体の断面",
  );
  const ang = p.params.angle_deg ?? 360;
  if (!Number.isFinite(ang) || ang <= 0 || ang > 360) throw new Error(`回転角は 0 より大きく 360 以下にしてください（現在 ${ang}）`);
  const sc = new Scope();
  try {
    const cs = sc.add(new (manifold().CrossSection)([prof], "Positive"));
    // CrossSection の (x, y) = (r, z)。revolve は y 軸を z 軸として回す。
    return cs.revolve(segs(p.params.segments), ang);
  } finally {
    sc.release();
  }
}

// ---- 板（sheet） --------------------------------------------------------------
// 中心線 path（部品の xz 平面の折れ線）。bend_radius は **中心線（板厚の中央）の曲げ半径**（内側半径 = R − t/2）。
// 断面（xz 平面の 2D 多角形）を作って y 方向に width だけ（中心 0）押し出す。
// 穴は (s, y) 平面（s = 曲げを含む中心線の弧長、y = 幅方向）で定義し、その位置の板厚方向に貫通させる。
// 曲げ部にかかる穴は拒否する。

interface LineEl {
  s0: number;
  s1: number;
  a: Vec2; // 始点（x, z）
  d: Vec2; // 単位方向
}

interface SheetGeom {
  outline: Vec2[]; // 断面（x, z）
  lines: LineEl[];
  totalLength: number;
}

export function sheetGeometry(p: Part): SheetGeom {
  const t = pos(p.params.thickness, "板厚");
  const h = t / 2;
  const R = p.params.bend_radius ?? 0;
  if (!Number.isFinite(R) || R < 0) throw new Error("曲げ半径は 0 以上にしてください");
  if (R > 0 && R < h - 1e-9) throw new Error(`曲げ半径（中心線）${R} mm が板厚の半分 ${h} mm より小さいと内側がつぶれます`);
  const raw = p.params.path;
  if (!raw || raw.length < 2) throw new Error("板の中心線の点が 2 つ未満です");
  for (const q of raw) if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) throw new Error("板の中心線に数値でない座標があります");
  // 重複点と直進する中間点を除く
  let P = cleanPolygon(raw);
  if (P.length >= 2 && Math.hypot(P[0][0] - raw[raw.length - 1][0], P[0][1] - raw[raw.length - 1][1]) < 1e-9 && raw.length > 2)
    throw new Error("板の中心線が閉じています（開いた折れ線にしてください）");
  if (P.length < 2) throw new Error("板の中心線の長さが 0 です");
  const dirOf = (a: Vec2, b: Vec2): Vec2 => {
    const l = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
  };
  {
    const Q: Vec2[] = [P[0]];
    for (let i = 1; i < P.length - 1; i++) {
      const d1 = dirOf(Q[Q.length - 1], P[i]);
      const d2 = dirOf(P[i], P[i + 1]);
      const cr = d1[0] * d2[1] - d1[1] * d2[0];
      const dt = d1[0] * d2[0] + d1[1] * d2[1];
      if (Math.abs(cr) < 1e-12 && dt > 0) continue;
      Q.push(P[i]);
    }
    Q.push(P[P.length - 1]);
    P = Q;
  }
  const n = P.length;
  const D: Vec2[] = [];
  const L: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    D.push(dirOf(P[i], P[i + 1]));
    L.push(Math.hypot(P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1]));
  }
  const N = D.map((d) => [-d[1], d[0]] as Vec2); // 左法線（2D で反時計回り 90°）
  // 頂点 i（1..n-2）の曲がり角
  const theta: number[] = new Array(n).fill(0);
  const tanD: number[] = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const a = D[i - 1];
    const b = D[i];
    const th = Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
    if (Math.abs(th) > (170 * Math.PI) / 180) throw new Error(`板の中心線が点 ${i + 1} で 170° を超えて折り返しています`);
    theta[i] = th;
    tanD[i] = R * Math.tan(Math.abs(th) / 2);
  }
  for (let i = 0; i < n - 1; i++) {
    if (tanD[i] + tanD[i + 1] > L[i] + 1e-9)
      throw new Error(`曲げ半径が大きすぎて中心線の区間 ${i + 1}（長さ ${L[i].toFixed(2)} mm）に曲げが収まりません`);
  }
  const nSeg = segs(p.params.segments);
  const left: Vec2[] = [];
  const right: Vec2[] = [];
  const lines: LineEl[] = [];
  let s = 0;
  left.push([P[0][0] + h * N[0][0], P[0][1] + h * N[0][1]]);
  right.push([P[0][0] - h * N[0][0], P[0][1] - h * N[0][1]]);
  for (let i = 0; i < n - 1; i++) {
    // 直線部
    const a: Vec2 = [P[i][0] + D[i][0] * tanD[i], P[i][1] + D[i][1] * tanD[i]];
    const len = L[i] - tanD[i] - tanD[i + 1];
    lines.push({ s0: s, s1: s + len, a, d: D[i] });
    s += len;
    if (i + 1 >= n - 1) break;
    const v = i + 1; // 次の頂点
    const th = theta[v];
    if (R === 0 || Math.abs(th) < 1e-12) {
      // 角（マイター）
      const m1 = N[v - 1];
      const m2 = N[v];
      const k = 1 + m1[0] * m2[0] + m1[1] * m2[1];
      const m: Vec2 = [(m1[0] + m2[0]) / k, (m1[1] + m2[1]) / k];
      left.push([P[v][0] + h * m[0], P[v][1] + h * m[1]]);
      right.push([P[v][0] - h * m[0], P[v][1] - h * m[1]]);
    } else {
      // 円弧（中心線の半径 R）
      const t1: Vec2 = [P[v][0] - D[v - 1][0] * tanD[v], P[v][1] - D[v - 1][1] * tanD[v]];
      const sg = th > 0 ? 1 : -1;
      const c: Vec2 = [t1[0] + sg * R * N[v - 1][0], t1[1] + sg * R * N[v - 1][1]];
      const phi0 = Math.atan2(t1[1] - c[1], t1[0] - c[0]);
      const k = Math.max(2, Math.ceil((Math.abs(th) / (2 * Math.PI)) * nSeg));
      for (let j = 0; j <= k; j++) {
        const phi = phi0 + (th * j) / k;
        const cp: Vec2 = [c[0] + R * Math.cos(phi), c[1] + R * Math.sin(phi)];
        const tg: Vec2 = [-sg * Math.sin(phi), sg * Math.cos(phi)];
        const nn: Vec2 = [-tg[1], tg[0]];
        left.push([cp[0] + h * nn[0], cp[1] + h * nn[1]]);
        right.push([cp[0] - h * nn[0], cp[1] - h * nn[1]]);
      }
      s += R * Math.abs(th);
    }
  }
  const last = P[n - 1];
  const nl = N[n - 2];
  left.push([last[0] + h * nl[0], last[1] + h * nl[1]]);
  right.push([last[0] - h * nl[0], last[1] - h * nl[1]]);
  const outline = cleanPolygon([...left, ...right.reverse()], 1e-7);
  if (findSelfIntersection(outline)) throw new Error("板の断面が自分自身と重なります（中心線が交差しているか、板厚に対して区間が短すぎます）");
  return { outline: toCCW(outline), lines, totalLength: s };
}

function holeSection(hole: SheetHole, idx: number, nSeg: number): CrossSection {
  const CS = manifold().CrossSection;
  const what = `板の穴 ${idx + 1}`;
  if (hole.shape === "circle") return CS.circle(pos(hole.d, `${what}の直径`) / 2, nSeg);
  if (hole.shape === "slot") {
    const w = pos(hole.w, `${what}の幅`);
    const l = pos(hole.l, `${what}の長さ`);
    if (l <= w) return CS.circle(w / 2, nSeg);
    const c = CS.circle(w / 2, nSeg);
    const a = c.translate([-(l - w) / 2, 0]);
    const b = c.translate([(l - w) / 2, 0]);
    const r = CS.hull([a, b]);
    c.delete();
    a.delete();
    b.delete();
    return r;
  }
  if (hole.shape === "poly") return new CS([validatePolygon(hole.pts, `${what}の輪郭`)], "Positive");
  throw new Error(`${what}の形 "${String(hole.shape)}" は未対応です`);
}

function buildSheet(p: Part): Manifold {
  const M = manifold();
  const w = pos(p.params.width, "板の幅");
  const t = pos(p.params.thickness, "板厚");
  const g = sheetGeometry(p);
  const sc = new Scope();
  try {
    const cs = sc.add(new M.CrossSection([g.outline], "Positive"));
    const flat = sc.add(cs.extrude(w, 0, 0, [1, 1], true));
    // 2D (X, Y) = (x, z)、押し出し方向 Z → −y: x 軸まわりに +90°
    const body = sc.add(flat.rotate([90, 0, 0]));
    const holes = p.params.sheet_holes ?? [];
    if (holes.length === 0) {
      return body.translate([0, 0, 0]); // body は Scope が解放するので複製を返す
    }
    const cutters: Manifold[] = [];
    const nSeg = segs(p.params.segments);
    holes.forEach((hole, i) => {
      if (!Number.isFinite(hole.s) || !Number.isFinite(hole.y)) throw new Error(`板の穴 ${i + 1} の位置 s・y が数値ではありません`);
      const hcs = sc.add(holeSection(hole, i, nSeg));
      const b = hcs.bounds();
      const sMin = hole.s + b.min[0];
      const sMax = hole.s + b.max[0];
      const eps = 1e-6;
      const el = g.lines.find((e) => e.s0 - eps <= sMin && sMax <= e.s1 + eps);
      if (!el) {
        if (sMin < -eps || sMax > g.totalLength + eps)
          throw new Error(`板の穴 ${i + 1}（s = ${hole.s}）が板の端（弧長 0〜${g.totalLength.toFixed(2)} mm）からはみ出しています`);
        throw new Error(`板の穴 ${i + 1}（s = ${hole.s}）が曲げ部にかかっています。平らな区間に置いてください`);
      }
      const cut = sc.add(hcs.extrude(t * 4, 0, 0, [1, 1], true));
      const cx = el.a[0] + el.d[0] * (hole.s - el.s0);
      const cz = el.a[1] + el.d[1] * (hole.s - el.s0);
      // 局所 (a, b, c) = (中心線方向, y, 板の法線)。列優先で manifold へ。
      const T = [el.d[0], 0, el.d[1], 0, 0, 1, 0, 0, -el.d[1], 0, el.d[0], 0, cx, hole.y, cz, 1] as MMat4;
      cutters.push(sc.add(cut.transform(T)));
    });
    const allCut = sc.add(M.Manifold.union(cutters));
    return body.subtract(allCut);
  } finally {
    sc.release();
  }
}

// ---- 公開 API -----------------------------------------------------------------

/** 部品座標の立体（transform 適用前）。不正なら日本語の例外。 */
export function buildPartLocal(p: Part): Manifold {
  const what = `部品「${p.name || p.id}」`;
  let m: Manifold;
  switch (p.kind) {
    case "box":
      m = buildBox(p);
      break;
    case "cylinder":
      m = buildCylinder(p);
      break;
    case "frustum":
      m = buildFrustum(p);
      break;
    case "extrude":
      m = buildExtrude(p);
      break;
    case "revolve":
      m = buildRevolve(p);
      break;
    case "sheet":
      m = buildSheet(p);
      break;
    default:
      throw new Error(`部品の種類 "${String((p as Part).kind)}" は未対応です`);
  }
  return checkResult(m, what);
}

/** obj 座標の立体（T_obj_part を適用済み）。 */
export function buildPart(p: Part): Manifold {
  const local = buildPartLocal(p);
  try {
    const tr = p.transform ?? { t_mm: [0, 0, 0], r_deg: [0, 0, 0] };
    for (const v of [...tr.t_mm, ...tr.r_deg]) if (!Number.isFinite(v)) throw new Error(`部品「${p.name || p.id}」の位置・回転に数値でない値があります`);
    return local.transform(mat4ToColumnMajor(mat4FromTransform(tr)) as MMat4);
  } finally {
    local.delete();
  }
}

export interface PartBuild {
  id: string;
  op: "add" | "subtract";
  manifold: Manifold | null;
  error?: string;
}

export interface AssembleResult {
  model: Manifold | null; // add の union から subtract の difference
  parts: PartBuild[];
  errors: { partId: string; message: string }[];
}

/**
 * 全部品を作って合成する。失敗した部品はその部品のエラーとして返し、残りで合成する。
 * visible は表示用の属性で、合成には全部品を使う。返す Manifold はすべて呼び出し側が delete() する（freeAssemble）。
 */
export function assemble(parts: Part[]): AssembleResult {
  const M = manifold().Manifold;
  const res: PartBuild[] = [];
  const errors: { partId: string; message: string }[] = [];
  for (const p of parts) {
    try {
      res.push({ id: p.id, op: p.op, manifold: buildPart(p) });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      res.push({ id: p.id, op: p.op, manifold: null, error: message });
      errors.push({ partId: p.id, message });
    }
  }
  const adds = res.filter((r) => r.op !== "subtract" && r.manifold).map((r) => r.manifold!);
  const subs = res.filter((r) => r.op === "subtract" && r.manifold).map((r) => r.manifold!);
  let model: Manifold | null = null;
  if (adds.length > 0) {
    try {
      const u = M.union(adds);
      if (subs.length > 0) {
        const s = M.union(subs);
        model = u.subtract(s);
        u.delete();
        s.delete();
      } else model = u;
      if (model.status() !== "NoError") {
        errors.push({ partId: "", message: `合成に失敗しました（${model.status()}）` });
        model.delete();
        model = null;
      } else if (model.isEmpty()) {
        errors.push({ partId: "", message: "合成した形が空です（切削部品がすべてを削っています）" });
        model.delete();
        model = null;
      }
    } catch (e) {
      errors.push({ partId: "", message: `合成に失敗しました: ${e instanceof Error ? e.message : String(e)}` });
      model = null;
    }
  }
  return { model, parts: res, errors };
}

export function freeAssemble(r: AssembleResult): void {
  r.model?.delete();
  for (const p of r.parts) p.manifold?.delete();
}

/** 新しい部品の既定値（mm・度、日本語名）。 */
export function defaultPart(kind: PartKind, id: string): Part {
  const base = { id, op: "add" as const, visible: true, transform: { t_mm: [0, 0, 0] as [number, number, number], r_deg: [0, 0, 0] as [number, number, number] }, kind };
  switch (kind) {
    case "box":
      return { ...base, name: "箱", params: { size: [50, 30, 10] } };
    case "cylinder":
      return { ...base, name: "円柱", params: { radius: 15, height: 30, segments: 64 } };
    case "frustum":
      return { ...base, name: "円錐台", params: { radius_bottom: 20, radius_top: 10, height: 30, segments: 64 } };
    case "extrude":
      return {
        ...base,
        name: "押し出し",
        params: { outer: [[-30, -20], [30, -20], [30, 20], [-30, 20]], holes: [], thickness: 5, symmetric: false },
      };
    case "revolve":
      // 容器: 外側 → 口 → 内側（肉厚 3 mm、上が開いている）
      return {
        ...base,
        name: "回転体",
        params: { profile: [[0, 0], [30, 0], [30, 50], [27, 50], [27, 3], [0, 3]], angle_deg: 360, segments: 64 },
      };
    case "sheet":
      return {
        ...base,
        name: "板",
        params: { path: [[-60, 0], [0, 0], [40, 20]], thickness: 2, width: 25, bend_radius: 5, sheet_holes: [], segments: 32 },
      };
  }
}
