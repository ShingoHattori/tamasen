import { beforeAll, describe, expect, it } from "vitest";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { Mat4, Part, Project, Vec2, Vec3 } from "../types";
import {
  applyParamEdit,
  backProjectToPlane,
  buildModel,
  checkMeasurements,
  defaultPart,
  exportGLB,
  initGeometry,
  isLocked,
  mat4Apply,
  mat4FromRt,
  mat4FromThree,
  mat4FromTransform,
  mat4Identity,
  mat4InvRigid,
  mat4Mul,
  mat4ToThree,
  planePoint,
  projectPoint,
  transformFromMat4,
} from "./index";
import { circlePts, polyArea, readGLB, slotPts } from "./testutil";

const T0 = { t_mm: [0, 0, 0] as Vec3, r_deg: [0, 0, 0] as Vec3 };
const part = (id: string, kind: Part["kind"], params: Part["params"], op: Part["op"] = "add", transform = T0): Part => ({
  id, name: id, op, visible: true, transform, kind, params,
});

const hole1 = circlePts(-30, 0, 6, 64);
const hole2 = slotPts(25, 0, 20, 8, 32);
const plateParts: Part[] = [
  part("plate", "extrude", { outer: [[-60, -30], [60, -30], [60, 30], [-60, 30]], holes: [hole1, hole2], thickness: 3 }),
];

beforeAll(async () => {
  await initGeometry();
});

describe("形状", () => {
  it("押し出しの板（丸穴＋長穴）: 水密・genus 2・体積", () => {
    const r = buildModel(plateParts);
    expect(r.errors).toEqual([]);
    const s = r.stats!;
    expect(s.watertight).toBe(true);
    expect(s.genus).toBe(2);
    const exact = 120 * 60 * 3 - (Math.PI * 36 + (12 * 8 + Math.PI * 16)) * 3;
    expect(Math.abs(s.volume_mm3 - exact) / exact).toBeLessThan(0.005);
    const poly = (120 * 60 - polyArea(hole1) - polyArea(hole2)) * 3;
    expect(s.volume_mm3).toBeCloseTo(poly, 3);
    expect(r.geometry!.getAttribute("position").count).toBe(s.triangles * 3);
    r.geometry!.dispose();
  });

  it("押し出し方向: 表側は z ∈ [0, t]、裏側は [−t, 0]、両側は [−t/2, t/2]", () => {
    const base = plateParts[0];
    for (const [dir, sym, lo, hi] of [["front", false, 0, 3], ["back", false, -3, 0], ["front", true, -1.5, 1.5]] as const) {
      const part: Part = { ...base, params: { ...base.params, direction: dir, symmetric: sym } };
      const r = buildModel([part]);
      expect(r.errors).toEqual([]);
      expect(r.stats!.bbox_min[2]).toBeCloseTo(lo, 6);
      expect(r.stats!.bbox_max[2]).toBeCloseTo(hi, 6);
      r.geometry!.dispose();
    }
  });

  it("輪郭の向き（時計回り）を自動で直す・自己交差は拒否", () => {
    const cw = part("a", "extrude", { outer: [[-60, 30], [60, 30], [60, -30], [-60, -30]], holes: [[...hole1].reverse()], thickness: 3 });
    const r = buildModel([cw]);
    expect(r.errors).toEqual([]);
    expect(r.stats!.genus).toBe(1);
    const bow = part("b", "extrude", { outer: [[0, 0], [10, 10], [10, 0], [0, 10]], thickness: 3 });
    const r2 = buildModel([bow]);
    expect(r2.geometry).toBeNull();
    expect(r2.errors[0].partId).toBe("b");
    expect(r2.errors[0].message).toMatch(/交差/);
    const out = part("c", "extrude", { outer: [[0, 0], [10, 0], [10, 10], [0, 10]], holes: [circlePts(10, 5, 2)], thickness: 1 });
    expect(buildModel([out]).errors[0].message).toMatch(/はみ出し/);
  });

  it("箱から円柱・箱を引く: genus 2", () => {
    const r = buildModel([
      part("p", "box", { size: [120, 60, 3] }),
      part("h1", "cylinder", { radius: 6, height: 10, segments: 64 }, "subtract", { t_mm: [-30, 0, 0], r_deg: [0, 0, 0] }),
      part("h2", "box", { size: [12, 8, 10] }, "subtract", { t_mm: [25, 0, 0], r_deg: [0, 0, 0] }),
      part("h3", "cylinder", { radius: 4, height: 10, segments: 64 }, "subtract", { t_mm: [31, 0, 0], r_deg: [0, 0, 0] }),
      part("h4", "cylinder", { radius: 4, height: 10, segments: 64 }, "subtract", { t_mm: [19, 0, 0], r_deg: [0, 0, 0] }),
    ]);
    expect(r.errors).toEqual([]);
    expect(r.stats!.watertight).toBe(true);
    expect(r.stats!.genus).toBe(2);
    expect(r.parts.length).toBe(5);
    expect(r.parts.every((p) => p.geometry)).toBe(true);
  });

  it("回転体の容器（外 r35 高 80 肉厚 3、上が開口）: genus 0・空洞あり", () => {
    const r = buildModel([
      part("cup", "revolve", { profile: [[0, 0], [35, 0], [35, 80], [32, 80], [32, 3], [0, 3]], angle_deg: 360, segments: 128 }),
    ]);
    expect(r.errors).toEqual([]);
    const s = r.stats!;
    expect(s.watertight).toBe(true);
    expect(s.genus).toBe(0);
    const shell = Math.PI * 35 * 35 * 80 - Math.PI * 32 * 32 * 77;
    const solid = Math.PI * 35 * 35 * 80;
    expect(Math.abs(s.volume_mm3 - shell) / shell).toBeLessThan(0.01);
    expect(s.volume_mm3).toBeLessThan(solid * 0.5);
    expect(s.bbox_max[2]).toBeCloseTo(80, 6);
    expect(s.bbox_min[2]).toBeCloseTo(0, 6);
  });

  it("板（曲げ 1 箇所＋穴）: 水密・genus 1、曲げ部の穴は拒否", () => {
    const params = { path: [[0, 0], [50, 0], [80, 30]] as Vec2[], thickness: 2, width: 20, bend_radius: 5, segments: 64, sheet_holes: [{ s: 20, y: 0, shape: "circle" as const, d: 6 }] };
    const r = buildModel([part("sh", "sheet", params)]);
    expect(r.errors).toEqual([]);
    expect(r.stats!.watertight).toBe(true);
    expect(r.stats!.genus).toBe(1);
    // 体積 ≈ 中心線長 × 板厚 × 幅 − 穴
    const th = Math.PI / 4;
    const d = 5 * Math.tan(th / 2);
    const len = 50 - d + (Math.hypot(30, 30) - d) + 5 * th;
    const expected = len * 2 * 20 - Math.PI * 9 * 2;
    expect(Math.abs(r.stats!.volume_mm3 - expected) / expected).toBeLessThan(0.005);
    const bad = buildModel([part("sh", "sheet", { ...params, sheet_holes: [{ s: 50, y: 0, shape: "circle", d: 6 }] })]);
    expect(bad.errors[0].message).toMatch(/曲げ部/);
    // 角（R = 0）・長穴
    const sharp = buildModel([part("s2", "sheet", { ...params, bend_radius: 0, sheet_holes: [{ s: 20, y: 0, shape: "slot", w: 4, l: 12 }] })]);
    expect(sharp.errors).toEqual([]);
    expect(sharp.stats!.genus).toBe(1);
    expect(sharp.stats!.watertight).toBe(true);
  });

  it("既定の部品はすべて作れる・失敗部品は他を止めない", () => {
    for (const k of ["box", "cylinder", "frustum", "extrude", "revolve", "sheet"] as const) {
      const r = buildModel([defaultPart(k, "x")]);
      expect(r.errors, k).toEqual([]);
      expect(r.stats!.watertight, k).toBe(true);
    }
    const r = buildModel([defaultPart("box", "a"), part("bad", "box", { size: [1, -1, 1] })]);
    expect(r.geometry).not.toBeNull();
    expect(r.errors.map((e) => e.partId)).toEqual(["bad"]);
    expect(r.parts[1].error).toMatch(/正の数/);
  });
});

describe("変換", () => {
  it("fromTransform / toTransform / invert / THREE 往復", () => {
    const tr = { t_mm: [12.5, -3, 40] as Vec3, r_deg: [10, -35, 120] as Vec3 };
    const m = mat4FromTransform(tr);
    const back = transformFromMat4(m);
    for (let i = 0; i < 3; i++) {
      expect(back.t_mm[i]).toBeCloseTo(tr.t_mm[i], 9);
      expect(back.r_deg[i]).toBeCloseTo(tr.r_deg[i], 9);
    }
    const I = mat4Mul(m, mat4InvRigid(m));
    I.forEach((v, i) => expect(v).toBeCloseTo(mat4Identity()[i], 12));
    // 固定軸 X→Y→Z: Rz·Ry·Rx の順（x 軸 → Rz(90) で y 軸）
    const p = mat4Apply(mat4FromRt([0, 0, 90], [0, 0, 0]), [1, 0, 0]);
    expect(p[1]).toBeCloseTo(1, 12);
    // THREE（列優先）: 並進は elements[12..14]
    const t3 = mat4ToThree(m);
    expect(t3.elements[12]).toBeCloseTo(12.5, 12);
    const v = new THREE.Vector3(1, 2, 3).applyMatrix4(t3);
    const q = mat4Apply(m, [1, 2, 3]);
    expect(v.x).toBeCloseTo(q[0], 9);
    expect(mat4FromThree(t3)).toEqual(m);
    // ジンバルロック
    const g = transformFromMat4(mat4FromRt([0, 90, 30], [0, 0, 0]));
    mat4FromTransform(g).forEach((x, i) => expect(x).toBeCloseTo(mat4FromRt([0, 90, 30], [0, 0, 0])[i], 9));
  });

  it("投影 → 平面への逆投影の往復（1e-6 mm 以内）", () => {
    const K = [913.09, 0, 640.5, 0, 913.09, 480.2, 0, 0, 1];
    const T_cam_marker: Mat4 = mat4FromRt([160, 20, -15], [10, -20, 350]);
    const T_marker_obj: Mat4 = mat4FromRt([5, -10, 30], [3, 4, 5]);
    const plane = { origin: [5, -2, 10] as Vec3, normal: [0.1, 0.3, 1] as Vec3, xAxis: [1, 0, 0] as Vec3 };
    for (const uv of [[0, 0], [12.3, -40], [-25, 33]] as Vec2[]) {
      const p = planePoint(plane, uv);
      const pix = projectPoint(K, T_cam_marker, T_marker_obj, p);
      expect(pix).not.toBeNull();
      const bp = backProjectToPlane(pix!, K, T_cam_marker, T_marker_obj, plane)!;
      expect(bp).not.toBeNull();
      for (let i = 0; i < 3; i++) expect(Math.abs(bp.p_obj[i] - p[i])).toBeLessThan(1e-6);
      expect(Math.abs(bp.uv_plane[0] - uv[0])).toBeLessThan(1e-6);
      expect(Math.abs(bp.uv_plane[1] - uv[1])).toBeLessThan(1e-6);
    }
    // カメラの後ろ
    expect(projectPoint(K, mat4Identity(), mat4Identity(), [0, 0, -10])).toBeNull();
  });
});

describe("書き出し", () => {
  it("GLB: m 単位・座標の選択・asset.extras・GLTFLoader で読み戻し", async () => {
    const T_marker_obj = mat4FromRt([0, 30, 90], [10, 20, -5]);
    const objBuf = await exportGLB(plateParts, T_marker_obj, "obj");
    const mkBuf = await exportGLB(plateParts, T_marker_obj, "marker");
    const a = readGLB(objBuf);
    const b = readGLB(mkBuf);
    expect(a.json.asset.extras.units).toBe("m");
    expect(a.json.asset.extras.frame).toBe("obj");
    expect(b.json.asset.extras.frame).toBe("marker");
    expect(a.positions.length).toBeGreaterThan(0);
    let maxAbs = 0;
    for (const v of a.positions) maxAbs = Math.max(maxAbs, Math.abs(v));
    expect(maxAbs).toBeCloseTo(0.06, 6); // 120 mm 幅の半分 = 0.06 m
    expect(b.positions.length).toBe(a.positions.length);
    for (let i = 0; i < a.positions.length; i += 3) {
      const p = mat4Apply(T_marker_obj, [a.positions[i] * 1000, a.positions[i + 1] * 1000, a.positions[i + 2] * 1000]);
      for (let k = 0; k < 3; k++) expect(Math.abs(b.positions[i + k] - p[k] * 0.001)).toBeLessThan(1e-6);
    }
    // Python（trimesh）での確認用。@types/node を入れないので node:fs は型なしで読む。
    const fs = (await import(/* @vite-ignore */ "node:fs" as string)) as { writeFileSync(p: string, d: Uint8Array): void };
    fs.writeFileSync("/tmp/modeler_plate.glb", new Uint8Array(mkBuf));
    const gltf = await new GLTFLoader().parseAsync(objBuf, "");
    let count = 0;
    gltf.scene.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) count += (o as THREE.Mesh).geometry.getAttribute("position").count;
    });
    expect(count).toBeGreaterThan(0);
  });
});

describe("実測寸法のロック", () => {
  it("ロックされたパラメータは変更できない", () => {
    const p = {
      shape: { parts: [defaultPart("box", "p1")] },
      measurements: [
        { id: "m1", label: "板厚", part: "p1", param: "size.2", value_mm: 10, locked: true },
        { id: "m2", label: "幅", part: "p1", param: "size.0", value_mm: 48, locked: false },
      ],
    } as unknown as Project;
    expect(isLocked(p, "p1", "size.2")).toBe(true);
    expect(isLocked(p, "p1", "size")).toBe(true);
    expect(isLocked(p, "p1", "size.0")).toBe(false);
    expect(applyParamEdit(p, "p1", "size.2", 3)).toMatch(/ロック/);
    expect(p.shape.parts[0].params.size![2]).toBe(10);
    expect(applyParamEdit(p, "p1", "size.0", 48)).toBeNull();
    expect(p.shape.parts[0].params.size![0]).toBe(48);
    expect(applyParamEdit(p, "p1", "size.7", 1)).toMatch(/範囲外/);
    expect(applyParamEdit(p, "zz", "size.0", 1)).toMatch(/見つかりません/);
    const c = checkMeasurements(p);
    expect(c.map((x) => x.ok)).toEqual([true, true]);
    p.shape.parts[0].params.size![0] = 50;
    expect(checkMeasurements(p)[1]).toMatchObject({ ok: false, actual: 50, expected: 48 });
  });
});
