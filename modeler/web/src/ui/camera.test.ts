// K_und → WebGL 投影行列の確認: 既知の 3D 点の投影が projectPoint() と 0.5 px 以内で一致すること。
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { projectionFromK, intrinsicsFromK, viewIntrinsics, glModelMatrix, ndcToPixel, pixelToNdc, makeCvCamera, setCvProjection } from "./camera";
import { projectPoint, mat4FromRt, mat4Mul } from "../geom/index";
import type { Vec3 } from "../types";

const K = [913.09, 0, 641.3, 0, 912.4, 478.9, 0, 0, 1];
const W = 1280,
  H = 960;

function glProject(T_cam_obj: number[], p: Vec3, k = intrinsicsFromK(K), w = W, h = H): [number, number] {
  const v = new THREE.Vector3(...p).applyMatrix4(glModelMatrix(T_cam_obj)).applyMatrix4(projectionFromK(k, w, h));
  return ndcToPixel(v, w, h);
}

describe("projectionFromK", () => {
  const poses = [
    { T_cam_marker: mat4FromRt([180, 0, 0], [0, 0, 400]), T_marker_obj: mat4FromRt([0, 0, 0], [0, 0, 0]) },
    { T_cam_marker: mat4FromRt([150, 20, -35], [35, -20, 520]), T_marker_obj: mat4FromRt([10, -5, 30], [12, -8, 4]) },
    { T_cam_marker: mat4FromRt([-120, 45, 80], [-60, 40, 300]), T_marker_obj: mat4FromRt([0, 90, 0], [0, 20, -15]) },
  ];
  const pts: Vec3[] = [
    [0, 0, 0],
    [25, -10, 5],
    [-40, 30, 12],
    [10, 60, -20],
  ];
  it("projectPoint と 0.5 px 以内で一致する", () => {
    for (const ps of poses) {
      const T = mat4Mul(ps.T_cam_marker, ps.T_marker_obj);
      for (const p of pts) {
        const ref = projectPoint(K, ps.T_cam_marker, ps.T_marker_obj, p);
        expect(ref).not.toBeNull();
        const [u, v] = glProject(T, p);
        expect(Math.abs(u - ref![0])).toBeLessThan(0.5);
        expect(Math.abs(v - ref![1])).toBeLessThan(0.5);
        // 実際にはほぼ厳密に一致する
        expect(Math.abs(u - ref![0])).toBeLessThan(1e-6 * 1000);
      }
    }
  });
  it("画像ビューの拡大・移動を畳み込んでも screen = (u + 0.5)·zoom + pan", () => {
    const ps = poses[1];
    const T = mat4Mul(ps.T_cam_marker, ps.T_marker_obj);
    const zoom = 0.73,
      panX = 41,
      panY = -17,
      VW = 1000,
      VH = 700;
    for (const p of pts) {
      const ref = projectPoint(K, ps.T_cam_marker, ps.T_marker_obj, p)!;
      const [su, sv] = glProject(T, p, viewIntrinsics(intrinsicsFromK(K), zoom, panX, panY), VW, VH);
      // glProject は OpenCV 画素（中心が整数）を返すので screen 連続座標は +0.5
      expect(Math.abs(su + 0.5 - ((ref[0] + 0.5) * zoom + panX))).toBeLessThan(1e-6);
      expect(Math.abs(sv + 0.5 - ((ref[1] + 0.5) * zoom + panY))).toBeLessThan(1e-6);
    }
  });
  it("カメラの後ろの点は描かれない（w < 0）", () => {
    const T = mat4FromRt([0, 0, 0], [0, 0, -100]);
    const v4 = new THREE.Vector4(0, 0, 0, 1).applyMatrix4(glModelMatrix(T)).applyMatrix4(projectionFromK(intrinsicsFromK(K), W, H));
    expect(v4.w).toBeLessThan(0);
  });
  it("Raycaster の光線が画素を通る（逆投影）", () => {
    const cam = makeCvCamera();
    setCvProjection(cam, intrinsicsFromK(K), W, H);
    const rc = new THREE.Raycaster();
    const u = 900.25,
      v = 210.5;
    rc.setFromCamera(pixelToNdc(u, v, W, H), cam);
    const d = rc.ray.direction;
    // GL → OpenCV
    const x = d.x / -d.z,
      y = -d.y / -d.z;
    expect(Math.abs(K[0] * x + K[2] - u)).toBeLessThan(1e-3);
    expect(Math.abs(K[4] * y + K[5] - v)).toBeLessThan(1e-3);
  });
});
