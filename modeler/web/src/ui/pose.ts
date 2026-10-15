// 画像ごとの撮影姿勢 T_cam_marker の変更（位置合わせモード専用）。T_marker_obj は決して変えない。
import type { Mat4, Project, ProjectImage, Vec3 } from "../types";
import { mat4Mul, mat4FromRt, mat4Identity } from "./geo";
import { store } from "./store";

export function poseRefusal(img: ProjectImage | null): string | null {
  if (!store.project) return "プロジェクトが開かれていません";
  if (!img) return "画像が選択されていません";
  if (store.ui.mode !== "align") return "姿勢は位置合わせモードでだけ変更できます";
  if (img.pose.locked) return "この画像の姿勢は確定（ロック）されています。変えるには「ロック解除」してください";
  if (img.role === "background") return "背景画像には姿勢がありません";
  return null;
}

export function T_cam_obj(p: Project, img: ProjectImage): Mat4 | null {
  if (!img.pose.T_cam_marker) return null;
  return mat4Mul(img.pose.T_cam_marker, p.T_marker_obj);
}

/** その画像だけの姿勢を置き換える。T_marker_obj と他画像の姿勢が変わらないことを確かめる。 */
export function applyPose(img: ProjectImage, T: Mat4, source: ProjectImage["pose"]["source"], quality?: ProjectImage["pose"]["quality"]): boolean {
  const p = store.project!;
  const before = JSON.stringify(p.T_marker_obj);
  const others = JSON.stringify(p.images.filter((im) => im.id !== img.id).map((im) => im.pose));
  img.pose.T_cam_marker = T.slice(0, 16);
  img.pose.source = source;
  if (quality) img.pose.quality = quality;
  if (JSON.stringify(p.T_marker_obj) !== before) throw new Error("内部エラー: 位置合わせで T_marker_obj が変わりました");
  if (JSON.stringify(p.images.filter((im) => im.id !== img.id).map((im) => im.pose)) !== others) throw new Error("内部エラー: 他の画像の姿勢が変わりました");
  return true;
}

/** 1 操作として姿勢を変える（Undo 単位）。拒否なら理由を表示して false。 */
export function commitPose(img: ProjectImage, T: Mat4, source: ProjectImage["pose"]["source"] = "manual", quality?: ProjectImage["pose"]["quality"]): boolean {
  const why = poseRefusal(img);
  if (why) {
    store.msg(why, "error");
    return false;
  }
  store.commit("姿勢", () => applyPose(img, T, source, quality), "pose");
  return true;
}

export const trans = (t: Vec3): Mat4 => mat4FromRt([0, 0, 0], t);
export const rotX = (deg: number): Mat4 => mat4FromRt([deg, 0, 0], [0, 0, 0]);
export const rotY = (deg: number): Mat4 => mat4FromRt([0, deg, 0], [0, 0, 0]);
export const rotZ = (deg: number): Mat4 => mat4FromRt([0, 0, deg], [0, 0, 0]);

/** cam 座標の点 c を中心に回転 R をかける: Trans(c)·R·Trans(-c)·T */
export function rotateAbout(T: Mat4, c: Vec3, R: Mat4): Mat4 {
  return mat4Mul(trans(c), mat4Mul(R, mat4Mul(trans([-c[0], -c[1], -c[2]]), T)));
}

/** 姿勢がない画像の初期値: 物体をカメラ正面 distance mm に置く（マーカー面がカメラを向く）。 */
export function defaultPose(distance = 400): Mat4 {
  return mat4Mul(trans([0, 0, distance]), rotX(180));
}

export { mat4Identity };
