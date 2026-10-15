// プロジェクトの型（正本）。DESIGN.md §1〜3 と server/project.py と一致させる。
// 長さはすべて mm、角度は度。変換 T_a_b は b → a、4×4 行優先の 16 要素。

export type Mat4 = number[]; // 16 要素、行優先
export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface MarkerSpec {
  dictionary: string; // 例 "DICT_4X4_50"
  id: number;
  size_mm: number; // 黒い正方形の一辺
}

export interface Camera {
  width: number;
  height: number;
  K: number[]; // 3×3 行優先（原画像）
  dist: number[]; // OpenCV の歪み係数
  K_und: number[]; // 歪み補正画像のピンホール 3×3
  source: "file" | "vision" | "manual" | "none";
}

export interface Transform {
  t_mm: Vec3;
  r_deg: Vec3; // T = Trans(t)·Rz(rz)·Ry(ry)·Rx(rx)
}

export type PartKind = "box" | "cylinder" | "frustum" | "extrude" | "revolve" | "sheet";

export interface SheetHole {
  s: number; // 中心線の弧長位置 mm
  y: number; // 幅方向 mm（中心 0）
  shape: "circle" | "slot" | "poly";
  d?: number; // circle の直径
  w?: number; // slot の幅
  l?: number; // slot の長さ（中心線方向）
  pts?: Vec2[]; // poly（(s, y) 平面）
}

export interface PartParams {
  // box
  size?: Vec3;
  // cylinder / frustum
  radius?: number;
  radius_bottom?: number;
  radius_top?: number;
  height?: number;
  segments?: number;
  // extrude
  outer?: Vec2[];
  holes?: Vec2[][];
  thickness?: number;
  symmetric?: boolean;
  // revolve
  profile?: Vec2[]; // (r, z)
  angle_deg?: number;
  // sheet
  path?: Vec2[]; // (x, z)
  width?: number;
  bend_radius?: number;
  sheet_holes?: SheetHole[];
}

export interface Part {
  id: string;
  name: string;
  op: "add" | "subtract";
  visible: boolean;
  transform: Transform;
  kind: PartKind;
  params: PartParams;
}

export interface Measurement {
  id: string;
  label: string;
  part: string; // Part.id
  param: string; // PartParams のキー（配列なら "size.0" のように添字）
  value_mm: number;
  locked: boolean;
}

export interface MarkerObs {
  id: number;
  corners_raw: Vec2[];
  corners_und: Vec2[];
  source: "auto" | "manual";
  refine?: "edges" | "subpix"; // サーバーが付ける: 四隅の磨き方
}

export interface PoseQuality {
  reproj_px?: number;
  ambiguity?: number;
  note?: string;
}

export interface ImagePose {
  T_cam_marker: Mat4 | null;
  source: "marker" | "manual" | "pnp" | "none";
  locked: boolean;
  quality: PoseQuality;
}

export interface Contour {
  id: string;
  kind: "outer" | "hole";
  points_und: Vec2[];
  status: "candidate" | "confirmed";
}

export interface Correspondence {
  uv_und: Vec2;
  p_obj: Vec3;
}

export interface ImageQuality {
  blur_var: number;
  overexposed_pct: number;
  underexposed_pct: number;
  ok: boolean;
  notes: string[];
}

export interface DepthInfo {
  file: string;
  unit: "mm" | "m";
  invalid: number;
  registered_to: "rgb_und" | "rgb_raw";
}

export interface ProjectImage {
  id: string;
  file: string;
  camera: string;
  width: number;
  height: number;
  orientation: number;
  role: "object" | "background";
  quality: ImageQuality;
  markers: MarkerObs[];
  pose: ImagePose;
  contours: Contour[];
  correspondences: Correspondence[];
  depth: DepthInfo | null;
  has_labels?: boolean; // サーバーが付ける: labels.png（人の修正）があるか
  import_info?: ImportInfo; // サーバーが書く: 取り込み時の情報
  preprocess_info?: PreprocessInfo; // サーバーが書く: 前処理の方式と注記
}

export interface ImportInfo {
  filename: string;
  exif_orientation: number; // 取り込み時の EXIF の向き（画素には適用済み）
  bytes: number;
  sha256: string;
}

export interface PreprocessInfo {
  mask_method?: "background_diff" | "grabcut" | "background_image";
  notes: string[];
  steps: string[];
  updated: string;
}

export interface Project {
  version: 1;
  id?: string;
  name: string;
  units: { length: "mm"; angle: "deg" };
  frames_doc: string;
  marker: MarkerSpec;
  cameras: Record<string, Camera>;
  T_marker_obj: Mat4;
  T_marker_obj_locked: boolean;
  shape: { parts: Part[] };
  measurements: Measurement[];
  attributes: { mass_g: number | null; material: string; notes: string };
  background_image: string | null;
  images: ProjectImage[];
  settings: { autosave_s: number };
  autosave_available?: boolean;
}

// ラベル値（labels*.png）
export const LABEL = { UNKNOWN: 0, BACKGROUND: 1, OBJECT: 2, HOLE: 3, MARKER: 4 } as const;
