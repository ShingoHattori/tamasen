// 位置合わせモード（選択画像のみ）: 数値での姿勢編集・対応点 PnP・確定（ロック）。
import type { Mat4, ProjectImage, Vec3 } from "../types";
import { api } from "./api";
import { h, clear, numInput, fmt, lockIcon, deferred } from "./dom";
import { mat4FromRt, mat4Mul, mat4Apply, transformFromMat4 } from "./geo";
import { modelCenterObj } from "./model";
import { commitPose, defaultPose, poseRefusal, rotateAbout, rotX, rotY, rotZ, trans, T_cam_obj } from "./pose";
import { runMarkerPose } from "./prepPanel";
import { busy } from "./shell";
import { store } from "./store";

interface PnpSolution {
  T_cam_marker: Mat4;
  rms_px?: number;
  residuals_px?: number[];
  ambiguous?: boolean;
  degenerate?: boolean;
  note?: string;
}

let solutions: { img: string; list: PnpSolution[]; warnings: string[] } | null = null;

function parseSolutions(res: unknown): { list: PnpSolution[]; warnings: string[] } {
  const r = (res ?? {}) as Record<string, unknown>;
  const raw = (Array.isArray(r.solutions) ? r.solutions : Array.isArray(res) ? res : []) as Record<string, unknown>[];
  const list = raw
    .filter((s) => Array.isArray(s.T_cam_marker))
    .map((s) => ({
      T_cam_marker: s.T_cam_marker as Mat4,
      rms_px: (s.rms_px ?? s.rms ?? s.reproj_px) as number | undefined,
      residuals_px: (s.residuals_px ?? s.residuals ?? s.per_point_px) as number[] | undefined,
      ambiguous: !!(s.ambiguous ?? r.ambiguous),
      degenerate: !!(s.degenerate ?? r.degenerate),
      note: (s.note ?? "") as string,
    }));
  const warnings: string[] = [];
  for (const k of ["warning", "warnings", "notes", "note", "message"]) {
    const v = r[k];
    if (typeof v === "string" && v) warnings.push(v);
    else if (Array.isArray(v)) warnings.push(...v.map(String));
  }
  if (r.degenerate) warnings.push("対応点の配置が退化しています（一直線・同一平面など）。点を増やすか散らしてください");
  if (r.ambiguous) warnings.push("解が一意に決まりません（候補が複数あります）。見た目で選んでください");
  return { list, warnings };
}

export function buildAlignPanel(el: HTMLElement): void {
  const render = deferred(() => {
    clear(el);
    const p = store.project;
    const img = store.image();
    if (!p) return;
    if (!img) {
      el.append(h("p", { class: "note" }, "画像を選んでください"));
      return;
    }
    const ui = store.ui;
    const locked = img.pose.locked;
    el.append(
      h("h3", null, `画像 ${img.id} の撮影姿勢`, locked ? lockIcon("確定済み") : null),
      h("p", { class: "note" }, `出所: ${srcLabel(img.pose.source)}${img.pose.quality?.reproj_px !== undefined ? `、再投影 ${fmt(img.pose.quality.reproj_px, 2)} px` : ""}${img.pose.quality?.note ? `、${img.pose.quality.note}` : ""}`),
      h("p", { class: "note" }, "ここで変わるのはこの画像の T_cam_marker だけです（形状・固定変換・他の画像は変わりません）。"),
      img.role === "background" ? h("p", { class: "note warn" }, "背景画像には姿勢がありません") : "",
      h(
        "div",
        { class: "row wrap" },
        locked
          ? h("button", { id: "btn-unlock", onclick: () => store.commit("ロック解除", () => (img.pose.locked = false), "pose") }, "ロック解除")
          : h("button", { id: "btn-lock", class: "primary", disabled: !img.pose.T_cam_marker, onclick: () => store.commit("姿勢の確定", () => (img.pose.locked = true), "pose") }, "確定（ロック）"),
        h("button", { id: "btn-reestimate", disabled: locked || !img.markers.length, title: img.markers.length ? "マーカーの四隅から姿勢を求め直す" : "マーカーが写っていません", onclick: () => void runMarkerPose(img) }, "再推定（マーカー）"),
        !img.pose.T_cam_marker ? h("button", { id: "btn-init-pose", disabled: locked, onclick: () => commitPose(img, defaultPose(), "manual") }, "初期姿勢を置く") : null,
      ),
      h("div", { class: "seg" },
        h("button", { class: ui.alignTool === "move" ? "active" : "", id: "tool-move", onclick: () => store.setUi({ alignTool: "move", corrPending: null }) }, "移動・回転"),
        h("button", { class: ui.alignTool === "corr" ? "active" : "", id: "tool-corr", onclick: () => store.setUi({ alignTool: "corr" }) }, "対応点"),
      ),
    );
    if (img.pose.T_cam_marker) el.append(numericPose(img));
    el.append(correspondences(img));
  });
  store.on((k) => {
    if (k !== "labels") render();
  });
  render();
}

function srcLabel(s: string): string {
  return ({ marker: "マーカー", manual: "手動", pnp: "対応点（PnP）", none: "なし" } as Record<string, string>)[s] ?? s;
}

function numericPose(img: ProjectImage): HTMLElement {
  const T = img.pose.T_cam_marker!;
  const tf = transformFromMat4(T);
  const ro = !!poseRefusal(img);
  const set = (t: Vec3, r: Vec3) => commitPose(img, mat4FromRt(r, t), "manual");
  const g = h("div", { class: "grid4", id: "pose-fields" }, h("span", null, ""), h("span", { class: "note" }, "x"), h("span", { class: "note" }, "y"), h("span", { class: "note" }, "z"));
  g.append(h("span", null, "位置 mm"));
  for (let i = 0; i < 3; i++)
    g.append(numInput(tf.t_mm[i], (v) => {
      const t = tf.t_mm.slice() as Vec3;
      t[i] = v;
      set(t, tf.r_deg);
    }, { step: 1, cls: `pt${i}`, readonly: ro, title: ro ? "この画像の姿勢は変更できません" : "" }));
  g.append(h("span", null, "回転 deg"));
  for (let i = 0; i < 3; i++)
    g.append(numInput(tf.r_deg[i], (v) => {
      const r = tf.r_deg.slice() as Vec3;
      r[i] = v;
      set(tf.t_mm, r);
    }, { step: 1, cls: `pr${i}`, readonly: ro, title: ro ? "この画像の姿勢は変更できません" : "" }));
  // 物体中心まわりの微調整ボタン（cam 座標）
  const nudge = (D: Mat4, rot: boolean) => {
    const p = store.project!;
    const pc = mat4Apply(T_cam_obj(p, img)!, modelCenterObj()) as Vec3;
    commitPose(img, rot ? rotateAbout(T, pc, D) : mat4Mul(D, T), "manual");
  };
  const b = (label: string, D: Mat4, rot: boolean) => h("button", { class: "small", disabled: ro, onclick: () => nudge(D, rot) }, label);
  return h(
    "section",
    { class: "tool-sec" },
    h("h4", null, "T_cam_marker（cam 座標、Trans·Rz·Ry·Rx）"),
    g,
    h("div", { class: "row wrap" }, h("span", { class: "note" }, "移動"), b("←", trans([-1, 0, 0]), false), b("→", trans([1, 0, 0]), false), b("↑", trans([0, -1, 0]), false), b("↓", trans([0, 1, 0]), false), b("近く", trans([0, 0, -5]), false), b("遠く", trans([0, 0, 5]), false)),
    h("div", { class: "row wrap" }, h("span", { class: "note" }, "回転"), b("視線 −", rotZ(-2), true), b("視線 +", rotZ(2), true), b("縦 −", rotX(-2), true), b("縦 +", rotX(2), true), b("横 −", rotY(-2), true), b("横 +", rotY(2), true)),
  );
}

function correspondences(img: ProjectImage): HTMLElement {
  const sec = h("section", { class: "tool-sec", id: "corr-sec" }, h("h4", null, `対応点（${img.correspondences.length} 組）`));
  const tbl = h("table", { class: "tbl" }, h("tr", null, h("th", null, "#"), h("th", null, "画像 u, v"), h("th", null, "モデル x, y, z mm"), h("th", null, "")));
  img.correspondences.forEach((c, i) =>
    tbl.append(
      h(
        "tr",
        null,
        h("td", null, String(i + 1)),
        h("td", null, `${fmt(c.uv_und[0], 1)}, ${fmt(c.uv_und[1], 1)}`),
        h("td", null, c.p_obj.map((v) => fmt(v, 1)).join(", ")),
        h("td", null, h("button", { class: "small danger", onclick: () => store.commit("対応点の削除", () => img.correspondences.splice(i, 1)) }, "×")),
      ),
    ),
  );
  sec.append(
    tbl,
    h("p", { class: "note" }, "「対応点」道具で、画像上の点 → モデル上の点（画像の重ね合わせ上か 3D ビュー）の順にクリック。4 組以上（6 組以上を推奨）。"),
    h(
      "div",
      { class: "row wrap" },
      h("button", { id: "btn-pnp", class: "primary", disabled: img.correspondences.length < 4, onclick: () => void runPnp(img) }, "対応点で推定"),
      h("button", { disabled: !img.correspondences.length, onclick: () => store.commit("対応点の全削除", () => (img.correspondences = [])) }, "全削除"),
    ),
  );
  if (solutions && solutions.img === img.id) {
    const box = h("div", { class: "solutions", id: "pnp-solutions" }, h("h5", null, `推定結果（${solutions.list.length} 件、まだ適用していません）`));
    for (const w of solutions.warnings) box.append(h("p", { class: "warn" }, w));
    if (!solutions.list.length) box.append(h("p", { class: "warn" }, "解が得られませんでした"));
    solutions.list.forEach((s, i) => {
      const tf = transformFromMat4(s.T_cam_marker);
      box.append(
        h(
          "div",
          { class: "solution" },
          h("strong", null, `解 ${i + 1}`),
          ` RMS ${fmt(s.rms_px, 2)} px`,
          s.ambiguous ? h("span", { class: "badge warn" }, "あいまい") : null,
          s.degenerate ? h("span", { class: "badge warn" }, "退化") : null,
          h("div", { class: "note" }, `位置 ${tf.t_mm.map((v) => fmt(v, 1)).join(", ")} mm / 回転 ${tf.r_deg.map((v) => fmt(v, 1)).join(", ")} deg`),
          s.residuals_px ? h("div", { class: "note" }, "点ごとの残差 px: " + s.residuals_px.map((r, j) => `${j + 1}:${fmt(r, 2)}`).join("  ")) : null,
          s.note ? h("div", { class: "note" }, s.note) : null,
          h("button", { class: "small primary", "data-solution": String(i), disabled: img.pose.locked, onclick: () => {
            if (commitPose(img, s.T_cam_marker, "pnp", { reproj_px: s.rms_px, note: s.ambiguous ? "対応点（あいまい）" : "対応点" })) {
              solutions = null;
              store.msg(`解 ${i + 1} を適用しました`, "ok");
              store.emit("ui");
            }
          } }, "この解を適用"),
        ),
      );
    });
    box.append(h("button", { class: "small", onclick: () => {
      solutions = null;
      store.emit("ui");
    } }, "キャンセル（手動に戻る）"));
    sec.append(box);
  }
  return sec;
}

async function runPnp(img: ProjectImage): Promise<void> {
  if (!store.pid || !store.project) return;
  const why = poseRefusal(img);
  if (why) {
    store.msg(why, "error");
    return;
  }
  const res = await busy("対応点で推定", () =>
    api.pnp(store.pid!, img.id, { correspondences: img.correspondences, T_marker_obj: store.project!.T_marker_obj, T_init: img.pose.T_cam_marker }),
  );
  if (res === undefined) return;
  const s = parseSolutions(res);
  solutions = { img: img.id, ...s };
  store.emit("ui");
}


