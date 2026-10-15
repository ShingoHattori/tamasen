// 前処理モードの道具: ブラシ・ラベル保存・GrabCut・輪郭・マーカー四隅。
import type { Contour, ImagePose, MarkerObs, ProjectImage, Vec2 } from "../types";
import { api } from "./api";
import { h, clear, fmt, lockIcon, deferred } from "./dom";
import { dropLabels, getLabels, labelsRedo, labelsUndo, loadLabels, saveLabels } from "./labels";
import { busy, normalize } from "./shell";
import { store, type PrepTool } from "./store";

export function poseFromResponse(res: unknown): ImagePose | null {
  const r = res as Record<string, unknown> | null;
  if (!r) return null;
  const cand = (r.pose ?? (r.image as Record<string, unknown> | undefined)?.pose ?? r) as Partial<ImagePose>;
  if (cand && Array.isArray(cand.T_cam_marker)) {
    return { T_cam_marker: cand.T_cam_marker, source: cand.source ?? "marker", locked: !!cand.locked, quality: cand.quality ?? {} };
  }
  return null;
}

/** サーバーが返した Image（または一部）をクライアントの状態へ取り込む。 */
export function mergeImage(img: ProjectImage, res: unknown, keys: (keyof ProjectImage)[]): boolean {
  const r = res as Record<string, unknown> | null;
  if (!r) return false;
  const src = (r.image && typeof r.image === "object" ? r.image : r) as Partial<ProjectImage>;
  let any = false;
  for (const k of keys) {
    if (src[k] !== undefined) {
      (img as unknown as Record<string, unknown>)[k] = JSON.parse(JSON.stringify(src[k]));
      any = true;
    }
  }
  return any;
}

/** 歪みモデル（OpenCV: k1,k2,p1,p2,k3）で歪み補正後の画素 → 原画像の画素。 */
export function distortPoint(uv: Vec2, K_und: number[], K: number[], dist: number[]): Vec2 {
  const x = (uv[0] - K_und[2]) / K_und[0];
  const y = (uv[1] - K_und[5]) / K_und[4];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = dist;
  const r2 = x * x + y * y;
  const rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [K[0] * xd + K[2], K[4] * yd + K[5]];
}

export async function runMarkerPose(img: ProjectImage): Promise<void> {
  if (!store.pid || !store.project) return;
  if (img.pose.locked) {
    store.msg("この画像の姿勢は確定（ロック）されています。再推定するには「ロック解除」してください", "error");
    return;
  }
  const m = img.markers.find((x) => x.id === store.project!.marker.id) ?? img.markers[0];
  if (!m) {
    store.msg("この画像にはマーカーの四隅がありません（前処理で手動指定できます）", "error");
    return;
  }
  const res = await busy("マーカーから姿勢", () => api.markerPose(store.pid!, img.id, { markers: img.markers, marker: store.project!.marker, corners_und: m.corners_und }));
  if (!res) return;
  const pose = poseFromResponse(res);
  if (!pose) {
    store.msg("サーバーの応答に姿勢がありません", "error");
    return;
  }
  const before = JSON.stringify(store.project.T_marker_obj);
  store.commit("マーカーから姿勢", () => {
    img.pose = { ...pose, locked: false };
  }, "pose");
  if (JSON.stringify(store.project.T_marker_obj) !== before) throw new Error("内部エラー: T_marker_obj が変わりました");
  store.msg(`マーカーから姿勢を求めました（再投影 ${fmt(pose.quality?.reproj_px, 2)} px）`, "ok");
}

export function buildPrepPanel(el: HTMLElement): void {
  document.addEventListener("corners-done", () => void finishCorners());
  const render = deferred(() => {
    clear(el);
    const img = store.image();
    const ui = store.ui;
    if (!store.project) return;
    if (!img) {
      el.append(h("p", { class: "note" }, "画像を選んでください"));
      return;
    }
    const L = getLabels(img.id);
    if (!L) void loadLabels(img.id);
    const tools: [PrepTool, string][] = [
      ["view", "表示のみ"],
      ["brush", "ブラシ"],
      ["contour", "輪郭編集"],
      ["corners", "マーカー四隅"],
    ];
    const roleSel = h("select", null, h("option", { value: "object", selected: img.role === "object" }, "物体"), h("option", { value: "background", selected: img.role === "background" }, "背景のみ"));
    roleSel.addEventListener("change", () => store.commit("画像の役割", () => (img.role = roleSel.value as ProjectImage["role"])));
    el.append(
      h("h3", null, `画像 ${img.id}`),
      h("div", { class: "row" }, h("span", null, "役割"), roleSel, h("span", { class: "note" }, `${img.width}×${img.height}`)),
      h("p", { class: "note" + (img.quality?.ok === false ? " warn" : "") }, `ぶれ指標 ${fmt(img.quality?.blur_var, 0)}、白飛び ${fmt(img.quality?.overexposed_pct, 1)}%、黒つぶれ ${fmt(img.quality?.underexposed_pct, 1)}%${img.quality?.notes?.length ? "（" + img.quality.notes.join("・") + "）" : ""}`),
      h("div", { class: "seg", id: "prep-tools" }, tools.map(([t, label]) => h("button", { class: ui.prepTool === t ? "active" : "", "data-tool": t, onclick: () => store.setUi({ prepTool: t, cornerDraft: [], selVertex: null }) }, label))),
    );

    // ブラシ
    const sec = h("section", { class: "tool-sec" });
    const brushes: [0 | 1 | 2 | 3, string, string][] = [
      [2, "前景", "b-fg"],
      [1, "背景", "b-bg"],
      [3, "穴", "b-hole"],
      [0, "消しゴム", "b-erase"],
    ];
    const radius = h("input", { type: "range", min: "1", max: "80", value: String(ui.brushRadius), id: "brush-radius" });
    radius.addEventListener("input", () => {
      store.ui.brushRadius = parseInt(radius.value);
      rlabel.textContent = `${radius.value} px`;
    });
    const rlabel = h("span", null, `${ui.brushRadius} px`);
    sec.append(
      h("h4", null, "マスク（ラベル）"),
      h("div", { class: "seg" }, brushes.map(([v, label, cls]) => h("button", { class: `${cls} ${ui.prepTool === "brush" && ui.brushLabel === v ? "active" : ""}`, "data-brush": String(v), onclick: () => store.setUi({ prepTool: "brush", brushLabel: v }) }, label))),
      h("div", { class: "row" }, h("span", null, "半径"), radius, rlabel),
      h("p", { class: "note" }, L ? `表示中: ${L.source === "labels" ? "修正済み（labels.png）" : L.source === "labels_auto" ? "自動候補（labels_auto.png）" : "なし"}${L.dirty ? "・未保存の塗りあり" : ""}` : "ラベルを読み込み中…"),
      h(
        "div",
        { class: "row wrap" },
        h("button", { disabled: !L?.undo.length, onclick: () => L && labelsUndo(L) }, "塗りを戻す"),
        h("button", { disabled: !L?.redo.length, onclick: () => L && labelsRedo(L) }, "塗りをやり直す"),
        h("button", { id: "btn-save-labels", class: L?.dirty ? "primary" : "", disabled: !L, onclick: async () => {
          if (!L) return;
          if (await busy("ラベルの保存", () => saveLabels(L))) store.msg("ラベルを保存しました", "ok");
        } }, "ラベルを保存"),
        h("button", { id: "btn-reset-labels", onclick: async () => {
          const r = await busy("候補に戻す", () => api.deleteLabels(store.pid!, img.id));
          if (r === undefined) return;
          img.has_labels = false;
          store.bumpFiles(img.id);
          dropLabels(img.id);
          await loadLabels(img.id, true);
          store.msg("自動候補に戻しました", "ok");
        } }, "候補に戻す"),
        h("button", { id: "btn-grabcut", onclick: () => void grabcut(img) }, "GrabCut 再計算"),
      ),
    );
    el.append(sec);

    // 輪郭
    const csec = h("section", { class: "tool-sec" });
    csec.append(
      h("h4", null, "輪郭（外周・穴）"),
      h("div", { class: "row wrap" },
        h("button", { id: "btn-make-contours", onclick: () => void makeContours(img) }, "輪郭を作る"),
        h("button", { id: "btn-confirm-all", disabled: !img.contours.some((c) => c.status === "candidate"), onclick: () => store.commit("輪郭をすべて確定", () => img.contours.forEach((c) => (c.status = "confirmed"))) }, "すべて確定"),
        h("button", { onclick: () => void reprocess(img) }, "前処理をやり直す"),
      ),
    );
    const list = h("table", { class: "tbl", id: "contour-table" }, h("tr", null, h("th", null, "ID"), h("th", null, "種類"), h("th", null, "頂点"), h("th", null, "状態"), h("th", null, "")));
    for (const c of img.contours) list.append(contourRow(img, c));
    if (!img.contours.length) list.append(h("tr", null, h("td", { colspan: "5", class: "note" }, "輪郭はまだありません")));
    csec.append(list, h("p", { class: "note" }, "橙 = 外周、桃 = 穴。破線は候補、実線は確定。"));
    el.append(csec);

    // マーカー
    const msec = h("section", { class: "tool-sec" });
    msec.append(h("h4", null, "マーカー"));
    if (img.markers.length)
      for (const m of img.markers) msec.append(h("p", { class: "note" }, `ID ${m.id}（${m.source === "manual" ? "手動" : "自動検出"}）四隅 ${m.corners_und.map((p) => `(${fmt(p[0], 0)},${fmt(p[1], 0)})`).join(" ")}`));
    else msec.append(h("p", { class: "note" }, "マーカーは見つかっていません"));
    msec.append(
      h("div", { class: "row wrap" },
        h("button", { id: "btn-corners", class: ui.prepTool === "corners" ? "active" : "", onclick: () => store.setUi({ prepTool: "corners", cornerDraft: [] }) }, "四隅を手動で指定"),
        h("button", { id: "btn-marker-pose", disabled: !img.markers.length || img.pose.locked, onclick: () => void runMarkerPose(img) }, "マーカーから姿勢"),
        img.pose.locked ? h("span", { class: "note" }, lockIcon(), " 姿勢は確定済み") : null,
      ),
      ui.prepTool === "corners" ? h("p", { class: "guide-inline" }, `${ui.cornerDraft.length}/4 — 左上→右上→右下→左下 の順に、黒い正方形の外側の角をクリック（Esc でやり直し）`) : "",
    );
    el.append(msec);
  });
  store.on((k) => {
    if (k !== "pose") render();
  });
  render();
}

function contourRow(img: ProjectImage, c: Contour): HTMLElement {
  const sel = store.ui.selContour === c.id;
  const kind = h("select", null, h("option", { value: "outer", selected: c.kind === "outer" }, "外周"), h("option", { value: "hole", selected: c.kind === "hole" }, "穴"));
  kind.addEventListener("change", () => store.commit("輪郭の種類", () => (c.kind = kind.value as Contour["kind"])));
  return h(
    "tr",
    { class: sel ? "sel" : "", "data-contour": c.id },
    h("td", null, h("a", { href: "#", onclick: (e: Event) => {
      e.preventDefault();
      store.setUi({ selContour: c.id, prepTool: store.ui.prepTool === "view" ? "contour" : store.ui.prepTool });
    } }, c.id)),
    h("td", null, kind),
    h("td", null, String(c.points_und.length)),
    h("td", null, h("button", { class: "small " + (c.status === "confirmed" ? "ok" : ""), "data-action": "toggle-confirm", onclick: () => store.commit("輪郭の確定", () => (c.status = c.status === "confirmed" ? "candidate" : "confirmed")) }, c.status === "confirmed" ? "確定済み（解除）" : "確定する")),
    h("td", null, h("button", { class: "small danger", onclick: () => store.commit("輪郭の削除", () => {
      img.contours = img.contours.filter((x) => x.id !== c.id);
      if (store.ui.selContour === c.id) store.ui.selContour = null;
    }) }, "削除")),
  );
}

async function grabcut(img: ProjectImage): Promise<void> {
  const L = getLabels(img.id);
  if (!L || !store.pid) return;
  const fg = L.strokes.filter((s) => s.label === 2).map((s) => ({ points: s.points, radius: s.r }));
  const bg = L.strokes.filter((s) => s.label === 1).map((s) => ({ points: s.points, radius: s.r }));
  // 塗りの結果（labels.png）も種として使えるよう先に保存する
  if (L.dirty && !(await busy("ラベルの保存", () => saveLabels(L)))) return;
  const res = await busy("GrabCut", () => api.grabcut(store.pid!, img.id, { fg_strokes: fg, bg_strokes: bg }));
  if (res === undefined) return;
  img.has_labels = true;
  store.bumpFiles(img.id);
  dropLabels(img.id);
  await loadLabels(img.id, true);
  store.msg("GrabCut でマスクを作り直しました", "ok");
}

async function makeContours(img: ProjectImage): Promise<void> {
  const L = getLabels(img.id);
  if (!store.pid) return;
  if (L?.dirty && !(await busy("ラベルの保存", () => saveLabels(L)))) return;
  const res = (await busy("輪郭を作る", () => api.contoursFromLabels(store.pid!, img.id))) as Record<string, unknown> | undefined;
  if (!res) return;
  const contours = (Array.isArray(res) ? res : (res.contours ?? (res.image as Record<string, unknown> | undefined)?.contours)) as Contour[] | undefined;
  if (!contours) {
    store.msg("サーバーの応答に輪郭がありません", "error");
    return;
  }
  store.commit("輪郭を作る", () => {
    img.contours = contours.map((c) => ({ ...c, status: c.status ?? "candidate" }));
  });
  store.setUi({ selContour: null });
  store.msg(`輪郭候補 ${contours.length} 本（外周 ${contours.filter((c) => c.kind === "outer").length}、穴 ${contours.filter((c) => c.kind === "hole").length}）`, "ok");
}

async function reprocess(img: ProjectImage): Promise<void> {
  if (!store.pid) return;
  const res = await busy("前処理のやり直し", () => api.preprocess(store.pid!, img.id, ["undistort", "quality", "markers", "mask", "edges", "contours"]));
  if (res === undefined) return;
  store.commit("前処理のやり直し", () => {
    mergeImage(img, res, ["quality", "markers", "contours", "has_labels"]);
  });
  normalize(store.project!);
  store.bumpFiles(img.id);
  dropLabels(img.id);
  void loadLabels(img.id, true);
  store.msg("前処理をやり直しました", "ok");
}

async function finishCorners(): Promise<void> {
  const img = store.image();
  const p = store.project;
  if (!img || !p || store.ui.cornerDraft.length !== 4) return;
  const pts = store.ui.cornerDraft.map((c) => [c[0], c[1]] as Vec2);
  const cam = p.cameras[img.camera];
  const raw = cam && cam.K?.[0] && cam.K_und?.[0] ? pts.map((q) => distortPoint(q, cam.K_und, cam.K, cam.dist ?? [])) : pts.map((q) => [q[0], q[1]] as Vec2);
  const obs: MarkerObs = { id: p.marker.id, corners_und: pts, corners_raw: raw, source: "manual" };
  store.commit("マーカー四隅（手動）", () => {
    img.markers = [...img.markers.filter((m) => m.id !== p.marker.id), obs];
  });
  store.setUi({ cornerDraft: [], prepTool: "view" });
  store.msg("マーカー四隅を設定しました", "ok");
  if (!img.pose.locked) await runMarkerPose(img);
  else store.msg("姿勢はロック中のため再推定しません", "info");
}
