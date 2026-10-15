// 画面の組み立て。
import "./ui/style.css";
import { h } from "./ui/dom";
import { ImageView } from "./ui/imageView";
import { View3D } from "./ui/view3d";
import { buildHeader, buildSettings, importFiles, openProject, startAutosave } from "./ui/shell";
import { buildPrepPanel } from "./ui/prepPanel";
import { buildShapePanel } from "./ui/shapePanel";
import { buildAlignPanel } from "./ui/alignPanel";
import { buildStrip } from "./ui/strip";
import { buildStatus } from "./ui/status";
import { startGeometry } from "./ui/model";
import { store, type OverlayMode } from "./ui/store";

const app = document.getElementById("app")!;
const header = h("header", { id: "hdr" });
const settings = h("div", { id: "settings-body" });
const prep = h("div", { id: "tools-prep", class: "tools" });
const shape = h("div", { id: "tools-shape", class: "tools" });
const align = h("div", { id: "tools-align", class: "tools" });
const shapeRo = h("div", { id: "shape-readonly" });
const side = h(
  "aside",
  { id: "side" },
  h("details", { open: true, class: "settings" }, h("summary", null, "プロジェクト"), settings),
  prep,
  shape,
  h("div", { id: "align-wrap", class: "tools" }, align, h("details", { class: "ro-shape" }, h("summary", null, "形状（読み取り専用）"), shapeRo)),
);
const layerBar = h("div", { class: "toolbar", id: "layerbar" });
const imgEl = h("div", { id: "imgview" });
const strip = h("div", { id: "strip" });
const v3bar = h("div", { class: "toolbar", id: "v3bar" });
const v3El = h("div", { id: "view3d" });
const status = h("footer", { id: "status" });
app.append(header, h("div", { id: "main" }, side, h("section", { id: "center" }, layerBar, imgEl, strip), h("section", { id: "right" }, v3bar, v3El)), status);

buildStatus(status);
buildHeader(header);
buildSettings(settings);
buildPrepPanel(prep);
buildShapePanel(shape, false);
buildShapePanel(shapeRo, true);
buildAlignPanel(align);
buildStrip(strip);
const iv = new ImageView(imgEl);
const v3 = new View3D(v3El);

// 画像ビューの表示切替
const layerDefs: [keyof typeof store.ui.layers, string][] = [
  ["photo", "元画像"],
  ["mask", "マスク"],
  ["contours", "輪郭"],
  ["edges", "エッジ"],
  ["model", "モデル重ね合わせ"],
  ["silhouette", "シルエット比較"],
];
for (const [k, label] of layerDefs) {
  const cb = h("input", { type: "checkbox", checked: store.ui.layers[k], "data-layer": k });
  cb.addEventListener("change", () => {
    store.ui.layers[k] = cb.checked;
    store.emit("ui");
  });
  layerBar.append(h("label", { class: "chk" }, cb, label));
}
const ov = h(
  "select",
  { id: "overlay-mode", title: "モデルの重ね方" },
  ([["translucent", "半透明"], ["wire", "ワイヤー"], ["outline", "輪郭線"], ["hidden", "非表示"]] as [OverlayMode, string][]).map(([v, l]) => h("option", { value: v, selected: v === store.ui.overlay }, l)),
);
ov.addEventListener("change", () => store.setUi({ overlay: ov.value as OverlayMode }));
layerBar.append(h("span", { class: "sep" }), h("label", null, "モデル ", ov), h("button", { class: "small", onclick: () => iv.fit() }, "全体表示"), h("span", { class: "note" }, "ホイール=拡大、Space+ドラッグ/中ボタン=移動"));

const depthCb = h("input", { type: "checkbox", id: "depth-points" });
depthCb.addEventListener("change", () => store.setUi({ depthPoints: depthCb.checked }));
const ghostCb = h("input", { type: "checkbox", checked: store.ui.showGhost });
ghostCb.addEventListener("change", () => store.setUi({ showGhost: ghostCb.checked }));
v3bar.append(h("strong", null, "3D ビュー"), h("button", { class: "small", onclick: () => v3.resetView() }, "視点リセット"), h("label", { class: "chk" }, ghostCb, "選択部品"), h("label", { class: "chk" }, depthCb, "深度点群"), h("span", { class: "note" }, "赤 = 切削部品、黄 = カメラ（選択）、緑 = 確定"));

// ドラッグ＆ドロップで取り込み
imgEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  imgEl.classList.add("drop");
});
imgEl.addEventListener("dragleave", () => imgEl.classList.remove("drop"));
imgEl.addEventListener("drop", (e) => {
  e.preventDefault();
  imgEl.classList.remove("drop");
  const files = Array.from(e.dataTransfer?.files ?? []).filter((f) => f.type.startsWith("image/"));
  const role = (document.getElementById("import-role") as HTMLSelectElement | null)?.value ?? "object";
  if (files.length) void importFiles(files, role);
});

// モードごとの道具だけを表示
store.on(() => {
  const m = store.ui.mode;
  prep.hidden = m !== "prep";
  shape.hidden = m !== "shape";
  document.getElementById("align-wrap")!.hidden = m !== "align";
});
store.emit("ui");

startGeometry();
startAutosave();

const params = new URLSearchParams(location.search);
let last: string | null = params.get("project");
try {
  last ??= localStorage.getItem("modeler.lastProject");
} catch {
  /* 無視 */
}
if (last) void openProject(last);

// テスト・デバッグ用
(window as unknown as Record<string, unknown>).__modeler = { store, iv, v3 };
