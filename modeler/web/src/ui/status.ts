// 状態表示（形状の統計・エラー・メッセージ）。
import { h, clear, fmt, deferred } from "./dom";
import { currentModel, geometryError, geometryReady } from "./model";
import { store } from "./store";

export function buildStatus(el: HTMLElement): void {
  const stats = h("span", { class: "stats", id: "model-stats" });
  const errs = h("span", { class: "errs", id: "model-errors" });
  const msg = h("span", { class: "msg", id: "status-msg" });
  el.append(stats, errs, msg);
  let timer = 0;
  store.setMessenger((text, kind = "info") => {
    msg.textContent = text;
    msg.className = "msg " + kind;
    window.clearTimeout(timer);
    if (kind !== "error") timer = window.setTimeout(() => (msg.textContent = ""), 6000);
    if (kind === "error") console.warn(text);
  });
  const render = deferred(() => {
    clear(stats);
    clear(errs);
    if (!store.project) return;
    if (!geometryReady()) {
      stats.textContent = geometryError() ?? "形状エンジンを読み込み中…";
      return;
    }
    const m = currentModel();
    const s = m?.stats;
    if (s) {
      stats.append(
        `体積 ${fmt(s.volume_mm3 / 1000, 2)} cm³`,
        h("span", { class: "sep" }),
        `表面積 ${fmt(s.area_mm2 / 100, 1)} cm²`,
        h("span", { class: "sep" }),
        `貫通穴の数 ${s.genus}`,
        h("span", { class: "sep" }),
        h("span", { class: s.watertight ? "ok" : "warn" }, `水密 ${s.watertight ? "はい" : "いいえ"}`),
        h("span", { class: "sep" }),
        `三角形 ${s.triangles}`,
        h("span", { class: "sep" }),
        `大きさ ${[0, 1, 2].map((i) => fmt(s.bbox_max[i] - s.bbox_min[i], 1)).join(" × ")} mm`,
      );
    } else stats.textContent = store.project.shape.parts.length ? "形状を作れません" : "部品なし";
    if (m?.errors?.length) errs.textContent = "形状エラー: " + m.errors.map((e) => `${e.partId ? e.partId + ": " : ""}${e.message}`).join(" / ");
  });
  store.on((k) => {
    if (k !== "pose" && k !== "labels") render();
  });
  render();
}
