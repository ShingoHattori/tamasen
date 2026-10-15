// 画像の一覧（サムネイル・品質・マーカー・姿勢の出所・ロック）。
import { fileUrl } from "./api";
import { h, clear, lockIcon, deferred } from "./dom";
import { store } from "./store";

export function buildStrip(el: HTMLElement): void {
  let lastKey = "";
  const render = deferred(() => {
    const p = store.project;
    const key = p
      ? JSON.stringify([store.pid, store.ui.selImage, p.marker.id, store.fileVersion, p.images.map((im) => [im.id, im.role, im.quality?.notes, im.markers.map((m) => m.id), im.pose.locked, im.pose.source, !!im.pose.T_cam_marker])])
      : "";
    if (key === lastKey) return;
    lastKey = key;
    clear(el);
    if (!p) return;
    if (!p.images.length) {
      el.append(h("div", { class: "note pad" }, "画像はまだありません"));
      return;
    }
    for (const im of p.images) {
      const notes = im.quality?.notes ?? [];
      const blur = notes.some((n) => n.includes("ぶれ"));
      const expo = notes.some((n) => n.includes("露出") || n.includes("白飛び") || n.includes("黒つぶれ")) || (im.quality?.overexposed_pct ?? 0) > 5 || (im.quality?.underexposed_pct ?? 0) > 20;
      const hasMarker = im.markers.some((m) => m.id === p.marker.id);
      const src = im.pose.T_cam_marker ? im.pose.source : "none";
      el.append(
        h(
          "div",
          { class: "thumb" + (im.id === store.ui.selImage ? " sel" : "") + (im.role === "background" ? " bg" : ""), "data-image": im.id, title: im.id, onclick: () => store.setUi({ selImage: im.id, selContour: null, selVertex: null, cornerDraft: [], corrPending: null }) },
          h("img", { src: fileUrl(store.pid!, im.id, "undistorted.png", store.fileVersion[im.id] ?? 0), loading: "lazy", alt: im.id }),
          h(
            "div",
            { class: "badges" },
            im.role === "background" ? h("span", { class: "badge" }, "背景") : null,
            blur ? h("span", { class: "badge warn" }, "ぶれ") : null,
            expo ? h("span", { class: "badge warn" }, "露出") : null,
            h("span", { class: "badge " + (hasMarker ? "ok" : "muted"), title: hasMarker ? "マーカーあり" : "マーカーなし" }, hasMarker ? "マーカー○" : "マーカー×"),
            h("span", { class: "badge src-" + src }, ({ marker: "姿勢:マーカー", manual: "姿勢:手動", pnp: "姿勢:対応点", none: "姿勢なし" } as Record<string, string>)[src]),
            im.pose.locked ? h("span", { class: "badge lock" }, lockIcon("姿勢を確定済み")) : null,
          ),
          h("div", { class: "thumb-id" }, im.id),
        ),
      );
    }
  });
  store.on((k) => {
    if (k !== "labels") render();
  });
  render();
}
