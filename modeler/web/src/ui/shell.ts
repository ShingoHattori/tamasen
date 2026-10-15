// 画面上部（プロジェクト・サンプル・保存・書き出し・Undo/Redo・モード）とプロジェクト設定。
import type { Project } from "../types";
import { api } from "./api";
import { exportGLB } from "./geo";
import { h, clear, modal, confirmDlg, fmt, deferred } from "./dom";
import { dropLabels, getLabels, saveLabels } from "./labels";
import { store, type Mode } from "./store";

export function busy<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
  document.body.classList.add("busy");
  store.msg(label + "…", "info");
  return fn().then(
    (v) => {
      document.body.classList.remove("busy");
      return v;
    },
    (e: Error) => {
      document.body.classList.remove("busy");
      store.msg(`${label}: ${e.message}`, "error");
      return undefined;
    },
  );
}

export async function openProject(pid: string): Promise<void> {
  if (store.dirty && !(await confirmDlg("未保存の変更", "保存していない変更があります。破棄して開きますか？", "破棄して開く", "やめる"))) return;
  const p = await busy("プロジェクトを開く", () => api.getProject(pid));
  if (!p) return;
  let proj: Project = p;
  let restored = false;
  if (p.autosave_available) {
    const yes = await confirmDlg("自動保存があります", "保存した内容より新しい自動保存があります。自動保存を復元しますか？（「いいえ」なら保存済みの内容を開きます）", "復元する", "いいえ");
    if (yes) {
      const a = await busy("自動保存の読み込み", () => api.getAutosave(pid));
      if (a) {
        proj = a;
        restored = true;
      }
    }
  }
  dropLabels();
  store.setProject(normalize(proj), pid);
  if (restored) {
    store.dirty = true;
    store.msg("自動保存を復元しました（「保存」で確定します）", "ok");
  } else store.msg(`「${proj.name}」を開きました`, "ok");
  try {
    localStorage.setItem("modeler.lastProject", pid);
  } catch {
    /* 無視 */
  }
  void refreshProjectList();
}

export function normalize(p: Project): Project {
  p.shape ??= { parts: [] };
  p.shape.parts ??= [];
  p.measurements ??= [];
  p.images ??= [];
  p.cameras ??= {};
  p.settings ??= { autosave_s: 20 };
  p.attributes ??= { mass_g: null, material: "", notes: "" };
  p.T_marker_obj ??= [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (const im of p.images) {
    im.contours ??= [];
    im.correspondences ??= [];
    im.markers ??= [];
    im.pose ??= { T_cam_marker: null, source: "none", locked: false, quality: {} };
    im.pose.quality ??= {};
  }
  return p;
}

export async function saveProject(): Promise<boolean> {
  const p = store.project,
    pid = store.pid;
  if (!p || !pid) return false;
  // 未保存のラベルも一緒に保存
  for (const im of p.images) {
    const L = getLabels(im.id);
    if (L?.dirty && !(await busy("ラベルの保存", () => saveLabels(L)))) return false;
  }
  const body = { ...p };
  delete body.autosave_available;
  const r = await busy("保存", () => api.saveProject(pid, body));
  if (r === undefined) return false;
  store.markSaved();
  store.msg("保存しました", "ok");
  return true;
}

let lastAutosave = Date.now();
export function startAutosave(): void {
  setInterval(() => {
    const p = store.project;
    if (!p || !store.pid || !store.autosaveDirty || store.inGesture) return;
    const every = Math.max(3, p.settings?.autosave_s || 20) * 1000;
    if (Date.now() - lastAutosave < every) return;
    lastAutosave = Date.now();
    const body = { ...p };
    delete body.autosave_available;
    store.autosaveDirty = false;
    api.saveProject(store.pid, body, true).then(
      () => store.msg("自動保存しました", "info"),
      (e: Error) => {
        store.autosaveDirty = true;
        store.msg(`自動保存に失敗: ${e.message}`, "error");
      },
    );
  }, 1000);
}

// ---------- ヘッダー ----------
let projSel: HTMLSelectElement;
let sampleSel: HTMLSelectElement;

export async function refreshProjectList(): Promise<void> {
  try {
    const list = await api.listProjects();
    clear(projSel);
    projSel.append(h("option", { value: "" }, "（プロジェクトを選ぶ）"));
    for (const p of list) projSel.append(h("option", { value: p.id, selected: p.id === store.pid }, `${p.name}（${p.id}）`));
  } catch (e) {
    store.msg(`プロジェクト一覧を取得できません: ${(e as Error).message}`, "error");
  }
}

async function refreshSamples(): Promise<void> {
  try {
    const list = await api.listSamples();
    clear(sampleSel);
    sampleSel.append(h("option", { value: "" }, "（サンプル）"));
    for (const s of list) {
      const name = typeof s === "string" ? s : s.name;
      const title = typeof s === "string" ? s : s.title || s.name;
      sampleSel.append(h("option", { value: name, title: typeof s === "string" ? "" : s.description ?? "" }, title));
    }
  } catch (e) {
    store.msg(`サンプル一覧を取得できません: ${(e as Error).message}`, "error");
  }
}

export function buildHeader(el: HTMLElement): void {
  projSel = h("select", { id: "project-select", title: "プロジェクト" });
  projSel.addEventListener("change", () => {
    if (projSel.value) void openProject(projSel.value);
  });
  sampleSel = h("select", { id: "sample-select" });
  const newBtn = h("button", {
    id: "btn-new",
    onclick: async () => {
      const inp = h("input", { type: "text", value: "新しい物体", id: "new-project-name" });
      const r = await modal("新規プロジェクト", h("label", null, "名前 ", inp), [
        { label: "やめる", value: "no" },
        { label: "作成", value: "yes", primary: true },
      ]);
      if (r !== "yes" || !inp.value.trim()) return;
      const p = await busy("プロジェクトの作成", () => api.createProject(inp.value.trim()));
      if (p?.id) await openProject(p.id);
    },
  }, "新規");
  const sampleBtn = h("button", {
    id: "btn-load-sample",
    onclick: async () => {
      if (!sampleSel.value) {
        store.msg("サンプルを選んでください", "error");
        return;
      }
      const p = await busy("サンプルの読み込み", () => api.loadSample(sampleSel.value));
      if (p?.id) {
        await refreshProjectList();
        await openProject(p.id);
      }
    },
  }, "サンプルを読み込む");
  const undoBtn = h("button", { id: "btn-undo", title: "元に戻す（Ctrl+Z）", onclick: () => store.undo() }, "元に戻す");
  const redoBtn = h("button", { id: "btn-redo", title: "やり直す（Ctrl+Shift+Z）", onclick: () => store.redo() }, "やり直す");
  const saveBtn = h("button", { id: "btn-save", class: "primary", onclick: () => void saveProject() }, "保存");
  const exportBtn = h("button", { id: "btn-export", onclick: () => void exportDialog() }, "書き出し");
  const modes: [Mode, string, string][] = [
    ["prep", "1 前処理", "画像ごと: マスク・輪郭・マーカー四隅"],
    ["shape", "2 形状編集", "全画像共通: 部品・寸法・押し出し・固定変換"],
    ["align", "3 位置合わせ", "選択画像のみ: その画像の撮影姿勢"],
  ];
  const modeBtns = modes.map(([m, label, tip]) => h("button", { class: "mode-btn", "data-mode": m, title: tip, onclick: () => setMode(m) }, label));
  const nameEl = h("span", { class: "proj-name" });
  el.append(
    h("div", { class: "hdr-row" }, h("strong", { class: "app-title" }, "modeler"), projSel, newBtn, sampleSel, sampleBtn, h("span", { class: "sep" }), undoBtn, redoBtn, saveBtn, exportBtn, nameEl),
    h("div", { class: "mode-row" }, h("span", { class: "mode-label" }, "モード"), modeBtns, h("span", { class: "mode-desc" })),
  );
  const update = () => {
    const ui = store.ui;
    for (const b of modeBtns) b.classList.toggle("active", b.dataset.mode === ui.mode);
    const d = el.querySelector(".mode-desc")!;
    d.textContent = modes.find((m) => m[0] === ui.mode)![2];
    undoBtn.disabled = !store.undoStack.length;
    redoBtn.disabled = !store.redoStack.length;
    saveBtn.disabled = !store.project;
    exportBtn.disabled = !store.project;
    nameEl.textContent = store.project ? `${store.project.name}${store.dirty ? "（未保存）" : ""}` : "";
    document.body.dataset.mode = ui.mode;
  };
  store.on(update);
  update();
  window.addEventListener("keydown", (e) => {
    const t = e.target as HTMLElement;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) store.redo();
      else store.undo();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
      e.preventDefault();
      store.redo();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      void saveProject();
    }
  });
  window.addEventListener("beforeunload", (e) => {
    if (store.dirty) e.preventDefault();
  });
  void refreshProjectList();
  void refreshSamples();
}

export function setMode(m: Mode): void {
  if (store.inGesture) return;
  store.setUi({ mode: m, cornerDraft: [], corrPending: null, selVertex: null, planePreview: m === "shape" ? store.ui.planePreview : null });
}

async function exportDialog(): Promise<void> {
  const p = store.project,
    pid = store.pid;
  if (!p || !pid) return;
  const sel = h("select", { id: "export-coords" }, h("option", { value: "marker", selected: true }, "marker 座標（既定）"), h("option", { value: "obj" }, "obj 座標"));
  const r = await modal(
    "書き出し",
    h("div", null, h("p", null, "GLB（m 単位、穴付きの表示形状）と model.json を書き出します。書き出しの前にプロジェクトを保存します。"), h("label", null, "座標 ", sel), h("p", { class: "note" }, "衝突形状は未対応です（凸包にはしません）。")),
    [
      { label: "やめる", value: "no" },
      { label: "書き出す", value: "yes", primary: true },
    ],
  );
  if (r !== "yes") return;
  const frame = sel.value as "marker" | "obj";
  if (!(await saveProject())) return;
  const glb = await busy("GLB の作成", () => exportGLB(p.shape.parts, p.T_marker_obj, frame));
  if (!glb) return;
  const res = (await busy("書き出し", () => api.exportModel(pid, glb, frame))) as Record<string, unknown> | undefined;
  if (!res) return;
  const model = (res.model ?? res.model_json) as Record<string, unknown> | undefined;
  const verification = res.verification ?? model?.verification;
  const collision = (res.collision ?? model?.collision) as { status?: string; reason?: string } | undefined;
  const files = res.files as { glb?: string; json?: string } | undefined;
  const body = h(
    "div",
    { id: "export-result" },
    h("p", null, `保存先: ${String(res.dir ?? res.path ?? res.export_dir ?? "exports/")}`),
    h("p", null, `GLB: ${fmt(glb.byteLength / 1024, 1)} KB`, files?.glb ? [" ", h("a", { href: files.glb, download: "model.glb" }, "model.glb")] : "", files?.json ? [" ", h("a", { href: files.json, target: "_blank" }, "model.json")] : ""),
    h("h4", null, "検証"),
    h("pre", null, verification !== undefined ? JSON.stringify(verification, null, 2) : "（サーバーから検証結果が返りませんでした）"),
    h("h4", null, "衝突形状"),
    h("p", { class: "note" }, collision ? `${collision.status === "not_generated" ? "生成していません（not_generated）" : collision.status}: ${collision.reason ?? ""}` : "生成していません（not_generated）"),
  );
  await modal("書き出し完了", body, [{ label: "閉じる", value: "ok", primary: true }]);
  store.msg("書き出しました", "ok");
}

// ---------- プロジェクト設定（サイドバー上部） ----------
export function buildSettings(el: HTMLElement): void {
  const render = deferred(() => {
    clear(el);
    const p = store.project;
    if (!p) {
      el.append(h("p", { class: "note" }, "プロジェクトを開くか、新規作成・サンプル読み込みをしてください。"));
      return;
    }
    const role = h("select", { id: "import-role" }, h("option", { value: "object" }, "物体"), h("option", { value: "background" }, "背景のみ"));
    const fileInp = h("input", { type: "file", multiple: true, accept: "image/*", id: "import-files" });
    fileInp.addEventListener("change", () => {
      const files = Array.from(fileInp.files ?? []);
      if (files.length) void importFiles(files, role.value);
      fileInp.value = "";
    });
    const camInp = h("input", { type: "file", accept: ".json,application/json", id: "camera-file" });
    camInp.addEventListener("change", async () => {
      const f = camInp.files?.[0];
      if (!f || !store.pid) return;
      const fd = new FormData();
      fd.append("file", f, f.name);
      const res = await busy("校正ファイルの適用", () => api.setCamera(store.pid!, fd));
      camInp.value = "";
      if (res) await reloadAfterServerChange("カメラの内部パラメータを設定し、歪み補正をやり直しました");
    });
    const cams = Object.entries(p.cameras);
    const marker = p.marker;
    const dict = h(
      "select",
      { id: "marker-dict" },
      ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_50", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"].map((d) => h("option", { value: d, selected: d === marker.dictionary }, d)),
    );
    dict.addEventListener("change", () => store.commit("マーカー辞書", (pp) => (pp.marker.dictionary = dict.value)));
    const mid = h("input", { type: "number", class: "num", value: String(marker.id), step: "1", min: "0", id: "marker-id" });
    mid.addEventListener("change", () => store.commit("マーカー ID", (pp) => (pp.marker.id = parseInt(mid.value) || 0)));
    const msz = h("input", { type: "number", class: "num", value: String(marker.size_mm), step: "0.1", min: "0", id: "marker-size" });
    msz.addEventListener("change", () => {
      const v = parseFloat(msz.value);
      if (v > 0) store.commit("マーカー寸法", (pp) => (pp.marker.size_mm = v));
    });
    const bg = h("select", { id: "background-image" }, h("option", { value: "" }, "（なし）"), p.images.filter((i) => i.role === "background").map((i) => h("option", { value: i.id, selected: p.background_image === i.id }, i.id)));
    bg.addEventListener("change", () => store.commit("背景画像", (pp) => (pp.background_image = bg.value || null)));
    const autos = h("input", { type: "number", class: "num", value: String(p.settings.autosave_s), min: "3", step: "1" });
    autos.addEventListener("change", () => store.commit("自動保存間隔", (pp) => (pp.settings.autosave_s = Math.max(3, parseInt(autos.value) || 20))));
    const mass = h("input", { type: "number", class: "num", value: p.attributes.mass_g ?? "", step: "0.1", placeholder: "g" });
    mass.addEventListener("change", () => store.commit("質量", (pp) => (pp.attributes.mass_g = mass.value === "" ? null : parseFloat(mass.value))));
    const mat = h("input", { type: "text", value: p.attributes.material });
    mat.addEventListener("change", () => store.commit("材質", (pp) => (pp.attributes.material = mat.value)));
    el.append(
      h(
        "div",
        { class: "row" },
        h("label", { class: "btn-like" }, "画像を取り込む", fileInp),
        role,
        h(
          "button",
          {
            id: "btn-capture",
            onclick: async () => {
              const r = await busy("OAK から撮影", () => api.capture(store.pid!));
              if (r) await reloadAfterServerChange("OAK から 1 枚取り込みました");
            },
          },
          "OAK から撮影",
        ),
      ),
      h("p", { class: "note" }, "画像は画像ビューへドラッグ＆ドロップでも取り込めます。"),
      h("details", { class: "sub" }, h("summary", null, "マーカー・カメラ・属性"),
        h("div", { class: "grid2" },
          h("span", null, "辞書"), dict,
          h("span", null, "ID"), mid,
          h("span", null, "一辺 mm"), msz,
          h("span", null, "背景画像"), bg,
          h("span", null, "自動保存 s"), autos,
          h("span", null, "質量 g"), mass,
          h("span", null, "材質"), mat,
        ),
        h("div", { class: "row" }, h("label", { class: "btn-like" }, "校正ファイル（JSON）", camInp),
          h("button", {
            onclick: async () => {
              const r = await busy("vision から内部パラメータ", () => api.setCamera(store.pid!, { from: "vision" }));
              if (r) await reloadAfterServerChange("vision から内部パラメータを設定しました");
            },
          }, "vision から取得")),
        cams.length
          ? cams.map(([name, c]) => h("p", { class: "note" }, `カメラ ${name}: ${c.width}×${c.height}、fx ${fmt(c.K_und?.[0], 1)}、出所 ${c.source}`))
          : h("p", { class: "note warn" }, "カメラの内部パラメータがありません（重ね合わせ・姿勢推定に必要）"),
      ),
    );
  });
  store.on((k) => {
    if (k === "load" || k === "project") render();
  });
  render();
}

export async function importFiles(files: File[], role = "object"): Promise<void> {
  if (!store.pid) {
    store.msg("先にプロジェクトを開いてください", "error");
    return;
  }
  const imgs = await busy(`画像 ${files.length} 枚の取り込み`, () => api.importImages(store.pid!, files, role));
  if (!imgs) return;
  store.commit("画像の取り込み", (p) => {
    for (const im of imgs) {
      const i = p.images.findIndex((x) => x.id === im.id);
      if (i >= 0) p.images[i] = im;
      else p.images.push(im);
    }
  });
  normalize(store.project!);
  if (imgs[0]) store.setUi({ selImage: imgs[0].id });
  store.msg(`${imgs.length} 枚取り込みました`, "ok");
}

/** サーバー側で project.json が変わる操作の後: 画像・カメラをサーバーの内容で更新（形状などの未保存の編集は保つ）。 */
export async function reloadAfterServerChange(msg: string): Promise<void> {
  if (!store.pid || !store.project) return;
  const fresh = await busy("再読み込み", () => api.getProject(store.pid!));
  if (!fresh) return;
  normalize(fresh);
  store.commit("サーバー側の変更", (p) => {
    p.cameras = fresh.cameras;
    const byId = new Map(p.images.map((im) => [im.id, im]));
    p.images = fresh.images.map((fi) => {
      const old = byId.get(fi.id);
      if (!old) return fi;
      // 歪み補正のやり直しなどで画素が変わるので、サーバーの値を採る（姿勢・輪郭・対応点を含む）
      return fi;
    });
  });
  for (const im of fresh.images) {
    store.bumpFiles(im.id);
    dropLabels(im.id);
  }
  store.emit("load");
  store.msg(msg, "ok");
}
