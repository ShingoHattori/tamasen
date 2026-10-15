// 形状編集モード: 部品一覧・数値編集・実測寸法・輪郭から押し出し・断面の点表・固定変換の編集。
// 位置合わせモードでは読み取り専用で表示する（fieldset disabled）。
import type { Contour, Mat4, Part, PartKind, Project, ProjectImage, SheetHole, Vec2, Vec3 } from "../types";
import { applyParamEdit, backProjectToPlane, checkMeasurements, defaultPart, isLocked, mat4FromRt, mat4InvRigid, mat4Apply, transformFromMat4 } from "./geo";
import { h, clear, numInput, modal, fmt, lockIcon, round, deferred } from "./dom";
import { store, newId, type PlaneSpec } from "./store";

const KIND_LABEL: Record<PartKind, string> = { box: "直方体", cylinder: "円柱", frustum: "円錐台", extrude: "押し出し", revolve: "回転体", sheet: "板（曲げ）" };

const SCALAR_PARAMS: Record<PartKind, [string, string][]> = {
  box: [["size.0", "幅 x"], ["size.1", "奥行 y"], ["size.2", "高さ z"]],
  cylinder: [["radius", "半径"], ["height", "高さ"], ["segments", "分割数"]],
  frustum: [["radius_bottom", "下半径"], ["radius_top", "上半径"], ["height", "高さ"], ["segments", "分割数"]],
  extrude: [["thickness", "厚さ"]],
  revolve: [["angle_deg", "回転角 deg"], ["segments", "分割数"]],
  sheet: [["thickness", "板厚"], ["width", "幅"], ["bend_radius", "曲げ半径"]],
};

function getParam(part: Part, path: string): number {
  const [k, i] = path.split(".");
  const v = (part.params as Record<string, unknown>)[k];
  if (i !== undefined && Array.isArray(v)) return Number(v[parseInt(i)]);
  return Number(v);
}

function editParam(partId: string, param: string, value: number): void {
  let err: string | null = null;
  store.commit("部品パラメータ", (p) => {
    err = applyParamEdit(p, partId, param, value);
  });
  if (err) store.msg(err, "error");
}

function editPart(label: string, part: Part, fn: (pt: Part) => void): void {
  store.commit(label, (p) => {
    const pt = p.shape.parts.find((x) => x.id === part.id);
    if (pt) fn(pt);
  });
}

export function buildShapePanel(el: HTMLElement, readonly: boolean): void {
  const extrudeState = { open: false, touched: false, contour: "", plane: "marker", offset: 0, thickness: 3, symmetric: false, direction: "back" as "front" | "back", op: "add" as Part["op"], name: "" };
  const render = deferred(() => {
    clear(el);
    const p = store.project;
    if (!p) return;
    const fs = h("fieldset", { class: "shape-fs", disabled: readonly });
    el.append(fs);
    if (readonly) fs.append(h("p", { class: "note" }, "位置合わせモードでは形状は読み取り専用です（変えるには「形状編集」へ）。"));
    // 輪郭から押し出し（最初に使う操作なので先頭に置き、部品がまだないときは開いておく）
    if (!readonly) {
      if (!p.shape.parts.length && !extrudeState.touched) extrudeState.open = true;
      fs.append(extrudeSection(p, extrudeState, render));
    }
    // 部品一覧
    const kindSel = h("select", { id: "add-kind" }, (Object.keys(KIND_LABEL) as PartKind[]).map((k) => h("option", { value: k }, KIND_LABEL[k])));
    fs.append(
      h("h4", null, "部品"),
      readonly ? "" : h("div", { class: "row" }, kindSel, h("button", { id: "btn-add-part", onclick: () => {
        const id = newId("p", p.shape.parts.map((x) => x.id));
        const part = defaultPart(kindSel.value as PartKind, id);
        part.name ||= `${KIND_LABEL[part.kind]} ${id}`;
        store.commit("部品の追加", (pp) => pp.shape.parts.push(part));
        store.setUi({ selPart: id });
      } }, "部品を追加")),
    );
    const tbl = h("table", { class: "tbl parts", id: "parts-table" });
    p.shape.parts.forEach((pt, i) => {
      const vis = h("input", { type: "checkbox", checked: pt.visible, title: "表示" });
      vis.addEventListener("change", () => editPart("表示", pt, (x) => (x.visible = vis.checked)));
      const op = h("select", null, h("option", { value: "add", selected: pt.op === "add" }, "追加"), h("option", { value: "subtract", selected: pt.op === "subtract" }, "切削"));
      op.addEventListener("change", () => editPart("合成方法", pt, (x) => (x.op = op.value as Part["op"])));
      tbl.append(
        h(
          "tr",
          { class: (store.ui.selPart === pt.id ? "sel " : "") + (pt.op === "subtract" ? "sub" : ""), "data-part": pt.id },
          h("td", null, vis),
          h("td", { class: "clickable", onclick: () => store.setUi({ selPart: pt.id }) }, `${pt.name || pt.id}`, h("span", { class: "note" }, ` ${KIND_LABEL[pt.kind]}`)),
          h("td", null, op),
          readonly
            ? null
            : h(
                "td",
                { class: "nowrap" },
                h("button", { class: "small", disabled: i === 0, title: "上へ", onclick: () => store.commit("並べ替え", (pp) => swap(pp.shape.parts, i, i - 1)) }, "↑"),
                h("button", { class: "small", disabled: i === p.shape.parts.length - 1, title: "下へ", onclick: () => store.commit("並べ替え", (pp) => swap(pp.shape.parts, i, i + 1)) }, "↓"),
                h("button", { class: "small danger", title: "削除", onclick: () => {
                  store.commit("部品の削除", (pp) => {
                    pp.shape.parts = pp.shape.parts.filter((x) => x.id !== pt.id);
                    pp.measurements = pp.measurements.filter((m) => m.part !== pt.id);
                  });
                  if (store.ui.selPart === pt.id) store.setUi({ selPart: p.shape.parts[0]?.id ?? null });
                } }, "削除"),
              ),
        ),
      );
    });
    if (!p.shape.parts.length) tbl.append(h("tr", null, h("td", { class: "note" }, "部品はまだありません")));
    fs.append(tbl);

    const part = store.part();
    if (part) fs.append(partEditor(p, part));
    fs.append(measurementsSection(p));
    if (!readonly) fs.append(fixedTransformSection(p));
  });
  store.on((k) => {
    if (k === "load" || k === "project" || k === "ui" || k === "model") render();
  });
  render();
}

function swap<T>(a: T[], i: number, j: number): void {
  const t = a[i];
  a[i] = a[j];
  a[j] = t;
}

function partEditor(p: Project, part: Part): HTMLElement {
  const box = h("section", { class: "tool-sec", id: "part-editor" });
  const name = h("input", { type: "text", value: part.name, id: "part-name" });
  name.addEventListener("change", () => editPart("名前", part, (x) => (x.name = name.value)));
  box.append(h("h4", null, `部品 ${part.id}（${KIND_LABEL[part.kind]}）`), h("div", { class: "row" }, h("span", null, "名前"), name));
  // 変換
  const tr = h("div", { class: "grid4" }, h("span", null, ""), h("span", { class: "note" }, "x"), h("span", { class: "note" }, "y"), h("span", { class: "note" }, "z"));
  tr.append(h("span", null, "位置 mm"));
  for (let i = 0; i < 3; i++) tr.append(numInput(part.transform.t_mm[i], (v) => editPart("位置", part, (x) => (x.transform.t_mm[i] = v)), { step: 0.5, cls: `t${i}` }));
  tr.append(h("span", null, "回転 deg"));
  for (let i = 0; i < 3; i++) tr.append(numInput(part.transform.r_deg[i], (v) => editPart("回転", part, (x) => (x.transform.r_deg[i] = v)), { step: 1, cls: `r${i}` }));
  box.append(tr, h("p", { class: "note" }, "T_obj_part = 平行移動·Rz·Ry·Rx"));
  // スカラー
  const sc = h("div", { class: "grid2" });
  for (const [param, label] of SCALAR_PARAMS[part.kind]) {
    const locked = isLocked(p, part.id, param);
    sc.append(h("span", null, label, locked ? lockIcon("実測値でロック中") : null), numInput(getParam(part, param), (v) => editParam(part.id, param, v), { step: param === "segments" ? 1 : 0.1, readonly: locked, title: locked ? "実測寸法でロックされています" : "", cls: `param-${param.replace(".", "-")}` }));
  }
  if (part.kind === "extrude") {
    const cur = part.params.symmetric ? "sym" : part.params.direction === "back" ? "back" : "front";
    const dir = dirSelect(cur, "part-direction");
    dir.addEventListener("change", () => editPart("押し出し方向", part, (x) => {
      x.params.symmetric = dir.value === "sym";
      x.params.direction = dir.value === "back" ? "back" : "front";
    }));
    sc.append(h("span", null, "押し出し方向"), dir);
  }
  box.append(sc);
  // 点表
  if (part.kind === "extrude") {
    box.append(pointEditor("外周 (x, y) mm", ["x", "y"], part.params.outer ?? [], (pts) => editPart("外周", part, (x) => (x.params.outer = pts)), true));
    (part.params.holes ?? []).forEach((hole, hi) => {
      box.append(
        pointEditor(`穴 ${hi + 1} (x, y) mm`, ["x", "y"], hole, (pts) => editPart("穴", part, (x) => (x.params.holes![hi] = pts)), true),
        h("button", { class: "small danger", onclick: () => editPart("穴の削除", part, (x) => x.params.holes!.splice(hi, 1)) }, `穴 ${hi + 1} を削除`),
      );
    });
    box.append(h("button", { class: "small", onclick: () => editPart("穴の追加", part, (x) => (x.params.holes = [...(x.params.holes ?? []), [[-2, -2], [2, -2], [2, 2], [-2, 2]]])) }, "穴を追加"));
  } else if (part.kind === "revolve") {
    box.append(pointEditor("断面 (r, z) mm（閉じた多角形、r ≥ 0）", ["r", "z"], part.params.profile ?? [], (pts) => editPart("断面", part, (x) => (x.params.profile = pts.map((q) => [Math.max(0, q[0]), q[1]] as Vec2))), true));
  } else if (part.kind === "sheet") {
    box.append(pointEditor("中心線 (x, z) mm（折れ線）", ["x", "z"], part.params.path ?? [], (pts) => editPart("中心線", part, (x) => (x.params.path = pts)), false));
    box.append(sheetHoles(part));
  }
  return box;
}

/** 点表 ＋ 簡単な 2D 折れ線エディタ（点のドラッグ・辺のダブルクリックで追加）。 */
function pointEditor(title: string, axes: [string, string], pts: Vec2[], onChange: (pts: Vec2[]) => void, closed: boolean): HTMLElement {
  const wrap = h("div", { class: "pt-editor" }, h("h5", null, title));
  const cv = h("canvas", { width: "300", height: "180", class: "poly-canvas" });
  wrap.append(cv);
  const tbl = h("table", { class: "tbl pts" }, h("tr", null, h("th", null, "#"), h("th", null, axes[0]), h("th", null, axes[1]), h("th", null, "")));
  pts.forEach((q, i) => {
    tbl.append(
      h(
        "tr",
        null,
        h("td", null, String(i + 1)),
        h("td", null, numInput(q[0], (v) => onChange(pts.map((r, j) => (j === i ? [v, r[1]] : r)) as Vec2[]), { step: 0.5 })),
        h("td", null, numInput(q[1], (v) => onChange(pts.map((r, j) => (j === i ? [r[0], v] : r)) as Vec2[]), { step: 0.5 })),
        h(
          "td",
          { class: "nowrap" },
          h("button", { class: "small", title: "後ろに点を挿入", onclick: () => {
            const n = pts[(i + 1) % pts.length] ?? q;
            const np = pts.slice();
            np.splice(i + 1, 0, [(q[0] + n[0]) / 2, (q[1] + n[1]) / 2]);
            onChange(np);
          } }, "+"),
          h("button", { class: "small danger", disabled: pts.length <= (closed ? 3 : 2), onclick: () => onChange(pts.filter((_, j) => j !== i)) }, "×"),
        ),
      ),
    );
  });
  wrap.append(tbl);
  if (!pts.length) wrap.append(h("button", { class: "small", onclick: () => onChange(closed ? [[0, 0], [10, 0], [10, 10]] : [[0, 0], [20, 0]]) }, "点を作る"));
  // 2D キャンバス
  const ctx = cv.getContext("2d")!;
  let xs = pts.map((q) => q[0]),
    ys = pts.map((q) => q[1]);
  if (!pts.length) {
    xs = [0, 10];
    ys = [0, 10];
  }
  const minx = Math.min(...xs),
    maxx = Math.max(...xs),
    miny = Math.min(...ys),
    maxy = Math.max(...ys);
  const span = Math.max(maxx - minx, maxy - miny, 1);
  const sc = Math.min(260 / span, 140 / span);
  const ox = 150 - ((minx + maxx) / 2) * sc,
    oy = 90 + ((miny + maxy) / 2) * sc;
  const toS = (q: Vec2): [number, number] => [ox + q[0] * sc, oy - q[1] * sc];
  const toW = (x: number, y: number): Vec2 => [round((x - ox) / sc, 2), round((oy - y) / sc, 2)];
  const work = pts.map((q) => [q[0], q[1]] as Vec2);
  const draw = () => {
    ctx.fillStyle = "#111318";
    ctx.fillRect(0, 0, 300, 180);
    ctx.strokeStyle = "#333";
    ctx.beginPath();
    const [ax, ay] = toS([0, 0]);
    ctx.moveTo(0, ay);
    ctx.lineTo(300, ay);
    ctx.moveTo(ax, 0);
    ctx.lineTo(ax, 180);
    ctx.stroke();
    ctx.strokeStyle = "#4ac0ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    work.forEach((q, i) => {
      const [x, y] = toS(q);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    if (closed) ctx.closePath();
    ctx.stroke();
    work.forEach((q, i) => {
      const [x, y] = toS(q);
      ctx.fillStyle = i === 0 ? "#ffd000" : "#fff";
      ctx.fillRect(x - 3, y - 3, 6, 6);
    });
    ctx.fillStyle = "#888";
    ctx.font = "10px sans-serif";
    ctx.fillText(`${axes[0]} →  ${axes[1]} ↑   ドラッグで移動・辺をダブルクリックで追加`, 4, 174);
  };
  draw();
  let dragI = -1;
  const pos = (e: MouseEvent): [number, number] => {
    const r = cv.getBoundingClientRect();
    return [((e.clientX - r.left) * 300) / r.width, ((e.clientY - r.top) * 180) / r.height];
  };
  cv.addEventListener("pointerdown", (e) => {
    const [x, y] = pos(e);
    dragI = work.findIndex((q) => {
      const [sx, sy] = toS(q);
      return Math.hypot(sx - x, sy - y) < 7;
    });
    if (dragI >= 0) cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (dragI < 0) return;
    const [x, y] = pos(e);
    work[dragI] = toW(x, y);
    draw();
  });
  cv.addEventListener("pointerup", () => {
    if (dragI >= 0) {
      dragI = -1;
      onChange(work.map((q) => [q[0], q[1]] as Vec2));
    }
  });
  cv.addEventListener("dblclick", (e) => {
    const [x, y] = pos(e);
    let best = -1,
      bd = 8;
    const n = work.length;
    for (let i = 0; i < (closed ? n : n - 1); i++) {
      const a = toS(work[i]),
        b = toS(work[(i + 1) % n]);
      const d = segDist(x, y, a, b);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    if (best >= 0) {
      const np = work.slice();
      np.splice(best + 1, 0, toW(x, y));
      onChange(np);
    }
  });
  return wrap;
}

function segDist(px: number, py: number, a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  const t = Math.max(0, Math.min(1, L2 ? ((px - a[0]) * dx + (py - a[1]) * dy) / L2 : 0));
  return Math.hypot(px - a[0] - t * dx, py - a[1] - t * dy);
}

function sheetHoles(part: Part): HTMLElement {
  const holes = part.params.sheet_holes ?? [];
  const set = (fn: (hs: SheetHole[]) => void) => editPart("板の穴", part, (x) => {
    x.params.sheet_holes = JSON.parse(JSON.stringify(x.params.sheet_holes ?? []));
    fn(x.params.sheet_holes!);
  });
  const tbl = h("table", { class: "tbl" }, h("tr", null, h("th", null, "形"), h("th", null, "弧長 s"), h("th", null, "幅方向 y"), h("th", null, "d / w"), h("th", null, "l"), h("th", null, "")));
  holes.forEach((ho, i) => {
    const shape = h("select", null, ["circle", "slot"].map((s) => h("option", { value: s, selected: ho.shape === s }, s === "circle" ? "円" : "長穴")));
    shape.addEventListener("change", () => set((hs) => {
      hs[i].shape = shape.value as SheetHole["shape"];
      if (hs[i].shape === "slot") {
        hs[i].w ??= 3;
        hs[i].l ??= 8;
      } else hs[i].d ??= 3;
    }));
    tbl.append(
      h(
        "tr",
        null,
        h("td", null, ho.shape === "poly" ? "多角形" : shape),
        h("td", null, numInput(ho.s, (v) => set((hs) => (hs[i].s = v)))),
        h("td", null, numInput(ho.y, (v) => set((hs) => (hs[i].y = v)))),
        h("td", null, numInput((ho.shape === "circle" ? ho.d : ho.w) ?? 0, (v) => set((hs) => (hs[i].shape === "circle" ? (hs[i].d = v) : (hs[i].w = v))))),
        h("td", null, ho.shape === "slot" ? numInput(ho.l ?? 0, (v) => set((hs) => (hs[i].l = v))) : ""),
        h("td", null, h("button", { class: "small danger", onclick: () => set((hs) => hs.splice(i, 1)) }, "×")),
      ),
    );
  });
  return h("div", null, h("h5", null, "板の穴"), tbl, h("button", { class: "small", onclick: () => set((hs) => hs.push({ s: 10, y: 0, shape: "circle", d: 3 })) }, "穴を追加"));
}

// ---------- 実測寸法 ----------
function measurementsSection(p: Project): HTMLElement {
  const sec = h("section", { class: "tool-sec", id: "measurements" }, h("h4", null, "実測寸法"));
  let status: unknown = null;
  try {
    status = checkMeasurements(p);
  } catch (e) {
    status = null;
  }
  const statusFor = (id: string, idx: number): string => {
    const s = status as unknown;
    let entry: Record<string, unknown> | undefined;
    if (Array.isArray(s)) entry = (s.find((x: Record<string, unknown>) => x && (x.id === id || x.measurementId === id)) ?? s[idx]) as Record<string, unknown> | undefined;
    else if (s && typeof s === "object") entry = (s as Record<string, Record<string, unknown>>)[id];
    if (!entry) return "";
    const ok = entry.ok ?? entry.status === "ok";
    const msg = (entry.message ?? entry.status ?? "") as string;
    const actual = entry.actual ?? entry.current ?? entry.value;
    return `${ok ? "一致" : "不一致"}${typeof actual === "number" ? `（現在 ${fmt(actual, 2)}）` : ""}${msg && msg !== "ok" ? ` ${msg}` : ""}`;
  };
  const tbl = h("table", { class: "tbl", id: "meas-table" }, h("tr", null, h("th", null, "名前"), h("th", null, "部品"), h("th", null, "パラメータ"), h("th", null, "値 mm"), h("th", null, "ロック"), h("th", null, "状態"), h("th", null, "")));
  p.measurements.forEach((m, i) => {
    const set = (fn: (mm: (typeof p.measurements)[number]) => void) => store.commit("実測寸法", (pp) => {
      const mm = pp.measurements.find((x) => x.id === m.id);
      if (mm) fn(mm);
    });
    const label = h("input", { type: "text", value: m.label, class: "w6" });
    label.addEventListener("change", () => set((mm) => (mm.label = label.value)));
    const partSel = h("select", null, p.shape.parts.map((pt) => h("option", { value: pt.id, selected: pt.id === m.part }, pt.name || pt.id)));
    partSel.addEventListener("change", () => set((mm) => (mm.part = partSel.value)));
    const pt = p.shape.parts.find((x) => x.id === m.part);
    const paramSel = h("select", null, (pt ? SCALAR_PARAMS[pt.kind].filter(([k]) => k !== "segments" && k !== "angle_deg") : []).map(([k, lb]) => h("option", { value: k, selected: k === m.param }, lb)));
    if (pt && !SCALAR_PARAMS[pt.kind].some(([k]) => k === m.param)) paramSel.append(h("option", { value: m.param, selected: true }, m.param));
    paramSel.addEventListener("change", () => set((mm) => (mm.param = paramSel.value)));
    const lock = h("input", { type: "checkbox", checked: m.locked, class: "meas-lock" });
    lock.addEventListener("change", () => {
      set((mm) => (mm.locked = lock.checked));
      if (lock.checked) {
        // ロックしたら参照先の値を実測値に合わせる
        const pp = store.project!;
        const target = pp.shape.parts.find((x) => x.id === m.part);
        if (target && Math.abs(getParam(target, m.param) - m.value_mm) > 1e-9) {
          const mm = pp.measurements.find((x) => x.id === m.id)!;
          mm.locked = false;
          let err: string | null = null;
          store.commit("実測値を形状へ", (q) => {
            err = applyParamEdit(q, m.part, m.param, m.value_mm);
            q.measurements.find((x) => x.id === m.id)!.locked = true;
          });
          if (err) store.msg(err, "error");
        }
      }
    });
    tbl.append(
      h(
        "tr",
        { "data-meas": m.id },
        h("td", null, label),
        h("td", null, partSel),
        h("td", null, paramSel),
        h("td", null, numInput(m.value_mm, (v) => set((mm) => (mm.value_mm = v)), { step: 0.1 })),
        h("td", null, lock),
        h("td", { class: "note" }, statusFor(m.id, i)),
        h("td", null, h("button", { class: "small danger", onclick: () => store.commit("実測寸法の削除", (pp) => (pp.measurements = pp.measurements.filter((x) => x.id !== m.id))) }, "×")),
      ),
    );
  });
  sec.append(tbl);
  const sel = store.part();
  sec.append(
    h("button", {
      id: "btn-add-meas",
      disabled: !p.shape.parts.length,
      onclick: () => {
        const part = sel ?? p.shape.parts[0];
        const param = SCALAR_PARAMS[part.kind][0][0];
        store.commit("実測寸法の追加", (pp) =>
          pp.measurements.push({ id: newId("m", pp.measurements.map((x) => x.id)), label: "寸法", part: part.id, param, value_mm: getParam(part, param), locked: false }),
        );
      },
    }, "実測寸法を追加"),
    h("p", { class: "note" }, "ロックした寸法の参照先は数値欄・自動処理で変更できません。"),
  );
  return sec;
}

// ---------- 輪郭から押し出し ----------
export function planeFor(p: Project, kind: string, offset: number): PlaneSpec {
  switch (kind) {
    case "YZ":
      return { origin: [offset, 0, 0], normal: [1, 0, 0], xAxis: [0, 1, 0] };
    case "ZX":
      return { origin: [0, offset, 0], normal: [0, 1, 0], xAxis: [0, 0, 1] };
    case "marker": {
      const T_obj_marker = mat4InvRigid(p.T_marker_obj);
      const o = mat4Apply(T_obj_marker, [0, 0, offset]);
      const R = (v: Vec3): Vec3 => {
        const a = mat4Apply(T_obj_marker, v),
          b = mat4Apply(T_obj_marker, [0, 0, 0]);
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
      };
      return { origin: o as Vec3, normal: R([0, 0, 1]), xAxis: R([1, 0, 0]) };
    }
    default:
      return { origin: [0, 0, offset], normal: [0, 0, 1], xAxis: [1, 0, 0] };
  }
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function norm(a: Vec3): Vec3 {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

function pointInPoly(pt: Vec2, poly: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i],
      [xj, yj] = poly[j];
    if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function holesInside(img: ProjectImage, outer: Contour): Contour[] {
  return img.contours.filter((c) => c.kind === "hole" && c.status === "confirmed" && c.points_und.length >= 3 && c.points_und.every((q) => pointInPoly(q, outer.points_und)));
}

/** 画像の輪郭を作図平面へ逆投影し、押し出し部品を作る。失敗なら理由（文字列）。 */
export function extrudeFromContour(p: Project, img: ProjectImage, contourId: string, plane: PlaneSpec, thickness: number, symmetric: boolean, op: Part["op"], name: string, direction: "front" | "back" = "front"): Part | string {
  if (!img.pose.T_cam_marker) return "この画像には姿勢がありません。先に位置合わせかマーカーで姿勢を決めてください";
  const cam = p.cameras[img.camera];
  if (!cam?.K_und?.[0]) return "カメラの内部パラメータがありません";
  const outer = img.contours.find((c) => c.id === contourId);
  if (!outer || outer.kind !== "outer" || outer.status !== "confirmed") return "確定した外周の輪郭を選んでください";
  const n = norm(plane.normal);
  const x0 = norm(plane.xAxis);
  const x = norm(sub(x0, [n[0] * dot(x0, n), n[1] * dot(x0, n), n[2] * dot(x0, n)]));
  const y = cross(n, x);
  const proj = (pts: Vec2[]): Vec2[] | string => {
    const out: Vec2[] = [];
    for (const uv of pts) {
      const r = backProjectToPlane(uv, cam.K_und, img.pose.T_cam_marker!, p.T_marker_obj, { origin: plane.origin, normal: n, xAxis: x });
      if (!r) return "輪郭の一部が作図平面と交わりません（平面がカメラと平行か、カメラの後ろ側です）";
      const d = sub(r.p_obj as Vec3, plane.origin);
      out.push([round(dot(d, x), 3), round(dot(d, y), 3)]);
    }
    return out;
  };
  const o = proj(outer.points_und);
  if (typeof o === "string") return o;
  const holes: Vec2[][] = [];
  for (const hc of holesInside(img, outer)) {
    const hp = proj(hc.points_und);
    if (typeof hp === "string") return hp;
    holes.push(hp);
  }
  const id = newId("p", p.shape.parts.map((q) => q.id));
  const part = defaultPart("extrude", id);
  part.name = name || `押し出し ${id}`;
  part.op = op;
  part.params = { ...part.params, outer: o, holes, thickness, symmetric, direction };
  // 部品座標 = 作図平面（x, y, 法線）
  const M: Mat4 = [x[0], y[0], n[0], plane.origin[0], x[1], y[1], n[1], plane.origin[1], x[2], y[2], n[2], plane.origin[2], 0, 0, 0, 1];
  part.transform = transformFromMat4(M);
  return part;
}

function extrudeSection(p: Project, st: { open: boolean; touched: boolean; contour: string; plane: string; offset: number; thickness: number; symmetric: boolean; direction: "front" | "back"; op: Part["op"]; name: string }, rerender: () => void): HTMLElement {
  const img = store.image();
  const det = h("details", { class: "tool-sec", id: "extrude-sec", open: st.open });
  det.addEventListener("toggle", () => {
    if (det.open === st.open) return;
    st.open = det.open;
    st.touched = true;
    store.ui.planePreview = det.open ? planeFor(p, st.plane, st.offset) : null;
    store.emit("model");
  });
  det.append(h("summary", null, "輪郭から押し出し"));
  if (!img) {
    det.append(h("p", { class: "note" }, "画像を選んでください"));
    return det;
  }
  const outers = img.contours.filter((c) => c.kind === "outer" && c.status === "confirmed");
  if (st.contour && !outers.some((c) => c.id === st.contour)) st.contour = "";
  if (!st.contour && outers[0]) st.contour = outers[0].id;
  if (store.ui.selContour && outers.some((c) => c.id === store.ui.selContour)) st.contour = store.ui.selContour;
  const csel = h("select", { id: "extrude-contour" }, outers.length ? outers.map((c) => h("option", { value: c.id, selected: c.id === st.contour }, `${c.id}（${c.points_und.length} 点）`)) : h("option", { value: "" }, "（確定した外周がありません）"));
  csel.addEventListener("change", () => {
    st.contour = csel.value;
    store.setUi({ selContour: csel.value || null });
  });
  if (!outers.length) {
    det.append(h("p", { class: "note warn" }, `画像 ${img.id} に確定した外周がありません。「1 前処理」で「輪郭を作る」→ 外周を選んで「確定」してから使います。`));
  } else if (!img.pose.T_cam_marker) {
    det.append(h("p", { class: "note warn" }, `画像 ${img.id} には撮影姿勢がありません（マーカーの見える画像を選ぶか、「3 位置合わせ」で姿勢を付けてから使います）。`));
  }
  const cur = outers.find((c) => c.id === st.contour);
  const holes = cur ? holesInside(img, cur) : [];
  const psel = h("select", { id: "extrude-plane" }, [["XY", "obj の XY 面"], ["YZ", "obj の YZ 面"], ["ZX", "obj の ZX 面"], ["marker", "マーカー面"]].map(([v, l]) => h("option", { value: v, selected: v === st.plane }, l)));
  const updPlane = () => {
    store.ui.planePreview = planeFor(p, st.plane, st.offset);
    store.emit("model");
  };
  psel.addEventListener("change", () => {
    st.plane = psel.value;
    updPlane();
  });
  const off = h("input", { type: "number", class: "num", value: String(st.offset), step: "0.5", id: "extrude-offset" });
  off.addEventListener("change", () => {
    st.offset = parseFloat(off.value) || 0;
    updPlane();
  });
  const th = h("input", { type: "number", class: "num", value: String(st.thickness), step: "0.1", min: "0.01", id: "extrude-thickness" });
  th.addEventListener("change", () => (st.thickness = parseFloat(th.value) || st.thickness));
  const sym = dirSelect(st.symmetric ? "sym" : st.direction, "extrude-direction");
  sym.addEventListener("change", () => {
    st.symmetric = sym.value === "sym";
    st.direction = sym.value === "back" ? "back" : "front";
  });
  const op = h("select", null, h("option", { value: "add", selected: st.op === "add" }, "追加"), h("option", { value: "subtract", selected: st.op === "subtract" }, "切削"));
  op.addEventListener("change", () => (st.op = op.value as Part["op"]));
  const nm = h("input", { type: "text", value: st.name, placeholder: "部品名" });
  nm.addEventListener("change", () => (st.name = nm.value));
  const hasPose = !!img.pose.T_cam_marker;
  det.append(
    h("p", { class: "note" }, `画像 ${img.id} の姿勢で輪郭を光線にし、選んだ平面との交点を部品の輪郭にします（斜めの画像を正投影とはみなしません）。`),
    hasPose ? "" : h("p", { class: "note warn" }, "この画像には姿勢がないため使えません"),
    h("div", { class: "grid2" },
      h("span", null, "外周"), csel,
      h("span", null, "穴"), h("span", { class: "note" }, holes.length ? holes.map((c) => c.id).join(", ") : "なし（確定した穴の輪郭が内側にあれば自動で含めます）"),
      h("span", null, "作図平面"), psel,
      h("span", null, "オフセット mm"), off,
      h("span", null, "厚さ mm"), th,
      h("span", null, "押し出し方向"), sym,
      h("span", null, "合成"), op,
      h("span", null, "名前"), nm,
    ),
    h("button", {
      id: "btn-extrude",
      class: "primary",
      disabled: !hasPose || !cur,
      onclick: () => {
        const plane = planeFor(p, st.plane, st.offset);
        const r = extrudeFromContour(p, img, st.contour, plane, st.thickness, st.symmetric, st.op, st.name, st.direction);
        if (typeof r === "string") {
          store.msg(r, "error");
          return;
        }
        store.commit("輪郭から押し出し", (pp) => pp.shape.parts.push(r));
        store.setUi({ selPart: r.id });
        store.msg(`押し出し部品 ${r.id} を作りました（外周 ${r.params.outer!.length} 点、穴 ${r.params.holes!.length}）`, "ok");
        rerender();
      },
    }, "押し出しを作成"),
  );
  if (st.open && !store.ui.planePreview) store.ui.planePreview = planeFor(p, st.plane, st.offset);
  return det;
}

// ---------- 固定変換 T_marker_obj ----------
function fixedTransformSection(p: Project): HTMLElement {
  const lock = h("input", { type: "checkbox", checked: p.T_marker_obj_locked, id: "tmo-lock" });
  lock.addEventListener("change", () => store.commit("固定変換のロック", (pp) => (pp.T_marker_obj_locked = lock.checked)));
  const tf = transformFromMat4(p.T_marker_obj);
  return h(
    "section",
    { class: "tool-sec" },
    h("h4", null, "固定変換 T_marker_obj（マーカー → 物体）"),
    h("p", { class: "note" }, `位置 ${tf.t_mm.map((v) => fmt(v, 1)).join(", ")} mm、回転 ${tf.r_deg.map((v) => fmt(v, 1)).join(", ")} deg`),
    h("div", { class: "row" },
      h("button", { id: "btn-edit-tmo", disabled: p.T_marker_obj_locked, title: p.T_marker_obj_locked ? "ロック中は編集できません" : "", onclick: () => void editFixedTransform() }, "固定変換を編集…"),
      h("label", null, lock, " ロック"),
      p.T_marker_obj_locked ? lockIcon() : null,
    ),
  );
}

async function editFixedTransform(): Promise<void> {
  const p = store.project;
  if (!p || store.ui.mode !== "shape") return;
  if (p.T_marker_obj_locked) {
    store.msg("固定変換はロックされています", "error");
    return;
  }
  const tf = transformFromMat4(p.T_marker_obj);
  const vals = { t: tf.t_mm.slice() as Vec3, r: tf.r_deg.slice() as Vec3 };
  const inputs = (arr: number[], step: number) =>
    arr.map((v, i) => {
      const inp = h("input", { type: "number", class: "num", step: String(step), value: String(round(v, 4)), "data-i": String(i) });
      inp.addEventListener("change", () => (arr[i] = parseFloat(inp.value) || 0));
      return inp;
    });
  const nMarker = p.images.filter((im) => im.pose.T_cam_marker && (im.pose.source === "marker" || im.markers.length)).length;
  const body = h(
    "div",
    { id: "tmo-dialog" },
    h("p", { class: "warn" }, `注意: この変換は全画像で共通です。変えると、マーカーから姿勢を決めた画像（${nMarker} 枚）すべての重ね合わせで物体が動きます。各画像の撮影姿勢 T_cam_marker は変わりません。`),
    h("div", { class: "grid4" }, h("span", null, ""), h("span", null, "x"), h("span", null, "y"), h("span", null, "z"), h("span", null, "位置 mm"), inputs(vals.t, 0.5), h("span", null, "回転 deg"), inputs(vals.r, 1)),
  );
  const r = await modal("固定変換 T_marker_obj を編集", body, [
    { label: "やめる", value: "no" },
    { label: "適用（全画像に影響）", value: "yes", primary: true },
  ]);
  if (r !== "yes") return;
  store.commit("固定変換の編集", (pp) => (pp.T_marker_obj = mat4FromRt(vals.r, vals.t)));
  store.msg("固定変換を変更しました（全画像の重ね合わせに反映）", "ok");
}

/** 押し出し方向の選択（表側 = 平面の法線の向き、裏側 = 逆向き、両側 = 対称）。マーカー面に描いたときは裏側が物体の中。 */
function dirSelect(cur: string, id: string): HTMLSelectElement {
  return h("select", { id },
    h("option", { value: "back", selected: cur === "back" }, "裏側へ（−法線、面の奥＝物体の中）"),
    h("option", { value: "front", selected: cur === "front" }, "表側へ（＋法線、カメラ側）"),
    h("option", { value: "sym", selected: cur === "sym" }, "両側（対称）"));
}
