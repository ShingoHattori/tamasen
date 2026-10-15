// 小さな DOM ヘルパー（UI フレームワークは使わない）。

type Child = Node | string | number | null | undefined | false | Child[];
type Attrs = Record<string, unknown>;

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs?: Attrs | null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") {
        el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      } else if (k === "class") {
        el.className = String(v);
      } else if (k === "style" && typeof v === "object") {
        Object.assign(el.style, v);
      } else if (k === "value" || k === "checked" || k === "disabled" || k === "readOnly" || k === "selected" || k === "title") {
        (el as unknown as Record<string, unknown>)[k] = v;
      } else if (k === "dataset" && typeof v === "object") {
        Object.assign(el.dataset, v);
      } else {
        el.setAttribute(k, v === true ? "" : String(v));
      }
    }
  }
  append(el, children);
  return el;
}

function append(el: Node, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function fmt(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

/** 数値欄。change（確定）時に onChange。readonly なら鍵マーク付きで編集不可。 */
export function numInput(value: number, onChange: (v: number) => void, opts: { step?: number; readonly?: boolean; title?: string; digits?: number; cls?: string } = {}): HTMLElement {
  const inp = h("input", {
    type: "number",
    class: "num " + (opts.cls ?? ""),
    step: String(opts.step ?? 0.1),
    value: Number.isFinite(value) ? String(round(value, opts.digits ?? 4)) : "",
    readOnly: !!opts.readonly,
    disabled: !!opts.readonly,
    title: opts.title ?? "",
  });
  inp.addEventListener("change", () => {
    const v = parseFloat(inp.value);
    if (Number.isFinite(v)) onChange(v);
  });
  if (opts.readonly) return h("span", { class: "locked-field" }, inp, lockIcon(opts.title || "実測値でロック中"));
  return inp;
}

export function round(x: number, d: number): number {
  const f = Math.pow(10, d);
  return Math.round(x * f) / f;
}

export function lockIcon(title = "ロック中"): SVGSVGElement {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "lock-icon");
  svg.setAttribute("aria-label", title);
  const t = document.createElementNS(ns, "title");
  t.textContent = title;
  svg.appendChild(t);
  const body = document.createElementNS(ns, "rect");
  body.setAttribute("x", "3");
  body.setAttribute("y", "7");
  body.setAttribute("width", "10");
  body.setAttribute("height", "8");
  body.setAttribute("rx", "1");
  body.setAttribute("fill", "currentColor");
  const shackle = document.createElementNS(ns, "path");
  shackle.setAttribute("d", "M5 7V5a3 3 0 0 1 6 0v2");
  shackle.setAttribute("fill", "none");
  shackle.setAttribute("stroke", "currentColor");
  shackle.setAttribute("stroke-width", "1.6");
  svg.append(body, shackle);
  return svg;
}

/** モーダル。ボタンの値で resolve（Esc・閉じるは null）。 */
export function modal<T extends string>(title: string, body: Node | string, buttons: { label: string; value: T; primary?: boolean; danger?: boolean }[]): Promise<T | null> {
  return new Promise((resolve) => {
    const dlg = h("dialog", { class: "modal" });
    const done = (v: T | null) => {
      dlg.close();
      dlg.remove();
      resolve(v);
    };
    dlg.append(
      h("h3", null, title),
      h("div", { class: "modal-body" }, body),
      h(
        "div",
        { class: "modal-buttons" },
        buttons.map((b) => h("button", { class: (b.primary ? "primary " : "") + (b.danger ? "danger" : ""), onclick: () => done(b.value), "data-value": b.value }, b.label)),
      ),
    );
    dlg.addEventListener("cancel", (e) => {
      e.preventDefault();
      done(null);
    });
    document.body.appendChild(dlg);
    dlg.showModal();
  });
}

export async function confirmDlg(title: string, message: string, ok = "はい", cancel = "いいえ"): Promise<boolean> {
  return (await modal(title, message, [{ label: cancel, value: "no" }, { label: ok, value: "yes", primary: true }])) === "yes";
}

/** 再描画を 1 回にまとめて後回しにする（change → blur の再入で DOM を壊さないため）。 */
export function deferred(fn: () => void): () => void {
  let queued = false;
  return () => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      fn();
    });
  };
}
