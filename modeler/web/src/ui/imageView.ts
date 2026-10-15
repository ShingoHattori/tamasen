// 画像ビュー: 写真（歪み補正後）・ラベル・エッジ・輪郭・モデルの重ね合わせ。ズーム・パン、
// 前処理（ブラシ・輪郭頂点・マーカー四隅）、位置合わせ（ドラッグで姿勢・対応点）の操作。
import * as THREE from "three";
import type { Contour, Mat4, ProjectImage, Vec2, Vec3 } from "../types";
import { mat4Apply, mat4Mul } from "./geo";
import { fileUrl } from "./api";
import { intrinsicsFromK, viewIntrinsics, makeCvCamera, setCvProjection, glModelMatrix, m4, pixelToNdc } from "./camera";
import { getLabels, loadLabels, strokeBegin, strokePoint } from "./labels";
import { currentModel, modelCenterObj, modelVersion } from "./model";
import { applyPose, poseRefusal, rotX, rotY, rotZ, rotateAbout, trans, T_cam_obj } from "./pose";
import { store, newId } from "./store";

const CORNER_NAMES = ["左上", "右上", "右下", "左下"];

type Drag =
  | { kind: "pan"; sx: number; sy: number; px: number; py: number }
  | { kind: "brush" }
  | { kind: "vertex"; contour: string; index: number }
  | { kind: "pose"; mode: "move" | "rollz" | "tilt"; sx: number; sy: number; T0: Mat4; pc: Vec3; cs: [number, number]; img: string };

export class ImageView {
  readonly el: HTMLElement;
  private base: HTMLCanvasElement;
  private glc: HTMLCanvasElement;
  private top: HTMLCanvasElement;
  private guide: HTMLElement;
  private renderer: THREE.WebGLRenderer | null = null;
  private scene = new THREE.Scene();
  private cam = makeCvCamera();
  private group = new THREE.Group();
  private mesh: THREE.Mesh<THREE.BufferGeometry, THREE.Material> = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial() as THREE.Material);
  private matTranslucent = new THREE.MeshLambertMaterial({ color: 0x40c0ff, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: true });
  private matWire = new THREE.MeshBasicMaterial({ color: 0x40ffd0, wireframe: true });
  private matSil = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  zoom = 1;
  panX = 0;
  panY = 0;
  private fittedFor: string | null = null;
  private photos = new Map<string, { img: HTMLImageElement; ok: boolean; v: number }>();
  private edges = new Map<string, { canvas: HTMLCanvasElement | null; v: number }>();
  private sil: { key: string; canvas: HTMLCanvasElement | null } = { key: "", canvas: null };
  private drag: Drag | null = null;
  private space = false;
  private cursor: [number, number] | null = null; // screen
  private raf = 0;

  constructor(el: HTMLElement) {
    this.el = el;
    el.classList.add("imgview");
    this.base = document.createElement("canvas");
    this.glc = document.createElement("canvas");
    this.top = document.createElement("canvas");
    this.guide = document.createElement("div");
    this.guide.className = "guide";
    for (const c of [this.base, this.glc, this.top]) c.className = "layer";
    this.top.tabIndex = 0;
    el.append(this.base, this.glc, this.top, this.guide);
    try {
      this.renderer = new THREE.WebGLRenderer({ canvas: this.glc, alpha: true, antialias: true, preserveDrawingBuffer: true });
      this.renderer.setPixelRatio(1);
      this.renderer.setClearColor(0x000000, 0);
    } catch (e) {
      store.msg("WebGL が使えないため、モデルの重ね合わせを表示できません", "error");
    }
    this.mesh.matrixAutoUpdate = false;
    this.group.matrixAutoUpdate = false;
    this.group.add(this.mesh);
    this.scene.add(this.group);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const dl = new THREE.DirectionalLight(0xffffff, 1.6);
    dl.position.set(0.3, 0.5, 1);
    this.scene.add(dl);

    new ResizeObserver(() => this.render()).observe(el);
    this.top.addEventListener("pointerdown", (e) => this.onDown(e));
    this.top.addEventListener("pointermove", (e) => this.onMove(e));
    this.top.addEventListener("pointerup", (e) => this.onUp(e));
    this.top.addEventListener("pointercancel", (e) => this.onUp(e));
    this.top.addEventListener("pointerleave", () => {
      this.cursor = null;
      this.render();
    });
    this.top.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.top.addEventListener("contextmenu", (e) => this.onContext(e));
    window.addEventListener("keydown", (e) => this.onKey(e, true));
    window.addEventListener("keyup", (e) => this.onKey(e, false));
    store.on((k) => {
      if (k === "load") {
        this.photos.clear();
        this.edges.clear();
        this.fittedFor = null;
        this.sil.key = "";
      }
      this.render();
    });
  }

  // ---------- 座標 ----------
  toScreen(u: number, v: number): [number, number] {
    return [(u + 0.5) * this.zoom + this.panX, (v + 0.5) * this.zoom + this.panY];
  }
  toImage(sx: number, sy: number): [number, number] {
    return [(sx - this.panX) / this.zoom - 0.5, (sy - this.panY) / this.zoom - 0.5];
  }
  private evPos(e: MouseEvent): [number, number] {
    const r = this.top.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  fit(): void {
    const img = store.image();
    if (!img) return;
    const W = this.el.clientWidth || 800;
    const H = this.el.clientHeight || 600;
    this.zoom = Math.min(W / img.width, H / img.height) * 0.98;
    this.panX = (W - img.width * this.zoom) / 2;
    this.panY = (H - img.height * this.zoom) / 2;
    this.render();
  }

  private K(img: ProjectImage): number[] | null {
    const c = store.project?.cameras[img.camera];
    if (!c || !c.K_und || c.K_und.length < 9 || !c.K_und[0]) return null;
    return c.K_und;
  }

  // ---------- 描画 ----------
  render(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  private photo(img: ProjectImage): HTMLImageElement | null {
    const pid = store.pid!;
    const v = store.fileVersion[img.id] ?? 0;
    let e = this.photos.get(img.id);
    if (!e || e.v !== v) {
      const im = new Image();
      e = { img: im, ok: false, v };
      const entry = e;
      im.onload = () => {
        entry.ok = true;
        this.render();
      };
      im.onerror = () => {
        // 歪み補正画像がなければ元画像
        if (!im.src.includes(img.file)) im.src = fileUrl(pid, img.id, img.file, v);
      };
      im.src = fileUrl(pid, img.id, "undistorted.png", v);
      this.photos.set(img.id, e);
    }
    return e.ok ? e.img : null;
  }

  private edgeCanvas(img: ProjectImage): HTMLCanvasElement | null {
    const pid = store.pid!;
    const v = store.fileVersion[img.id] ?? 0;
    let e = this.edges.get(img.id);
    if (!e || e.v !== v) {
      e = { canvas: null, v };
      const entry = e;
      this.edges.set(img.id, e);
      const im = new Image();
      im.onload = () => {
        const c = document.createElement("canvas");
        c.width = im.naturalWidth;
        c.height = im.naturalHeight;
        const ctx = c.getContext("2d", { willReadFrequently: true })!;
        ctx.drawImage(im, 0, 0);
        const d = ctx.getImageData(0, 0, c.width, c.height);
        for (let i = 0; i < d.data.length; i += 4) {
          const on = d.data[i] > 0;
          d.data[i] = 255;
          d.data[i + 1] = 90;
          d.data[i + 2] = 0;
          d.data[i + 3] = on ? 220 : 0;
        }
        ctx.putImageData(d, 0, 0);
        entry.canvas = c;
        this.render();
      };
      im.src = fileUrl(pid, img.id, "edges.png", v);
    }
    return e.canvas;
  }

  private draw(): void {
    const W = Math.max(1, this.el.clientWidth);
    const H = Math.max(1, this.el.clientHeight);
    for (const c of [this.base, this.top]) {
      if (c.width !== W || c.height !== H) {
        c.width = W;
        c.height = H;
      }
    }
    if (this.renderer) {
      const s = this.renderer.getSize(new THREE.Vector2());
      if (s.x !== W || s.y !== H) this.renderer.setSize(W, H, false);
      this.glc.style.width = W + "px";
      this.glc.style.height = H + "px";
    }
    const bctx = this.base.getContext("2d")!;
    const tctx = this.top.getContext("2d")!;
    bctx.fillStyle = "#15171b";
    bctx.fillRect(0, 0, W, H);
    tctx.clearRect(0, 0, W, H);
    const img = store.image();
    const ui = store.ui;
    if (!store.project || !img) {
      this.renderer?.clear();
      this.guide.textContent = store.project ? "画像がありません。「画像を取り込む」かここへドロップしてください" : "プロジェクトを開いてください";
      return;
    }
    if (this.fittedFor !== img.id) {
      this.fittedFor = img.id;
      this.fit();
    }
    const iw = img.width * this.zoom;
    const ih = img.height * this.zoom;
    bctx.imageSmoothingEnabled = this.zoom < 2;
    if (ui.layers.photo) {
      const ph = this.photo(img);
      if (ph) bctx.drawImage(ph, this.panX, this.panY, iw, ih);
    } else {
      bctx.fillStyle = "#2a2d33";
      bctx.fillRect(this.panX, this.panY, iw, ih);
    }
    if (ui.layers.mask) {
      const L = getLabels(img.id);
      if (L) bctx.drawImage(L.canvas, this.panX, this.panY, iw, ih);
      else void loadLabels(img.id);
    }
    if (ui.layers.edges) {
      const ec = this.edgeCanvas(img);
      if (ec) bctx.drawImage(ec, this.panX, this.panY, iw, ih);
    }

    // モデル
    const K = this.K(img);
    const Tco = T_cam_obj(store.project, img);
    const model = currentModel();
    const geom = model?.geometry ?? null;
    const showModel = ui.layers.model && !!K && !!Tco && !!geom;
    if (this.renderer) {
      this.renderer.setRenderTarget(null);
      this.renderer.clear();
      if (showModel && (ui.overlay === "translucent" || ui.overlay === "wire")) {
        this.mesh.geometry = geom!;
        this.mesh.material = ui.overlay === "wire" ? this.matWire : this.matTranslucent;
        this.group.matrix.copy(glModelMatrix(Tco!));
        this.group.matrixWorld.copy(this.group.matrix);
        this.mesh.matrixWorld.copy(this.group.matrix);
        this.group.matrixWorldNeedsUpdate = false;
        setCvProjection(this.cam, viewIntrinsics(intrinsicsFromK(K!), this.zoom, this.panX, this.panY), W, H);
        this.renderer.render(this.scene, this.cam);
      }
    }
    if (showModel && (ui.overlay === "outline" || ui.layers.silhouette) && ui.overlay !== "hidden") {
      const sc = this.silhouette(img, K!, Tco!, geom!);
      if (sc) {
        tctx.imageSmoothingEnabled = false;
        tctx.drawImage(sc, this.panX, this.panY, iw, ih);
      }
    }

    // 輪郭
    if (ui.layers.contours) for (const c of img.contours) this.drawContour(tctx, c, c.id === ui.selContour);

    // マーカー四隅
    for (const m of img.markers) {
      if (!m.corners_und || m.corners_und.length !== 4) continue;
      this.poly(tctx, m.corners_und, true, m.id === store.project.marker.id ? "#00e5ff" : "#888", 1.5, []);
      m.corners_und.forEach((p, i) => {
        const [x, y] = this.toScreen(p[0], p[1]);
        tctx.fillStyle = "#00e5ff";
        tctx.font = "11px sans-serif";
        tctx.fillText(String(i), x + 4, y - 4);
      });
    }
    if (ui.mode === "prep" && ui.prepTool === "corners") {
      ui.cornerDraft.forEach((p, i) => {
        const [x, y] = this.toScreen(p[0], p[1]);
        tctx.strokeStyle = "#ff0";
        tctx.lineWidth = 2;
        tctx.beginPath();
        tctx.arc(x, y, 6, 0, Math.PI * 2);
        tctx.stroke();
        tctx.fillStyle = "#ff0";
        tctx.fillText(CORNER_NAMES[i], x + 8, y - 8);
      });
      if (ui.cornerDraft.length > 1) this.poly(tctx, ui.cornerDraft, false, "#ff0", 1, [4, 3]);
    }

    // 対応点
    if (ui.mode === "align" || img.correspondences.length) {
      img.correspondences.forEach((c, i) => {
        const [x, y] = this.toScreen(c.uv_und[0], c.uv_und[1]);
        this.cross(tctx, x, y, "#ff3b3b");
        tctx.fillStyle = "#ff3b3b";
        tctx.fillText(String(i + 1), x + 6, y + 12);
        if (K && Tco) {
          const pc = mat4Apply(Tco, c.p_obj);
          if (pc[2] > 0) {
            const u = (K[0] * pc[0]) / pc[2] + K[2];
            const v = (K[4] * pc[1]) / pc[2] + K[5];
            const [mx, my] = this.toScreen(u, v);
            tctx.strokeStyle = "rgba(80,160,255,0.9)";
            tctx.setLineDash([3, 3]);
            tctx.beginPath();
            tctx.moveTo(x, y);
            tctx.lineTo(mx, my);
            tctx.stroke();
            tctx.setLineDash([]);
            tctx.fillStyle = "#4aa0ff";
            tctx.beginPath();
            tctx.arc(mx, my, 3.5, 0, Math.PI * 2);
            tctx.fill();
          }
        }
      });
      if (ui.corrPending) {
        const [x, y] = this.toScreen(ui.corrPending[0], ui.corrPending[1]);
        this.cross(tctx, x, y, "#ffea00");
      }
    }

    // ブラシの円
    if (ui.mode === "prep" && ui.prepTool === "brush" && this.cursor) {
      tctx.strokeStyle = ["#fff", "#4aa0ff", "#20e070", "#ff40d0"][ui.brushLabel];
      tctx.lineWidth = 1;
      tctx.beginPath();
      tctx.arc(this.cursor[0], this.cursor[1], ui.brushRadius * this.zoom, 0, Math.PI * 2);
      tctx.stroke();
    }
    this.guide.textContent = this.guideText(img, !!K, !!Tco, !!geom);
  }

  private guideText(img: ProjectImage, hasK: boolean, hasPose: boolean, hasGeom: boolean): string {
    const ui = store.ui;
    const notes: string[] = [];
    if (ui.mode === "prep") {
      if (ui.prepTool === "brush") notes.push(`ブラシ（${["消しゴム", "背景", "前景", "穴"][ui.brushLabel]}、半径 ${ui.brushRadius} px）: ドラッグで塗る`);
      if (ui.prepTool === "contour") notes.push("輪郭編集: 頂点をドラッグで移動・辺をクリックで頂点を追加・右クリックか Delete で削除");
      if (ui.prepTool === "corners") {
        const n = ui.cornerDraft.length;
        notes.push(n < 4 ? `マーカー四隅 ${n + 1}/4: 「${CORNER_NAMES[n]}」の角をクリック（順に 左上→右上→右下→左下、Esc でやり直し）` : "四隅を指定しました");
      }
    } else if (ui.mode === "align") {
      if (img.pose.locked) notes.push("この画像の姿勢は確定（ロック）されています");
      else if (ui.alignTool === "corr") notes.push(ui.corrPending ? "対応点 2/2: モデル上の対応する点をクリック（この重ね合わせ上か 3D ビュー）。Esc で取消" : "対応点 1/2: 画像上の点をクリック");
      else if (hasPose) notes.push("ドラッグ=画像面内で移動、Shift+ドラッグ=視線軸まわり回転、Ctrl+ドラッグ=縦横軸まわり回転、Alt+ホイール=奥行き、矢印キー/PageUp/PageDown");
      else notes.push("姿勢がありません。左の「初期姿勢を置く」か対応点で推定してください");
    }
    if (ui.layers.model && !hasK) notes.push("カメラの内部パラメータがありません");
    if (ui.layers.model && hasK && !hasPose && img.role === "object") notes.push("この画像には姿勢がありません");
    if (ui.layers.model && !hasGeom && store.project?.shape.parts.length) notes.push("形状がまだありません");
    return notes.join("　/　");
  }

  private silhouette(img: ProjectImage, K: number[], Tco: Mat4, geom: THREE.BufferGeometry): HTMLCanvasElement | null {
    if (!this.renderer) return null;
    const key = `${img.id}|${modelVersion()}|${Tco.map((x) => x.toFixed(4)).join(",")}|${K.join(",")}`;
    if (this.sil.key === key) return this.sil.canvas;
    const w = img.width,
      h = img.height;
    const rt = new THREE.WebGLRenderTarget(w, h);
    const cam = makeCvCamera();
    setCvProjection(cam, intrinsicsFromK(K), w, h);
    const mesh = new THREE.Mesh(geom, this.matSil);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(glModelMatrix(Tco));
    mesh.matrixWorld.copy(mesh.matrix);
    const sc = new THREE.Scene();
    sc.add(mesh);
    mesh.matrixWorldAutoUpdate = false;
    this.renderer.setRenderTarget(rt);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(sc, cam);
    const buf = new Uint8Array(w * h * 4);
    this.renderer.readRenderTargetPixels(rt, 0, 0, w, h, buf);
    this.renderer.setRenderTarget(null);
    rt.dispose();
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const sy = h - 1 - y; // GL は下から
      for (let x = 0; x < w; x++) mask[y * w + x] = buf[(sy * w + x) * 4 + 3] > 127 ? 1 : 0;
    }
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d")!;
    const id = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        const edge = x === 0 || y === 0 || x === w - 1 || y === h - 1 || !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w];
        if (edge) {
          id.data[i * 4] = 0;
          id.data[i * 4 + 1] = 240;
          id.data[i * 4 + 2] = 255;
          id.data[i * 4 + 3] = 255;
        }
      }
    ctx.putImageData(id, 0, 0);
    this.sil = { key, canvas: c };
    return c;
  }

  private poly(ctx: CanvasRenderingContext2D, pts: Vec2[] | [number, number][], closed: boolean, color: string, width: number, dash: number[]): void {
    if (!pts.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = this.toScreen(p[0], p[1]);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    if (closed) ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  private cross(ctx: CanvasRenderingContext2D, x: number, y: number, color: string): void {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 6, y);
    ctx.lineTo(x + 6, y);
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x, y + 6);
    ctx.stroke();
  }

  private drawContour(ctx: CanvasRenderingContext2D, c: Contour, sel: boolean): void {
    const color = c.kind === "outer" ? "#ffa000" : "#ff40d0";
    this.poly(ctx, c.points_und, true, color, sel ? 2.5 : 1.5, c.status === "confirmed" ? [] : [6, 4]);
    if (sel && store.ui.mode === "prep" && store.ui.prepTool === "contour") {
      c.points_und.forEach((p, i) => {
        const [x, y] = this.toScreen(p[0], p[1]);
        ctx.fillStyle = i === store.ui.selVertex ? "#fff" : color;
        ctx.fillRect(x - 3.5, y - 3.5, 7, 7);
      });
    }
  }

  // ---------- 輪郭のヒットテスト ----------
  private hitVertex(sx: number, sy: number, c: Contour): number {
    let best = -1,
      bd = 8;
    c.points_und.forEach((p, i) => {
      const [x, y] = this.toScreen(p[0], p[1]);
      const d = Math.hypot(x - sx, y - sy);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  }
  private hitEdge(sx: number, sy: number, c: Contour, tol = 6): number {
    const n = c.points_und.length;
    let best = -1,
      bd = tol;
    for (let i = 0; i < n; i++) {
      const a = this.toScreen(c.points_und[i][0], c.points_und[i][1]);
      const b = this.toScreen(c.points_und[(i + 1) % n][0], c.points_und[(i + 1) % n][1]);
      const d = segDist(sx, sy, a, b);
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return best;
  }
  private pickContour(sx: number, sy: number): Contour | null {
    const img = store.image();
    if (!img) return null;
    for (const c of img.contours) if (this.hitVertex(sx, sy, c) >= 0 || this.hitEdge(sx, sy, c) >= 0) return c;
    return null;
  }

  // ---------- 入力 ----------
  private onDown(e: PointerEvent): void {
    this.top.focus();
    const [sx, sy] = this.evPos(e);
    const img = store.image();
    if (e.button === 1 || (e.button === 0 && this.space)) {
      e.preventDefault();
      this.drag = { kind: "pan", sx, sy, px: this.panX, py: this.panY };
      this.top.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0 || !img || !store.project) return;
    const ui = store.ui;
    const [u, v] = this.toImage(sx, sy);
    if (ui.mode === "prep") {
      if (ui.prepTool === "brush") {
        const L = getLabels(img.id);
        if (!L) {
          store.msg("ラベルを読み込み中です", "info");
          return;
        }
        strokeBegin(L, ui.brushLabel, ui.brushRadius);
        strokePoint(L, u, v);
        this.drag = { kind: "brush" };
        this.top.setPointerCapture(e.pointerId);
        this.render();
      } else if (ui.prepTool === "contour") {
        const sel = img.contours.find((c) => c.id === ui.selContour);
        if (sel) {
          const vi = this.hitVertex(sx, sy, sel);
          if (vi >= 0) {
            store.gestureBegin();
            store.ui.selVertex = vi;
            this.drag = { kind: "vertex", contour: sel.id, index: vi };
            this.top.setPointerCapture(e.pointerId);
            this.render();
            return;
          }
          const ei = this.hitEdge(sx, sy, sel);
          if (ei >= 0) {
            store.gestureBegin();
            sel.points_und.splice(ei + 1, 0, [u, v]);
            store.ui.selVertex = ei + 1;
            this.drag = { kind: "vertex", contour: sel.id, index: ei + 1 };
            this.top.setPointerCapture(e.pointerId);
            store.emit("pose");
            return;
          }
        }
        const c = this.pickContour(sx, sy);
        store.setUi({ selContour: c?.id ?? null, selVertex: null });
      } else if (ui.prepTool === "corners") {
        if (ui.cornerDraft.length >= 4) ui.cornerDraft = [];
        ui.cornerDraft.push([u, v]);
        store.emit("ui");
        if (ui.cornerDraft.length === 4) document.dispatchEvent(new CustomEvent("corners-done"));
      } else {
        const c = this.pickContour(sx, sy);
        if (c) store.setUi({ selContour: c.id });
      }
    } else if (ui.mode === "shape") {
      const c = this.pickContour(sx, sy);
      if (c) store.setUi({ selContour: c.id });
    } else if (ui.mode === "align") {
      if (ui.alignTool === "corr") {
        this.corrClick(img, u, v, sx, sy);
        return;
      }
      const why = poseRefusal(img);
      if (why) {
        store.msg(why, "error");
        return;
      }
      if (!img.pose.T_cam_marker) {
        store.msg("姿勢がありません。「初期姿勢を置く」か対応点で推定してください", "error");
        return;
      }
      const Tco = T_cam_obj(store.project, img)!;
      const pc = mat4Apply(Tco, modelCenterObj()) as Vec3;
      const K = this.K(img);
      const cs: [number, number] = K && pc[2] > 0 ? this.toScreen((K[0] * pc[0]) / pc[2] + K[2], (K[4] * pc[1]) / pc[2] + K[5]) : [sx, sy];
      store.gestureBegin();
      this.drag = { kind: "pose", mode: e.shiftKey ? "rollz" : e.ctrlKey || e.metaKey ? "tilt" : "move", sx, sy, T0: img.pose.T_cam_marker.slice(), pc, cs, img: img.id };
      this.top.setPointerCapture(e.pointerId);
    }
  }

  private corrClick(img: ProjectImage, u: number, v: number, sx: number, sy: number): void {
    const ui = store.ui;
    if (!ui.corrPending) {
      store.setUi({ corrPending: [u, v] });
      return;
    }
    const p = this.raycastModel(img, sx, sy);
    if (!p) {
      store.msg("重ね合わせたモデルに当たりません。モデルの上をクリックするか、3D ビューで点を選んでください", "error");
      return;
    }
    addCorrespondence(img, p);
  }

  /** 画面上の点から重ね合わせモデルへ光線を飛ばし、当たった点を obj 座標で返す。 */
  raycastModel(img: ProjectImage, sx: number, sy: number): Vec3 | null {
    const K = this.K(img);
    const Tco = store.project ? T_cam_obj(store.project, img) : null;
    const geom = currentModel()?.geometry;
    if (!K || !Tco || !geom) return null;
    const W = this.top.width,
      H = this.top.height;
    const cam = makeCvCamera();
    setCvProjection(cam, viewIntrinsics(intrinsicsFromK(K), this.zoom, this.panX, this.panY), W, H);
    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }));
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(glModelMatrix(Tco));
    mesh.matrixWorld.copy(mesh.matrix);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(pixelToNdc(sx - 0.5, sy - 0.5, W, H), cam);
    const hit = rc.intersectObject(mesh, false)[0];
    if (!hit) return null;
    const local = hit.point.clone().applyMatrix4(mesh.matrix.clone().invert());
    return [local.x, local.y, local.z];
  }

  private onMove(e: PointerEvent): void {
    const [sx, sy] = this.evPos(e);
    this.cursor = [sx, sy];
    const d = this.drag;
    const img = store.image();
    if (!d) {
      if (store.ui.mode === "prep" && store.ui.prepTool === "brush") this.render();
      return;
    }
    if (d.kind === "pan") {
      this.panX = d.px + (sx - d.sx);
      this.panY = d.py + (sy - d.sy);
      this.sil.key = this.sil.key; // シルエットは画像解像度なので作り直し不要
      this.render();
    } else if (d.kind === "brush" && img) {
      const L = getLabels(img.id);
      if (L) {
        const [u, v] = this.toImage(sx, sy);
        strokePoint(L, u, v);
        this.render();
      }
    } else if (d.kind === "vertex" && img) {
      const c = img.contours.find((x) => x.id === d.contour);
      if (c) {
        c.points_und[d.index] = this.toImage(sx, sy);
        store.emit("pose");
      }
    } else if (d.kind === "pose" && img && img.id === d.img && store.project) {
      const K = this.K(img);
      const fx = K ? K[0] : 1000,
        fy = K ? K[4] : 1000;
      const dxi = (sx - d.sx) / this.zoom;
      const dyi = (sy - d.sy) / this.zoom;
      let T: Mat4;
      if (d.mode === "move") {
        const z = Math.max(1, d.pc[2]);
        T = mat4Mul(trans([(dxi * z) / fx, (dyi * z) / fy, 0]), d.T0);
      } else if (d.mode === "rollz") {
        const a0 = Math.atan2(d.sy - d.cs[1], d.sx - d.cs[0]);
        const a1 = Math.atan2(sy - d.cs[1], sx - d.cs[0]);
        T = rotateAbout(d.T0, d.pc, rotZ(((a1 - a0) * 180) / Math.PI));
      } else {
        const k = 0.4;
        T = rotateAbout(d.T0, d.pc, mat4Mul(rotX(dyi * this.zoom * k), rotY(-dxi * this.zoom * k)));
      }
      applyPose(img, T, "manual");
      store.emit("pose");
    }
  }

  private onUp(e: PointerEvent): void {
    const d = this.drag;
    this.drag = null;
    try {
      this.top.releasePointerCapture(e.pointerId);
    } catch {
      /* 無視 */
    }
    if (!d) return;
    if (d.kind === "brush") store.emit("labels");
    if (d.kind === "vertex" || d.kind === "pose") store.gestureEnd();
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    const img = store.image();
    const [sx, sy] = this.evPos(e);
    if (e.altKey && store.ui.mode === "align" && img && store.project) {
      const why = poseRefusal(img);
      if (why || !img.pose.T_cam_marker) {
        store.msg(why ?? "姿勢がありません", "error");
        return;
      }
      const Tco = T_cam_obj(store.project, img)!;
      const pc = mat4Apply(Tco, modelCenterObj());
      const dist = Math.hypot(pc[0], pc[1], pc[2]) || 1;
      const step = (e.shiftKey ? 0.002 : 0.01) * dist * (e.deltaY > 0 ? 1 : -1);
      const dir: Vec3 = [(pc[0] / dist) * step, (pc[1] / dist) * step, (pc[2] / dist) * step];
      store.gestureBegin();
      applyPose(img, mat4Mul(trans(dir), img.pose.T_cam_marker), "manual");
      store.gestureEnd();
      return;
    }
    const f = Math.exp(-e.deltaY * 0.0015);
    const nz = Math.min(40, Math.max(0.05, this.zoom * f));
    const k = nz / this.zoom;
    this.panX = sx - (sx - this.panX) * k;
    this.panY = sy - (sy - this.panY) * k;
    this.zoom = nz;
    this.render();
  }

  private onContext(e: MouseEvent): void {
    e.preventDefault();
    const img = store.image();
    if (!img || store.ui.mode !== "prep" || store.ui.prepTool !== "contour") return;
    const [sx, sy] = this.evPos(e);
    const sel = img.contours.find((c) => c.id === store.ui.selContour);
    if (!sel) return;
    const vi = this.hitVertex(sx, sy, sel);
    if (vi >= 0) deleteVertex(sel.id, vi);
  }

  private onKey(e: KeyboardEvent, down: boolean): void {
    const t = e.target as HTMLElement;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
    if (e.code === "Space" && !typing) {
      this.space = down;
      if (down && (t === this.top || t === document.body)) e.preventDefault();
      this.top.style.cursor = down ? "grab" : "";
      return;
    }
    if (!down || typing || !store.project) return;
    const ui = store.ui;
    const img = store.image();
    if (e.key === "Escape") {
      if (ui.cornerDraft.length) store.setUi({ cornerDraft: [] });
      if (ui.corrPending) store.setUi({ corrPending: null });
      return;
    }
    if ((e.key === "Delete" || e.key === "Backspace") && ui.mode === "prep" && ui.prepTool === "contour" && ui.selContour && ui.selVertex !== null) {
      e.preventDefault();
      deleteVertex(ui.selContour, ui.selVertex);
      return;
    }
    if (ui.mode === "align" && img && img.pose.T_cam_marker && ui.alignTool === "move") {
      const step = e.shiftKey ? 10 : 1;
      const rstep = e.shiftKey ? 5 : 0.5;
      let D: Mat4 | null = null;
      const pc = mat4Apply(T_cam_obj(store.project, img)!, modelCenterObj()) as Vec3;
      const rot = e.ctrlKey || e.metaKey;
      switch (e.key) {
        case "ArrowLeft":
          D = rot ? rotY(rstep) : trans([-step, 0, 0]);
          break;
        case "ArrowRight":
          D = rot ? rotY(-rstep) : trans([step, 0, 0]);
          break;
        case "ArrowUp":
          D = rot ? rotX(-rstep) : trans([0, -step, 0]);
          break;
        case "ArrowDown":
          D = rot ? rotX(rstep) : trans([0, step, 0]);
          break;
        case "PageUp":
          D = trans([0, 0, -step]);
          break;
        case "PageDown":
          D = trans([0, 0, step]);
          break;
        case ",":
          D = rotZ(-rstep);
          break;
        case ".":
          D = rotZ(rstep);
          break;
      }
      if (!D) return;
      e.preventDefault();
      const why = poseRefusal(img);
      if (why) {
        store.msg(why, "error");
        return;
      }
      const isRot = rot || e.key === "," || e.key === ".";
      const T = isRot ? rotateAbout(img.pose.T_cam_marker, pc, D) : mat4Mul(D, img.pose.T_cam_marker);
      store.commit("姿勢（キー）", () => applyPose(img, T, "manual"), "pose");
    }
  }

  /** 3D ビューの描画から参照する（現在の画像ビューのズーム）。 */
  snapshot(): string {
    return this.glc.toDataURL();
  }
}

export function addCorrespondence(img: ProjectImage, p_obj: Vec3): void {
  const pend = store.ui.corrPending;
  if (!pend) return;
  store.ui.corrPending = null;
  store.commit("対応点の追加", () => {
    img.correspondences.push({ uv_und: [pend[0], pend[1]], p_obj: [p_obj[0], p_obj[1], p_obj[2]] });
  });
  store.msg(`対応点 ${img.correspondences.length} 組目を追加しました`, "ok");
}

export function deleteVertex(contourId: string, index: number): void {
  const img = store.image();
  const c = img?.contours.find((x) => x.id === contourId);
  if (!img || !c) return;
  if (c.points_und.length <= 3) {
    store.msg("輪郭の頂点は 3 つ以上必要です", "error");
    return;
  }
  store.commit("頂点の削除", () => {
    c.points_und.splice(index, 1);
  });
  store.setUi({ selVertex: null });
}

export function newContourId(img: ProjectImage): string {
  return newId("c", img.contours.map((c) => c.id));
}

function segDist(px: number, py: number, a: [number, number], b: [number, number]): number {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const L2 = dx * dx + dy * dy;
  let t = L2 ? ((px - a[0]) * dx + (py - a[1]) * dy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

export { m4 };
