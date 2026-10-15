// アプリ状態・Undo/Redo（プロジェクト JSON のスナップショット）・変更通知。
import type { Project, ProjectImage, Part } from "../types";

export type Mode = "prep" | "shape" | "align";
export type OverlayMode = "translucent" | "wire" | "outline" | "hidden";
export type PrepTool = "view" | "brush" | "contour" | "corners";
export type AlignTool = "move" | "corr";

export interface PlaneSpec {
  origin: [number, number, number];
  normal: [number, number, number];
  xAxis: [number, number, number];
}

export interface UiState {
  mode: Mode;
  selImage: string | null;
  selPart: string | null;
  layers: { photo: boolean; mask: boolean; contours: boolean; edges: boolean; model: boolean; silhouette: boolean };
  overlay: OverlayMode;
  prepTool: PrepTool;
  brushLabel: 0 | 1 | 2 | 3;
  brushRadius: number;
  selContour: string | null;
  selVertex: number | null;
  cornerDraft: [number, number][];
  alignTool: AlignTool;
  corrPending: [number, number] | null; // 対応点: 画像点を選んで、モデル点待ち
  planePreview: PlaneSpec | null;
  depthPoints: boolean;
  showGhost: boolean;
}

export type ChangeKind = "load" | "project" | "pose" | "ui" | "labels" | "model";

type Listener = (kind: ChangeKind) => void;

const MAX_UNDO = 120;

export class Store {
  pid: string | null = null;
  project: Project | null = null;
  ui: UiState = {
    mode: "prep",
    selImage: null,
    selPart: null,
    layers: { photo: true, mask: true, contours: true, edges: false, model: true, silhouette: true },
    overlay: "translucent",
    prepTool: "view",
    brushLabel: 2,
    brushRadius: 8,
    selContour: null,
    selVertex: null,
    cornerDraft: [],
    alignTool: "move",
    corrPending: null,
    planePreview: null,
    depthPoints: false,
    showGhost: true,
  };
  undoStack: string[] = [];
  redoStack: string[] = [];
  /** 最後の保存（PUT）以降の変更があるか */
  dirty = false;
  /** 最後の自動保存以降の変更があるか */
  autosaveDirty = false;
  /** ファイル（画像）キャッシュ破棄用。サーバー側で画像類が変わったら増やす */
  fileVersion: Record<string, number> = {};
  private listeners: Listener[] = [];
  private gestureSnap: string | null = null;
  private msgFn: (text: string, kind?: "info" | "error" | "ok") => void = () => {};

  on(fn: Listener): void {
    this.listeners.push(fn);
  }
  emit(kind: ChangeKind): void {
    for (const l of this.listeners) l(kind);
  }
  setMessenger(fn: (text: string, kind?: "info" | "error" | "ok") => void): void {
    this.msgFn = fn;
  }
  msg(text: string, kind: "info" | "error" | "ok" = "info"): void {
    this.msgFn(text, kind);
  }

  setProject(p: Project, pid: string): void {
    this.project = p;
    this.pid = pid;
    this.undoStack = [];
    this.redoStack = [];
    this.dirty = false;
    this.autosaveDirty = false;
    if (!p.images.some((im) => im.id === this.ui.selImage)) this.ui.selImage = p.images[0]?.id ?? null;
    if (!p.shape.parts.some((pt) => pt.id === this.ui.selPart)) this.ui.selPart = p.shape.parts[0]?.id ?? null;
    this.ui.selContour = null;
    this.ui.selVertex = null;
    this.ui.cornerDraft = [];
    this.ui.corrPending = null;
    this.emit("load");
  }

  private snap(): string {
    const p = { ...(this.project as Project) };
    delete p.autosave_available;
    return JSON.stringify(p);
  }

  private markChanged(): void {
    this.dirty = true;
    this.autosaveDirty = true;
  }

  /** 1 操作として project を変更する（変更前を Undo に積む）。 */
  commit(label: string, fn: (p: Project) => void, kind: ChangeKind = "project"): void {
    if (!this.project) return;
    const before = this.snap();
    fn(this.project);
    const after = this.snap();
    if (after === before) {
      this.emit(kind);
      return;
    }
    this.undoStack.push(before);
    if (this.undoStack.length > MAX_UNDO) this.undoStack.shift();
    this.redoStack = [];
    this.markChanged();
    void label;
    this.emit(kind);
  }

  /** ドラッグなど連続操作の開始（終了時に 1 つの Undo 単位にまとめる）。 */
  gestureBegin(): void {
    if (this.project && this.gestureSnap === null) this.gestureSnap = this.snap();
  }
  gestureEnd(): void {
    if (!this.project || this.gestureSnap === null) return;
    const before = this.gestureSnap;
    this.gestureSnap = null;
    if (this.snap() !== before) {
      this.undoStack.push(before);
      if (this.undoStack.length > MAX_UNDO) this.undoStack.shift();
      this.redoStack = [];
      this.markChanged();
    }
    this.emit("project");
  }
  get inGesture(): boolean {
    return this.gestureSnap !== null;
  }

  undo(): void {
    if (!this.project || !this.undoStack.length) return;
    this.redoStack.push(this.snap());
    this.restore(this.undoStack.pop()!);
  }
  redo(): void {
    if (!this.project || !this.redoStack.length) return;
    this.undoStack.push(this.snap());
    this.restore(this.redoStack.pop()!);
  }
  private restore(s: string): void {
    const p = JSON.parse(s) as Project;
    this.project = p;
    this.markChanged();
    if (this.ui.selContour && !this.image()?.contours.some((c) => c.id === this.ui.selContour)) this.ui.selContour = null;
    this.ui.selVertex = null;
    this.emit("project");
  }

  image(id: string | null = this.ui.selImage): ProjectImage | null {
    if (!this.project || !id) return null;
    return this.project.images.find((im) => im.id === id) ?? null;
  }
  part(id: string | null = this.ui.selPart): Part | null {
    if (!this.project || !id) return null;
    return this.project.shape.parts.find((p) => p.id === id) ?? null;
  }
  setUi(patch: Partial<UiState>): void {
    Object.assign(this.ui, patch);
    this.emit("ui");
  }
  bumpFiles(iid: string): void {
    this.fileVersion[iid] = (this.fileVersion[iid] ?? 0) + 1;
  }
  markSaved(): void {
    this.dirty = false;
    this.autosaveDirty = false;
    this.emit("ui");
  }
}

export const store = new Store();

export function newId(prefix: string, existing: string[]): string {
  let i = 1;
  while (existing.includes(prefix + i)) i++;
  return prefix + i;
}
