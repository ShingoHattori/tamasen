// サーバー API（DESIGN.md §4）。エラーはサーバーの日本語メッセージをそのまま例外にする。
import type { Project, ProjectImage } from "../types";

export class ApiError extends Error {}

async function parse(res: Response): Promise<unknown> {
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  const obj = data as { ok?: boolean; error?: string } | null;
  if (!res.ok || (obj && obj.ok === false)) {
    const plain = text && !obj ? text.replace(/<(script|style|title)[^>]*>[\s\S]*?<\/\1>/gi, " ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 200) : "";
    const msg = obj?.error || (plain ? `HTTP ${res.status}: ${plain}` : "") || `HTTP ${res.status} ${res.statusText}`;
    throw new ApiError(msg);
  }
  return data;
}

export async function req<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData || body instanceof Blob || body instanceof ArrayBuffer) {
    init.body = body as BodyInit;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new ApiError(`サーバーに接続できません（${(e as Error).message}）`);
  }
  return (await parse(res)) as T;
}

export interface ProjectSummary {
  id: string;
  name: string;
  images?: number;
  updated?: string | number;
}

function asArray<T>(x: unknown, key: string): T[] {
  if (Array.isArray(x)) return x as T[];
  const o = x as Record<string, unknown> | null;
  if (o && Array.isArray(o[key])) return o[key] as T[];
  return [];
}

function asProject(x: unknown): Project {
  const o = x as Record<string, unknown>;
  if (o && typeof o === "object" && "project" in o && o.project && typeof o.project === "object") {
    const p = o.project as Project;
    if (o.autosave_available !== undefined && p.autosave_available === undefined) p.autosave_available = o.autosave_available as boolean;
    if (o.id !== undefined && p.id === undefined) p.id = o.id as string;
    return p;
  }
  return o as unknown as Project;
}

const enc = encodeURIComponent;
const P = (pid: string) => `/api/projects/${enc(pid)}`;
const I = (pid: string, iid: string) => `${P(pid)}/images/${enc(iid)}`;

export const api = {
  listProjects: async () => asArray<ProjectSummary>(await req("GET", "/api/projects"), "projects"),
  createProject: async (name: string) => asProject(await req("POST", "/api/projects", { name })),
  getProject: async (pid: string) => asProject(await req("GET", P(pid))),
  getAutosave: async (pid: string) => asProject(await req("GET", `${P(pid)}?autosave=1`)),
  saveProject: (pid: string, p: Project, autosave = false) => req("PUT", P(pid) + (autosave ? "?autosave=1" : ""), p),
  listSamples: async () => asArray<{ name: string; title?: string; description?: string } | string>(await req("GET", "/api/samples"), "samples"),
  loadSample: async (name: string) => asProject(await req("POST", `/api/samples/${enc(name)}/load`)),
  importImages: async (pid: string, files: File[], role: string) => {
    const fd = new FormData();
    for (const f of files) fd.append("files[]", f, f.name);
    fd.append("role", role);
    return asArray<ProjectImage>(await req("POST", `${P(pid)}/images`, fd), "images");
  },
  preprocess: (pid: string, iid: string, steps: string[], options: Record<string, unknown> = {}) => req<unknown>("POST", `${I(pid, iid)}/preprocess`, { steps, options }),
  putLabels: (pid: string, iid: string, png: Blob) => req("PUT", `${I(pid, iid)}/labels`, png),
  deleteLabels: (pid: string, iid: string) => req("DELETE", `${I(pid, iid)}/labels`),
  grabcut: (pid: string, iid: string, body: unknown) => req<unknown>("POST", `${I(pid, iid)}/grabcut`, body),
  contoursFromLabels: (pid: string, iid: string) => req<unknown>("POST", `${I(pid, iid)}/contours_from_labels`, {}),
  markerPose: (pid: string, iid: string, body: unknown) => req<unknown>("POST", `${I(pid, iid)}/marker_pose`, body),
  pnp: (pid: string, iid: string, body: unknown) => req<unknown>("POST", `${I(pid, iid)}/pnp`, body),
  depthPoints: (pid: string, iid: string, step = 4) => req<unknown>("GET", `${I(pid, iid)}/depth_points?step=${step}`),
  capture: (pid: string) => req<unknown>("POST", `${P(pid)}/capture`, {}),
  setCamera: (pid: string, body: FormData | Record<string, unknown>) => req<unknown>("POST", `${P(pid)}/camera`, body),
  exportModel: (pid: string, glb: ArrayBuffer, coords: string) => {
    const fd = new FormData();
    fd.append("glb", new Blob([glb], { type: "model/gltf-binary" }), "model.glb");
    fd.append("coords", coords);
    return req<unknown>("POST", `${P(pid)}/export`, fd);
  },
};

export function fileUrl(pid: string, iid: string, name: string, bust?: number | string): string {
  return `/files/${enc(pid)}/images/${enc(iid)}/${name}` + (bust !== undefined ? `?v=${bust}` : "");
}
