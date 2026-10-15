// 実測寸法によるロック（DESIGN.md §3）。measurements[].locked の参照先パラメータは変更できない。
// param は PartParams のキー。配列の要素は "size.0"、"outer.2.1" のように添字をつなぐ。
import type { Part, Project } from "../types";

function related(a: string, b: string): boolean {
  return a === b || a.startsWith(b + ".") || b.startsWith(a + ".");
}

/** その部品のパラメータ（またはその親・子のパス）がロックされた実測寸法に結びついているか。 */
export function isLocked(p: Project, partId: string, param: string): boolean {
  return (p.measurements ?? []).some((m) => m.locked && m.part === partId && related(m.param, param));
}

/** パスで params の数値を読む。数値でなければ null。 */
export function getParam(part: Part, param: string): number | null {
  let cur: unknown = part.params;
  for (const k of param.split(".")) {
    if (cur === null || typeof cur !== "object") return null;
    cur = (cur as Record<string, unknown>)[k];
  }
  return typeof cur === "number" && Number.isFinite(cur) ? cur : null;
}

/**
 * 数値パラメータを編集する。成功なら p を書き換えて null、拒否なら日本語の理由を返す（p は変えない）。
 * ロック中の実測寸法の参照先、存在しない部品・パス、数値でない値を拒否する。
 */
export function applyParamEdit(p: Project, partId: string, param: string, value: number): string | null {
  const part = p.shape.parts.find((x) => x.id === partId);
  if (!part) return `部品 ${partId} が見つかりません`;
  if (!Number.isFinite(value)) return `値が数値ではありません（${value}）`;
  if (isLocked(p, partId, param)) {
    const m = p.measurements.find((m) => m.locked && m.part === partId && related(m.param, param));
    return `「${m?.label ?? param}」は実測寸法でロックされています（${m?.value_mm} mm）。変更するには先にロックを外してください`;
  }
  const keys = param.split(".");
  if (keys.length === 0 || keys.some((k) => k === "")) return `パラメータ名「${param}」が正しくありません`;
  let cur: unknown = part.params;
  for (let i = 0; i < keys.length - 1; i++) {
    if (cur === null || typeof cur !== "object") return `パラメータ「${param}」がありません`;
    cur = (cur as Record<string, unknown>)[keys[i]];
  }
  if (cur === null || typeof cur !== "object") return `パラメータ「${param}」がありません`;
  const last = keys[keys.length - 1];
  const obj = cur as Record<string, unknown>;
  if (Array.isArray(obj)) {
    const idx = Number(last);
    if (!Number.isInteger(idx) || idx < 0 || idx >= obj.length) return `パラメータ「${param}」の添字が範囲外です`;
  }
  const old = obj[last];
  if (old !== undefined && typeof old !== "number") return `パラメータ「${param}」は数値ではありません`;
  obj[last] = value;
  return null;
}

export interface MeasurementCheck {
  id: string;
  label: string;
  expected: number;
  actual: number | null; // 参照先がない・数値でないなら null
  ok: boolean; // |actual − expected| ≤ tol_mm
}

/** 実測値と現在のパラメータの食い違いを調べる。 */
export function checkMeasurements(p: Project, tol_mm = 1e-6): MeasurementCheck[] {
  return (p.measurements ?? []).map((m) => {
    const part = p.shape.parts.find((x) => x.id === m.part);
    const actual = part ? getParam(part, m.param) : null;
    return {
      id: m.id,
      label: m.label,
      expected: m.value_mm,
      actual,
      ok: actual !== null && Math.abs(actual - m.value_mm) <= tol_mm,
    };
  });
}
