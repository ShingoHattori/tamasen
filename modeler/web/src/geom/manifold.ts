// manifold-3d（WASM）の遅延読み込み。ブラウザ（Vite）と node（vitest）の両方で動く。
// ブラウザ: `?url` で Vite が wasm を資産として出力した URL を locateFile で渡す。
// node: 既定の解決（manifold.js の import.meta.url 隣の manifold.wasm を fs で読む）に任せる。
import Module from "manifold-3d";
import type { ManifoldToplevel } from "manifold-3d";
import wasmUrl from "manifold-3d/manifold.wasm?url";

let wasm: ManifoldToplevel | null = null;
let loading: Promise<ManifoldToplevel> | null = null;

const isNode = !!(globalThis as { process?: { versions?: { node?: string } } }).process?.versions?.node;

/** WASM を一度だけ読み込む（2 回目以降は同じものを返す）。 */
export function loadManifold(): Promise<ManifoldToplevel> {
  if (wasm) return Promise.resolve(wasm);
  if (!loading) {
    const opts: Record<string, unknown> = isNode ? {} : { locateFile: () => wasmUrl };
    loading = (Module as unknown as (o: Record<string, unknown>) => Promise<ManifoldToplevel>)(opts).then((m) => {
      m.setup();
      wasm = m;
      return m;
    });
    loading.catch(() => {
      loading = null; // 失敗したら次回やり直す
    });
  }
  return loading;
}

/** 読み込み済みの WASM。未読み込みなら例外（先に loadManifold / initGeometry を await すること）。 */
export function manifold(): ManifoldToplevel {
  if (!wasm) throw new Error("形状エンジン（manifold-3d）がまだ読み込まれていません。initGeometry() を先に待ってください");
  return wasm;
}
