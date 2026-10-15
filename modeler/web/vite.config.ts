import { defineConfig } from "vite";

// 開発時は Python サーバー（既定 8096）へ /api と /files を中継する。本番は `npm run build` の dist/ を同じサーバーが配る。
export default defineConfig({
  base: "./",
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8096", "/files": "http://127.0.0.1:8096" } },
  build: { outDir: "dist", emptyOutDir: true, target: "es2022" },
  optimizeDeps: { exclude: ["manifold-3d"] },
});
