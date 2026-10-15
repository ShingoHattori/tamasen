# modeler

実物の画像に人が形を合わせて 3D モデルを作るローカル Web アプリ。設計と契約は [DESIGN.md](DESIGN.md)。

## 使い方（最短）

robot1 では `~/tamasen/modeler` で動かす（2026-10-15 から `--host 0.0.0.0 --port 8096 --allow-from 127.0.0.1/32,192.168.32.0/24,172.16.0.0/12`、
pid は `~/tamasen/run/modeler.pid`、ログは `~/tamasen/run/modeler.log`）。ブラウザで `http://win10.hpcs-sb1124.sec.llhi.org:8096/`。

1. 「新規」でプロジェクトを作り、画像を取り込む（ファイル選択・ドラッグ＆ドロップ・「OAK から撮影」）。背景だけの画像は役割「背景」で取り込み、
   「マーカー・カメラ・属性」で背景画像に指定すると、背景差分でマスク候補と穴候補を作り直す。
2. **前処理**: マスク・穴候補をブラシで直す → 「輪郭を作る」→ 外周・穴の頂点を直して「確定」。マーカーが検出できないときは四隅を手で指定。
3. **形状編集**: 「輪郭から押し出し」（作図平面を選ぶ: obj の XY/YZ/ZX ＋オフセット、マーカー面）→ 厚さ（mm）。箱・円柱・円錐台・回転体・曲げ板・切削も追加できる。
   実測寸法を入れてロックすると、その値は変えられなくなる。
4. **位置合わせ**: マーカーの見えない画像を選び、ドラッグ／数値／対応点（PnP）でその画像の姿勢だけを合わせる → 「確定（ロック）」。
5. 「保存」（自動保存もある）→「書き出し」で `exports/<日時>/model.glb`（m、既定はマーカー座標）と `model.json`。

合成サンプル（「サンプルを読み込む」）: 穴 2 つの板（マーカーあり・裏側あり・背景画像あり）、口と肉厚のある容器。**合成データ**であり、実機の精度は確認していない。

### 制限（未対応）
- 衝突形状は作らない（`model.json` の `collision.status = "not_generated"`）。凸包は穴・空洞を失うため。凸分解は未実装。
- OAK の直接撮影は vision サービスの `/raw.png`（RGB、1280×960）を読む方式。深度は vision が出していないので、深度はファイルで取り込む。
- 対応点のモデル側の点は合成済みモデルの表面だけで選べる（部品ごとは不可）。形状の自動フィットはしない（姿勢推定だけ）。
- 画面の JS は 1 ファイル約 750 kB（分割していない）。

### 使用ライブラリ（固定版）
画面: three 0.186.1、manifold-3d 3.5.4（WASM。押し出し・回転体・ブーリアン）、vite 8.3.1、vitest 5.0.3、typescript 5.9.3（`web/package.json`、`.npmrc` は legacy-peer-deps）。
サーバー: `requirements.txt`（numpy 2.4.6、opencv-contrib-python 5.0.0.93、manifold3d 3.2.1、trimesh 5.1.1、scipy 1.17.1、pytest 8.4.2）。
robot1 には node がないので、画面はビルド済みの `web/dist` をコミットしている（変更したら Mac で `npm run build` してコミット）。

## サーバー（`server/`）

Python の標準ライブラリだけの HTTP サーバー（API と画面の配信）。既定 `127.0.0.1:8096`。

```
modeler/.venv/bin/python modeler/server/app.py --port 8096
# 手元から robot1 のサーバーを見るとき
ssh -N -L 8096:127.0.0.1:8096 win10.hpcs-sb1124.sec.llhi.org      # → http://127.0.0.1:8096/
```

| オプション | 既定 | |
|---|---|---|
| `--host` | 127.0.0.1 | 0.0.0.0 で LAN に出すときは `--allow-from` も指定する |
| `--port` | 8096 | |
| `--projects-dir` | modeler/projects | プロジェクト一式（git 管理外） |
| `--vision-url` | http://127.0.0.1:8090 | OAK の vision サービス。使うのは `GET /state.json` と `GET /raw.png` だけ。動いていなくても他の機能は使える |
| `--allow-from` | 127.0.0.1/32,::1/128 | これ以外のアドレスからの要求は 403 |
| `--static` | modeler/web/dist | 画面（`cd modeler/web && npm run build`）。開発時は `npm run dev`（5173 番、/api と /files を 8096 へ中継） |

- **アームの API は呼ばない**（`/arm/...` に要求を出すコードはない）。カメラも動かさない（vision の画像を読むだけ）。
- 環境: `uv venv --python 3.11 modeler/.venv && uv pip install --python modeler/.venv/bin/python -r modeler/requirements.txt`
- 構成: `app.py`（HTTP・各 API）、`project.py`（保存形式・検証・ロック判定・原子的な書き込み）、
  `preprocess.py`（EXIF の向き、歪み補正 `K_und = K`、ArUco 検出、品質、ラベル候補、穴候補、エッジ、輪郭、GrabCut、マーカー姿勢・PnP、深度）

### 前処理の要点

- ラベル候補: 背景のみの画像（role = background）があれば差分、なければ GrabCut。穴を埋める処理はしない（3×3 の open/close だけ）。
  物体に囲まれた領域は、背景の差分が小さい（背景画像あり）か、物体のすぐ外の背景の色に近い（なし）ときだけ「穴」(3) にする。
  暗いだけの領域は穴にしない（物体 2 のまま。本当に穴なら穴ブラシで直す）。
- 背景画像を後から取り込んだときは、既存の画像で「前処理のやり直し」（`preprocess`）を押すと差分を使う。
- GrabCut の候補は、マーカーに触れる連結成分（マーカーが見えないときは最大の成分）だけを残す。手は物体として残るので、背景ブラシで消す。

## 合成サンプル（`samples/`）

**合成データ（実物の写真ではない）**。K 既知・弱い樽型の歪み・雑然とした背景・一部の画像に手。

```
modeler/.venv/bin/python modeler/samples/make_samples.py          # 画像・truth.json・取り込み済みプロジェクト（数秒）
```

- `plate_synthetic`: 板 120×60×3 mm、貫通穴 2 つ（φ12、長穴 20×8）、DICT_4X4_50 ID 7（30 mm）。黒い塗装の円（穴ではない）。裏側 2 枚・背景のみ 1 枚
- `container_synthetic`: カップ（外半径 35・高さ 80・肉厚 3、上が開口）、側面のカードに ID 11（30 mm）。開口を覗く画像・マーカーが見えない画像
- 画面の「サンプルを開く」（`POST /api/samples/<name>/load`）は取り込み済みプロジェクトを複製する。形状・固定変換・姿勢は空から作る（真値は入れない）

## 試験

```
modeler/.venv/bin/python -m pytest modeler/tests -q          # 10 秒程度。実機・vision サービスは使わない
```
