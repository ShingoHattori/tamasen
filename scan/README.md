# scan — iPad 深度スキャンからマーカー座標の 3D モデルを作る

手で回した物体（ArUco DICT_4X4_50 を 1 枚以上貼る）を iPad で撮り、RGB と位置合わせ済み深度から、
**指定したマーカーの座標系**で表したメッシュを作る。MuJoCo（`sim/`）に入れる前提。

マーカー座標は `oak/markers.toml` と同じ: 原点 = 黒い正方形の中心、x 右・y 上・z 面から外向き、
四隅順 0..3 = 印刷の左上・右上・右下・左下。出力の長さはすべて **m**。

## 環境

```
uv venv --python 3.11 scan/.venv
uv pip install --python scan/.venv/bin/python -r scan/requirements.txt
```
oak/.venv とは別（触らない）。Linux（robot1）でも同じ手順。

## 使い方

```
scan/.venv/bin/python scan/process.py <session_dir> --marker-id 2 --marker-mm 30 --skin-mask [--out DIR]
```
- `--marker-sizes 3:20,5:25` ほかのマーカーも使う（同時に写ったフレームから基準との相対姿勢を求め、基準が見えないフレームに使う）
- `--crop-mm x0,x1,y0,y1,z0,z1` マーカー座標の切り出し箱（既定 ±150 mm）。顔・背景が近いときは狭める
- `--skin-mask` 肌色の画素を捨てる（物体が肌色に近いときは使わない）。`--skin-dilate-px` で広げる
- `--min-depth 0.25` LiDAR（ARKit sceneDepth）は 25 cm 未満が不安定。`--min-conf 2` で信頼度 high のみ
- `--no-track` マーカーが見えないフレームを使わない / `--no-fill` 見えなかった所を塞がない
- そのほかの閾値は `--help`

合成データでの試験（約 50 秒）:
```
scan/.venv/bin/python scan/make_synthetic.py --run     # scan/output/synthetic/… に生成・処理・評価
```

## 処理

1. **マーカー姿勢**: 検出した四隅を歪み LUT で戻し、IPPE_SQUARE の 2 解を求める。マーカー内の深度に平面を当て、
   2 解のうち法線が深度に近い方を選ぶ（曖昧・反転を深度で解決）。法線は RGB と深度を不確かさで重みづけ、
   奥行きは「四隅の対角線の交点を通る光線 ∩ 深度平面」で決める。深度が使えず誤差比が大きい解、小さいマーカー
   （一辺 25 px 未満）、直前から法線が 60° 以上跳ぶ解は棄却。
2. **追跡**: マーカーで姿勢が決まったフレーム（アンカー）から前後へ、モデルへの ICP（点-面 ＋ 最後の段だけ色つき）で
   たどる。外れたら前回の物体位置のまわりを FPFH + RANSAC で再捕捉（直方体の裏返しに備え主軸 180° 回転も候補）。
   どの候補も「モデルが観測面より手前に出る割合（素通し矛盾）」で裏返しを弾く。アンカーに挟まれた区間で前向きの鎖が
   次のアンカーまで届けば、ずれを区間に比例配分（ループ補正）。2 パス（2 回目は 1 回目の結果で作ったモデル）。
3. **深度**: 段差（隣との差が視線 78° 相当を超える）と輪郭 2 画素を捨て、LUT で歪みを除いた深度を点群に。
4. **手の除去**: 2 mm ボクセルごとに各フレームへ投影し「表面として見えた」「手前が空いて見えた」を数えて投票。
   手は物体に対して動くので消える。最大の塊に対して小さい塊も捨てる。
5. **姿勢の微修正**: 点-面 ICP（マーカー姿勢を事前分布にしたガウス・ニュートン、上限つき）。補正の平均は差し引く。
6. **TSDF → メッシュ**: マーカー座標の TSDF。見ていないボクセルは表面の凸包の内側なら中身とみなして閉じる
   （必ず watertight。見えなかった凹みは埋まる）。2 万三角形以下へ間引き。

## セッション形式（アプリ側が守ること）

`scan-YYYYMMDD-HHMMSS/session.json`: `camera`（lidar / truedepth）、`video_w/h`、`depth_w/h`、`fps`、`source` など。
`frames` は使わず、`frames/*.json` を数える。

`frames/NNNNNN.jpg`（歪みあり RGB）、`.depth`（float32 LE、m、RGB と同じ視野、0/NaN/inf は無効）、
`.conf`（任意、uint8、ARKit の信頼度）、`.json`:
`t`、`K`（3x3、`K_ref` の画素）、`K_ref`、`dist_center`、`lut`、`inv_lut`（空なら歪みなし）、
`K_video`（あれば優先）、`T_world_cam_arkit`（記録のみ、今は使わない）。

前提:
- 画素座標は「画素 i が [i, i+1) を占める」連続座標で、解像度の違いは単純な倍率。深度画素 (u, v) の中心は
  映像の ((u + 0.5)·video_w/depth_w, (v + 0.5)·video_h/depth_h)（半画素ずらし。LiDAR の 7.5 倍では 3 映像画素の差になる）。
- `K`・`dist_center`・LUT は同じ参照寸法 `K_ref`。`K_ref`・映像・深度の縦横比は同じ（映像を切り抜いた場合は未対応で止める）。
- 歪みは Apple の lensDistortionPointForPoint の定義（`lens.py` の冒頭）。`lut` は歪みなし → 歪みあり、`inv_lut` はその逆。
- 深度は光軸方向の距離（z、m）。

## 出力（`<session>/result/`）

| ファイル | 内容 |
|---|---|
| `mesh.ply` / `mesh.stl` / `mesh.obj` | マーカー座標・m のメッシュ（頂点色つきは ply） |
| `hull.stl` | 凸包（MuJoCo の衝突用） |
| `points.ply` | 投票で残った点（マーカー座標） |
| `result.json` | 使ったフレーム数（マーカー / 追跡 / 棄却とその理由）、各フレームの姿勢、投票・ICP・追跡の統計、bbox、直方体当てはめ（`box_fit`: 寸法と RMS） |
| `overlay/*.jpg` | モデルを RGB へ重ねた確認画像（赤 = マーカーの正方形、軸 赤 x・緑 y・青 z） |
| `preview.png` | 3 方向の正射影（10 mm 格子、マーカーの正方形つき） |

MuJoCo へは、マーカー座標を物体の body 座標として `mesh.stl`（視覚）・`hull.stl`（衝突）をそのまま置ける（単位 m）。

## 精度の目安と限界

- 合成（LiDAR 256x192、20 mm マーカー、手の遮蔽あり）: 見えた面の誤差 平均 0.1〜0.35 mm、メッシュ全体 平均 0.4 mm・
  p95 1.2 mm、体積 160.2 cm³（真値 162.0）、マーカー姿勢 0.27 mm / 0.5°（中央値）。
- 実データ（TrueDepth、`scan-20261015-132738`、67 × 78 × 31 mm の紙箱、ID 2・30 mm）: 結果は result.json の `box_fit`。
  両手の指先で常に同じ所（長辺の両端）を持っているため、指の一部が投票で消えきらず、長辺が長めに出る。
  持ち替えながら回す・指の当たる位置を変える・肌色マスクを広げる、のどれかで改善する。
- iPad の LiDAR（sceneDepth）は滑らかすぎて小物の角が丸まり、25 cm 未満は不安定。小物は TrueDepth の方がよい。

## 置いた物体を iPad を動かして撮る（フォトグラメトリ、`photogrammetry/`）

手で回す方式は指が残り、マーカー 1 枚では裏面がつながらないため、物体を机に置いて iPad（背面 LiDAR + ARKit）で
周りを撮り、裏返してもう一度撮る方式を試作した（`scan-20261015-163500`、黄色の箱 67 × 78 × 31 mm、ID 0・3 が 20 mm、
ID 0 の 30 mm 版、机に ID 2 40 mm）。重い処理は robot1（`~/tamasen-scan/scan`、GPU は MuJoCo EGL）、Object Capture だけ Mac。

1. `static_fuse.py` — ARKit 姿勢で区間ごとに深度を積む（物体の中心と机の高さを得るため。LiDAR の形そのものは粗い）
2. `photogrammetry/prep_masks_static.py` — 机より上・物体のまわりの粗いマスク → `refine_masks.py`（robot1、GrabCut、約 14 s）
3. `photogrammetry/objcap`（Mac、`swiftc -O objcap.swift -o objcap`）— Apple Object Capture。約 1.5 分。裏返し前後も自動でつながる
4. `photogrammetry/markers_on_mesh.py`（robot1 GPU、約 4 s）— メッシュを 60 方向から描いて ArUco を検出し、既知の一辺で尺度、
   基準マーカーで座標を決める
5. `photogrammetry/carve.py`（robot1、約 1.5 分）— 写真のマスクによる visual hull でメッシュの膨らみを削る
6. 確認: `fit_box_silhouette.py`（写真の輪郭だけで直方体を当てる）、`measure_box.py`（向かい合う面の距離）

結果（2026-10-15）: Object Capture のメッシュは 84.9 × 70.2 × 35.0 mm（マーカー尺度）と膨らむ。深度の有無・マスクを詰めても変わらない。
ARKit 軌跡の尺度はマーカー尺度より 3.7 % 小さく、その尺度でメッシュの輪郭は写真と一致（IoU 0.92）。写真の輪郭だけの直方体当てはめは
81.9 × 68.9 × 30.9 mm（67 × 78 × 31 固定より IoU が高い）、visual hull で削ったメッシュは 82.9 × 69.2 × 32.6 mm。長辺が 78 に対して約 +4 mm 残る。
