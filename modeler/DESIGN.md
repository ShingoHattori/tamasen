# modeler — 実物画像に人が形を合わせて作るデジタルツイン用 3D モデラー（設計・契約）

固定した OAK-D Lite の前で、マーカーを固定した物体の位置・向きを人が変えて撮った画像（マーカーが見えない裏側も含む）に、
人が形状を合わせて 3D モデルを作るローカル Web アプリ。全自動復元はしない。RGB のみで全工程が完了し、深度は任意の参考情報。
有料 API・クラウド送信・重い学習モデルは使わない。UI は日本語。

この文書はサーバー（Python）・形状生成（TS）・画面（TS）の**共通の契約**。型の正本は `web/src/types.ts`、
Python 側は `server/project.py` に同じ定義を持つ。変更するときは両方とこの文書を直す。

## 1. 座標系・変換・単位（全コード・保存形式で共通）

| 記号 | 定義 |
|---|---|
| `cam` | カメラ（OpenCV）: x 右、y 下、z 前方。原点は光学中心。**歪み補正後の画像**に対するピンホール |
| `marker` | 基準マーカー: 原点 = 黒い正方形の中心、x 右、y 上、z はマーカー面から外向き。四隅の順 0..3 = 印刷の左上・右上・右下・左下（OpenCV ArUco と同じ）。`oak/markers.toml` と同じ定義 |
| `obj` | 物体モデル座標。部品はすべてこの座標で定義する |
| `img` | 画素: u 右、v 下。画素中心が整数座標（OpenCV と同じ）。原画像と歪み補正画像の 2 種類があり、どちらかを明記する（`*_raw` / `*_und`） |

- 変換 `T_a_b` は **b 座標の点を a 座標へ移す**: `p_a = T_a_b · p_b`。4×4 同次行列、**行優先**の 16 要素配列で保存する。
- 投影: `p_cam = T_cam_marker[i] · T_marker_obj · p_obj`、`uv_und = K_und · p_cam / z`。
- **長さの単位はプロジェクト内ですべて mm**（形状パラメータ、変換の並進、実測寸法、深度点群の表示）。角度は保存は **度**。
- 書き出し: GLB は glTF の規約どおり **m**（mm × 0.001）、座標は `obj` か `marker`（書き出し時に選ぶ。既定 `marker`）。
  glTF は y 上が慣例だが、**変換しない**（JSON に軸定義を明記する）。
- 4 つのデータを分離して持つ:
  1. 形状 `shape`（全画像で共通、`obj` 座標）
  2. 固定変換 `T_marker_obj`（全画像で共通。形状編集モードの「固定変換の編集」でだけ変わる）
  3. 画像ごとの撮影姿勢 `images[i].pose.T_cam_marker`（その画像の位置合わせでだけ変わる）
  4. カメラの内部パラメータ・歪み `cameras[c]`

## 2. プロジェクトの保存形式

`<projects_dir>/<project_id>/` に一式を置く。サーバーが管理する。

```
project.json               # 下記。自動保存は project.autosave.json（一定間隔・変更時）、保存で project.json に確定
images/<image_id>/
  original.<ext>           # 取り込んだ元画像（変更しない）。EXIF の向きが 1 以外なら、回転を適用した画素を original.png に書き、
  upload.<ext>             #   送られたバイト列そのものを upload.<ext> に残す（向きが 1 なら upload.* はない）
  undistorted.png          # 歪み補正（校正なしなら original と同じ画素をコピー）。K_und = K（下記）
  labels_auto.png          # 自動候補のラベル（下記の値、8 bit、歪み補正画像の解像度）
  labels.png               # 人が修正したラベル（なければ labels_auto を使う）。「候補に戻す」で削除
  edges.png                # エッジ候補（Canny、8 bit）
  depth.png / depth.npy    # 任意。depth.png は 16 bit mm、depth.npy は float32 m（どちらかを meta に記録）
exports/<timestamp>/model.glb, model.json
```

- **歪み補正画像のピンホール `K_und` は `K` と同じ**（getOptimalNewCameraMatrix は使わない）。画素の大きさ・主点が原画像と同じで、
  校正なしなら undistorted = original。原画像の外から来る画素（糸巻き型の四隅など）は黒、ラベルは 0。
- マーカーの四隅は原画像で検出し、歪み補正画像の上で四辺に直線を当てはめて磨く（`refine: "edges"`。失敗したら cornerSubPix のまま `"subpix"`）。
  `corners_raw` はそこから歪みを掛け直した値。
- ラベルの PNG（`PUT .../labels`）は 8 bit の 1 チャンネル、または RGB(A) で **R にラベル値**（0〜4）。大きさは歪み補正画像と同じ。

ラベル値: `0` 未定義、`1` 背景、`2` 物体、`3` 穴として見える背景（物体の内側に見える背景）、`4` マーカー（物体外形とは別に保持）。
マーカー領域は `labels` にも描くが、四隅は `markers[].corners_*` に別途保持するので、形状から除外しても検出情報は失われない。

`project.json`（`version: 1`）:

```jsonc
{
  "version": 1,
  "name": "ヘラ",
  "units": { "length": "mm", "angle": "deg" },
  "frames_doc": "T_a_b maps b to a, row-major 4x4; cam=OpenCV(x right,y down,z fwd); marker=center of black square, x right,y up,z out",
  "marker": { "dictionary": "DICT_4X4_50", "id": 7, "size_mm": 30.0 },   // 基準マーカー。size は黒い正方形の一辺
  "cameras": { "oak": { "width": 1280, "height": 960, "K": [9 floats 行優先], "dist": [k1,k2,p1,p2,k3,...], "K_und": [9], "source": "file|vision|manual|none" } },
  "T_marker_obj": [16],          // 固定変換（既定は単位行列）
  "T_marker_obj_locked": false,
  "shape": { "parts": [Part, ...] },          // §3
  "measurements": [ { "id": "...", "label": "板厚", "part": "p1", "param": "thickness", "value_mm": 2.0, "locked": true } ],
  "attributes": { "mass_g": null, "material": "", "notes": "" },
  "background_image": null,      // 背景のみの画像の image_id（差分の背景除去に使う）
  "images": [ Image, ... ],
  "settings": { "autosave_s": 20 }
}
```

`Image`:

```jsonc
{
  "id": "im_0003", "file": "original.jpg", "camera": "oak", "width": 1280, "height": 960,
  "orientation": 1,                    // EXIF。取り込み時に回転を適用し、適用後は 1（元の値は import_info.exif_orientation）
  "role": "object" | "background",
  "quality": { "blur_var": 123.4, "overexposed_pct": 0.2, "underexposed_pct": 1.0, "ok": true, "notes": ["ぶれ"] },
  "markers": [ { "id": 7, "corners_raw": [[u,v]x4], "corners_und": [[u,v]x4], "source": "auto|manual", "refine": "edges|subpix" } ],
  "pose": {
    "T_cam_marker": [16] | null,
    "source": "marker|manual|pnp|none",
    "locked": false,                   // 確定。ロック中はこの画像の姿勢を変える操作を全部拒否
    "quality": { "reproj_px": 0.4, "ambiguity": 0.1, "note": "" }
  },
  "contours": [ { "id": "c1", "kind": "outer|hole", "points_und": [[u,v],...], "status": "candidate|confirmed" } ],
  "correspondences": [ { "uv_und": [u,v], "p_obj": [x,y,z] } ],   // 対応点による姿勢推定用（p_obj は mm）
  "depth": null | { "file": "depth.png", "unit": "mm", "invalid": 0, "registered_to": "rgb_und" },
  // 以下はサーバーが書く参考情報（任意）
  "import_info": { "filename": "IMG_1.jpg", "exif_orientation": 6, "bytes": 123456, "sha256": "..." },
  "preprocess_info": { "mask_method": "background_diff|grabcut|background_image", "notes": ["..."], "steps": ["..."], "updated": "2026-10-15 18:00:00" }
}
```

- サーバーが画像を足す・前処理する・姿勢を書く（marker_pose）・カメラを変えるときは、project.json と（それより新しければ）
  project.autosave.json の両方に同じ変更を入れる。保存（PUT、autosave なし）は project.json を書いて自動保存を消す。

## 3. 形状（編集可能なパラメータを保存する。三角形メッシュは保存しない）

`Part`（共通）:
```jsonc
{ "id": "p1", "name": "刃", "op": "add" | "subtract",       // subtract は切削形状（他部品との差分）
  "visible": true, "transform": { "t_mm": [x,y,z], "r_deg": [rx,ry,rz] },   // T_obj_part = Trans(t)·Rz·Ry·Rx（内因的 XYZ ではなく固定軸 X→Y→Z の順に回す）
  "kind": "box" | "cylinder" | "frustum" | "extrude" | "revolve" | "sheet",
  "params": { ... } }
```

| kind | params（mm・度） | 部品座標での置き方 |
|---|---|---|
| `box` | `size: [x,y,z]` | 中心が原点 |
| `cylinder` | `radius, height, segments` | 軸 = +z、z ∈ [-h/2, h/2] |
| `frustum` | `radius_bottom, radius_top, height, segments` | 同上（底面 z=-h/2） |
| `extrude` | `outer: [[x,y],...]`, `holes: [[[x,y],...], ...]`, `thickness`, `symmetric: bool` | 輪郭は部品の xy 平面、押し出しは +z（symmetric なら ±t/2）。穴は保持される |
| `revolve` | `profile: [[r,z],...]`（閉じた断面、r ≥ 0）, `angle_deg: 360`, `segments` | z 軸まわりに回す。容器は肉厚のある断面（外側→口→内側）で描けば空洞と口が残る |
| `sheet` | `path: [[x,z],...]`（中心線、部品の xz 平面の折れ線）, `thickness`, `width`, `bend_radius`（0 で角）, `sheet_holes: [{ "s": 弧長位置, "y": 幅方向, "shape": "circle|slot|poly", "d"/"w,l"/"pts" }]` | 中心線を y 方向に `width` だけ押し出した板。区間ごとの折れ曲がり（ヘラの柄と刃）。穴は板厚方向に貫通 |

- `sheet` の `bend_radius` は板の**中心線**の曲げ半径（内側半径 = R − t/2。0 < R < t/2 は不可）。`sheet_holes` の `poly` の点は穴の (s, y) からの相対、`slot` の `l` は中心線方向の全長・`w` は幅。曲げ部にかかる穴は不可。
- `visible` は表示だけの設定で、合成・書き出しには常に全部品を使う（非表示の subtract も削る）。
- 合成: `add` の部品を順に union、その後 `subtract` を difference（部品の並び順）。結果は 1 つの多様体（穴・空洞はそのまま）。
- 穴・空洞は**形状として**持つ（透明テクスチャで代用しない）。「メッシュを閉じる」処理はしない。
- 輪郭の作図平面（extrude の部品座標）は**明示的に**決める: 画像でなぞった点は、その画像の姿勢で光線を作り、
  指定した 3D 平面（`obj` 座標の平面: 原点・法線・面内 x 軸）との交点として部品座標へ写す。斜めの画像を正投影とはみなさない。
  平面の既定候補: obj の XY/YZ/ZX 面（＋オフセット）、マーカー面、既存部品の面。
- 実測寸法 `measurements[].locked = true` の参照先パラメータは、UI の数値欄・ドラッグ・自動処理で変更できない。

## 4. サーバー API（Python、`server/app.py`、既定 127.0.0.1:8096、`--host/--port/--projects-dir/--vision-url/--allow-from`）

すべて JSON（画像は multipart / バイナリ）。エラーは `{"ok": false, "error": "日本語の説明"}`。成功は `{"ok": true, ...}`。
状態コード: 400 入力の誤り、403 `--allow-from` 外、404 なし、409 ロック中、422 画像処理できない、502 vision サービスに届かない。
アームの API（`/arm/...`）は呼ばない。vision サービスで使うのは `GET /state.json`（`intrinsics.K`・`dist_coeffs`・`image`）と `GET /raw.png` だけ。

| メソッド・パス | 内容 |
|---|---|
| `GET /api/projects` | 一覧 |
| `POST /api/projects` `{name}` | 新規 → project |
| `GET /api/projects/<pid>` | `{project}`（autosave が新しければ `autosave_available: true` を付ける。各画像に `has_labels`）。`?autosave=1` で自動保存の内容 |
| `PUT /api/projects/<pid>` | project.json を保存（全体）。`?autosave=1` なら autosave 側へ。形の検証あり。ロック中（保存済みと新しい内容の両方で locked）の姿勢・`T_marker_obj` を変える内容は 409。サーバーにない画像 id は 400 |
| `POST /api/projects/<pid>/images` multipart `files[]`, `role` | 取り込み → 前処理 → 追加した Image[] |
| `POST /api/projects/<pid>/images/<iid>/preprocess` `{steps: [...], options}` | 前処理のやり直し（labels_auto・edges・markers・contours 候補・quality）→ `{image}`。steps ⊂ undistort, markers, quality, mask, edges, contours（省略で全部）。options: dictionary, marker_id, replace_manual, use_background, rect [x,y,w,h], grabcut_iters, diff_threshold, canny_lo/hi, epsilon_px。labels.png（人の修正）は変えない |
| `GET /files/<pid>/images/<iid>/<name>` | 画像類（original, undistorted.png, labels_auto.png, labels.png, edges.png, depth.*）。`/files/<pid>/exports/<ts>/model.*` も |
| `PUT /api/projects/<pid>/images/<iid>/labels` body = PNG | 修正したラベルを保存 |
| `DELETE /api/projects/<pid>/images/<iid>/labels` | 候補に戻す |
| `POST /api/projects/<pid>/images/<iid>/grabcut` `{fg_strokes, bg_strokes}` or uses labels.png as seeds | 前景・背景指定からマスクを再計算（ラベル 3・4・0 は保持）→ labels.png に保存。stroke = `{points: [[u,v],...], radius}`（und 画素）。`seed: "none"` で既存ラベルを種にしない |
| `POST /api/projects/<pid>/images/<iid>/contours_from_labels` | ラベルから外周・穴の輪郭候補（頂点を間引く、穴は物体領域内のラベル 3 から） |
| `POST /api/projects/<pid>/images/<iid>/marker_pose` `{corners_und?, corners_raw?, marker_id?, write?}` | マーカー四隅（manual を優先、なければ auto。本文で四隅を渡してもよい）から IPPE_SQUARE で姿勢 → pose（source=marker）をプロジェクトに書く（`write: false` で計算だけ）。`solutions` に 2 解、`pose.quality.ambiguity` = 誤差(best)/誤差(2 番目)。ロック中は 409 |
| `POST /api/projects/<pid>/images/<iid>/pnp` `{correspondences, T_marker_obj?}` | 対応点で姿勢推定（p_obj は `T_marker_obj` でマーカー座標へ）。結果は**確定しない**候補として返す（`solutions`、各 `T_cam_marker`・`rms_px`・`residuals_px`・`methods`、全体の `ambiguous`・`degenerate`・`coplanar`・`notes`）。degenerate: 4 点未満・3D がほぼ一直線・画像上の広がりが小さい。ambiguous: 2 番目の RMS ≤ 1.5 × max(best, 0.5 px) |
| `POST /api/projects/<pid>/images/<iid>/depth?registered_to=rgb_raw\|rgb_und&invalid=0` body = PNG/NPY（multipart `file` も可） | 深度の追加。既定 rgb_raw（原画像に位置合わせ済み）とみなし、歪み補正と同じ写像（最近傍）で rgb_und に写して保存 |
| `GET /api/projects/<pid>/images/<iid>/depth_points?step=4` | 深度点群（cam 座標 mm、無効値除外） |
| `POST /api/projects/<pid>/capture` `{role?}` | OAK（vision サービス `--vision-url` の `/raw.png` と `/state.json` の内部パラメータ）から 1 枚取り込む。カメラ oak が未校正なら vision の値を設定する。未接続なら 502 とエラー文 |
| `POST /api/projects/<pid>/camera` | 校正を設定 → そのカメラの全画像の歪み補正をやり直す（labels.png・und 基準の深度・輪郭・対応点は原画像経由で新しい und へ写す。ロックしていない marker 由来の姿勢は解き直す。ロック中は変えない）。本文は JSON `{K(9 または 3×3), dist, width, height, camera?}`、multipart `file`（校正 JSON、source=file）、または `{"from": "vision"}`。→ `{camera, notes, project}` |
| `POST /api/projects/<pid>/export` multipart `glb` + `options`（JSON 文字列: `{coords: "marker"\|"obj", project?}`） | ブラウザで作った GLB を受け取り、model.json（単位・座標・マーカー・固定変換 mm と m・部品・実測・属性・`collision: not_generated`・trimesh による検証: watertight・オイラー数・種数 = 貫通穴の数・体積・bbox）と一緒に exports/ へ保存 → `{dir, files, model}` |
| `GET /api/samples` / `POST /api/samples/<name>/load` | 合成サンプル（`samples/<name>/`、名前・JSON に「合成データ」と明記）をプロジェクトとして読み込む。形状・固定変換・姿勢は空（真値 truth.json は入れない） |
| `GET /api/health` | 生存確認 |

## 5. 画面（`web/`、TypeScript + Three.js + manifold-3d）

- モードは 3 つで、常に画面上部に表示し切り替える。モードごとに操作できるものだけを有効にする:
  1. **前処理**（画像ごと）: 表示切替（元／マスク／輪郭／エッジ）、前景・背景・穴ブラシ、マーカー四隅の手動指定、輪郭頂点の追加・移動・削除、候補の確定
  2. **形状編集**（全画像共通）: 部品の追加・数値編集・移動回転、輪郭から押し出し（作図平面を明示選択）、回転体・板の断面編集、
     実測寸法の入力とロック、固定変換 `T_marker_obj` の編集（別ボタンで明示的に開く）
  3. **位置合わせ**（選択画像のみ）: その画像の `T_cam_marker` だけを移動・回転・対応点 PnP で変える。確定でロック
- 画像ビュー（写真とモデルの重ね合わせ: 半透明・ワイヤー・輪郭線・写真のみ・モデルのみ、ズーム・パン、外周と穴の輪郭の比較）と
  3D ビュー（オービット、部品、マーカー、カメラ位置、深度点群）を並べる。
- Undo/Redo はプロジェクト状態のスナップショット（形状・変換・姿勢・ラベル以外の JSON）で行う。
- 書き出し: GLB（表示形状、穴付きのまま）と model.json。衝突形状は**未対応**と明記し、凸包にはしない（`collision: {"status": "not_generated", "reason": "..."}`）。
