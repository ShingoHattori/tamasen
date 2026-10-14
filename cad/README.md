# cad — reBot Arm B601-DM の CAD と、稼働中モデルとの座標対応

ID 1 用マーカー台の設計に使う。手首・接続部・グリッパーの STEP を **`end_link` 座標**で書き出す。
あわせて、CAD の部品座標と稼働中 URDF のリンク座標の対応を、数値で確認した結果を残す（2026-10-13）。

**稼働中モデルは置き換えない。** 公開モデルが更新されたことだけを理由に、SDK の URDF（`rebot/…/reBot-DevArm_fixend.urdf`）や
シミュレーターの MJCF（`sim/models/borot/`）を差し替えない。このディレクトリは参照と照合のためだけに使う。

## 出所

| | |
|---|---|
| リポジトリ | https://github.com/Seeed-Projects/reBot-DevArm |
| commit | `9ac616955640702ed7836a636a17c22242dc7c38`（2026-10-13 06:51 UTC） |
| 取得先 | リポジトリ直下の `reBot-DevArm/`（git 管理外、`.gitignore`）。`git clone` して上記 commit を checkout |
| ライセンス | ハードウェア CERN-OHL-W-2.0（上流の `LICENSE`） |

使ったファイル（sha256）:

| ファイル | sha256 | 用途 |
|---|---|---|
| `hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260425.step` | `db2835ed6099a16a515f2b46bb9dd1bccd6ef561b206cebfaed5b25f8a71a4ba` | **DM 版の全体アセンブリ**（座標対応・書き出しの元） |
| `hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260625.step` | `e99525ef0d2af1ab00a4dd6febc1904f345228d9965aa60638f9bb61fa1bb9da` | 使わない（下記の注意を参照） |
| `Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf` | `bce0d4740ec97a27e30d521e890416231c6d0117093fa997d1ae7ac95204ffb6` | 公開 URDF。メッシュは `DM/meshes/` |

### 注意: DM ディレクトリの「20260625」は RS 版
- `reBot_B601_DM_v1.1_20260625.step` は、ファイル名とディレクトリが DM なのに、最上位の名前が `reBot_B601_RS_v1.0_20260625` になっている。
  サブアセンブリも `02-RS-LINK-1_ASM` などで、モーターは RobStride（`9-M-RS00`、`9-M-RS06`、`RSM*`）。座標系も Y 軸が上で、DM 版と違う。
  この内容は上流 commit `90bd264`（2026-07-09「Opensource reBot Arm RS」）で入った。
- **DM 版の正しいアセンブリは `…_20260425.step`** のほう。Damiao `DM-J4310` / `DM-J4340P` を含み、上流 readme の v1.1 の変更内容
  （Cable Restraint、関節 1 = 4340P、`02_Base_Reinforcement_Part`）とも合う。このファイルの内部の最上位名は `reBot_B601_DM_v1.1_20260625` で、
  `70ff050`（2026-07-01「update some structural parts of DM」）でも更新されている。
- 上流 readme の「ファイル構成」にある `reBot_B601_DM_v1.1_20260415.step` は存在しない。

## 稼働中モデルとの関係

| モデル | 使っている所 | 公開 DM URDF との比較 |
|---|---|---|
| SDK `reBot-DevArm_fixend.urdf`（上流 reBotArm_control_py `40ab6ce` を取り込んだもの） | `rebot/arm_state.py`（Pinocchio FK）、`oak/urdf_fk.py`、`oak/handeye.py` | 関節 1〜6 と `end_joint` の origin・axis・limit が**文字列まで同一**。リンクメッシュ（`base_link`〜`link6`）も一致（p99 ≤ 0.07 mm）。`end_link.STL` は、公開モデルのグリッパー本体に**閉じた指**（スライド 0 mm）を合わせた 1 つのメッシュ（中央値 0.000 mm、p99 1.1 mm）。指の関節はない |
| Borot-Arm_Mujoco `3f3e98d` の MJCF（`sim/models/borot/`） | `sim/`（MuJoCo） | 取り込んだメッシュ 41 個はすべて `Rebot_Arm_description/DM/meshes/` と**バイト単位で同一**。`end_link` は SDK FK と一致（`sim/verify_sim.py`、< 1e-6 mm） |

したがって `base_link`〜`end_link` の座標は、三つのモデルで同じものとして扱える。

## CAD 座標と URDF リンク座標の対応（実測ではなく、モデル同士の照合）

DM 版の全体 STEP（`…_20260425.step`）を、公開 URDF のメッシュと点からメッシュまでの距離で照合した。

- **CAD の世界座標（mm）= `base_link` を +z へ 3.295 mm 平行移動したもの。回転はない。**
  点の変換は `p_cad = 1000 · p_base + (0, 0, 3.295)`。CAD 上の台座の底面は z = 3.3 mm にある。
- **CAD の組立姿勢は q = (−π, 0, 0, 0, 0, 0) rad**（関節 1 が 180°、ほかは 0、指は閉）。関節 1 の制限 ±2.8 rad の外にある（CAD 上だけの姿勢）。
- この 2 条件で、`link2`〜`end_link` と指の残差は中央値 0.005〜0.07 mm（p90 ≤ 0.2 mm）。リンクごとに剛体 ICP をかけても、補正は ≤ 0.2 mm・≤ 0.1°。
  `base_link` と `link1` だけ、形状に数 mm の違いがある（v1.1 の台座補強と関節 1 のモーター変更。URDF のメッシュは旧形状と思われる）。

CAD のサブアセンブリと URDF リンクの対応（2 mm 以内に入る点の割合から判定）:

| URDF リンク | CAD サブアセンブリ | 主な部品 |
|---|---|---|
| `link5` | `Link5` | `03_Link5`、`DM-J4310:4`（関節 6 のモーター）、`02_Motor_Front_Spacer`、Cable Restraint |
| `link6` | `Link6` | `02_Gripper_Connector_A` / `_B`、`DM-J4310:3`（グリッパー駆動モーター）、Cable Restraint |
| `end_link` | `Gripper`（指以外） | `01_Rail_Bracket`、`02_Slider_Bracket`、`02_Gear_Connector`、`RAIL-170`、`GEAR-1M16C`、`02_Rack`、`SLIDER` |
| `finger_left_link` / `finger_right_link` | `Gripper` の `01_Finger`、`02_Slider_Extension` | 公開 URDF だけにある指の関節（SDK では `end_link` に固定） |

`end_link` は `end_joint`（fixed）で `link6` に固定されている。そのため `Link6` と `Gripper` の部品は、`end_link` から見るとすべて動かない。
`Link5` は関節 6 の回転で `end_link` に対して回る。

`end_link` 座標の向き: +x = 掌から指先の方向（指先は x ≈ 0、レール台座の後端は x = −103.4 mm）、y = 指の開閉方向、z = それと直交する方向。
`T_link6_end_link` = 回転 [[0, 0, 1], [0, −1, 0], [1, 0, 0]]、並進 (0, 0, 155.39) mm（URDF の rpy を丸めた分、約 1e-4 rad の誤差がある）。

`end_link` 座標で見た主な部品の範囲（mm）:

| 部品 | x | y | z |
|---|---|---|---|
| `01_Rail_Bracket` | −103.4 〜 −73.4 | ±92.0 | ±34.0 |
| `02_Slider_Bracket`（金属、後端） | −103.4 〜 −100.4 | ±88.1 | ±34.2 |
| `01_Finger`（閉） | −65.0 〜 −0.2 | 0 〜 ±31.4 | ±19.6 |
| `02_Gripper_Connector_B` | −160.4 〜 −150.9 | ±28.4 | ±28.4 |
| `DM-J4310:3`（グリッパー駆動） | −148.4 〜 −102.4 | ±28.5 | ±28.5 |
| `03_Link5`（関節 6 = 0 のとき） | −211.4 〜 −151.4 | ±25.0 | −22 〜 +38 |

±z 面の平坦部: `01_Rail_Bracket` の z = ±34.04 面は x −100.4〜−83.3、y ±12.5（427 mm²）。後端の `02_Slider_Bracket` が z = ±34.24 で **0.2 mm 出っ張っている**。

## マーカー台 — 取り付けリンク・固定変換・マーカー座標（**設計値。未製作・未実測**）

生成: `cad/tools/marker_mount.py`（OCP で作図。出力は `cad/out/marker_mount/` の STEP・STL・`marker_mount.json`・`preview.png`）。
寸法はすべて `end_link` 座標（mm）。基準の値は CAD の BRep から読んだ。

**形**: グリッパーの金属背板 `02_Slider_Bracket` の上端に掛ける L 字（厚さ t = 4 mm、PLA/PETG 印刷の想定）。
- 縦の板: 背板の背面（x = −103.374）に当てる。背板の上側 2 本の皿ネジ `KA3-12`（y = ±11、z = +30）で共締めする。
  - ネジは **皿頭のタッピンねじ 3 × 16 mm** に替える（元の `KA3*12mm` と同じ系列。上流の部品表のリンク先は十字穴付きタッピンねじ）。元の 12 mm に t を足した長さ。
    相手のレール台座は印刷部品で、雌ねじのない Ø2.7 の下穴に元のネジがねじ山を切っている。小ねじ（ISO 10642 など）では効かない。
    頭が t だけ後ろへ下がるので、先端は今と同じ x = −91.374 に来て、既にあるねじ山をなぞる。ねじ山の形が違うタッピンねじは樹脂のねじ山を崩すおそれがあるので、同じ系列を選ぶ。
    下穴は x = −88.374 まであり、3 mm 余る（18 mm でも底には当たらないが、余裕は 1 mm で、新しいねじ山を切ることになる）。
  - 背面の皿の窪み（90°、背面で Ø6.0、深さ 1.4。古いネジの頭が抜けて空く）に、**台の内側の円錐突起**（45°、根元 Ø5.8、高さ 0.8、Ø3.4 の通し穴）がはまる。
    突起は窪みより低くしてある。そのため x 方向は平らな面が背板に密着して決まり、y・z 方向は円錐が決める（片側 0.1 mm の隙間。`--boss-clearance` で変更できる）。
  - 新しいネジの皿頭は、台の外側の面の皿穴（90°、Ø6.4）に収まる。
  - 下端はグリッパー駆動モーター（DM-J4310、半径 28.5）を半径 29.0 の円弧で避ける。
- 上の板: z = 34.24（背板の上端。`01_Rail_Bracket` の上面は 0.2 mm 低い 34.04）から上へ t。大きさは 40 × 40 mm（x = −107.374〜−67.374、y = ±20）。
  内側の角は 45° × 1.0 mm で面取りし、背板の角（バリ・丸み）に乗り上げないようにしている（`--relief`）。
- 印刷の向き: **縦の板の外側（皿穴のある面）をベッドに置く**。円錐の突起 2 つが上を向き、マーカーを貼る板は縦に立つ。
  突起・背板に当たる面は上面になって平らに出る。皿穴はベッド側から 45° で細くなり、モーター逃げのアーチは平面形状、内側の角の面取りは 45° の斜面なので、
  サポートは要らない想定（試し刷りで確認する）。マーカーを貼る面は縦の壁になるので積層の筋が出る。
- マーカー: **ID 1、黒枠 30 mm**（印刷シートの 30 mm 版）。紙を 40 × 40 mm（台と同じ大きさ、マーカーが中央）に切り、台の縁に合わせて貼る。

**取り付けリンク・固定変換・マーカー座標**
- 取り付けリンク: **`end_link`**（グリッパー本体。顎の開閉では動かない。`end_joint` で `link6` に固定）。
- マーカー座標（`oak/pose.py` と同じ）: 原点 = 黒い正方形の中心、x = 印刷の右、y = 印刷の上、z = 面から外向き。
- 黒い正方形の中心: `end_link` 座標で **(−87.374, 0, 38.24) mm**。紙の厚み（約 0.1 mm）と印刷の誤差は含まない。
- 軸: マーカー x = `end_link` +y、マーカー y = `end_link` −x（印刷の上が手首側）、マーカー z = `end_link` +z。
- `T_end_link_marker` = 回転 [[0, −1, 0], [1, 0, 0], [0, 0, 1]]、並進 (−87.374, 0, 38.24) mm。
  `link6` で表すと、並進 (38.24, 0, 68.02) mm、回転の列 = (0, −1, 0), (0, 0, −1), (1, 0, 0)（`T_link6_end_link` を掛けた値）。

**干渉の確認**（`marker_mount.py` の出力。外す 2 本の `KA3-12` は除く）
- 背板とは面で接する（重なりの体積は 0）。
- `01_Rail_Bracket` とは 0.2 mm、モーター DM-J4310 とは 0.495 mm、最も近いネジ（`KM3-7`）とは 2.4 mm 離れている。
- 関節 6 を −180°〜170°（10° 刻み）回しても、`link5` の部品とは当たらない。
- 指は z ≤ 19.6、スライダーは z ≤ 19.2 にあり、台（z ≥ 34.24）の下を通る。台の前端 x = −67.374 は、把持中心（x ≈ −25）でつかんだブロックより後ろにある。
  ただし、つかむ向きによっては背の高いブロックが当たるかもしれない。全関節の範囲と経路でのアーム全体との干渉は、シミュレーターでまだ確認していない。

**シミュレーターへの追加と全体の干渉**（2026-10-14、robot1 の作業ツリー `~/tamasen-mount`、ブランチ `marker-mount`）
- `sim/scene.toml [marker_mount]`（`enabled`、`status = "design"`）。`sim/scene.py` がメモリ上で `end_link` に足す: 表示メッシュ（`sim/models/marker_mount/`、
  `marker_mount.py` の粗い STL）、衝突の箱 2 つ（台・縦板、group 4 = リンク凸包と同じ扱い）、黒枠 30 mm の正方形、`marker1` site（マーカー座標）。
  既存の床・障害物・自己衝突・ブロック・経路シミュレーションの判定にそのまま入る。公開 MJCF は変更しない。
- `sim/check_mount.py`（台の有無で比較）: 関節の範囲から 20,000 姿勢（各関節 25 % は制限ぎりぎり）。台なしで接触のない 11,610 姿勢のうち、
  台で新たに接触したのは **10（0.09 %）**。相手は link2 7、カメラスタンド（余裕込みの箱）7、床 5、link1 1（重複あり）、最大の食い込みは 6.2 mm（床）。
  いずれも関節 1 が ±155° 以上、関節 4 が ±90° 付近など、グリッパーが自分の腕や床に押し付けられた姿勢。link の形は粗い凸包なので、実物より厳しく出る。
  把持計画（36 配置 × 24 候補）は台の有無で変わらない（可 133 → 133）。全関節 0（折りたたみ）・模擬姿勢・実機の最終受信姿勢では、台に最も近いのは link4 で 40 mm。
  `verify_sim.py` も台ありで全項目が従来どおり（模擬配置と立てたブロックの把持成功、狭い顎の失敗）。
- `sim/render_mount.py`: 台の有無の比較、アーム全体、最も近づいた姿勢の画像（`sim/output/mount_render/`）。
- 実機の関節角はこの確認の間読めなかった（アームの電源を切っていた）。

**注意**
- 公式の D405 カメラ取り付け具（`hardware/camera-mounts/b601-camera-mounts/D405_305_Mount.step`）にも、y = ±11 mm 間隔の M3 穴と、モーターを避ける円弧がある。
  同じ背板のネジを使う方式と思われる（形からの推定。上流の文書では確認していない）。
- 台に替えると、いまの ID 1（20 mm、テープ留め）と `markers.toml [marker.1]`（`size_mm` = 20）、`[marker.1.grip]`（実測）は使えなくなる。
  寸法の更新と把持中心の測り直しが必要になる。
- 設計値は CAD からの値で、実物との一致は製作後に確かめる（hand-eye が推定する `T_ee_m1` と比べられる）。

## 干渉確認に使うもの
- マーカー台の設計: `out/wrist_gripper_in_end_link.step`（`Gripper` + `Link6` + `Link5` の 90 部品（ネジ類を含む）、**`end_link` 座標、mm**、部品名つき）。
  `Link5` は関節 6 = 0 の位置にある。関節 6 の軸は `end_link` の x 軸（y = z = 0）で、`link6` の原点は x = −155.39 mm。
  `Link5` の部品は x = −221.4〜−151.4 mm、軸からの半径は 49.3 mm 以内にある。台（x ≥ −107.374 mm）は、関節 6 をどの角度にしても `Link5` と重ならない。
  ほかのリンク（関節 5 の曲げで近づく `link4` など）との干渉は、アーム全体で確認する（下）。
- 台座からマーカーまでの変換と、アーム全体との干渉: 稼働中の FK（SDK URDF = MuJoCo と同一）に `T_end_link_marker` を足す。
  全体形状は `reBot_B601_DM_v1.1_20260425.step` を上の式で `base_link` 座標に直す。

## 再現手順（Mac で確認）

```
git clone https://github.com/Seeed-Projects/reBot-DevArm.git && git -C reBot-DevArm checkout 9ac6169
uv venv -p 3.12 cad/.venv && uv pip install -p cad/.venv/bin/python -r cad/requirements.txt
P=cad/.venv/bin/python; R=reBot-DevArm; DM=$R/hardware/reBot_B601_DM/reBot_B601_DM_v1.1_20260425.step; U=$R/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf
$P -I cad/tools/step_tree.py $DM cad/out/dm0425_tree.json 2                                  # 部品ツリー
$P -I cad/tools/step_export.py $DM cad/out/dm0425 '^(Gripper|Link6|Link5):1/[^/]+/[^/]+$'     # STL（CAD 座標）
$P -I cad/tools/register.py $U cad/out/dm0425           # 関節角 q_cad の当てはめ（約 2 分）
$P -I cad/tools/fit_offset.py $U cad/out/dm0425 0.0033  # q = (−π,0,…) で T_cad_base → (0,0,3.295) mm
$P -I cad/tools/check_links.py $U cad/out/dm0425        # リンクごとの残差と ICP 補正（約 3 分）
$P -I cad/tools/parts_in_link.py $U cad/out/dm0425 -100.8,2.2,34.0   # end_link 座標の部品範囲、点に最も近い部品
$P -I cad/tools/misc_checks.py $U rebot/reBotArm_control_py/urdf/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf cad/out/dm0425
$P -I cad/tools/step_reframe.py $DM cad/out/wrist_gripper_in_end_link.step "$M" Gripper:1 Link6:1 Link5:1
$P -I cad/tools/marker_mount.py cad/out/wrist_gripper_in_end_link.step cad/out/marker_mount [--t 4] [--marker 30]   # マーカー台と干渉確認
$P -I cad/tools/render_mount.py cad/out/dm0425 cad/out/marker_mount/marker_mount.stl cad/out/marker_mount/preview.png $U
```
`$M` は `T_end_link_cad` の上 3 行（mm、12 個の数）。q = (−π, 0, …) の FK と +3.295 mm から計算する:
`-1 0 7.346e-6 -260.4756 0 -1 -1e-4 0.01946 7.346e-6 -1e-4 1 -194.9938`（有効数字は `step_reframe` の実行時に使った値）。
