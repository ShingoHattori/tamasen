# tamasen — 作業コンテキスト（2026-10-16 更新）

reBot Arm B601-DM と OAK カメラで、ArUco マーカー付きの物体（直方体ブロック、rod_jig など）とグリッパーを認識し、
把持するためのリポジトリ。**2026-10-16 から Claude Code も robot1 の上で動かす**（`~/tamasen` で `claude`）。
Mac は Xcode（iPad アプリ `scan/ios`）と Apple Object Capture（`scan/photogrammetry/objcap`）にだけ使う。

## 作業の進め方（依頼者の指示）
- 返答・報告は日本語で短く。時間をかけない・余計なことをしない・最短で結果を出す。並列にできるものは sub agent / background job に回す。
- 区切りごとに main へコミットしてよい（**push はしない**）。コミットの末尾に `Co-Authored-By: Claude <noreply@anthropic.com>` 系の行。
- 依頼者が現地でアームを触れる時間は限られる。実機で試す前にシミュレーションと UI を済ませておき、現地では押すだけにする。
- **robot1 では vision（OAK）・arm（50 Hz 制御）・sim が常に動いている。重い計算で巻き込まない**:
  `nice`、`OMP_NUM_THREADS` などを小さく、並列 4〜6、1 プロセスのメモリ上限（`prlimit --as` など）。負荷平均 < 8 を目安。
  2026-10-15 12:16 UTC、検証スクリプトがメモリ 15 GB を使い OOM → OAK が X_LINK_ERROR で止まり、vision が 13 時間映像を止めたままになった。
  重い処理の後は `curl -s 127.0.0.1:8090/state.json` の `fps` と `t`（古くないか）を確かめる。止まっていたら（アームのトルクが切れているのを確認して）`./run.sh restart vision`。
- ssh 越しの `pkill -f` / `pgrep -f` は自分のコマンド行にも一致するので使わない（pid ファイルを使う）。

## マシン

| | robot1（作業機） | Mac（これまでの開発機） |
|---|---|---|
| 接続 | `ssh win10.hpcs-sb1124.sec.llhi.org`（user `cely`、192.168.32.20 / DHCP） | — |
| OS | Ubuntu Server 24.04.3、kernel 6.8.0-147-generic | macOS |
| GPU | RTX 2080 SUPER、`nvidia-driver-595-open`（ubuntu-drivers 推奨版、ビルド済み署名モジュール、DKMS なし）。CUDA Toolkit は未導入 | — |
| アーム | USB-CAN（HDSC CDC `2e88:4603`）。2026-10-15 時点で `/dev/ttyACM1` | — |
| カメラ | OAK-D-Lite `1844301091A70A1300`（ハブ経由 USB3、30 fps。起動前に `03e7:2485` の USB2 で見えるのは正常） | — |

- robot1 はデュアルブート（Windows あり）。起動順と GRUB の既定は Ubuntu。Secure Boot は無効。
- GPU ドライバ作業の記録: robot1 の `/var/log/gpu-driver-work/install-20261013-080453.log`。
- `/etc/sudoers.d/99-nopasswd-current-user`（cely NOPASSWD: ALL）がある。ドライバ作業用に作られたもので、
  残すか消すかはまだ決まっていない。不要になったら `sudo rm` で削除する。
- GitHub: `git@github.com:ShingoHattori/tamasen.git`。robot1 の公開鍵は登録済みで、`~/tamasen` に clone 済み。

## リポジトリ構成

- `oak/` — OAK（DepthAI 3.x）のツール群。**`oak/TUTORIAL.md` の Traps を必ず読むこと**（パイプラインが無言で止まる落とし穴が多い）。
  - `oak.py`（list / serve）、`capture_still.py`、`record_video.py`、`codes.py` — 既存
  - `vision.py` + `static/` + `pose.py` + `world.py` + `recorder.py` + `markers.toml` — 認識・Web UI（後述）
- `marker/` — 印刷用 ArUco シート。辞書は **DICT_4X4_50**。`aruco_a4_compact*.pdf` には ID 0〜3 の 20/30/40/50(/60) mm 版がある（PDF の解析で確認）。
- `cad/` — 公開 CAD（Seeed-Projects/reBot-DevArm `9ac6169`、`reBot-DevArm/` に clone、git 管理外）と稼働中 URDF の座標対応、ID 1 マーカー台の取り付け案（`end_link`）。**DM の全体 STEP は `…_20260425.step`**（`…_20260625.step` は中身が RS 版）。詳細は `cad/README.md`。
- `rebot/teach.py` — ダイレクトティーチング（記録・再生）。`motorbridge` を使う。
- `rebot/reBotArm_control_py/` — 上流 reBotArm_control_py の `40ab6ce`（上流に LICENSE ファイルなし。MIT は未確認） に、
  ローカルの重力補償・校正の変更を加えたもの。通常のファイルとして取り込んでいる（submodule ではない）。
- venv はコミットしていない（`.gitignore`）。依存関係は `oak/pyproject.toml`、`oak/requirements.txt`、`rebot/reBotArm_control_py/uv.lock` から作り直す。

## 絶対の規則（依頼者の指示、2026-10-13）
- **Claude Code から実機アームを動かさない。** 実機の動作（有効化・移動・把持・トルク切り）は、Web UI のボタンを依頼者が押したときだけ。
  Claude は `--allow-motion` 付きで arm サービスを起動しない。実機の `/arm/api/*`（POST）を呼ばない。テストでも呼ばない（試験は `--fake` を別ポートで）。
- 実機を動かすときは、必ず事前に経路を MuJoCo でシミュレートしてチェックし、妥当だったものだけを実行する。
  実行中・保持中は常に「計画 − 実測」のフィードバック補正をかける（重力補正の差でずれるため）。
- カメラスタンドが映像の下端にある。不用意な動作でぶつかる（シミュレーターには `scene.toml` の camera_stand / camera_head として概略の箱で登録）。
- カメラは動かさない（hand-eye 校正が壊れる）。vision・sim の再起動はできればアームのトルクが切れているときに行う。

## 起動・停止（robot1）
```
./run.sh start            # vision(8090) + arm 読み取り専用(8091) + sim(8092)
./run.sh stop | status | restart vision | log sim
```
実機操作を有効にするときは **依頼者が** `ARM_ARGS="--port /dev/ttyACM1 --allow-motion" ./run.sh restart arm`。

| ポート | サービス |
|---|---|
| 8090 | vision（`--host 0.0.0.0 --allow-from 127.0.0.1/32,192.168.32.0/24,172.16.0.0/12`）。`/state.json`、`/raw.png`、`/arm/…`・`/sim/…` を中継 |
| 8091 | arm（`/dev/ttyACM1`、motion 許可は依頼者が起動したときだけ） |
| 8092 | sim |
| 8096 | modeler（`modeler/`、pid は `run/modeler.pid`、ログ `run/modeler.log`） |
| 8093 | 2026-10-13 から残っている試験用 fake arm（`--fake`）。使っていない |
| 8097–8099 | 試験用 fake arm / test sim / test vision（replay）。`/tmp/autotest_up.sh`・`/tmp/autotest_down.sh` |

## そのほかのディレクトリ（2026-10-15〜16 に追加）
- `scan/` — iPad（TamasenScan アプリ、`scan/ios`、xcodegen、Mac でビルド）で物体をスキャンし、マーカー座標の 3D モデルにする試み（`scan/README.md`）。
  手持ちで回す方式（TrueDepth）と、置いて iPad を動かす方式（LiDAR+ARKit → Object Capture → robot1 の GPU でマーカー検出と visual hull）。寸法はまだ数 mm 大きい。
- `modeler/` — 写真に人が形を合わせて 3D モデルを作るローカル Web アプリ（`modeler/README.md`、`DESIGN.md`）。robot1:8096 で稼働。合成サンプルのみ検証済み、実画像はテスト箱（ID 2）で途中まで。
- `cad/tools/rod_jig.py` → `cad/out/rod_jig/`（棒をつかむ治具。git 管理外）。`sim/models/rod_jig/` にコピー、`sim/rod_jig_scene.py`・`sim/check_rod_jig.py` で把持のシミュレーション。
  治具のマーカーは **ID 13**（黒 30 mm、余白込み 40 mm）、`oak/markers.toml [marker.13]` に貼った向き（`T_marker_jig`）を登録済み。
  結果: 差し込み直前（各段）の視覚補正で約 1 mm に詰めれば成功 93〜98 %、補正なしは 20〜42 %。平置きは手のひらが床に当たる → 台が要る（2026-10-16 依頼者が台に載せた）。
  台の上（2026-10-16）: 下面 ≈ 107 mm・傾き 4°・入口面は台座 x から約 90°。台の高さ 60〜120 mm のどれでも経路あり（床すき間 ≥ 47 mm、ID 1 と ID 13 が全行程で見える）。
  **引き継ぎ（2026-10-16、Mac 側セッションから）** — UI から箱と同様に治具をつかむための統合は途中:
  - 済: （ が既定。true で本番 sim のシーンに治具と台が入る。置き方 support_height_m / tilt、差し込み開き 44 mm、深さ 45 mm、一段 5 mm）、
    （[jig] の読み込み、markers.toml [marker.13] の T_marker_jig、 など）、（治具 46 凸片・台・CAD 指）、
     の  /  /  / （未試験。平置きは床すき間で拒否、台なら通る想定）、（検証。）。
    箱の  は変更前後で同じ結果（standing_block の滑り 0.026→0.029 mm のみ）。
  - 未:  に治具の同期（vision の ID 13 → T_base_jig）・治具の計画・自動シーケンス（接近前 → 近接 → 一段ごとに視覚補正しながら差し込み → 閉じる → 持ち上げ）・UI のボタン、
    fake arm 8097 / test sim 8098 / test vision 8099（）での通し試験、本番 sim の再起動（[jig] enabled = true）。
  - 方針: 位置合わせは各段で常にかけてよい（依頼者）。治具が動かなければ押し込みの許容は上がるはず → 差し込みの最後は力（接触）で止める、開きは 44〜46 mm で調整可。

## 構成（2026-10-13 時点、commit は git log 参照）
- `oak/vision.py` — RGB のみ（ステレオ深度は使わない）。1280×960 で検出、ブラウザへは 960 幅の縮小 MJPEG。
  四隅・ID・軸、マーカー姿勢（両 IPPE 解を保持、ok / ambiguous / flip / rejected）、ID 2 暫定 world（毎フレーム追従 / 登録して固定、ずれ検知）、
  ID 0・3 それぞれからのブロック姿勢（平均しない）、静止画保存（元画像 + 注釈 + 状態）、録画（FFV1、上限 2000 MB / 120 s、明示操作のみ）、`--replay`。
- `oak/markers.toml` — 寸法・取り付け・校正値を *_status（unset / sample / measured / calibrated）付きで管理。内部は m・rad。
- `rebot/arm_state.py` — 既定は読み取り専用（Damiao 0xCC 要求だけ。shutdown() は使わない＝全モーター無効化を送るため）。
  `--allow-motion` で Web UI からの操作を受け付ける（有効化 MOVE、停止、トルク切り RELEASE、シミュレーター確認済み経路の実行、把持実行は前提未充足のため拒否）。
  デッドマン（ページの生存確認 0.6 s）、追従フィードバック（Kp 0.5 / Ki 1.0、補正上限 0.2 rad）、逸脱 0.08 rad / 手先 20 mm で停止。
  全 6 関節を MIT モード（kp・kd は SDK yaml）で保持し、SDK の公式 dynamics の重力 g(q) をフィードフォワード（有効化後 0.5 s で立ち上げ）、移動中は計画速度も加える。
  POS_VEL だと保持中に joint4・joint5 が 11 mrad のスティックスリップ、joint6 がハンチング（2026-10-14 実測）。joint4・6 は MIT 化で静止を確認済み、全関節化は依頼者の判断。
  移動中は `rebot/mit_tuning.yaml` のクーロン摩擦（joint2 0.90 / joint3 1.71 N m）を進む向きに加える。このファイルは有効化のたびに読み直す（再起動不要）。
  衝突停止（`ContactMonitor`）: 抵抗 = モーター実測トルク − 重力 ff − 摩擦 ff を計画の進む向きに測った値。大きさではなく「増え続けている」ことで止める
  （直近 1.5 s の最小値〔0 未満は 0〕からの増加が rise_nm を超えてまだ増加中、または進んだ量に比例して増えている）。値は `mit_tuning.yaml contact:`。
  絶対値の上限は 08:08 に休止姿勢からの持ち上げ（ストッパーが支えていた重さをモーターが引き受けた）で誤停止したためやめた。
  **2026-10-15 依頼者の指示で無効（`mit_tuning.yaml contact.enabled: false`、arm 再起動済み）**。校正の自動巡回中に joint1 で誤停止したため。逸脱停止は有効のまま。
  止まったら今の位置で保持して fault（要確認解除）。可動端 3° 以内の関節は除外。`rebot/contact_replay.py` で記録を再生・模擬接触を試せる。
  fake には壁の注入（`/api/test/inject {"wall": {joint: モーター角}}`）がある。トレースに各関節の抵抗 `res` を記録。
  移動ごとの 50 Hz 記録（計画・実測・速度・トルク・指令）が `run/arm_trace_last.jsonl` に残る。
  関節の零点補正（ソフトウェアのみ）: `rebot/joint_offsets.yaml`（モデル角 = モーター角 + offset）。2026-10-14 の hand-eye 校正で推定した値
  （joint2 −0.79、joint3 +1.01、joint4 −0.36、**joint5 −5.97 deg**。joint5 は定規の手合わせでも約 −7 deg を確認）。state.json の pos はモデル角、pos_motor が生の値。
  校正サンプルは記録時の offset を `joint_offsets_rad` に残し、handeye.py はモーター角に戻して解く。
  hand-eye の採用（/api/handeye/adopt）は、解と組の関節零点ずれで `joint_offsets.yaml` を書き換える（前の内容は `joint_offsets_before_<session>.yaml`）。
  「校正を自動で」ボタン: 作業範囲の校正姿勢（校正用 12・評価用 3 ＋中継）へ順に 経路シミュレーション → 実行 → 静止 → 記録、有効化時の姿勢へ戻って解き、採用は確認ダイアログで。
- `sim/` — MuJoCo 3.15.0（`sim/.venv`、EGL でオフスクリーン描画）。公開モデル Borot-Arm_Mujoco `3f3e98d` をそのまま取り込み、`scene.py` で床・ブロック・凸包衝突・アクチュエータを追加。
  現実同期（物理なし）/ シミュレーション試行（接近・閉じる・持ち上げ）、把持候補の判定、実機経路の動的シミュレーション。`verify_sim.py` で検証。
  hand-eye 採用時は把持候補に傾き（0, ±20, ±35°）を加えて 120 通りにし、接近前・近接（把持の 3 cm 手前）で ID 1 がカメラから見える
  （向き 55° 以内、画面端 40 px 以上内側、一辺 20 px 以上、カメラへの光線 5 本がブロック・腕・障害物に遮られない）ものだけを通す。
  接近前→近接は 10 mm ごとに全点で見えることを要求し、近接→把持で見えない点・平均の向き（0.08/deg）を順位に入れる（見えるほど上位）。机すき間 17 mm を保つ。近接→把持は 0.10 rad/s。
  「カメラで位置合わせ」（/api/visual_correct）: ID 1 の 10 フレームで FK 把持点との差を求め、残りの目標をずらして IK を解き直す（30 mm 超は拒否、累積しない）。
  自動シーケンス（「掴んで初期位置へ戻る」）の接近前→近接→把持は 10 mm 刻み（`check_path` の `step:<goal>:<mm>`）で、各段の前に ID 1 で視覚補正し直す
  （ID 1 が見えない段は前回の補正のまま、補正が前回から 8 mm 超跳んだら停止）。各段も経路シミュレーションを通したものだけ実行。
  2026-10-14 07:42 の事故（真上からの盲目接近で指がブロックに刺さった）を受けた変更。実機の手順: 計画 → 接近前 → 位置合わせ → 近接 → 位置合わせ → 把持（低速）→ 挟む → 持ち上げ。
  ID 1 マーカー台（取り付け済み）を `scene.toml [marker_mount]` で end_link に追加（`check_mount.py` / `render_mount.py`、詳細は `cad/README.md`）。
- 依存: oak/.venv depthai 3.10.0 / opencv 5.0.0.93 / numpy 2.4.6、rebot venv motorbridge 0.5.6 / pinocchio 3.9.0、sim/.venv は `sim/requirements.txt`。three.js 0.170.0 を `oak/static/vendor` に同梱。

## 確定したこと
- ID 3 の面は **78 × 31 mm**（依頼者確認。当初の「67 × 31」は誤り）。物体座標: +Z = ID 0 面（67×78）、−X = ID 3 面（78×31）。
- 取り付け: ID 0・ID 3 とも corner = bottom-left、x_axis = left（面図は markers.toml）。根拠は `oak/evidence/mount-20261013/`。
  ライブで両方 ok のとき ID 0 由来と ID 3 由来のブロック姿勢の差は回転 中央値 3.8°（最大 6°）、位置 中央値 13 mm（最大 25 mm、ID 0 が 16 px と小さいため）。
- 内部パラメータ: 工場校正（EEPROM）を ImgTransformation で 1280×960 に変換（fx 913.09）。robot1 上で旧 vision.py のステレオ深度による参考確認: ID 2 = 40.6 mm（印刷 40）、ID 1 = 19.8 mm。
- 実機関節: 7 軸とも読み取り専用で応答（状態 0x0 = 無効）。MuJoCo の end_link と SDK の Pinocchio FK が実関節角で一致（< 1e-6 mm）。

## 未確定・次の段階
0. （2026-10-16）hand-eye の再校正が未採用（2026-10-15 にカメラが約 35 mm ずれた疑い。「校正を自動で」は衝突停止で 2 回途中停止、今は衝突停止を無効化済み）。
   link5 のマーカー台 `cad/out/marker_mount_l5_1m_z` の ArUco ID は未定（vision・sim への登録も未着手）。
1. ID 2 カードの物理的固定と厚み測定 → 「基準を登録」。
2. 済: カメラ⇔アーム台座（`oak/calib/handeye_active.json`。sim はこれを使い、`scene.toml [placement]` は未使用のまま）。
3. 済: ID 1 → 把持中心（`markers.toml [marker.1.grip]`）。
4. 済: グリッパー開口（最大 100 mm、比例、遊び約 1 mm）。済: hand-eye 校正（位置中央値 3.7 mm）・把持中心・関節零点補正。
5. ブロック質量・摩擦の実測。カメラスタンドをシミュレーターへ登録。
6. 把持精度の検証（検出・シミュレーション成功とは別に報告する）。視認性つき計画・視覚補正・衝突停止は実機で未試験。
7. 保留: ステレオ深度をブロック推定に足すか（実機のずれを見てから決める）。
