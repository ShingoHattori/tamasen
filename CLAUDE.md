# tamasen — 作業コンテキスト（2026-10-13 時点の引き継ぎ）

reBot Arm B601-DM と OAK カメラで、ArUco マーカー付きの直方体ブロックとグリッパーを認識し、
最終的に把持するためのリポジトリ。以後の作業は **robot1** 上で行う。

## マシン

| | robot1（作業機） | Mac（これまでの開発機） |
|---|---|---|
| 接続 | `ssh win10.hpcs-sb1124.sec.llhi.org`（user `cely`、192.168.32.20 / DHCP） | — |
| OS | Ubuntu Server 24.04.3、kernel 6.8.0-147-generic | macOS |
| GPU | RTX 2080 SUPER、`nvidia-driver-595-open`（ubuntu-drivers 推奨版、ビルド済み署名モジュール、DKMS なし）。CUDA Toolkit は未導入 | — |
| アーム | USB-CAN が `/dev/ttyACM0`（HDSC CDC `2e88:4603`） | — |
| カメラ | **未接続**。OAK-D-Lite を Mac から付け替える必要あり | OAK-D-Lite `1844301091A70A1300` |

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
- `rebot/teach.py` — ダイレクトティーチング（記録・再生）。`motorbridge` を使う。
- `rebot/reBotArm_control_py/` — 上流 reBotArm_control_py の `40ab6ce`（上流に LICENSE ファイルなし。MIT は未確認） に、
  ローカルの重力補償・校正の変更を加えたもの。通常のファイルとして取り込んでいる（submodule ではない）。
- venv はコミットしていない（`.gitignore`）。依存関係は `oak/pyproject.toml`、`oak/requirements.txt`、`rebot/reBotArm_control_py/uv.lock` から作り直す。

## 絶対の規則（依頼者の指示、2026-10-13）
- **Claude Code から実機アームを動かさない。** 実機の動作（有効化・移動・把持・トルク切り）は、Web UI のボタンを依頼者が押したときだけ。
  Claude は `--allow-motion` 付きで arm サービスを起動しない。実機の `/arm/api/*`（POST）を呼ばない。テストでも呼ばない（試験は `--fake` を別ポートで）。
- 実機を動かすときは、必ず事前に経路を MuJoCo でシミュレートしてチェックし、妥当だったものだけを実行する。
  実行中・保持中は常に「計画 − 実測」のフィードバック補正をかける（重力補正の差でずれるため）。
- カメラスタンドが映像の下端にある。不用意な動作でぶつかる（シミュレーターには未登録 = 台座⇔カメラ未校正のため）。

## 起動・停止（robot1）
```
./run.sh start            # vision(8090) + arm 読み取り専用(8091) + sim(8092)。すべて 127.0.0.1
./run.sh stop | status | restart vision | log sim
ssh -N -L 8090:127.0.0.1:8090 win10.hpcs-sb1124.sec.llhi.org   # 手元で http://127.0.0.1:8090/
```
実機操作を有効にするときは **依頼者が** `ARM_ARGS="--port /dev/ttyACM0 --allow-motion" ./run.sh restart arm`。

## 構成（2026-10-13 時点、commit は git log 参照）
- `oak/vision.py` — RGB のみ（ステレオ深度は使わない）。1280×960 で検出、ブラウザへは 960 幅の縮小 MJPEG。
  四隅・ID・軸、マーカー姿勢（両 IPPE 解を保持、ok / ambiguous / flip / rejected）、ID 2 暫定 world（毎フレーム追従 / 登録して固定、ずれ検知）、
  ID 0・3 それぞれからのブロック姿勢（平均しない）、静止画保存（元画像 + 注釈 + 状態）、録画（FFV1、上限 2000 MB / 120 s、明示操作のみ）、`--replay`。
- `oak/markers.toml` — 寸法・取り付け・校正値を *_status（unset / sample / measured / calibrated）付きで管理。内部は m・rad。
- `rebot/arm_state.py` — 既定は読み取り専用（Damiao 0xCC 要求だけ。shutdown() は使わない＝全モーター無効化を送るため）。
  `--allow-motion` で Web UI からの操作を受け付ける（有効化 MOVE、停止、トルク切り RELEASE、シミュレーター確認済み経路の実行、把持実行は前提未充足のため拒否）。
  デッドマン（ページの生存確認 0.6 s）、追従フィードバック（Kp 0.5 / Ki 1.0、補正上限 0.2 rad）、逸脱 0.08 rad / 手先 20 mm で停止。
- `sim/` — MuJoCo 3.15.0（`sim/.venv`、EGL でオフスクリーン描画）。公開モデル Borot-Arm_Mujoco `3f3e98d` をそのまま取り込み、`scene.py` で床・ブロック・凸包衝突・アクチュエータを追加。
  現実同期（物理なし）/ シミュレーション試行（接近・閉じる・持ち上げ）、把持候補 24 通りの判定、実機経路の動的シミュレーション。`verify_sim.py` で検証。
- 依存: oak/.venv depthai 3.10.0 / opencv 5.0.0.93 / numpy 2.4.6、rebot venv motorbridge 0.5.6 / pinocchio 3.9.0、sim/.venv は `sim/requirements.txt`。three.js 0.170.0 を `oak/static/vendor` に同梱。

## 確定したこと
- ID 3 の面は **78 × 31 mm**（依頼者確認。当初の「67 × 31」は誤り）。物体座標: +Z = ID 0 面（67×78）、−X = ID 3 面（78×31）。
- 取り付け: ID 0・ID 3 とも corner = bottom-left、x_axis = left（面図は markers.toml）。根拠は `oak/evidence/mount-20261013/`。
  ライブで両方 ok のとき ID 0 由来と ID 3 由来のブロック姿勢の差は回転 中央値 3.8°（最大 6°）、位置 中央値 13 mm（最大 25 mm、ID 0 が 16 px と小さいため）。
- 内部パラメータ: 工場校正（EEPROM）を ImgTransformation で 1280×960 に変換（fx 913.09）。robot1 上で旧 vision.py のステレオ深度による参考確認: ID 2 = 40.6 mm（印刷 40）、ID 1 = 19.8 mm。
- 実機関節: 7 軸とも読み取り専用で応答（状態 0x0 = 無効）。MuJoCo の end_link と SDK の Pinocchio FK が実関節角で一致（< 1e-6 mm）。

## 未確定・次の段階
1. ID 2 カードの物理的固定と厚み測定 → 「基準を登録」。
2. カメラ⇔アーム台座の位置合わせ（`sim/scene.toml [placement]`）。ID 1 を付けたアームを複数姿勢に動かす必要がある → 動作は依頼者が Web UI から。
3. ID 1 → 把持中心の校正（`markers.toml [marker.1.grip]`）。
4. グリッパーのモーター角 ↔ 開口の対応と最大開口の実測（`sim/scene.toml [jaw]`。モデルは 0〜100 mm、資料に 57 mm の記述あり）。
5. ブロック質量・摩擦の実測。カメラスタンドをシミュレーターへ登録。
6. 把持精度の検証（検出・シミュレーション成功とは別に報告する）。
