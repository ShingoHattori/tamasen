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
  - `vision.py` + `vision.html` + `pose.py` + `markers.toml` — 今回追加した ArUco 確認画面（後述）
- `marker/` — 印刷用 ArUco シート。辞書は **DICT_4X4_50**。`aruco_a4_compact*.pdf` には ID 0〜3 の 20/30/40/50(/60) mm 版がある（PDF の解析で確認）。
- `rebot/teach.py` — ダイレクトティーチング（記録・再生）。`motorbridge` を使う。
- `rebot/reBotArm_control_py/` — 上流 `vectorBH6/reBotArm_control_py`（MIT）の `40ab6ce` に、
  ローカルの重力補償・校正の変更を加えたもの。通常のファイルとして取り込んでいる（submodule ではない）。
- venv はコミットしていない（`.gitignore`）。依存関係は `oak/pyproject.toml`、`oak/requirements.txt`、`rebot/reBotArm_control_py/uv.lock` から作り直す。

## 現在の課題: 実カメラでの物体・手先認識と 3D 表示

### 制約（依頼者の指示）
- **初期段階ではモーター指令を送らない。**
- 既存の実装を確認し、その上に追加する。
- ブロックのマーカー取り付け変換は**実測値だけ**から作る。面の寸法から推測しない。未計測の項目は一覧で示す。
- ID 1 → 把持中心の変換が未校正なら、未校正だと明示する。
- 作業台座標・アーム台座座標へは、校正するまで変換しない（見かけ上の変換で表示しない）。
- 「検出できた」と「把持精度が検証できた」を区別して報告する。
- ID 0 が見えないのは異常ではない（現在は ID 3 の面がカメラに向いている）。

### 実物
- 直方体ブロック 67 × 78 × 31 mm。物体座標の X / Y / Z = 67 / 78 / 31。
- ID 0: 67 × 78 面、ID 3: 67 × 31 面（どちらも面の角の近く）。両方とも同じ物体のマーカー。
- ID 1: グリッパー根元の、顎の開閉では動かない部分。

### 実装済み（Mac で動作確認済み、commit `b8cdf79`）
`python vision.py --device 18443 [--host 0.0.0.0] [--port 8090]` でブラウザに以下を表示する。
- 実映像に四隅（黄 = corner 0）・ID・座標軸。取り付け登録後は投影したブロックの線（ID 0 由来 = シアン、ID 3 由来 = マゼンタ）。
- three.js による 3D 表示（**カメラ座標系のみ**）: 観測したマーカー、登録後は実寸ブロックと面上の登録マーカー、ID 0 / ID 3 それぞれから求めたブロック姿勢と、両者の差（mm / °）。
- ID 1 の観測姿勢と、把持中心の校正状態。
- 内部パラメータの出所と使用画像への適合（下記）、深度/PnP 比による整合確認。
- `markers.toml` は起動中も変更を検知して読み直す。未記入の項目はページの「未登録・未校正の項目」に一覧表示される。

物体座標の取り決め（`markers.toml` に記載）: 原点 = ブロック中心、+Z = ID 0 面の外向き、−Y = ID 3 面の外向き、
+X = ID 3 面を正面にして ID 0 面を上にしたときの右。取り付けの数式は `pose.py` の `_block_mount`。
合成データで往復誤差がほぼ 0 になることを確認済み。

### Mac での確認結果（2026-10-13）
- ID 1 と ID 3 を検出した（30 fps、1280×960）。机上の ID 2（別シート）も見えていた。
- **内部パラメータ**: OAK の工場校正（EEPROM、3840×2160 で保存）を、フレームの `ImgTransformation` から使用画像（センサーモード 2104×1560 → 1280×960）向けに取得している。fx = 913.09。
  校正値を単純に縮小すると fx = 989（+8.3 %）になり誤る（TUTORIAL 3.15）。
- 深度から逆算した参考の実寸: ID 1 ≈ 19.9 mm、ID 3 ≈ 20.1 mm、ID 2 ≈ 40.8 mm。印刷サイズ（20 / 40 mm）と 1〜2 % で一致し、内部パラメータの整合が裏付けられる。ただし**登録値は実測すること**。
- 姿勢の安定性（約 0.5 m、ほぼ正面から 6 秒間）: ID 1 の法線の揺れは中央値 0.5°、最大 1.1°。ID 3 は中央値 5.8°、最大 9.2°。
  IPPE の曖昧度は最大 0.98。20 mm のマーカーを正面から見ると、面の向きの 2 解が近づいて揺れる。
  対策の候補: カメラを近づける・斜めから見る、深度の平面で 2 解を選び分ける、時間方向のフィルタ。

### 未登録（依頼者の実測待ち）— `oak/markers.toml` に記入する
- ID 0 / ID 3: `size_mm`（黒枠外寸）、`corner`（寄っている面の角）、`offset_x_mm`・`offset_y_mm`（その角の辺からマーカー黒枠までの距離）、
  `x_axis`（確認画面の赤軸が面図で向く方向）。面図の見方は `markers.toml` の冒頭に記載。
- ID 1: `size_mm`。把持中心への変換（`[marker.1.grip]`）は未校正。

### 次の段階
1. **robot1 へ移す**（下記）。実映像と 3D 表示で、ブロックの寸法・面の向き・マーカーの取り付け位置が一致することを依頼者と確認する ← 最初の到達点
2. カメラとアーム台座の位置合わせ（hand-eye）。ID 1 をアームに付けたまま複数の姿勢を取る必要があり、モーター指令を伴うため、**着手前に依頼者の許可を取る**。
3. ID 1 → 把持中心の校正。
4. 把持精度の検証（検出とは別に報告する）。

## robot1 のセットアップ状況（2026-10-13）
- [x] uv 0.13.0（`~/.local/bin`、`.bashrc` / `.profile` で PATH に追加済み）
- [x] `oak/.venv`: Python 3.11、`requirements.txt` どおり（depthai 3.10.0 / opencv-python 5.0.0 / numpy 2.4.6）。import 確認済み（libGL の問題なし）
- [x] `rebot/reBotArm_control_py/.venv`: `uv sync --frozen`、Python 3.10.22（pyproject が <3.12 を要求）、motorbridge 0.5.6。`rebot/teach.py` もこの venv で動く想定
- [x] udev: `/etc/udev/rules.d/80-movidius.rules` — `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0660", GROUP="plugdev"`（cely は plugdev 所属）
- [x] OAK-D-Lite を robot1 に接続（08:28）。cely 権限（plugdev）で `oak.py list` が通り、`UsbSpeed.SUPER`。カーネルドライバは不要（depthai が libusb で直接開く）
- [ ] **アームの USB-CAN（`/dev/ttyACM0`）が外れている**。付け替えのときに、USB-CAN と USB-LAN（RTL8153）が載ったハブ（1-2）ごと抜けた。アームを使う前に接続し直す
- [ ] `oak/.venv/bin/python vision.py --device 18443 --host 0.0.0.0`
      → 手元のブラウザで `http://192.168.32.20:8090/` を開く（届かなければ `ssh -L 8090:localhost:8090 win10.hpcs-sb1124.sec.llhi.org`）。
      3D 表示は three.js を unpkg から読み込むため、**ブラウザ側に**インターネット接続が必要
- [ ] Linux 上の DepthAI の挙動（TUTORIAL 3.3〜3.4 の再列挙・強制終了まわり）は Mac でしか確認していない。改めて確認する
