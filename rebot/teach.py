#!/usr/bin/env python3
"""
reBot Arm B601-DM ダイレクトティーチング（記録・再生）

使い方:
  python teach.py probe   --port /dev/tty.usbmodemXXXX            # 状態が読めるか確認（動かない）
  python teach.py record  --port /dev/tty.usbmodemXXXX -o demo.csv # 手で動かして記録（Ctrl-Cで終了）
  python teach.py replay  --port /dev/tty.usbmodemXXXX -i demo.csv # 記録を再生

注意:
  - motorbridge-gateway（Studio）はバスを占有するので、実行前に必ず止めること
  - record中は KP=0 なので、J2/J3 は自重で下がる。手で支えながら動かす
  - 終了時にトルクが抜けるので、アームを支えてから Enter を押す
"""
import argparse
import csv
import sys
import time

from motorbridge import Controller, Mode

# (関節番号, CAN_ID, MASTER_ID, モデル)
# ★ モデルは Studio の各カードの「Model」欄に合わせて必ず修正すること
JOINTS = [
    (1, 0x01, 0x11, "4340P"),
    (2, 0x02, 0x12, "4340P"),
    (3, 0x03, 0x13, "4340P"),
    (4, 0x04, 0x14, "4310"),
    (5, 0x05, 0x15, "4310"),
    (6, 0x06, 0x16, "4310"),
    (7, 0x07, 0x17, "4310"),
]

HZ = 50                 # 記録・再生の周期
REC_KD = 0.3            # 記録時のダンピング（大きいほど重く、落ちにくい）
PLAY_KP = [20, 20, 20, 10, 10, 10, 5]  # 再生時の剛性（関節ごと）
PLAY_KD = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3]
PLAY_HZ = 100           # 再生時の指令周期（記録点の間を補間して送る）
SMOOTH = 5              # 記録データの移動平均の窓幅（サンプル数、1で無効）
APPROACH_VEL = 0.3      # 開始姿勢への移動速度 [rad/s]
MAX_STEP = 0.15         # 1周期あたりの最大変化量 [rad]。超えたら再生中止


def read_pos(m):
    """get_state() の返り値から位置を取り出す（属性・dict どちらにも対応）"""
    s = m.get_state()
    if s is None:
        return None
    if isinstance(s, dict):
        return s.get("pos")
    return getattr(s, "pos", None)


def open_arm(port):
    ctrl = Controller.from_dm_serial(port, 921600)
    motors = [ctrl.add_damiao_motor(cid, mid, model) for _, cid, mid, model in JOINTS]
    return ctrl, motors


def read_all(motors):
    for m in motors:
        m.request_feedback()
    time.sleep(0.01)
    return [read_pos(m) for m in motors]


def safe_shutdown(ctrl, motors):
    try:
        input("\n>>> アームを手で支えてから Enter を押すと、トルクを切ります...")
    except (EOFError, KeyboardInterrupt):
        pass
    for m in motors:
        try:
            m.disable()
        except Exception as e:
            print(f"disable失敗: {e}")
    if hasattr(ctrl, "close"):
        ctrl.close()


def cmd_probe(args):
    ctrl, motors = open_arm(args.port)
    try:
        for _ in range(10):
            q = read_all(motors)
            print(" ".join(f"J{j[0]}:{p:+.3f}" if p is not None else f"J{j[0]}:None"
                           for j, p in zip(JOINTS, q)))
            time.sleep(0.2)
        print("raw get_state (J1):", motors[0].get_state())
    finally:
        if hasattr(ctrl, "close"):
            ctrl.close()


def cmd_record(args):
    ctrl, motors = open_arm(args.port)
    rows = []
    try:
        ctrl.enable_all()
        for m in motors:
            m.ensure_mode(Mode.MIT, 1000)
        print(f"記録開始（{HZ}Hz）。手で動かしてください。Ctrl-C で終了。")
        t0 = time.time()
        dt = 1.0 / HZ
        while True:
            tick = time.time()
            q = []
            for m in motors:
                # KP=0: 位置保持なし。KD だけで軽くダンピング
                m.send_mit(0.0, 0.0, 0.0, REC_KD, 0.0)
                q.append(read_pos(m))
            if None not in q:
                rows.append([tick - t0] + q)
                if len(rows) % HZ == 0:
                    print(f"\r{len(rows)/HZ:5.1f}s  " +
                          " ".join(f"{p:+.2f}" for p in q), end="", flush=True)
            time.sleep(max(0.0, dt - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"j{j[0]}" for j in JOINTS])
            w.writerows(rows)
        print(f"\n{len(rows)} サンプルを {args.output} に保存しました")
        safe_shutdown(ctrl, motors)


def smooth(traj, w):
    """関節ごとに移動平均をかける（時刻はそのまま）"""
    if w <= 1:
        return traj
    n, h = len(traj), w // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = traj[lo:hi]
        out.append([traj[i][0]] + [sum(r[j] for r in seg) / len(seg)
                                    for j in range(1, len(traj[0]))])
    return out


def resample(traj, hz):
    """記録点を線形補間して hz の等間隔にし、速度（差分）も求める"""
    dt = 1.0 / hz
    t_end = traj[-1][0] - traj[0][0]
    out, k = [], 0
    for i in range(int(t_end / dt) + 1):
        t = traj[0][0] + i * dt
        while k < len(traj) - 2 and traj[k + 1][0] < t:
            k += 1
        a, b = traj[k], traj[k + 1]
        span = b[0] - a[0]
        r = 0.0 if span <= 0 else min(1.0, max(0.0, (t - a[0]) / span))
        q = [a[j] + (b[j] - a[j]) * r for j in range(1, len(a))]
        v = [0.0 if span <= 0 else (b[j] - a[j]) / span for j in range(1, len(a))]
        out.append((i * dt, q, v))
    return out


def cmd_replay(args):
    with open(args.input) as f:
        r = csv.reader(f)
        next(r)
        traj = [[float(x) for x in row] for row in r]
    if len(traj) < 2:
        sys.exit("軌道が短すぎます")

    dur = traj[-1][0] - traj[0][0]
    print(f"記録: {len(traj)} サンプル / {dur:.1f}s（実効 {len(traj)/max(dur,1e-6):.1f} Hz）")

    # 記録データ上で急な変化がないか先にチェック
    for a, b in zip(traj, traj[1:]):
        if max(abs(x - y) for x, y in zip(a[1:], b[1:])) > MAX_STEP:
            sys.exit(f"t={b[0]:.2f}s に急な変化があります。記録を確認してください")

    plan = resample(smooth(traj, SMOOTH), PLAY_HZ)

    ctrl, motors = open_arm(args.port)
    try:
        q_now = read_all(motors)
        if None in q_now:
            sys.exit("状態が読めません。probe で確認してください")

        ctrl.enable_all()
        for m in motors:
            m.ensure_mode(Mode.MIT, 1000)

        def send(q, v=None):
            for i, m in enumerate(motors):
                m.send_mit(q[i], 0.0 if v is None else v[i],
                           PLAY_KP[i], PLAY_KD[i], 0.0)

        # 1) 現在姿勢から開始姿勢へゆっくり補間
        q_start = plan[0][1]
        dist = max(abs(a - b) for a, b in zip(q_start, q_now))
        n = max(1, int(dist / APPROACH_VEL * PLAY_HZ))
        print(f"開始姿勢へ移動中（{n/PLAY_HZ:.1f}s）...")
        for k in range(1, n + 1):
            a = k / n
            send([c + (s - c) * a for c, s in zip(q_now, q_start)])
            time.sleep(1.0 / PLAY_HZ)

        # Enter 待ちの間も姿勢を保持し続ける
        import threading
        go = threading.Event()
        threading.Thread(target=lambda: (input(">>> Enter で再生開始（Ctrl-C で中止）"), go.set()),
                         daemon=True).start()
        while not go.is_set():
            send(q_start)
            time.sleep(1.0 / PLAY_HZ)

        # 2) 補間済み軌道を PLAY_HZ で再生（速度フィードフォワード付き）
        t0 = time.perf_counter()
        late = 0
        for t, q, v in plan:
            while time.perf_counter() - t0 < t:
                pass  # sleep より正確なビジーウェイト
            if time.perf_counter() - t0 > t + 2.0 / PLAY_HZ:
                late += 1
            send(q, v)
        if late:
            print(f"警告: {late}/{len(plan)} 周期で送信が遅れました。PLAY_HZ を下げてください")
        print("再生終了。最後の姿勢を保持しています。")
        q_last = plan[-1][1]
        for _ in range(PLAY_HZ):
            send(q_last)
            time.sleep(1.0 / PLAY_HZ)
    except KeyboardInterrupt:
        print("\n中止しました")
    finally:
        safe_shutdown(ctrl, motors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["probe", "record", "replay"])
    p.add_argument("--port", required=True)
    p.add_argument("-o", "--output", default="demo.csv")
    p.add_argument("-i", "--input", default="demo.csv")
    args = p.parse_args()
    {"probe": cmd_probe, "record": cmd_record, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    main()
