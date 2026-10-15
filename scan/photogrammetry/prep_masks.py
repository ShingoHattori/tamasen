"""フォトグラメトリ（Apple Object Capture）用に、各フレームの物体マスクを作る。

マスク = 肌色でなく、深度が「画面中央付近で一番手前の面 + --depth-band」以内の画素のうち、中央に最も近い連結成分。
手で持った物体の後ろの顔・背景（10 cm 以上奥）と手を落とす。出力は <out>/NNNNNN.png（映像解像度、0/255）。
使い方: scan/.venv/bin/python scan/photogrammetry/prep_masks.py <session> <out> [--step 2]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from process import skin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--depth-band", type=float, default=0.10, help="一番手前の面からこの奥行き (m) までを物体とみなす")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = sorted(glob.glob(os.path.join(a.session, "frames", "*.json")))[:: a.step]
    kept = 0
    for jf in js:
        j = json.load(open(jf))
        vw, vh, dw, dh = j["video_w"], j["video_h"], j["depth_w"], j["depth_h"]
        d = np.fromfile(jf[:-4] + "depth", np.float32).reshape(dh, dw)
        img = cv2.imread(jf[:-4] + "jpg")
        sk = cv2.resize(skin(img, 6).astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
        ok = np.isfinite(d) & (d > 0.1) & ~sk
        c = np.zeros_like(ok)
        c[dh // 5: dh * 4 // 5, dw // 5: dw * 4 // 5] = True
        if (ok & c).sum() < 200:
            continue
        near = np.percentile(d[ok & c], 3)
        m = (ok & (d < near + a.depth_band)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        # 中央に近く、十分大きい成分
        score = [(np.hypot(*(cen[i] - (dw / 2, dh / 2))) - 0.02 * np.sqrt(st[i, 4]), i) for i in range(1, n) if st[i, 4] > 300]
        if not score:
            continue
        i = min(score)[1]
        m = (lab == i).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = cv2.resize(m, (vw, vh), interpolation=cv2.INTER_LINEAR)
        m = cv2.erode((m > 127).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        cv2.imwrite(os.path.join(a.out, os.path.basename(jf)[:-5] + ".png"), m)
        kept += 1
    print(f"masks {kept}/{len(js)} -> {a.out}")


if __name__ == "__main__":
    main()
