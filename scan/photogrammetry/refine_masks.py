"""LiDAR から作った粗いマスクを、RGB の GrabCut で物体の輪郭に詰める（robot1 で並列に回す想定）。

粗いマスクを縮めた内側 = 確実に物体、広げた外側 = 確実に背景、その間 = 不明として GrabCut を初期化する。
処理は 1/2 解像度。出力は映像解像度の 0/255 PNG。
使い方: OMP_NUM_THREADS=1 refine_masks.py <session> <coarse_masks> <out> [--jobs 8]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

cv2.setNumThreads(1)


def one(args):
    img_path, m_path, out_path, inner, outer = args
    img = cv2.imread(img_path)
    m = cv2.imread(m_path, 0)
    if img is None or m is None:
        return 0
    h, w = m.shape
    sm = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    mm = cv2.resize(m, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 127
    k_in = np.ones((inner, inner), np.uint8)
    k_out = np.ones((outer, outer), np.uint8)
    sure_fg = cv2.erode(mm.astype(np.uint8), k_in) > 0
    maybe = cv2.dilate(mm.astype(np.uint8), k_out) > 0
    gc = np.full(mm.shape, cv2.GC_BGD, np.uint8)
    gc[maybe] = cv2.GC_PR_BGD
    gc[mm] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    if sure_fg.sum() < 50:
        return 0
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    ys, xs = np.nonzero(maybe)
    y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
    sub = gc[y0:y1, x0:x1].copy()
    cv2.grabCut(sm[y0:y1, x0:x1], sub, None, bg, fg, 4, cv2.GC_INIT_WITH_MASK)
    gc[y0:y1, x0:x1] = sub
    res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(res)
    if n > 1:
        res = (lab == 1 + st[1:, 4].argmax()).astype(np.uint8) * 255
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite(out_path, cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("coarse")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--inner", type=int, default=31, help="確実に物体とみなす縮め幅（1/2 解像度の画素）")
    ap.add_argument("--outer", type=int, default=41, help="不明とみなす広げ幅")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = sorted(f for f in os.listdir(a.coarse) if f.endswith(".png"))
    jobs = [(os.path.join(a.session, "frames", n[:-4] + ".jpg"), os.path.join(a.coarse, n), os.path.join(a.out, n), a.inner, a.outer)
            for n in names]
    with ProcessPoolExecutor(a.jobs) as ex:
        k = sum(ex.map(one, jobs, chunksize=4))
    print(f"refined {k}/{len(jobs)} -> {a.out}")


if __name__ == "__main__":
    main()
