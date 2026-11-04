# -*- coding: utf-8 -*-
"""Find hard test cases (missed scratch / dust) and dump boxed crops."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import device as dev  # noqa: F401  (must precede ultralytics)
from ultralytics import YOLO

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
IMG_DIR = ROOT / "dataset" / "images" / "test"
LBL_DIR = ROOT / "dataset" / "labels" / "test"
OUT = ROOT / "outputs" / "hardcases"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = CODE / "runs" / "gpu_v8s_1024" / "weights" / "best.pt"
CONF = 0.25
IOU_MATCH = 0.5


def load_gt(stem):
    rows = []
    p = LBL_DIR / f"{stem}.txt"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        v = line.split()
        if len(v) >= 5:
            rows.append((int(v[0]), *map(float, v[1:5])))
    return rows


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

model = YOLO(str(MODEL))
names = model.names
imgs = sorted(IMG_DIR.glob("TGV_10_*.bmp"))
if args.limit:
    imgs = imgs[: args.limit]
print("images", len(imgs))

rows = []
for img_path in imgs:
    stem = img_path.stem
    res = model.predict(str(img_path), imgsz=1024, conf=0.05, iou=0.6,
                        device=0, verbose=False)[0]
    h, w = res.orig_shape
    preds = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            preds.append((int(k), (float(x1), float(y1), float(x2), float(y2)), float(c)))
    gt = load_gt(stem)
    for k, cx, cy, bw, bh in gt:
        g = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        same = [p for p in preds if p[0] == k]
        best = max((iou(g, p[1]) for p in same), default=0.0)
        hit = [p for p in same if iou(g, p[1]) >= IOU_MATCH]
        best_conf = max((p[2] for p in same if iou(g, p[1]) >= 0.1), default=0.0)
        if not hit:
            area = (g[2] - g[0]) * (g[3] - g[1])
            rows.append({
                "stem": stem, "cls": names[k], "x1": g[0], "y1": g[1],
                "x2": g[2], "y2": g[3], "w": g[2] - g[0], "h": g[3] - g[1],
                "area": area, "best_iou": best, "best_conf_near": best_conf,
            })

rows.sort(key=lambda r: -r["area"])
with (OUT / "missed.csv").open("w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)

print("missed", len(rows))
for r in rows[:12]:
    print(f'{r["stem"]:12s} {r["cls"]:7s} {r["w"]:5.0f}x{r["h"]:4.0f}px '
          f'conf_near={r["best_conf_near"]:.2f}')


def crop_with_box(row, margin=1.6, size=640, box=True):
    img = cv2.imread(str(IMG_DIR / f'{row["stem"]}.bmp'))
    cx = (row["x1"] + row["x2"]) / 2
    cy = (row["y1"] + row["y2"]) / 2
    half = max(row["w"], row["h"]) * margin / 2
    x1 = int(max(0, cx - half)); y1 = int(max(0, cy - half))
    x2 = int(min(img.shape[1], cx + half)); y2 = int(min(img.shape[0], cy + half))
    crop = img[y1:y2, x1:x2].copy()
    if box:
        p = 3
        cv2.rectangle(crop,
                      (int(row["x1"]) - x1 - p, int(row["y1"]) - y1 - p),
                      (int(row["x2"]) - x1 + p, int(row["y2"]) - y1 + p),
                      (0, 165, 224), 2)  # amber BGR
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    tag = f'{row["cls"]}_{row["stem"]}_{int(row["w"])}x{int(row["h"])}'
    cv2.imwrite(str(OUT / f'{tag}_box.png'), crop)
    # raw version
    img2 = cv2.imread(str(IMG_DIR / f'{row["stem"]}.bmp'))
    raw = img2[y1:y2, x1:x2]
    raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f'{tag}_raw.png'), raw)


for r in rows[:10]:
    crop_with_box(r)
print("crops written to", OUT)
