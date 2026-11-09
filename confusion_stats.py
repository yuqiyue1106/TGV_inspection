"""Compute class-level confusion counts on the held-out test split."""
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "code/runs/gpu_v8s_1024/weights/best.pt"
IMAGES = ROOT / "dataset/images/test"
LABELS = ROOT / "dataset/labels/test"
OUT = ROOT / "code/runs/eval_test_gpu_v8s_1024/confusion.json"

CONF = 0.25
IOU_MATCH = 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(path, w, h):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


model = YOLO(str(WEIGHTS))
files = sorted(IMAGES.glob("*.bmp"))
conf = [[0] * 3 for _ in range(2)]  # rows: gt dust/scratch; cols: pred dust/scratch/none
fp = [0, 0]
gt_total = [0, 0]
pred_total = [0, 0]

for f in files:
    res = model.predict(str(f), imgsz=1024, conf=CONF, iou=0.7, device=0,
                        verbose=False, max_det=300)[0]
    h, w = res.orig_shape
    gts = load_gt(LABELS / f"{f.stem}.txt", w, h)
    preds = []
    for b in res.boxes:
        c = int(b.cls.item())
        xyxy = b.xyxy[0].tolist()
        preds.append((float(b.conf.item()), c, xyxy))
    preds.sort(key=lambda x: -x[0])
    used = [False] * len(gts)
    for pc in (0, 1):
        for _, c, pb in [p for p in preds if p[1] == pc]:
            pred_total[pc] += 1
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if used[i] or g[0] != pc:
                    continue
                v = iou(pb, g[1:])
                if v > best:
                    best, bi = v, i
            if bi >= 0 and best >= IOU_MATCH:
                used[bi] = True
                conf[pc][pc] += 1
            else:
                fp[pc] += 1
    for i, g in enumerate(gts):
        gt_total[g[0]] += 1
        if used[i]:
            continue
        other = 1 - g[0]
        swap = any(p[1] == other and iou(p[2], g[1:]) >= IOU_MATCH for p in preds)
        conf[g[0]][other if swap else 2] += 1

out = {
    "images": len(files), "conf": CONF, "iou_match": IOU_MATCH,
    "gt_total": gt_total, "pred_total": pred_total,
    "matrix_gt_rows_pred_cols_dust_scratch_miss": conf,
    "fp": fp,
    "tp": [conf[0][0], conf[1][1]],
    "fn": [conf[0][2], conf[1][2]],
}
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
