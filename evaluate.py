#!/usr/bin/env python3
"""Validate a trained model and sweep confidence thresholds for production use."""

import argparse
import csv
import json
from pathlib import Path

# device comes first on purpose: ultralytics imports torch, and the MPS fallback switch
# inside device.py only has any effect if it is set before that happens.
import device as dev
import numpy as np
import yaml
from ultralytics import YOLO

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=root / "code" / "runs" / "gpu_v8s_1024" / "weights" / "best.pt")
    p.add_argument("--data", type=Path, default=root / "dataset" / "data.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=None,
                   help="per-backend preset when unset: 4 on CUDA, 2 on MPS and CPU")
    p.add_argument("--device", default="auto", help="auto / 0 (CUDA) / mps / cpu")
    p.add_argument("--conf-min", type=float, default=0.01, help="lowest confidence kept for the sweep")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.5, help="IoU used to call a prediction a TP")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--out", type=Path, default=code_dir / "runs" / "eval")
    return p.parse_args()


def load_split(data_yaml):
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entries = cfg["val"] if isinstance(cfg["val"], (list, tuple)) else [cfg["val"]]
    images = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = base / path
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS))
        elif path.suffix == ".txt":
            images.extend(Path(line.strip()) for line in path.read_text().splitlines() if line.strip())
        else:
            images.append(path)
    names = [cfg["names"][i] for i in sorted(cfg["names"])]
    return images, names


def label_path_for(image):
    parts = list(image.resolve().parts)
    try:
        idx = max(i for i, part in enumerate(parts) if part == "images")
    except ValueError:
        return None
    return Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")


def load_gt(image, width, height):
    path = label_path_for(image)
    boxes, classes = [], []
    if path and path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts[0], *map(float, parts[1:])
            x1 = max((x - w / 2) * width, 0)
            y1 = max((y - h / 2) * height, 0)
            x2 = min((x + w / 2) * width, width)
            y2 = min((y + h / 2) * height, height)
            boxes.append([x1, y1, x2, y2])
            classes.append(int(cls))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(pred_boxes, pred_conf, gt_boxes, match_iou):
    order = np.argsort(-pred_conf)
    used = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for i in order:
        ious = iou_one_to_many(pred_boxes[i], gt_boxes)
        if len(ious):
            j = int(np.argmax(ious))
            if ious[j] >= match_iou and not used[j]:
                used[j] = True
                tp += 1
                continue
        tp += 0
    fp = len(pred_boxes) - tp
    fn = int((~used).sum())
    return tp, fp, fn


def main():
    args = parse_args()
    args.device = dev.resolve(args.device)
    dev.announce(args.device)
    if args.batch is None:
        args.batch = dev.preset(args.device)["batch"]
    args.out.mkdir(parents=True, exist_ok=True)
    images, names = load_split(args.data)

    model = YOLO(str(args.model))

    print("== standard validation ==")
    metrics = model.val(
        data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch,
        device=args.device, conf=0.001, iou=args.iou, plots=True,
        project=str(args.out.resolve()), name=args.model.parent.parent.name + "_val", exist_ok=True,
    )
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

    results = model.predict(
        images, imgsz=args.imgsz, batch=args.batch, device=args.device,
        conf=args.conf_min, iou=args.iou, verbose=False, max_det=300,
    )

    preds = []
    for image, result in zip(images, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            preds.append((np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)))
            continue
        preds.append((
            boxes.xyxy.cpu().numpy().astype(np.float32),
            boxes.conf.cpu().numpy().astype(np.float32),
            boxes.cls.cpu().numpy().astype(np.int64),
        ))

    gts = []
    for image, result in zip(images, results):
        height, width = result.orig_shape
        gts.append(load_gt(image, width, height))

    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    rows = []
    for cls_id, name in enumerate(names):
        for thr in thresholds:
            tp = fp = fn = 0
            for (pbox, pconf, pcls), (gbox, gcls) in zip(preds, gts):
                keep = (pcls == cls_id) & (pconf >= thr)
                cls_gt = gbox[gcls == cls_id]
                a, b, c = match(pbox[keep], pconf[keep], cls_gt, args.match_iou)
                tp, fp, fn = tp + a, fp + b, fn + c
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            rows.append({
                "class": name, "threshold": float(thr),
                "precision": round(precision, 4), "recall": round(recall, 4),
                "fp_per_image": round(fp / len(images), 3),
                "tp": tp, "fp": fp, "fn": fn,
            })

    csv_path = args.out / "threshold_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n== confidence sweep ==")
    print(f"{'class':8s} {'thr':>5s} {'P':>6s} {'R':>6s} {'FP/img':>7s}")
    for cls_id, name in enumerate(names):
        for row in rows:
            if row["class"] != name:
                continue
            if row["threshold"] in (0.05, 0.25, 0.5, 0.75, 0.9):
                print(f"{name:8s} {row['threshold']:5.2f} {row['precision']:6.3f} "
                      f"{row['recall']:6.3f} {row['fp_per_image']:7.3f}")

    print(f"\n== recommended thresholds (recall >= {args.target_recall:.2f}) ==")
    summary = {}
    for cls_id, name in enumerate(names):
        candidates = [r for r in rows if r["class"] == name and r["recall"] >= args.target_recall]
        if not candidates:
            best = max((r for r in rows if r["class"] == name), key=lambda r: r["recall"])
            print(f"{name:8s} no threshold reaches target; max recall {best['recall']:.3f} at {best['threshold']:.2f}")
            summary[name] = {"threshold": best["threshold"], "recall": best["recall"],
                             "precision": best["precision"], "fp_per_image": best["fp_per_image"]}
            continue
        best = max(candidates, key=lambda r: (r["precision"], -r["fp_per_image"]))
        print(f"{name:8s} conf={best['threshold']:.2f}  P={best['precision']:.3f} "
              f"R={best['recall']:.3f}  FP/img={best['fp_per_image']:.3f}")
        summary[name] = {"threshold": best["threshold"], "recall": best["recall"],
                         "precision": best["precision"], "fp_per_image": best["fp_per_image"]}

    (args.out / "summary.json").write_text(json.dumps({
        "model": str(args.model), "images": len(images), "classes": names,
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        "match_iou": args.match_iou, "recommended": summary,
    }, indent=2))
    print(f"\nreport: {csv_path}")


if __name__ == "__main__":
    main()
