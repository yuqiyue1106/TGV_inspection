#!/usr/bin/env python3
"""Evaluate a trained model on the held-out test split, broken down by aperture group and label origin."""

import argparse
import csv
import functools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import ultralytics.utils.nms as _nms
import yaml
from ultralytics import YOLO

from evaluate import IMG_EXTS, load_gt, match

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=CODE / "runs" / "official_v8s_1024" / "weights" / "best.pt")
    p.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--conf-min", type=float, default=0.05, help="lowest confidence kept for the sweep")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, default=CODE / "runs" / "eval_test_official")
    p.add_argument("--limit", type=int, default=0, help="debug: only score the first N images")
    return p.parse_args()


def split_images(data_yaml, split):
    cfg = yaml.safe_load(data_yaml.read_text())
    base = Path(cfg.get("path") or data_yaml.parent)
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entry = Path(cfg[split])
    if not entry.is_absolute():
        entry = base / entry
    return sorted(p for p in entry.iterdir() if p.suffix.lower() in IMG_EXTS), \
        [cfg["names"][i] for i in sorted(cfg["names"])]


def machine_stems():
    """Stems written by the pre-label model; everything else is human-annotated."""
    stems = set()
    for manifest in CODE.glob("runs/**/manifest_*.json"):
        data = json.loads(manifest.read_text())
        stems.update(data.get("entries", {}))
    return stems


def group_of(stem):
    return stem.rsplit("_", 1)[0]


def score(preds, gts, names, match_iou, conf_thr=0.0):
    """Greedy matching per class; returns per-class counters."""
    out = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(len(names))}
    for (pbox, pconf, pcls), (gbox, gcls) in zip(preds, gts):
        for cls in out:
            keep = (pcls == cls) & (pconf >= conf_thr)
            tp, fp, fn = match(pbox[keep], pconf[keep], gbox[gcls == cls], match_iou)
            out[cls]["tp"] += tp
            out[cls]["fp"] += fp
            out[cls]["fn"] += fn
    return out


def relax_nms_time_limit():
    """Disable ultralytics' per-call NMS deadline (utils/nms.py: time_limit = 2 + 0.05*bs).

    When it trips, NMS breaks out of the batch loop and every remaining image silently gets
    zero boxes, which wrecks the per-image breakdown on a 176-image call.
    """
    original = _nms.non_max_suppression
    if getattr(original, "_tgv_no_deadline", False):
        return
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        kwargs["max_time_img"] = 3600.0
        return original(*args, **kwargs)
    wrapper._tgv_no_deadline = True
    _nms.non_max_suppression = wrapper


def predict_all(model, images, args, conf, chunk=8):
    results = []
    for start in range(0, len(images), chunk):
        part = images[start:start + chunk]
        results.extend(model.predict(part, imgsz=args.imgsz, batch=chunk, device=args.device,
                                     conf=conf, iou=args.iou, verbose=False, max_det=300))
    return results


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images, names = split_images(args.data, args.split)
    relax_nms_time_limit()
    model = YOLO(str(args.model))

    print(f"== {args.split}: {len(images)} images, model={args.model.name} ==")
    metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=4,
                        device=args.device, conf=0.001, iou=args.iou, plots=True,
                        project=str(args.out.resolve()), name=f"{args.split}_val", exist_ok=True)
    if args.limit:
        images = images[:args.limit]
    official = {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}
    per_class_official = {}
    index_of_class = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    for cls_id, name in enumerate(names):
        if cls_id not in index_of_class:
            continue
        i = index_of_class[cls_id]
        per_class_official[name] = {
            "precision": float(metrics.box.p[i]), "recall": float(metrics.box.r[i]),
            "mAP50": float(metrics.box.ap50[i]), "mAP50-95": float(metrics.box.ap[i]),
        }
    print(f"official mAP50={official['mAP50']:.4f}  mAP50-95={official['mAP50-95']:.4f}")

    results = predict_all(model, images, args, args.conf_min)
    preds, gts = [], []
    for image, result in zip(images, results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            preds.append((np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)))
        else:
            preds.append((boxes.xyxy.cpu().numpy().astype(np.float32),
                          boxes.conf.cpu().numpy().astype(np.float32),
                          boxes.cls.cpu().numpy().astype(np.int64)))
        height, width = result.orig_shape
        gts.append(load_gt(image, width, height))
    total_pred = sum(len(p[0]) for p in preds)
    print(f"predict: {total_pred} boxes on {len(images)} images ({total_pred / max(len(images), 1):.1f}/img)")

    rows = []

    def add_row(scope, subset, counts, n_images):
        for cls, name in enumerate(names):
            c = counts[cls]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            rows.append({
                "scope": scope, "subset": subset, "class": name, "images": n_images,
                "gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                "shapes_per_image": round((tp + fp) / n_images, 2) if n_images else 0.0,
            })

    add_row("all", args.split, score(preds, gts, names, args.match_iou, args.conf), len(images))

    by_group = defaultdict(list)
    by_origin = defaultdict(list)
    machine = machine_stems()
    for i, image in enumerate(images):
        by_group[group_of(image.stem)].append(i)
        by_origin["machine" if image.stem in machine else "hand"].append(i)

    for group, idx in sorted(by_group.items()):
        counts = score([preds[i] for i in idx], [gts[i] for i in idx], names, args.match_iou, args.conf)
        add_row("aperture", group, counts, len(idx))
    for origin, idx in sorted(by_origin.items()):
        counts = score([preds[i] for i in idx], [gts[i] for i in idx], names, args.match_iou, args.conf)
        add_row("label_origin", origin, counts, len(idx))

    sweep_rows = []
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        counts = score(preds, gts, names, args.match_iou, thr)
        for cls, name in enumerate(names):
            c = counts[cls]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            sweep_rows.append({
                "threshold": thr, "class": name, "gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                "fp_per_image": round(fp / len(images), 2),
                "boxes_per_image": round((tp + fp) / len(images), 2),
            })
    with open(args.out / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)

    csv_path = args.out / "breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n== breakdown @ conf={args.conf}, match IoU={args.match_iou} ==")
    print(f"{'scope':12s} {'subset':9s} {'class':8s} {'img':>4s} {'gt':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'P':>6s} {'R':>6s} {'box/img':>7s}")
    for r in rows:
        print(f"{r['scope']:12s} {r['subset']:9s} {r['class']:8s} {r['images']:4d} {r['gt']:5d} "
              f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['shapes_per_image']:7.2f}")

    print(f"\n== threshold sweep (all {len(images)} images) ==")
    print(f"{'thr':>5s} {'class':8s} {'gt':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'FP/img':>7s}")
    for r in sweep_rows:
        print(f"{r['threshold']:5.2f} {r['class']:8s} {r['gt']:5d} {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['fp_per_image']:7.2f}")

    summary = {
        "model": str(args.model), "split": args.split, "images": len(images),
        "conf": args.conf, "match_iou": args.match_iou,
        "official": official, "official_per_class": per_class_official,
        "breakdown": rows, "sweep": sweep_rows,
        "hand_images": len(by_origin.get("hand", [])),
        "machine_images": len(by_origin.get("machine", [])),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nreport: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
