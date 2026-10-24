#!/usr/bin/env python3
"""Run detection on TGV plate images, with optional tiled inference for full resolution."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COLORS = {0: (0, 190, 0), 1: (0, 0, 255)}


def parse_args():
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=code_dir / "runs" / "tgv_v8s_1024" / "weights" / "best.pt")
    p.add_argument("--source", type=Path, required=True, help="image file or directory")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU during inference")
    p.add_argument("--merge-iou", type=float, default=0.5, help="NMS IoU when merging tiles")
    p.add_argument("--merge-ios", type=float, default=0.8,
                   help="intersection/min-area threshold; merges boxes cut by tile borders")
    p.add_argument("--device", default="mps")
    p.add_argument("--tiled", action="store_true", help="slice the image into overlapping tiles")
    p.add_argument("--slice-size", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=code_dir / "runs" / "predict")
    p.add_argument("--no-save-txt", action="store_true")
    return p.parse_args()


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def window_starts(length, size, step):
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, step))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def merge_detections(boxes, scores, classes, iou_thr, ios_thr):
    """Class-wise greedy merge using both IoU and intersection-over-smaller-area.

    A defect crossing a tile border shows up as a truncated box in one tile and a
    fuller box in the neighbour; their IoU is low but the small box lies almost
    entirely inside the large one, which the IoS rule catches.
    """
    keep = []
    for cls in np.unique(classes):
        order = np.flatnonzero(classes == cls)
        order = order[np.argsort(-scores[order])]
        kept = []
        for i in order:
            box = boxes[i]
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
            suppressed = False
            for j in kept:
                other = boxes[j]
                x1, y1 = max(box[0], other[0]), max(box[1], other[1])
                x2, y2 = min(box[2], other[2]), min(box[3], other[3])
                inter = max(x2 - x1, 0) * max(y2 - y1, 0)
                if inter <= 0:
                    continue
                other_area = max((other[2] - other[0]) * (other[3] - other[1]), 1e-6)
                iou = inter / (area + other_area - inter)
                ios = inter / min(area, other_area)
                if iou >= iou_thr or ios >= ios_thr:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(i)
        keep.extend(kept)
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return boxes[keep], scores[keep], classes[keep]


def tiled_infer(model, image, args):
    height, width = image.shape[:2]
    size = args.slice_size
    step = max(1, int(size * (1 - args.overlap)))
    tiles, offsets = [], []
    for y in window_starts(height, size, step):
        for x in window_starts(width, size, step):
            tiles.append(np.ascontiguousarray(image[y:y + size, x:x + size]))
            offsets.append((x, y))
    results = model.predict(tiles, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False, max_det=300)
    boxes, scores, classes = [], [], []
    for result, (x, y) in zip(results, offsets):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        b = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        b[:, [0, 2]] += x
        b[:, [1, 3]] += y
        boxes.append(b)
        scores.append(result.boxes.conf.cpu().numpy().astype(np.float32))
        classes.append(result.boxes.cls.cpu().numpy().astype(np.int64))
    if not boxes:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes = np.concatenate(classes)
    return merge_detections(boxes, scores, classes, args.merge_iou, args.merge_ios)


def full_infer(model, image, args):
    result = model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                           device=args.device, verbose=False, max_det=300)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    return (result.boxes.xyxy.cpu().numpy().astype(np.float32),
            result.boxes.conf.cpu().numpy().astype(np.float32),
            result.boxes.cls.cpu().numpy().astype(np.int64))


def draw(image, boxes, scores, classes, names):
    canvas = image.copy()
    thickness = max(2, round(image.shape[1] / 1200))
    font_scale = max(0.6, image.shape[1] / 3200)
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        color = COLORS.get(int(cls), (0, 165, 255))
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{names[int(cls)]} {score:.2f}"
        cv2.putText(canvas, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    model = YOLO(str(args.model))
    names = model.names
    source = args.source
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS) if source.is_dir() else [source]
    if not images:
        raise SystemExit(f"no images under {source}")

    out_dir = args.out / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in images:
        image = read_image(path)
        if args.tiled:
            boxes, scores, classes = tiled_infer(model, image, args)
        else:
            boxes, scores, classes = full_infer(model, image, args)

        height, width = image.shape[:2]
        annotated = draw(image, boxes, scores, classes, names)
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not args.no_save_txt:
            lines = []
            for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
                xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                bw, bh = (x2 - x1) / width, (y2 - y1) / height
                lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (out_dir / f"{path.stem}.txt").write_text("\n".join(lines))

        counts = {names[int(c)]: int((classes == c).sum()) for c in np.unique(classes)}
        print(f"{path.name}: {counts if counts else 'no detection'}")

    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
