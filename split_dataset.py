#!/usr/bin/env python3
"""Build an Ultralytics dataset directory from images/ + labels/ using symlinks."""

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args():
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, default=root / "images")
    p.add_argument("--labels-dir", type=Path, default=root / "labels")
    p.add_argument("--out-dir", type=Path, default=root / "dataset")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.0,
                   help="held-out test split; never used for training or model selection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="rebuild an existing dataset dir")
    return p.parse_args()


def read_label(path):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts[0], *map(float, parts[1:])
        boxes.append((int(cls), x, y, w, h))
    return boxes


def load_classes(labels_dir):
    classes_file = labels_dir / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
        if names:
            return names
    return ["dust", "scratch"]


def ensure_clean(out_dir, force):
    image_root = out_dir / "images"
    label_root = out_dir / "labels"
    for d in (image_root, label_root):
        if d.exists():
            if not force:
                raise SystemExit(f"{d} already exists; rerun with --force to rebuild")
            shutil.rmtree(d)
    for split in ("train", "val", "test"):
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (label_root / split).mkdir(parents=True, exist_ok=True)


def link(src, dst):
    os.symlink(src.resolve(), dst)


def main():
    args = parse_args()
    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    samples = []
    for img in images:
        label = args.labels_dir / f"{img.stem}.txt"
        if label.exists():
            samples.append((img, label, read_label(label)))
    if not samples:
        raise SystemExit("no image/label pairs found")

    class_ids = sorted({c for _, _, boxes in samples for c, *_ in boxes})
    names = load_classes(args.labels_dir)
    missing = [c for c in class_ids if c >= len(names)]
    if missing:
        raise SystemExit(f"class ids {missing} have no entry in {args.labels_dir / 'classes.txt'}")
    names = names[: max(class_ids) + 1] if class_ids else names

    # stratify by plate group *and* presence of every non-zero class, so each split
    # sees the same aperture mix and the rare class stays represented everywhere
    def stratum(sample):
        img, _, boxes = sample
        group = img.stem.rsplit("_", 1)[0]
        present = {c for c, *_ in boxes}
        return (group,) + tuple(c in present for c in range(1, len(names)))

    groups = {}
    for sample in samples:
        groups.setdefault(stratum(sample), []).append(sample)

    def take(group, frac):
        n = round(len(group) * frac)
        if len(group) > 1:
            n = min(max(n, 1), len(group) - 1)
        else:
            n = 0
        return n

    rng = random.Random(args.seed)
    test, val, train = [], [], []
    for key in sorted(groups, reverse=True):
        group = groups[key]
        rng.shuffle(group)
        n_test = take(group, args.test_frac) if args.test_frac > 0 else 0
        test.extend(group[:n_test])
        rest = group[n_test:]
        n_val = take(rest, args.val_frac / (1 - args.test_frac)) if args.val_frac > 0 else 0
        val.extend(rest[:n_val])
        train.extend(rest[n_val:])
    for items in (train, val, test):
        rng.shuffle(items)

    ensure_clean(args.out_dir, args.force)
    for split, items in (("train", train), ("val", val), ("test", test)):
        for img, label, _ in items:
            link(img, args.out_dir / "images" / split / img.name)
            link(label, args.out_dir / "labels" / split / label.name)

    data = {
        "path": str(args.out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    import yaml

    with open(args.out_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train": [p.name for p, _, _ in train],
        "val": [p.name for p, _, _ in val],
        "test": [p.name for p, _, _ in test],
    }
    (args.out_dir / "split.json").write_text(json.dumps(manifest, indent=2))

    def summary(items):
        boxes = Counter(c for _, _, b in items for c, *_ in b)
        no_scratch = sum(1 for _, _, b in items if not any(c == 1 for c, *_ in b))
        groups = Counter(p.stem.rsplit("_", 1)[0] for p, _, _ in items)
        return (f"{len(items):4d} images, {dict(sorted(boxes.items()))} boxes, "
                f"{no_scratch} without scratch, groups={dict(sorted(groups.items()))}")

    print(f"dataset -> {args.out_dir}")
    print("train:", summary(train))
    print("val:  ", summary(val))
    if test:
        print("test: ", summary(test))
    print("classes:", {i: n for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
