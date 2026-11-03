#!/usr/bin/env python3
"""Build an Ultralytics dataset directory from images/ + labels/ with hard links."""

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
    p.add_argument("--mode", default="auto", choices=["auto", "hardlink", "symlink", "copy"],
                   help="hardlink needs no admin on Windows and behaves like the old symlinks; "
                        "copy is the fallback when the split sits on another volume")
    p.add_argument("--restratify", action="store_true",
                   help="draw a fresh split instead of reusing the existing split.json")
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


def link(src, dst, mode="auto"):
    """Point dst at src. Hard links behave like the symlinks this tree used on macOS
    (same inode, edits propagate) but need no developer-mode/admin on Windows."""
    tries = {"hardlink": ("hardlink",), "symlink": ("symlink",), "copy": ("copy",),
             "auto": ("hardlink", "symlink")}[mode]
    for kind in tries:
        try:
            if kind == "hardlink":
                os.link(src.resolve(), dst)
            elif kind == "symlink":
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            return kind
        except OSError:
            continue
    raise SystemExit(f"cannot link {src} -> {dst}; rerun with --mode copy")


def load_manifest(path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not all(key in data for key in ("train", "val", "test")):
        return None
    return data


def restore(manifest, samples):
    by_stem = {img.stem: (img, label, boxes) for img, label, boxes in samples}
    splits = []
    for key in ("train", "val", "test"):
        picked = []
        for name in manifest[key]:
            sample = by_stem.get(Path(name).stem)
            if sample is None:
                raise SystemExit(f"{key}: {name} is in the manifest but has no image/label pair")
            picked.append(sample)
        splits.append(picked)
    return splits


# stratify by plate group *and* presence of every non-zero class, so each split
# sees the same aperture mix and the rare class stays represented everywhere
def stratify(samples, names, args):
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
    return train, val, test


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

    manifest = None if args.restratify else load_manifest(args.out_dir / "split.json")
    if manifest:
        train, val, test = restore(manifest, samples)
        print(f"split reused from {args.out_dir / 'split.json'} "
              f"(seed {manifest.get('seed')}, val_frac {manifest.get('val_frac')})")
    else:
        train, val, test = stratify(samples, names, args)

    used_modes = set()
    ensure_clean(args.out_dir, args.force)
    for split, items in (("train", train), ("val", val), ("test", test)):
        for img, label, _ in items:
            used_modes.add(link(img, args.out_dir / "images" / split / img.name, args.mode))
            used_modes.add(link(label, args.out_dir / "labels" / split / label.name, args.mode))

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

    record = dict(manifest or {})
    record.setdefault("seed", args.seed)
    record.setdefault("val_frac", args.val_frac)
    record.setdefault("test_frac", args.test_frac)
    record["train"] = [p.name for p, _, _ in train]
    record["val"] = [p.name for p, _, _ in val]
    record["test"] = [p.name for p, _, _ in test]
    (args.out_dir / "split.json").write_text(json.dumps(record, indent=2))

    def summary(items):
        boxes = Counter(c for _, _, b in items for c, *_ in b)
        no_scratch = sum(1 for _, _, b in items if not any(c == 1 for c, *_ in b))
        groups = Counter(p.stem.rsplit("_", 1)[0] for p, _, _ in items)
        return (f"{len(items):4d} images, {dict(sorted(boxes.items()))} boxes, "
                f"{no_scratch} without scratch, groups={dict(sorted(groups.items()))}")

    print(f"dataset -> {args.out_dir}  ({'+'.join(sorted(used_modes))})")
    print("train:", summary(train))
    print("val:  ", summary(val))
    if test:
        print("test: ", summary(test))
    print("classes:", {i: n for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
