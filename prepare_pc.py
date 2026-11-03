#!/usr/bin/env python3
"""Rebuild the dataset split and data.yaml after copying the project to another machine.

The split directories here are symlinks into the flat images/ and labels/ folders, so they break
as soon as the folder is copied off macOS. This rebuilds them as hardlinks (or real copies) from
dataset/split.json, which keeps the exact same 80/10/10 split.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DATASET = ROOT / "dataset"
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"


def build_one(src, dst, copy):
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def rebuild(split, files, dry_run, copy):
    stats = {"link": 0, "copy": 0, "skip": 0, "missing": 0}
    for sub, src_dir, suffix in (("images", IMAGES, ""),
                                 ("labels", LABELS, ".txt")):
        dst_dir = DATASET / sub / split
        if not dst_dir.exists() and not dry_run:
            dst_dir.mkdir(parents=True)
        for name in files:
            src = src_dir / name if not suffix else src_dir / (Path(name).stem + suffix)
            if not src.exists():
                stats["missing"] += 1
                continue
            dst = dst_dir / src.name
            if dst.is_symlink() or dst.exists():
                if dst.exists() and dst.resolve() == src.resolve():
                    stats["skip"] += 1
                    continue
                if not dry_run:
                    dst.unlink()
            if not dry_run:
                stats[build_one(src, dst, copy)] += 1
    return stats


def broken_links():
    count = 0
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            for path in (DATASET / sub / split).glob("*"):
                if path.is_symlink() and not path.exists():
                    count += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="report only, change nothing")
    p.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    args = p.parse_args()

    split_file = DATASET / "split.json"
    if not split_file.exists():
        sys.exit(f"{split_file} not found")
    split = json.loads(split_file.read_text())

    print(f"root: {ROOT}")
    for name in ("train", "val", "test"):
        files = split[name]
        stats = rebuild(name, files, args.check, args.copy)
        print(f"{name:5s} {len(files):5d} images  link={stats['link']:5d} "
              f"copy={stats['copy']:5d} skip={stats['skip']:5d} missing={stats['missing']}")
    if broken_links():
        print(f"broken symlinks still present: {broken_links()}")

    if args.check:
        print("\n--check: nothing written")
    else:
        import yaml
        yaml_path = DATASET / "data.yaml"
        cfg = yaml.safe_load(yaml_path.read_text())
        cfg["path"] = str(DATASET).replace("\\", "/")
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"\nrewrote {yaml_path} -> path: {cfg['path']}")
        for cache in DATASET.glob("labels/*.cache"):
            cache.unlink()
            print(f"removed stale {cache.name}")

    try:
        # device before torch: it sets PYTORCH_ENABLE_MPS_FALLBACK for the MPS case.
        import device as dev
        import torch
        print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()} "
              f"mps={dev.mps_available()}")
        print(f"auto device -> {dev.describe(dev.resolve('auto'))}")
        print(f"preset        -> {dev.preset(dev.resolve('auto'))}")
    except ImportError:
        print("\ntorch not installed yet")

    print("\nnext: python -c \"from ultralytics import YOLO; m=YOLO('runs/official_v8s_1024/weights/best.pt');"
          " print(m.val(data='../dataset/data.yaml', split='test', imgsz=1024)[\"box\"].map)\"")
    print("      expect mAP50-95 = 0.4401 (same as on the Mac)")


if __name__ == "__main__":
    main()
