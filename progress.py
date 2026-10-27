#!/usr/bin/env python3
"""Print training progress and ETA from a run's results.csv."""

import argparse
import csv
from pathlib import Path

import yaml


def parse_args():
    code_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=code_dir / "runs" / "prelabel_v8s_1024")
    p.add_argument("--last", type=int, default=10, help="how many recent epochs to print")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = args.run / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found (training not started yet)")

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise SystemExit("no epochs recorded yet")

    total = 200
    args_yaml = args.run / "args.yaml"
    if args_yaml.exists():
        total = int(yaml.safe_load(args_yaml.read_text()).get("epochs", total))

    key = lambda r, name: float(r[name])
    done = len(rows)
    last = rows[-1]
    secs = key(last, "time")
    per_epoch = secs / done
    eta_min = per_epoch * (total - done) / 60

    print(f"{args.run.name}: epoch {done}/{total}  "
          f"({done / total * 100:5.1f}%)  elapsed {secs / 3600:5.2f}h  "
          f"{per_epoch / 60:.1f} min/epoch  ETA {eta_min / 60:5.2f}h")
    print(f"{'ep':>4} {'box':>7} {'cls':>7} {'dfl':>7} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
    for r in rows[-args.last:]:
        print(f"{int(key(r, 'epoch')):4d} {key(r, 'train/box_loss'):7.3f} {key(r, 'train/cls_loss'):7.3f} "
              f"{key(r, 'train/dfl_loss'):7.3f} {key(r, 'metrics/precision(B)'):6.3f} "
              f"{key(r, 'metrics/recall(B)'):6.3f} {key(r, 'metrics/mAP50(B)'):7.4f} "
              f"{key(r, 'metrics/mAP50-95(B)'):9.4f}")

    best = max(rows, key=lambda r: key(r, "metrics/mAP50-95(B)"))
    print(f"best so far: epoch {int(key(best, 'epoch'))}  "
          f"mAP50={key(best, 'metrics/mAP50(B)'):.4f}  mAP50-95={key(best, 'metrics/mAP50-95(B)'):.4f}")


if __name__ == "__main__":
    main()
