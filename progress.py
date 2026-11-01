#!/usr/bin/env python3
"""Print training progress and ETA from a run's results.csv + log."""

import argparse
import csv
import re
from pathlib import Path

import yaml

EPOCH_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+[\d.]+G\s+.*?(\d+)%\s.*?(\d+)/(\d+)\s+([\d.]+)s/it.*?<(\S+)$")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_args():
    code_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=code_dir / "runs" / "prelabel_v8s_1024")
    p.add_argument("--log", type=Path, default=code_dir / "runs" / "train_prelabel.log")
    p.add_argument("--last", type=int, default=10, help="how many recent epochs to print")
    return p.parse_args()


def live_status(log_path):
    """Most recent tqdm line: training progress inside the current epoch."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")[-20000:]
    for line in reversed(re.split(r"[\r\n]+", text)):
        m = EPOCH_LINE.match(ANSI.sub("", line).strip())
        if m:
            return {
                "epoch": int(m.group(1)), "total": int(m.group(2)),
                "pct": int(m.group(3)), "batch": int(m.group(4)), "batches": int(m.group(5)),
                "remaining": m.group(7),
            }
    return None


def training_finished(log_path):
    """train.py prints 'best checkpoint:' right before it exits."""
    if not log_path.exists():
        return False
    return "best checkpoint:" in log_path.read_text(errors="ignore")[-4000:]


def main():
    args = parse_args()
    csv_path = args.run / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found (training not started yet)")

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise SystemExit("no epochs recorded yet")

    total = 100
    args_yaml = args.run / "args.yaml"
    if args_yaml.exists():
        total = int(yaml.safe_load(args_yaml.read_text()).get("epochs", total))

    key = lambda r, name: float(r[name])
    done = len(rows)
    last = rows[-1]
    secs = key(last, "time")
    per_epoch = secs / done
    eta_min = per_epoch * (total - done) / 60

    finished = done >= total or training_finished(args.log)
    if finished:
        reason = "早停" if done < total else "跑满"
        print(f"{args.run.name}: 训练已结束（{reason}于 {done}/{total} epoch，用时 {secs / 3600:.2f}h）")
    else:
        print(f"{args.run.name}: 已完成 {done}/{total} epoch ({done / total * 100:5.1f}%)  "
              f"elapsed {secs / 3600:5.2f}h  {per_epoch / 60:.1f} min/epoch  "
              f"ETA {eta_min / 60:5.2f}h")
        live = live_status(args.log)
        if live and live["epoch"] > done:
            print(f"正在训练 epoch {live['epoch']}/{live['total']}: "
                  f"batch {live['batch']}/{live['batches']} ({live['pct']}%), "
                  f"本 epoch 剩余约 {live['remaining']}")
        elif live and live["epoch"] == done:
            print(f"正在验证/收尾 epoch {live['epoch']} ...")

    print(f"{'ep':>4} {'box':>7} {'cls':>7} {'dfl':>7} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
    for r in rows[-args.last:]:
        print(f"{int(key(r, 'epoch')):4d} {key(r, 'train/box_loss'):7.3f} {key(r, 'train/cls_loss'):7.3f} "
              f"{key(r, 'train/dfl_loss'):7.3f} {key(r, 'metrics/precision(B)'):6.3f} "
              f"{key(r, 'metrics/recall(B)'):6.3f} {key(r, 'metrics/mAP50(B)'):7.4f} "
              f"{key(r, 'metrics/mAP50-95(B)'):9.4f}")

    best = max(rows, key=lambda r: key(r, "metrics/mAP50-95(B)"))
    print(f"best so far: epoch {int(key(best, 'epoch'))}  "
          f"mAP50={key(best, 'metrics/mAP50(B)'):.4f}  mAP50-95={key(best, 'metrics/mAP50-95(B)'):.4f}")
    if finished:
        ckpt = args.run / "weights" / "best.pt"
        print(f"best checkpoint: {ckpt}" + ("" if ckpt.exists() else "  (缺失!)"))


if __name__ == "__main__":
    main()
