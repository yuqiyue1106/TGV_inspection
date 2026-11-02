#!/usr/bin/env python3
"""Mirror a run's per-epoch results.csv into TensorBoard scalars, live.

Ultralytics only writes TensorBoard events when its global `tensorboard`
setting is enabled *before* training starts (`yolo settings tensorboard=True`).
This bridges a training that is already running -- or one that already
finished -- into event files under code/runs/tb_logs/<run>/.

    python tb_bridge.py --run official_v8s_1024 --pid 1234 --follow
    python tb_bridge.py --all          # one-shot pass over every run

Then: python -m tensorboard.main --logdir runs/tb_logs --port 6006
"""

import argparse
import json
import os
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from export_metrics import RUNS_DIR, read_run

SCALARS = [
    "mAP50", "mAP50_95", "precision", "recall", "best_so_far_mAP50_95",
    "train_box_loss", "train_cls_loss", "train_dfl_loss",
    "val_box_loss", "val_cls_loss", "val_dfl_loss",
    "lr_pg0",
]

TB_ROOT = RUNS_DIR / "tb_logs"


def is_alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def sync_run(run, writer):
    """Append epochs newer than the last one written; return the newest epoch."""
    results_csv = RUNS_DIR / run / "results.csv"
    if not results_csv.exists():
        return None

    state_file = TB_ROOT / run / "last_epoch.json"
    last = 0
    if state_file.exists():
        try:
            last = int(json.loads(state_file.read_text())["epoch"])
        except (ValueError, KeyError):
            last = 0

    rows = [r for r in read_run(results_csv) if r["epoch"] > last]
    for row in rows:
        step = row["epoch"]
        for key in SCALARS:
            value = row.get(key)
            if isinstance(value, float):
                writer.add_scalar(key, value, step)
        writer.add_scalar("epoch_min", row["epoch_min"], step)

    if rows:
        writer.flush()
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"epoch": rows[-1]["epoch"]}))
    return rows[-1]["epoch"] if rows else last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run directory name under code/runs")
    p.add_argument("--all", action="store_true", help="one-shot sync of every run")
    p.add_argument("--pid", type=int, default=None,
                   help="follow until this process exits")
    p.add_argument("--interval", type=float, default=30, help="poll seconds")
    args = p.parse_args()

    if args.all:
        runs = [p.parent.name for p in sorted(RUNS_DIR.glob("*/results.csv"))]
    elif args.run:
        runs = [args.run]
    else:
        raise SystemExit("pass --run <name> or --all")

    writers = {}
    try:
        for run in runs:
            if (RUNS_DIR / run / "results.csv").exists():
                writers[run] = SummaryWriter(log_dir=str(TB_ROOT / run), flush_secs=5)

        while True:
            for run, writer in writers.items():
                newest = sync_run(run, writer)
                if newest is not None:
                    print(f"{run}: epoch {newest}", flush=True)
            if args.pid is None or not is_alive(args.pid):
                break
            time.sleep(args.interval)
    finally:
        for writer in writers.values():
            writer.close()
        print(f"tensorboard events -> {TB_ROOT}")


if __name__ == "__main__":
    main()
