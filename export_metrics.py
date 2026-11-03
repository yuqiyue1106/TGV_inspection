#!/usr/bin/env python3
"""Turn Ultralytics results.csv files into tidy, plotting-friendly CSVs.

For every code/runs/<run>/results.csv this writes
  code/runs/metrics_<run>.csv   per-epoch table with derived timing columns
and a combined
  code/runs/metrics_all_runs.csv

Safe to run while training: it only reads results.csv, which Ultralytics
appends to at the end of each epoch.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RUNS_DIR = CODE_DIR / "runs"

# ultralytics column -> tidy column
RENAME = {
    "train/box_loss": "train_box_loss",
    "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr_pg0",
    "lr/pg1": "lr_pg1",
    "lr/pg2": "lr_pg2",
}

COLUMNS = ["run", "epoch", "epoch_end", "elapsed_min", "epoch_min", "is_best",
           "best_so_far_mAP50_95",
           "train_box_loss", "train_cls_loss", "train_dfl_loss",
           "precision", "recall", "mAP50", "mAP50_95",
           "val_box_loss", "val_cls_loss", "val_dfl_loss",
           "lr_pg0", "lr_pg1", "lr_pg2"]


def clean_header(name):
    return name.strip().lstrip("\ufeff")


def read_run(results_csv):
    """Return [dict] of tidy rows, or [] when the file is empty/unreadable."""
    try:
        with results_csv.open(newline="") as f:
            reader = csv.reader(f)
            header = [clean_header(h) for h in next(reader)]
            raw = [row for row in reader if row and row[0].strip()]
    except (StopIteration, OSError):
        return []

    idx = {name: i for i, name in enumerate(header)}
    if "epoch" not in idx or "time" not in idx:
        return []

    rows = []
    best = float("-inf")
    prev_time = 0.0
    for row in raw:
        def get(name):
            i = idx.get(name)
            if i is None or i >= len(row):
                return ""
            return row[i].strip()

        try:
            epoch = int(float(get("epoch")))
            elapsed = float(get("time"))
        except ValueError:
            continue

        tidy = {"run": results_csv.parent.name, "epoch": epoch,
                "elapsed_min": round(elapsed / 60, 3),
                "epoch_min": round((elapsed - prev_time) / 60, 3)}
        prev_time = elapsed

        for src, dst in RENAME.items():
            value = get(src)
            if value == "":
                tidy[dst] = ""
                continue
            try:
                tidy[dst] = round(float(value), 6)
            except ValueError:
                tidy[dst] = value

        score = tidy.get("mAP50_95")
        if isinstance(score, float):
            if score > best:
                best = score
            tidy["is_best"] = int(score == best)
            tidy["best_so_far_mAP50_95"] = best
        else:
            tidy["is_best"] = ""
            tidy["best_so_far_mAP50_95"] = ""

        # results.csv is written at the end of each epoch, so mtime is the
        # end of the last completed epoch.
        tidy["epoch_end"] = ""
        rows.append(tidy)

    if rows:
        try:
            last_end = datetime.fromtimestamp(results_csv.stat().st_mtime)
            run_start = last_end - timedelta(seconds=prev_time)
            for row in rows:
                end = run_start + timedelta(minutes=row["elapsed_min"])
                row["epoch_end"] = end.strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            pass
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    all_rows = []
    for results_csv in sorted(RUNS_DIR.glob("*/results.csv")):
        rows = read_run(results_csv)
        if not rows:
            continue
        write_csv(RUNS_DIR / f"metrics_{results_csv.parent.name}.csv", rows)
        all_rows.extend(rows)
        last = rows[-1]
        print(f"{last['run']:24s} epoch {last['epoch']:>3}  "
              f"mAP50={last.get('mAP50', '')!s:>7}  "
              f"mAP50-95={last.get('mAP50_95', '')!s:>7}  "
              f"{last['elapsed_min'] / 60:.2f}h")

    write_csv(RUNS_DIR / "metrics_all_runs.csv", all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {RUNS_DIR / 'metrics_all_runs.csv'}")


if __name__ == "__main__":
    main()
