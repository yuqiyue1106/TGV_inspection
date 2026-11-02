#!/usr/bin/env python3
"""Guard the official training run: restart on death, evaluate when it finishes."""

import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
RUN = CODE / "runs" / "official_v8s_1024"
LOG = CODE / "runs" / "train_official_v8s_1024.log"
CHECK_SECONDS = 60
MAX_RESTARTS = 15
STALL_MINUTES = 25
HEARTBEAT_MINUTES = 10

BASE_CMD = [
    sys.executable, str(CODE / "train.py"),
    "--resume", "--tensorboard",
    "--project", str(CODE / "runs"), "--name", RUN.name,
    "--batch", "8", "--imgsz", "1024", "--patience", "30",
    "--close-mosaic", "10", "--workers", "0", "--cache", "ram", "--device", "mps",
]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def trainer_pids():
    out = subprocess.run(["pgrep", "-f", "train[.]py"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


def tensorboard_pids():
    out = subprocess.run(["pgrep", "-f", "tensorboard[.]main"], capture_output=True, text=True)
    return [p for p in out.stdout.split() if p]


def epoch_count():
    csv_path = RUN / "results.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f))


def best_map():
    csv_path = RUN / "results.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        return max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    except Exception:
        return None


def log_tail(size=8000):
    if not LOG.exists():
        return ""
    with open(LOG, "rb") as f:
        f.seek(max(0, LOG.stat().st_size - size))
        return f.read().decode(errors="ignore")


def finished():
    return "best checkpoint:" in log_tail()


def resumable(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"checkpoint {path.name} unreadable: {exc}")
        return False
    ok = ck.get("epoch", -1) >= 0 and ck.get("optimizer") is not None
    if not ok:
        log(f"checkpoint {path.name} lacks epoch/optimizer state")
    return ok


def pick_checkpoint():
    for name in ("last.pt", "best.pt"):
        path = RUN / "weights" / name
        if path.exists() and resumable(path):
            return path
    return None


def launch(checkpoint):
    cmd = BASE_CMD[:2] + ["--model", str(checkpoint)] + BASE_CMD[2:]
    with open(LOG, "a", buffering=1) as out:
        return subprocess.Popen(cmd, cwd=CODE, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def start_tensorboard():
    with open(CODE / "runs" / "tensorboard.log", "a", buffering=1) as out:
        return subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(CODE / "runs"),
             "--host", "127.0.0.1", "--port", "6006", "--reload_interval", "30"],
            cwd=CODE, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)


def run_step(name, cmd, attempts=2):
    for attempt in range(1, attempts + 1):
        log(f"run {name} (attempt {attempt})")
        result = subprocess.run(cmd, cwd=CODE)
        if result.returncode == 0:
            return True
        log(f"{name} exited {result.returncode}")
        time.sleep(30)
    return False


def evaluate():
    best = RUN / "weights" / "best.pt"
    if not best.exists():
        log(f"no {best}, cannot evaluate")
        return
    run_step("eval_test", [sys.executable, str(CODE / "eval_test.py"), "--model", str(best),
                           "--out", str(CODE / "runs" / "eval_test_official")])
    run_step("report", [sys.executable, str(CODE / "report.py")])
    run_step("export_metrics", [sys.executable, str(CODE / "export_metrics.py")])
    log("results -> runs/eval_test_official/summary.json, runs/official_v8s_1024/report/report.md")


def environment_warnings():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    if "Battery Power" in batt:
        log("WARNING running on battery; sleep/undervoltage will kill the run")
    free_gb = shutil.disk_usage(CODE).free / 1e9
    if free_gb < 5:
        log(f"WARNING only {free_gb:.1f} GB disk free")


def main():
    last_epoch = epoch_count()
    last_beat = time.time()
    restarts = stalled = 0
    log(f"supervisor start: pids={trainer_pids()} epochs={last_epoch}")
    environment_warnings()

    while True:
        if finished():
            log("training finished")
            break

        pids = trainer_pids()
        if not pids:
            checkpoint = pick_checkpoint()
            if checkpoint is None:
                log("no usable checkpoint, cannot resume")
                return 1
            restarts += 1
            if restarts > MAX_RESTARTS:
                log(f"giving up after {restarts} restarts")
                return 1
            log(f"trainer gone, restart #{restarts} from {checkpoint.name} at epoch {epoch_count()}")
            launch(checkpoint)
            time.sleep(90)
            continue

        epochs = epoch_count()
        if epochs > last_epoch:
            last_epoch, restarts, stalled = epochs, 0, 0

        if time.time() - last_beat > HEARTBEAT_MINUTES * 60:
            best = best_map()
            best_txt = f" best e{best['epoch']} mAP50-95={float(best['metrics/mAP50-95(B)']):.4f}" if best else ""
            log(f"heartbeat: {epochs} epochs, log idle {(time.time() - LOG.stat().st_mtime) / 60:.0f} min{best_txt}")
            last_beat = time.time()

        if not tensorboard_pids():
            log("tensorboard gone, restarting")
            start_tensorboard()

        if LOG.exists():
            idle = (time.time() - LOG.stat().st_mtime) / 60
            if idle > STALL_MINUTES:
                stalled += 1
                log(f"no log output for {idle:.0f} min (check {stalled}/2)")
                if stalled >= 2:
                    log(f"killing stalled trainer {pids}")
                    subprocess.run(["kill", "-9", *pids])
                    stalled = 0
                    time.sleep(10)
                    continue
        time.sleep(CHECK_SECONDS)

    evaluate()
    log("supervisor exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
