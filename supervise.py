#!/usr/bin/env python3
"""Guard a training run: restart on death, evaluate when it finishes.

Start the run once with train.py, then hand it to this. Process discovery,
detaching and killing go through psutil so the same script works on Windows
(where pgrep/pmset do not exist) and on macOS/Linux.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
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
    "--imgsz", "1024", "--patience", "30",
    "--close-mosaic", "10",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="official_v8s_1024", help="run directory name under code/runs")
    p.add_argument("--device", default="auto", help="auto / 0 (CUDA) / cpu / mps")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    return p.parse_args()


def command(args):
    return BASE_CMD + ["--project", str(CODE / "runs"), "--name", args.run,
                       "--batch", str(args.batch), "--workers", str(args.workers),
                       "--cache", args.cache, "--device", args.device]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def matching_pids(needle):
    """Pids whose command line contains needle, excluding this supervisor."""
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if needle in cmdline and proc.info["pid"] != os.getpid():
            pids.append(proc.info["pid"])
    return pids


def trainer_pids():
    return matching_pids("train.py")


def tensorboard_pids():
    # "tensorboard" alone also matches the trainer's --tensorboard flag, so pin to the module.
    return matching_pids("tensorboard.main")


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


def spawn(cmd, log_path):
    with open(log_path, "a", buffering=1) as out:
        kwargs = dict(cwd=str(CODE), stdin=subprocess.DEVNULL, stdout=out,
                      stderr=subprocess.STDOUT)
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(cmd, **kwargs)


def launch(args, checkpoint):
    cmd = command(args)
    cmd[2:2] = ["--model", str(checkpoint)]
    return spawn(cmd, LOG)


def start_tensorboard(args):
    return spawn([sys.executable, "-m", "tensorboard.main", "--logdir", str(CODE / "runs"),
                  "--host", "127.0.0.1", "--port", "6006", "--reload_interval", "30"],
                 CODE / "runs" / "tensorboard.log")


def run_step(name, cmd, attempts=2):
    for attempt in range(1, attempts + 1):
        log(f"run {name} (attempt {attempt})")
        result = subprocess.run(cmd, cwd=str(CODE))
        if result.returncode == 0:
            return True
        log(f"{name} exited {result.returncode}")
        time.sleep(30)
    return False


def evaluate(args):
    best = RUN / "weights" / "best.pt"
    if not best.exists():
        log(f"no {best}, cannot evaluate")
        return
    run_step("eval_test", [sys.executable, str(CODE / "eval_test.py"), "--model", str(best),
                           "--device", args.device,
                           "--out", str(CODE / "runs" / f"eval_test_{args.run}")])
    run_step("report", [sys.executable, str(CODE / "report.py"), "--run", str(RUN),
                        "--eval", str(CODE / "runs" / f"eval_test_{args.run}" / "summary.json")])
    run_step("export_metrics", [sys.executable, str(CODE / "export_metrics.py")])
    log(f"results -> runs/eval_test_{args.run}/summary.json, runs/{args.run}/report/report.md")


def environment_warnings():
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged:
        log(f"WARNING running on battery ({battery.percent:.0f}% left); sleep will kill the run")
    if os.name == "nt" and battery is not None:
        log("hint: keep the machine awake with `powercfg /change standby-timeout-ac 0`")
    free_gb = shutil.disk_usage(CODE).free / 1e9
    if free_gb < 5:
        log(f"WARNING only {free_gb:.1f} GB disk free")


def main():
    global RUN, LOG
    args = parse_args()
    RUN = CODE / "runs" / args.run
    LOG = CODE / "runs" / f"train_{args.run}.log"

    last_epoch = epoch_count()
    last_beat = time.time()
    restarts = stalled = 0
    log(f"supervisor start: run={args.run} device={args.device} "
        f"pids={trainer_pids()} epochs={last_epoch}")
    environment_warnings()

    while True:
        if finished():
            log("training finished")
            break

        pids = trainer_pids()
        if not pids:
            checkpoint = pick_checkpoint()
            if checkpoint is None:
                log(f"no usable checkpoint under {RUN / 'weights'}, cannot resume; "
                    f"start the run with train.py first")
                return 1
            restarts += 1
            if restarts > MAX_RESTARTS:
                log(f"giving up after {restarts} restarts")
                return 1
            log(f"trainer gone, restart #{restarts} from {checkpoint.name} at epoch {epoch_count()}")
            launch(args, checkpoint)
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
            start_tensorboard(args)

        if LOG.exists():
            idle = (time.time() - LOG.stat().st_mtime) / 60
            if idle > STALL_MINUTES:
                stalled += 1
                log(f"no log output for {idle:.0f} min (check {stalled}/2)")
                if stalled >= 2:
                    log(f"killing stalled trainer {pids}")
                    for pid in pids:
                        try:
                            psutil.Process(pid).kill()
                        except psutil.NoSuchProcess:
                            pass
                    stalled = 0
                    time.sleep(10)
                    continue
        time.sleep(CHECK_SECONDS)

    evaluate(args)
    log("supervisor exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
