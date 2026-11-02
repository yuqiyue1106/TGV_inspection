#!/usr/bin/env python3
"""Train YOLOv8 on the TGV dust/scratch dataset (Apple Silicon MPS)."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(code_dir / "weights" / "yolov8s.pt"),
                   help="pretrained .pt checkpoint or a model .yaml")
    p.add_argument("--data", type=Path, default=root / "dataset" / "data.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="mps", help="mps / cpu / 0 (CUDA)")
    p.add_argument("--workers", type=int, default=0, help="0 is the safe choice on macOS")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="auto",
                   help="'auto' ignores --lr0; pass AdamW/SGD to honour it when fine-tuning")
    p.add_argument("--lr0", type=float, default=0.01, help="initial LR (only used when --optimizer != auto)")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--tensorboard", action="store_true",
                   help="write TensorBoard events into the run directory")
    p.add_argument("--project", type=Path, default=code_dir / "runs")
    p.add_argument("--name", default="tgv_v8s_1024")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = args.epochs
        args.name = args.name + "_smoke"

    model = YOLO(args.model)
    if args.tensorboard:
        from ultralytics.utils import SETTINGS
        previous = SETTINGS["tensorboard"]
        SETTINGS.update({"tensorboard": True})
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        patience=args.patience,
        seed=args.seed,
        amp=args.device != "mps",  # AMP is not supported on MPS
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        # augmentation: scratch orientation is arbitrary, keep mosaic on by default
        degrees=10,
        fliplr=0.5,
        flipud=0.5,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=args.close_mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        resume=args.resume,
    )
    if args.tensorboard:
        SETTINGS.update({"tensorboard": previous})
    best = args.project / args.name / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")


if __name__ == "__main__":
    main()
