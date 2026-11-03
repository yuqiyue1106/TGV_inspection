#!/usr/bin/env python3
"""Train YOLOv8 on the TGV dust/scratch dataset (CUDA, MPS or CPU)."""

import argparse
from pathlib import Path

# device comes first on purpose: ultralytics imports torch, and the MPS fallback switch
# inside device.py only has any effect if it is set before that happens.
import device as dev
from ultralytics import YOLO


def default_model():
    """weights/ is git-ignored, so a fresh clone on the Mac has no local checkpoint.

    Falling back to the bare model name lets ultralytics download it instead of
    crashing on a path that only exists on the machine that trained last.
    """
    local = Path(__file__).resolve().parent / "weights" / "yolov8s.pt"
    return str(local) if local.exists() else "yolov8s.pt"


def parse_args():
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="pretrained .pt checkpoint or a model .yaml "
                        "(default: weights/yolov8s.pt, or a download when absent)")
    p.add_argument("--data", type=Path, default=root / "dataset" / "data.yaml")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="kept identical on every backend so runs stay comparable")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=None,
                   help="per-backend preset when unset: 4 on CUDA, 2 on MPS and CPU")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--device", default="auto", help="auto / 0 (CUDA) / mps / cpu")
    p.add_argument("--workers", type=int, default=None,
                   help="0 keeps everything in one process, the safe choice on Windows "
                        "and when --cache ram; unset takes the per-backend preset")
    p.add_argument("--cache", default=None, choices=["ram", "disk", "false"],
                   help="ram is fine on the CUDA box, disk keeps a Mac within unified memory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", default="auto",
                   help="'auto' ignores --lr0; pass AdamW/SGD to honour it when fine-tuning")
    p.add_argument("--lr0", type=float, default=0.01, help="initial LR (only used when --optimizer != auto)")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--cos-lr", action="store_true",
                   help="cosine LR schedule instead of the default linear decay")
    p.add_argument("--lrf", type=float, default=None,
                   help="final LR as a fraction of lr0 (ultralytics default: 0.01)")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="optimizer weight decay (ultralytics default: 0.0005)")
    p.add_argument("--cls-pw", type=float, default=0.0,
                   help="inverse-frequency class weight power, 0 disables it; "
                        "0.5 raises the weight of the rarer class by about 1.5x here")
    p.add_argument("--box-gain", type=float, default=None, help="box loss gain (default 7.5)")
    p.add_argument("--cls-gain", type=float, default=None, help="cls loss gain (default 0.5)")
    p.add_argument("--dfl-gain", type=float, default=None, help="dfl loss gain (default 1.5)")
    p.add_argument("--degrees", type=float, default=None, help="rotation range (default 10)")
    p.add_argument("--translate", type=float, default=None, help="translation range (default 0.1)")
    p.add_argument("--scale", type=float, default=None, help="scale jitter (default 0.5)")
    p.add_argument("--mosaic", type=float, default=None, help="mosaic probability (default 1.0)")
    p.add_argument("--mixup", type=float, default=None, help="mixup probability (default 0.0)")
    p.add_argument("--cutmix", type=float, default=None, help="cutmix probability (default 0.0)")
    p.add_argument("--hsv-h", type=float, default=None, help="hue jitter (default 0.015)")
    p.add_argument("--hsv-s", type=float, default=None, help="saturation jitter (default 0.7)")
    p.add_argument("--hsv-v", type=float, default=None, help="value jitter (default 0.4)")
    p.add_argument("--multi-scale", type=float, default=None,
                   help="multi-scale training range as a fraction of imgsz (0 disables)")
    p.add_argument("--tensorboard", action="store_true",
                   help="write TensorBoard events into the run directory")
    p.add_argument("--project", type=Path, default=code_dir / "runs")
    p.add_argument("--name", default="tgv_v8s_1024")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    gpu = dev.resolve(args.device)

    # Anything the caller left out comes from the backend preset, so the same command
    # line is safe on the CUDA box and on a laptop that shares memory with the GPU.
    prof = dev.preset(gpu)
    for name in ("batch", "workers", "cache"):
        if getattr(args, name) is None:
            setattr(args, name, prof[name])

    if args.model is None:
        args.model = default_model()

    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = args.epochs
        args.name = args.name + "_smoke"

    model = YOLO(args.model)
    if args.tensorboard:
        from ultralytics.utils import SETTINGS
        previous = SETTINGS["tensorboard"]
        SETTINGS.update({"tensorboard": True})
    dev.announce(gpu)
    print(f"preset: batch={args.batch} workers={args.workers} cache={args.cache} "
          f"amp={dev.amp_enabled(gpu)}")

    # Only forward what the caller changed, so the defaults stay ultralytics' own.
    extra = {}
    for key, value in (
        ("box", args.box_gain), ("cls", args.cls_gain), ("dfl", args.dfl_gain),
        ("lrf", args.lrf), ("weight_decay", args.weight_decay),
        ("multi_scale", args.multi_scale), ("mixup", args.mixup), ("cutmix", args.cutmix),
    ):
        if value is not None:
            extra[key] = value
    if args.cos_lr:
        extra["cos_lr"] = True

    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=gpu,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        patience=args.patience,
        seed=args.seed,
        amp=dev.amp_enabled(gpu),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        val=True,
        # augmentation: scratch orientation is arbitrary, keep mosaic on by default
        degrees=10 if args.degrees is None else args.degrees,
        translate=0.1 if args.translate is None else args.translate,
        scale=0.5 if args.scale is None else args.scale,
        mosaic=1.0 if args.mosaic is None else args.mosaic,
        hsv_h=0.015 if args.hsv_h is None else args.hsv_h,
        hsv_s=0.7 if args.hsv_s is None else args.hsv_s,
        hsv_v=0.4 if args.hsv_v is None else args.hsv_v,
        fliplr=0.5,
        flipud=0.5,
        close_mosaic=args.close_mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        cls_pw=args.cls_pw,
        resume=args.resume,
        **extra,
    )
    if args.tensorboard:
        SETTINGS.update({"tensorboard": previous})
    best = args.project / args.name / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")


if __name__ == "__main__":
    main()
