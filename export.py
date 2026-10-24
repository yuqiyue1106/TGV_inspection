#!/usr/bin/env python3
"""Export a trained checkpoint for deployment."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    code_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=code_dir / "runs" / "tgv_v8s_1024" / "weights" / "best.pt")
    p.add_argument("--format", default="onnx", choices=["onnx", "coreml", "engine", "torchscript"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--half", action="store_true", help="FP16; use on GPU targets only")
    p.add_argument("--dynamic", action="store_true", help="dynamic input shapes (ONNX)")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.model))
    path = model.export(
        format=args.format, imgsz=args.imgsz, opset=args.opset,
        half=args.half, dynamic=args.dynamic,
    )
    print(f"exported: {path}")
    if args.format == "onnx":
        print("run with onnxruntime; keep the same imgsz and tiling as in training")


if __name__ == "__main__":
    main()
