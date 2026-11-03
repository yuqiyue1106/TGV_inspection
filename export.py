#!/usr/bin/env python3
"""Export a trained checkpoint for deployment.

Which target formats are reachable depends on the machine you run this on:
  onnx / torchscript  build anywhere
  coreml              macOS only (needs `pip install coremltools`)
  engine (TensorRT)   NVIDIA CUDA only
"""

import argparse
import sys
from pathlib import Path

# device comes first on purpose: ultralytics imports torch, and the MPS fallback switch
# inside device.py only has any effect if it is set before that happens.
import device as dev
from ultralytics import YOLO


def parse_args():
    code_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=code_dir / "runs" / "tgv_v8s_1024" / "weights" / "best.pt")
    p.add_argument("--format", default="onnx", choices=["onnx", "coreml", "engine", "torchscript"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--half", action="store_true", help="FP16; CUDA hosts only")
    p.add_argument("--dynamic", action="store_true", help="dynamic input shapes (ONNX)")
    return p.parse_args()


def check_target(args):
    """Fail with a readable message instead of a stack trace from deep inside ultralytics."""
    if args.format == "engine" and not dev.cuda_available():
        raise SystemExit("--format engine needs TensorRT, so it only builds on an NVIDIA CUDA "
                         "host; export onnx or torchscript here instead")
    if args.format == "coreml" and sys.platform != "darwin":
        raise SystemExit("--format coreml only builds on macOS (coremltools refuses to install "
                         "elsewhere); export onnx on this machine")
    if args.half and not dev.cuda_available():
        raise SystemExit("--half writes an FP16 graph meant for an NVIDIA GPU; drop it or run "
                         "the export on the CUDA box")


def main():
    args = parse_args()
    check_target(args)
    model = YOLO(str(args.model))
    path = model.export(
        format=args.format, imgsz=args.imgsz, opset=args.opset,
        half=args.half, dynamic=args.dynamic,
    )
    print(f"exported: {path}")
    if args.format == "onnx":
        print("run with onnxruntime; keep the same imgsz and tiling as in training")
    if args.format == "coreml":
        print("deploy with coremltools/mlpackage on the M4 Neural Engine")


if __name__ == "__main__":
    main()
