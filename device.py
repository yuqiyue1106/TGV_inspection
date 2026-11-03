#!/usr/bin/env python3
"""Map a --device string onto what ultralytics and torch expect.

Kept in one place because the original scripts hardcoded "mps" and this tree now
runs on a CUDA box; "auto" lets the same command line work on either machine.
"""

import torch


def resolve(spec):
    """Return the value to hand to ultralytics; None means "pick the best available"."""
    if spec in (None, "", "auto"):
        return None
    text = str(spec).strip().lower()
    if text in ("cpu", "mps"):
        return text
    if text in ("cuda", "gpu", "nvidia"):
        return 0
    if "," in text:
        return [int(part) for part in text.split(",") if part.strip()]
    return int(text)


def is_cuda(device):
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    if isinstance(device, (list, tuple)):
        return bool(device)
    if isinstance(device, int):
        return device < torch.cuda.device_count()
    return False


def amp_enabled(device):
    """Mixed precision only pays off on CUDA; it is unsupported on MPS and CPU."""
    return is_cuda(device)


def describe(device):
    if device is None:
        return f"auto -> {'cuda:' + str(torch.cuda.get_device_name(0)) if is_cuda(None) else 'cpu'}"
    if isinstance(device, int) and is_cuda(device):
        return f"cuda:{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
