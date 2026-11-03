#!/usr/bin/env python3
"""Map a --device string onto what ultralytics and torch expect.

Kept in one place because the original scripts hardcoded "mps" and this tree now
runs on a CUDA box; "auto" lets the same command line work on either machine.

Two things a single command line cannot express also live here:

  * the MPS operator fallback switch, which has to be set *before* torch is
    imported, so no script may import torch ahead of this module, and
  * the per-backend resource presets, because an NVIDIA card, an Apple Silicon
    GPU and a bare CPU do not want the same batch / workers / cache.
"""

import os
import platform
import subprocess
import sys

# Deliberately before `import torch`: this is what stops an Apple Silicon run from
# dying mid-epoch on an operator the Metal backend has not implemented.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (the env var above has to be set first)


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


def cuda_available():
    return bool(torch.cuda.is_available())


def mps_available():
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def auto_target():
    """Mirror what ultralytics picks for an empty device: CUDA > MPS > CPU."""
    if cuda_available():
        return 0
    if mps_available():
        return "mps"
    return "cpu"


def is_cuda(device):
    if not cuda_available():
        return False
    if device is None:
        return True
    if isinstance(device, (list, tuple)):
        return bool(device)
    if isinstance(device, int):
        return device < torch.cuda.device_count()
    return False


def backend(device):
    """Collapse a resolved device onto "cuda" / "mps" / "cpu"."""
    if device is None:
        device = auto_target()
    if isinstance(device, str):
        text = device.strip().lower()
        if text.startswith("cuda"):
            return "cuda"
        if text == "mps":
            return "mps"
        return "cpu"
    return "cuda" if is_cuda(device) else "cpu"


def amp_enabled(device):
    """Mixed precision only pays off on CUDA; it is unsupported on MPS and CPU."""
    return backend(device) == "cuda"


def preset(device):
    """Resource defaults per backend, so one command line works on both machines.

    The CUDA row deliberately matches what this tree has already been trained with,
    so nothing changes on the Windows box; only the Mac gets conservative values.
    imgsz stays out of here on purpose -- every checkpoint in runs/ was trained at
    1024, and silently lowering it would make the two machines incomparable.
    """
    kind = backend(device)
    cores = os.cpu_count() or 4
    if kind == "cuda":
        return {"backend": kind, "batch": 4, "workers": 0, "cache": "ram"}
    if kind == "mps":
        # Unified memory has to hold the dataset *and* the Metal working set.
        return {"backend": kind, "batch": 2, "workers": min(4, cores), "cache": "disk"}
    # A bare CPU run lives on the MacBook Air M4 (10 cores, no accelerator): keep the
    # memory footprint small and let every core feed the dataloader instead.
    return {"backend": kind, "batch": 2, "workers": min(8, cores), "cache": "disk"}


def _chip_name():
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            name = out.stdout.strip()
            if out.returncode == 0 and name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
        return "Apple Silicon"
    # Windows hands back "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel" through
    # platform.processor(); the architecture is all the log line needs.
    return platform.machine()


def describe(device):
    """Human-readable line for the log, covering all three backends."""
    kind = backend(device)
    if kind == "cuda":
        if device is None:
            idx = 0
        elif isinstance(device, (list, tuple)):
            idx = device[0] if device else 0
        elif isinstance(device, int):
            idx = device
        else:
            idx = 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if kind == "mps":
        return f"mps ({_chip_name()} Metal GPU)"
    if device is None:
        return f"cpu ({_chip_name()}, no CUDA or MPS backend found)"
    return f"cpu ({_chip_name()})"


def mismatch(device):
    """Return a warning when an explicit --device has no backend on this machine."""
    if device is None or isinstance(device, str) and device.strip().lower() == "auto":
        return None
    if isinstance(device, (int, list, tuple)) and not cuda_available():
        return (f"--device {device} asks for CUDA, but torch.cuda.is_available() is False. "
                f"Either the CPU-only torch build is installed (see requirements-cuda.txt) "
                f"or there is no NVIDIA driver. Use auto / mps / cpu instead.")
    if isinstance(device, str) and device.strip().lower() == "mps" and not mps_available():
        return (f"--device mps asks for Apple Metal, but MPS is unavailable here "
                f"(needs macOS on arm64 plus torch>=2.4). Use auto or cpu.")
    return None


def announce(device):
    """Print the resolved device plus any mismatch warning; returns the device."""
    warning = mismatch(device)
    if warning:
        print(f"WARNING: {warning}", flush=True)
    print(f"device: {describe(device)}", flush=True)
    return device
