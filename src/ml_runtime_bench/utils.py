from __future__ import annotations

import contextlib
import os
import platform
import random

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def validate_dtype_device(dtype: torch.dtype, device: torch.device) -> None:
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError(
            "float16 CPU execution is not supported consistently; use float32 or bfloat16"
        )
    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("bfloat16 MPS support is not reliable for this suite")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    if device.type == "cpu" and resource is not None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return value / (1024**2)
        return value / 1024
    return None


@contextlib.contextmanager
def inference_context() -> Iterator[None]:
    with torch.inference_mode():
        yield


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_model_name() -> str | None:
    processor = platform.processor().strip()
    if processor:
        return processor
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return None


def system_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model_name(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "git_revision": git_revision(),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    try:
        import onnxruntime as ort

        metadata["onnxruntime"] = ort.__version__
        metadata["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        metadata["onnxruntime"] = None
    try:
        import triton

        metadata["triton"] = triton.__version__
    except ImportError:
        metadata["triton"] = None
    return metadata
