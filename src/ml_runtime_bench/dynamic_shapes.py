from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.models import TinyDecoderLM
from ml_runtime_bench.utils import (
    inference_context,
    resolve_device,
    resolve_dtype,
    seed_everything,
    synchronize,
    validate_dtype_device,
)


def _timed_call(
    model: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
) -> float:
    synchronize(input_ids.device)
    start = time.perf_counter_ns()
    model(input_ids)
    synchronize(input_ids.device)
    return (time.perf_counter_ns() - start) / 1e6


def run_dynamic_shape_experiment(
    *,
    output_dir: Path,
    lengths: list[int],
    batch_size: int,
    device_name: str,
    dtype_name: str,
    dynamic: bool,
    model_config: ModelConfig | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    if not lengths:
        raise ValueError("lengths cannot be empty")
    config = model_config or ModelConfig(max_seq_len=max(512, max(lengths)))
    if max(lengths) > config.max_seq_len:
        raise ValueError("a requested length exceeds model.max_seq_len")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name)
    validate_dtype_device(dtype, device)
    seed_everything(seed)

    torch._dynamo.reset()
    model = TinyDecoderLM(config).to(device=device, dtype=dtype).eval()
    compiled = torch.compile(model, backend="inductor", dynamic=dynamic)
    observations: list[dict[str, Any]] = []

    with inference_context():
        for index, length in enumerate(lengths):
            input_ids = torch.randint(0, config.vocab_size, (batch_size, length), device=device)
            first_ms = _timed_call(compiled, input_ids)
            repeat_ms = _timed_call(compiled, input_ids)
            counters = getattr(torch._dynamo.utils, "counters", {})
            stats = dict(counters.get("stats", {})) if hasattr(counters, "get") else {}
            observations.append(
                {
                    "index": index,
                    "sequence_length": length,
                    "first_call_ms": first_ms,
                    "repeat_call_ms": repeat_ms,
                    "dynamo_stats": stats,
                }
            )

    payload = {
        "dynamic": dynamic,
        "batch_size": batch_size,
        "dtype": dtype_name,
        "device": str(device),
        "lengths": lengths,
        "observations": observations,
    }
    (output_dir / "dynamic_shapes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Dynamic-shape experiment",
        "",
        f"- `torch.compile(dynamic={dynamic})`",
        f"- Batch size: `{batch_size}`",
        f"- Shape order: `{lengths}`",
        "",
        "| Index | Sequence | First call ms | Immediate repeat ms | Unique graphs | Calls captured |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in observations:
        stats = item["dynamo_stats"]
        lines.append(
            f"| {item['index']} | {item['sequence_length']} | {item['first_call_ms']:.2f} | "
            f"{item['repeat_call_ms']:.2f} | {stats.get('unique_graphs', '—')} | "
            f"{stats.get('calls_captured', '—')} |"
        )
    lines.extend(
        [
            "",
            "Repeated sequence lengths are intentional. A large first-call cost for a new shape followed by a much smaller repeat call suggests compilation or specialization overhead. Internal Dynamo counters are diagnostic and may vary by PyTorch version.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
