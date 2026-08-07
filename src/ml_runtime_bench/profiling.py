from __future__ import annotations

import copy
from pathlib import Path

import torch

from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.models import TinyDecoderLM
from ml_runtime_bench.runtimes import CompileRuntime, EagerRuntime
from ml_runtime_bench.utils import (
    inference_context,
    resolve_device,
    resolve_dtype,
    seed_everything,
    synchronize,
    validate_dtype_device,
)


def run_profile(
    *,
    model_config: ModelConfig,
    mode: str,
    device_name: str,
    dtype_name: str,
    batch_size: int,
    sequence_length: int,
    warmup_iterations: int,
    profile_iterations: int,
    output_dir: Path,
    compile_mode: str = "default",
    dynamic: bool | None = None,
    seed: int = 7,
) -> None:
    if mode not in {"eager", "compile"}:
        raise ValueError("profile mode must be eager or compile")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = resolve_dtype(dtype_name)
    validate_dtype_device(dtype, device)
    seed_everything(seed)

    model = TinyDecoderLM(model_config).to(device=device, dtype=dtype).eval()
    input_ids = torch.randint(
        0, model_config.vocab_size, (batch_size, sequence_length), device=device
    )
    runtime = (
        EagerRuntime()
        if mode == "eager"
        else CompileRuntime(compile_mode=compile_mode, dynamic=dynamic)
    )
    runtime.prepare(copy.deepcopy(model), input_ids, output_dir / "artifacts")

    with inference_context():
        for _ in range(warmup_iterations):
            runtime.run(input_ids)
        synchronize(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            acc_events=True,
        ) as profiler:
            for _ in range(profile_iterations):
                runtime.run(input_ids)
        synchronize(device)

    profiler.export_chrome_trace(str(output_dir / "trace.json"))
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    table = profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=40)
    (output_dir / "top_ops.txt").write_text(
        "\n".join(line.rstrip() for line in table.splitlines()) + "\n",
        encoding="utf-8",
    )
