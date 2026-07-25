from __future__ import annotations

import copy
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ml_runtime_bench.config import SuiteConfig
from ml_runtime_bench.models import TinyDecoderLM
from ml_runtime_bench.runtimes import CompileRuntime, EagerRuntime, OnnxRuntime
from ml_runtime_bench.runtimes.base import RuntimeAdapter
from ml_runtime_bench.utils import (
    inference_context,
    peak_memory_mb,
    reset_peak_memory,
    resolve_device,
    resolve_dtype,
    seed_everything,
    synchronize,
    system_metadata,
    validate_dtype_device,
)


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float


@dataclass
class BenchmarkResult:
    status: str
    mode: str
    batch_size: int
    sequence_length: int
    dtype: str
    device: str
    setup_ms: float | None = None
    cold_start_ms: float | None = None
    latency: LatencySummary | None = None
    throughput_tokens_per_second: float | None = None
    peak_memory_mb: float | None = None
    correctness_max_abs_error: float | None = None
    runtime_metadata: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(values_ms: list[float]) -> LatencySummary:
    return LatencySummary(
        mean_ms=statistics.fmean(values_ms),
        p50_ms=statistics.median(values_ms),
        p95_ms=percentile(values_ms, 0.95),
        min_ms=min(values_ms),
        max_ms=max(values_ms),
        stdev_ms=statistics.pstdev(values_ms),
    )


def create_runtime(mode: str, config: SuiteConfig) -> RuntimeAdapter:
    benchmark = config.benchmark
    if mode == "eager":
        return EagerRuntime()
    if mode == "compile":
        return CompileRuntime(
            compile_mode=benchmark.compile_mode,
            fullgraph=benchmark.fullgraph,
            dynamic=benchmark.dynamic,
        )
    if mode == "onnx":
        return OnnxRuntime(dynamic=bool(benchmark.dynamic))
    raise ValueError(f"unsupported mode: {mode}")


def _time_single_call(
    runtime: RuntimeAdapter, input_ids: torch.Tensor
) -> tuple[float, torch.Tensor]:
    synchronize(input_ids.device)
    start = time.perf_counter_ns()
    output = runtime.run(input_ids)
    synchronize(input_ids.device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return elapsed_ms, output


def _benchmark_one(
    *,
    mode: str,
    config: SuiteConfig,
    base_state: dict[str, torch.Tensor],
    reference_model: TinyDecoderLM,
    input_ids: torch.Tensor,
    artifact_dir: Path,
) -> BenchmarkResult:
    benchmark = config.benchmark
    device = input_ids.device
    dtype = resolve_dtype(benchmark.dtype)
    model = TinyDecoderLM(config.model).to(device=device, dtype=dtype).eval()
    model.load_state_dict(copy.deepcopy(base_state))
    runtime = create_runtime(mode, config)

    setup_start = time.perf_counter_ns()
    runtime.prepare(model, input_ids, artifact_dir)
    setup_ms = (time.perf_counter_ns() - setup_start) / 1e6

    reset_peak_memory(device)
    with inference_context():
        cold_ms, cold_output = _time_single_call(runtime, input_ids)

        with torch.no_grad():
            reference = reference_model(input_ids)
        reference_for_compare = reference.to(cold_output.device)
        max_error = float((cold_output.float() - reference_for_compare.float()).abs().max().item())
        torch.testing.assert_close(
            cold_output.float(),
            reference_for_compare.float(),
            atol=benchmark.correctness_atol,
            rtol=benchmark.correctness_rtol,
        )

        for _ in range(benchmark.warmup_iterations):
            runtime.run(input_ids)
        synchronize(device)

        samples_ms: list[float] = []
        for _ in range(benchmark.benchmark_iterations):
            elapsed_ms, _ = _time_single_call(runtime, input_ids)
            samples_ms.append(elapsed_ms)

    latency = summarize_latencies(samples_ms)
    tokens = input_ids.shape[0] * input_ids.shape[1]
    throughput = tokens / (latency.mean_ms / 1000)
    return BenchmarkResult(
        status="ok",
        mode=mode,
        batch_size=input_ids.shape[0],
        sequence_length=input_ids.shape[1],
        dtype=benchmark.dtype,
        device=str(device),
        setup_ms=setup_ms,
        cold_start_ms=cold_ms,
        latency=latency,
        throughput_tokens_per_second=throughput,
        peak_memory_mb=peak_memory_mb(device),
        correctness_max_abs_error=max_error,
        runtime_metadata=runtime.metadata(),
    )


def run_suite(config: SuiteConfig, output_dir: Path, *, strict: bool = False) -> dict[str, Any]:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.benchmark.device)
    dtype = resolve_dtype(config.benchmark.dtype)
    validate_dtype_device(dtype, device)
    seed_everything(config.benchmark.seed)

    base_model = TinyDecoderLM(config.model).to(device=device, dtype=dtype).eval()
    base_state = copy.deepcopy(base_model.state_dict())
    results: list[BenchmarkResult] = []

    for batch_size in config.benchmark.batch_sizes:
        for sequence_length in config.benchmark.sequence_lengths:
            input_ids = torch.randint(
                low=0,
                high=config.model.vocab_size,
                size=(batch_size, sequence_length),
                device=device,
            )
            reference_model = TinyDecoderLM(config.model).to(device=device, dtype=dtype).eval()
            reference_model.load_state_dict(copy.deepcopy(base_state))

            for mode in config.benchmark.modes:
                artifact_dir = output_dir / "artifacts" / f"{mode}-b{batch_size}-s{sequence_length}"
                try:
                    result = _benchmark_one(
                        mode=mode,
                        config=config,
                        base_state=base_state,
                        reference_model=reference_model,
                        input_ids=input_ids,
                        artifact_dir=artifact_dir,
                    )
                except Exception as exc:  # benchmark reports must preserve unsupported/error states
                    if strict:
                        raise
                    result = BenchmarkResult(
                        status="skipped",
                        mode=mode,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        dtype=config.benchmark.dtype,
                        device=str(device),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)

    return {
        "schema_version": 1,
        "suite": config.to_dict(),
        "system": system_metadata(),
        "results": [result.to_dict() for result in results],
    }
