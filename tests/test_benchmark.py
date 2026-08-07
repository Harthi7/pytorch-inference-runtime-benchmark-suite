from pathlib import Path

import torch

from ml_runtime_bench.benchmark import (
    assess_correctness,
    percentile,
    run_suite,
    summarize_latencies,
)
from ml_runtime_bench.config import BenchmarkConfig, ModelConfig, SuiteConfig


def test_percentile_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_latency_summary() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0])
    assert summary.p50_ms == 2.0
    assert summary.mean_ms == 2.0


def test_eager_smoke_suite(tmp_path: Path) -> None:
    config = SuiteConfig(
        name="test",
        model=ModelConfig(
            vocab_size=128,
            max_seq_len=16,
            d_model=32,
            n_heads=4,
            n_layers=1,
            mlp_ratio=2.0,
        ),
        benchmark=BenchmarkConfig(
            batch_sizes=[1],
            sequence_lengths=[8],
            modes=["eager"],
            device="cpu",
            dtype="float32",
            warmup_iterations=0,
            benchmark_iterations=2,
        ),
    )
    payload = run_suite(config, tmp_path)
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["latency"]["p50_ms"] > 0


def test_correctness_accepts_candidate_closer_to_high_precision_reference() -> None:
    eager = torch.tensor([0.10, -0.10])
    candidate = torch.tensor([0.05, -0.05])
    oracle = torch.tensor([0.0, 0.0])

    assessment = assess_correctness(
        candidate,
        eager,
        oracle,
        atol=0.01,
        rtol=0.0,
        accuracy_reference_dtype="float32",
    )

    assert assessment.parity_passed is False
    assert assessment.accuracy_rmse < assessment.eager_accuracy_rmse
    assert assessment.acceptable is True


def test_correctness_rejects_candidate_worse_than_eager_reference() -> None:
    eager = torch.tensor([0.05, -0.05])
    candidate = torch.tensor([0.10, -0.10])
    oracle = torch.tensor([0.0, 0.0])

    assessment = assess_correctness(
        candidate,
        eager,
        oracle,
        atol=0.01,
        rtol=0.0,
        accuracy_reference_dtype="float32",
    )

    assert assessment.parity_passed is False
    assert assessment.accuracy_rmse > assessment.eager_accuracy_rmse
    assert assessment.acceptable is False
