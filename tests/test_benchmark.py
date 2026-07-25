from pathlib import Path

from ml_runtime_bench.benchmark import percentile, run_suite, summarize_latencies
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
