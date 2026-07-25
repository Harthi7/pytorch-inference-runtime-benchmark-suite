from __future__ import annotations

import json
from pathlib import Path


def plot_results(results_path: Path, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install the 'viz' optional dependency group") from exc

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    completed = [result for result in payload["results"] if result.get("status") == "ok"]
    if not completed:
        raise ValueError("no completed results to plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['mode']}\nB{r['batch_size']} S{r['sequence_length']}" for r in completed]
    p50_values = [r["latency"]["p50_ms"] for r in completed]
    fig = plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, p50_values)
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Inference latency by runtime and shape")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_p50.png", dpi=160)
    plt.close(fig)

    throughput = [r["throughput_tokens_per_second"] for r in completed]
    fig = plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, throughput)
    ax.set_ylabel("input tokens / second")
    ax.set_title("Transformer prefill throughput")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "throughput.png", dpi=160)
    plt.close(fig)
