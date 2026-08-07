from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    latency = result.get("latency") or {}
    row = {
        "status": result.get("status"),
        "mode": result.get("mode"),
        "batch_size": result.get("batch_size"),
        "sequence_length": result.get("sequence_length"),
        "dtype": result.get("dtype"),
        "device": result.get("device"),
        "setup_ms": result.get("setup_ms"),
        "cold_start_ms": result.get("cold_start_ms"),
        "mean_ms": latency.get("mean_ms"),
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "min_ms": latency.get("min_ms"),
        "max_ms": latency.get("max_ms"),
        "stdev_ms": latency.get("stdev_ms"),
        "throughput_tokens_per_second": result.get("throughput_tokens_per_second"),
        "peak_memory_mb": result.get("peak_memory_mb"),
        "correctness_max_abs_error": result.get("correctness_max_abs_error"),
        "correctness_parity_passed": result.get("correctness_parity_passed"),
        "correctness_accuracy_reference_dtype": result.get("correctness_accuracy_reference_dtype"),
        "correctness_accuracy_max_abs_error": result.get("correctness_accuracy_max_abs_error"),
        "correctness_accuracy_mean_abs_error": result.get("correctness_accuracy_mean_abs_error"),
        "correctness_accuracy_rmse": result.get("correctness_accuracy_rmse"),
        "correctness_eager_accuracy_rmse": result.get("correctness_eager_accuracy_rmse"),
        "correctness_argmax_match": result.get("correctness_argmax_match"),
        "error": result.get("error"),
    }
    return row


def write_csv(payload: dict[str, Any], path: Path) -> None:
    rows = [_flatten_result(result) for result in payload["results"]]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _comparison_lines(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        key = (result["batch_size"], result["sequence_length"])
        grouped.setdefault(key, {})[result["mode"]] = result

    for (batch, sequence), modes in sorted(grouped.items()):
        eager = modes.get("eager")
        if not eager:
            continue
        eager_p50 = eager["latency"]["p50_ms"]
        for mode in ("compile", "onnx"):
            candidate = modes.get(mode)
            if not candidate:
                continue
            candidate_p50 = candidate["latency"]["p50_ms"]
            speedup = eager_p50 / candidate_p50
            overhead = max(
                0.0,
                (candidate.get("setup_ms") or 0.0)
                + (candidate.get("cold_start_ms") or 0.0)
                - (eager.get("setup_ms") or 0.0)
                - (eager.get("cold_start_ms") or 0.0),
            )
            saved = eager_p50 - candidate_p50
            if saved > 0:
                break_even = overhead / saved
                break_even_text = f"estimated break-even after {break_even:.0f} calls"
            else:
                break_even_text = "no steady-state break-even because it was not faster"
            lines.append(
                f"- B={batch}, S={sequence}: **{mode}** was {speedup:.2f}x eager p50; "
                f"{break_even_text}."
            )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    system = payload["system"]
    suite = payload["suite"]
    lines = [
        f"# Benchmark report: {suite['name']}",
        "",
        "## Environment",
        "",
        f"- Platform: `{system.get('platform')}`",
        f"- Python: `{system.get('python')}`",
        f"- PyTorch: `{system.get('torch')}`",
        f"- Device: `{system.get('cuda_device_name') or system.get('cpu_model') or system.get('device')}`",
        f"- ONNX Runtime: `{system.get('onnxruntime')}`",
        f"- Triton: `{system.get('triton')}`",
        "",
        "## Results",
        "",
        "| Status | Mode | Batch | Sequence | Setup ms | Cold ms | p50 ms | p95 ms | Tokens/s | Peak MB | Max abs vs eager | Parity | RMSE vs accuracy ref | Argmax vs ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        latency = result.get("latency") or {}
        lines.append(
            "| {status} | {mode} | {batch} | {sequence} | {setup} | {cold} | {p50} | {p95} | {throughput} | {memory} | {error} | {parity} | {accuracy_rmse} | {argmax_match} |".format(
                status=result.get("status"),
                mode=result.get("mode"),
                batch=result.get("batch_size"),
                sequence=result.get("sequence_length"),
                setup=_fmt(result.get("setup_ms")),
                cold=_fmt(result.get("cold_start_ms")),
                p50=_fmt(latency.get("p50_ms")),
                p95=_fmt(latency.get("p95_ms")),
                throughput=_fmt(result.get("throughput_tokens_per_second"), 0),
                memory=_fmt(result.get("peak_memory_mb")),
                error=_fmt(result.get("correctness_max_abs_error"), 6),
                parity=_fmt(result.get("correctness_parity_passed")),
                accuracy_rmse=_fmt(result.get("correctness_accuracy_rmse"), 6),
                argmax_match=(
                    _fmt(result.get("correctness_argmax_match") * 100, 2) + "%"
                    if result.get("correctness_argmax_match") is not None
                    else "—"
                ),
            )
        )
    lines.extend(["", "## Comparisons", ""])
    comparisons = _comparison_lines(payload["results"])
    lines.extend(comparisons or ["No comparable eager/candidate pairs completed."])

    skipped = [result for result in payload["results"] if result.get("status") != "ok"]
    if skipped:
        lines.extend(["", "## Skipped or failed modes", ""])
        for result in skipped:
            lines.append(
                f"- `{result['mode']}` B={result['batch_size']} S={result['sequence_length']}: "
                f"{result.get('error')}"
            )

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Setup and cold-start costs are reported separately from steady-state latency.",
            "- Token throughput is input tokens processed by a full forward/prefill pass, not generated tokens per second.",
            "- ONNX Runtime timing includes NumPy host input/output for `session.run`.",
            "- `Max abs vs eager` and `Parity` measure implementation parity against eager execution in the benchmark dtype.",
            "- For FP16/BF16, numerical accuracy is also measured against the same rounded model weights executed in FP32.",
            "- If reduced-precision eager parity fails, a candidate is accepted only when its RMSE against the FP32 reference is no worse than eager's RMSE against that same reference.",
            "- Argmax agreement is reported as a diagnostic only; this synthetic model is for runtime experiments, not model-quality evaluation.",
            "- These results apply only to the recorded hardware, software versions, shapes, and dtype.",
            "",
        ]
    )
    return "\n".join(lines)


def write_suite_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(payload, output_dir / "results.json")
    write_csv(payload, output_dir / "results.csv")
    (output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
