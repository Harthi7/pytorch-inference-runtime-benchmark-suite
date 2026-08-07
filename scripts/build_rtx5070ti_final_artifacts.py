#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path("results/rtx-5070-ti")
OUT = ROOT / "final"
CHARTS = OUT / "charts"

COMPILER_WARM = {
    "FP16": [ROOT / f"fp16-final-warm-{i}" / "results.json" for i in (1, 2, 3)],
    "BF16": [ROOT / f"bf16-final-warm-{i}" / "results.json" for i in (1, 2, 3)],
    "FP32": [ROOT / f"fp32-final-warm-{i}" / "results.json" for i in (1, 2, 3)],
}

COMPILER_COLD = {
    "FP16": ROOT / "fp16-final-cold" / "results.json",
    "BF16": ROOT / "bf16-final-cold" / "results.json",
    "FP32": ROOT / "fp32-final-cold" / "results.json",
}

ONNX_RUNS = [ROOT / f"onnx-fp32-final-run{i}" / "results.json" for i in (1, 2, 3)]
TRITON_RUNS = [ROOT / f"triton-rmsnorm-final-run{i}" / "triton_rmsnorm.json" for i in (1, 2, 3)]

EXPECTED_SHAPES = [(b, s) for b in (1, 4, 8) for s in (64, 128, 256, 512)]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required final artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_map(path: Path, expected_modes: set[str]) -> dict[tuple[int, int, str], dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected top-level results list")

    mapping: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"{path}: non-ok row found: {row}")
        key = (int(row["batch_size"]), int(row["sequence_length"]), str(row["mode"]))
        if key in mapping:
            raise ValueError(f"{path}: duplicate result key {key}")
        mapping[key] = row

    expected = {(b, s, mode) for b, s in EXPECTED_SHAPES for mode in expected_modes}
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path}: shape/mode mismatch; missing={missing}, extra={extra}")
    return mapping


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def ratio(a: float, b: float) -> float:
    if b == 0:
        return math.inf
    return a / b


def aggregate_compiler() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warm_rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []

    for dtype_label, paths in COMPILER_WARM.items():
        runs = [result_map(path, {"eager", "compile"}) for path in paths]

        for b, s in EXPECTED_SHAPES:
            eager = [run[(b, s, "eager")] for run in runs]
            comp = [run[(b, s, "compile")] for run in runs]

            per_run_speedup = [
                ratio(float(e["latency"]["p50_ms"]), float(c["latency"]["p50_ms"]))
                for e, c in zip(eager, comp, strict=True)
            ]

            warm_rows.append(
                {
                    "dtype": dtype_label,
                    "batch_size": b,
                    "sequence_length": s,
                    "eager_p50_ms_median": median([float(x["latency"]["p50_ms"]) for x in eager]),
                    "compile_p50_ms_median": median([float(x["latency"]["p50_ms"]) for x in comp]),
                    "compile_speedup_median": median(per_run_speedup),
                    "compile_speedup_run1": per_run_speedup[0],
                    "compile_speedup_run2": per_run_speedup[1],
                    "compile_speedup_run3": per_run_speedup[2],
                    "eager_p95_ms_median": median([float(x["latency"]["p95_ms"]) for x in eager]),
                    "compile_p95_ms_median": median([float(x["latency"]["p95_ms"]) for x in comp]),
                    "eager_tokens_s_median": median(
                        [float(x["throughput_tokens_per_second"]) for x in eager]
                    ),
                    "compile_tokens_s_median": median(
                        [float(x["throughput_tokens_per_second"]) for x in comp]
                    ),
                    "eager_peak_mb_median": median([float(x["peak_memory_mb"]) for x in eager]),
                    "compile_peak_mb_median": median([float(x["peak_memory_mb"]) for x in comp]),
                    "compile_parity_all_runs": all(
                        bool(x.get("correctness_parity_passed", True)) for x in comp
                    ),
                    "compile_max_abs_error_max": max(
                        float(x.get("correctness_max_abs_error", 0.0)) for x in comp
                    ),
                    "compile_accuracy_rmse_max": max(
                        float(x.get("correctness_accuracy_rmse", 0.0)) for x in comp
                    ),
                    "compile_argmax_match_min": min(
                        float(x.get("correctness_argmax_match", 1.0)) for x in comp
                    ),
                }
            )

        cold = result_map(COMPILER_COLD[dtype_label], {"eager", "compile"})
        for b, s in EXPECTED_SHAPES:
            eager = cold[(b, s, "eager")]
            comp = cold[(b, s, "compile")]
            cold_rows.append(
                {
                    "dtype": dtype_label,
                    "batch_size": b,
                    "sequence_length": s,
                    "eager_cold_start_ms": float(eager["cold_start_ms"]),
                    "compile_setup_ms": float(comp["setup_ms"]),
                    "compile_cold_start_ms": float(comp["cold_start_ms"]),
                    "cold_steady_speedup": ratio(
                        float(eager["latency"]["p50_ms"]),
                        float(comp["latency"]["p50_ms"]),
                    ),
                }
            )

    return warm_rows, cold_rows


def aggregate_onnx() -> list[dict[str, Any]]:
    runs = [result_map(path, {"eager", "onnx"}) for path in ONNX_RUNS]
    rows: list[dict[str, Any]] = []

    for b, s in EXPECTED_SHAPES:
        eager = [run[(b, s, "eager")] for run in runs]
        onnx = [run[(b, s, "onnx")] for run in runs]
        per_run_ratio = [
            ratio(float(e["latency"]["p50_ms"]), float(o["latency"]["p50_ms"]))
            for e, o in zip(eager, onnx, strict=True)
        ]
        rows.append(
            {
                "batch_size": b,
                "sequence_length": s,
                "eager_p50_ms_median": median([float(x["latency"]["p50_ms"]) for x in eager]),
                "onnx_p50_ms_median": median([float(x["latency"]["p50_ms"]) for x in onnx]),
                "onnx_vs_eager_ratio_median": median(per_run_ratio),
                "onnx_ratio_run1": per_run_ratio[0],
                "onnx_ratio_run2": per_run_ratio[1],
                "onnx_ratio_run3": per_run_ratio[2],
                "eager_p95_ms_median": median([float(x["latency"]["p95_ms"]) for x in eager]),
                "onnx_p95_ms_median": median([float(x["latency"]["p95_ms"]) for x in onnx]),
                "eager_tokens_s_median": median(
                    [float(x["throughput_tokens_per_second"]) for x in eager]
                ),
                "onnx_tokens_s_median": median(
                    [float(x["throughput_tokens_per_second"]) for x in onnx]
                ),
                "eager_peak_mb_median": median([float(x["peak_memory_mb"]) for x in eager]),
                "onnx_peak_mb_median": median([float(x["peak_memory_mb"]) for x in onnx]),
                "onnx_setup_ms_median": median([float(x["setup_ms"]) for x in onnx]),
                "onnx_cold_start_ms_median": median([float(x["cold_start_ms"]) for x in onnx]),
                "onnx_parity_all_runs": all(
                    bool(x.get("correctness_parity_passed", False)) for x in onnx
                ),
                "onnx_max_abs_error_max": max(
                    float(x.get("correctness_max_abs_error", 0.0)) for x in onnx
                ),
                "onnx_argmax_match_min": min(
                    float(x.get("correctness_argmax_match", 1.0)) for x in onnx
                ),
            }
        )
    return rows


def aggregate_triton() -> list[dict[str, Any]]:
    runs = [load_json(path)["results"] for path in TRITON_RUNS]
    by_hidden: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        if len(run) != 3:
            raise ValueError("Each final Triton run must contain exactly 3 results")
        for row in run:
            by_hidden[int(row["hidden_size"])].append(row)

    if sorted(by_hidden) != [1024, 2048, 4096]:
        raise ValueError(f"Unexpected Triton hidden sizes: {sorted(by_hidden)}")

    rows: list[dict[str, Any]] = []
    for hidden in sorted(by_hidden):
        items = by_hidden[hidden]
        if len(items) != 3:
            raise ValueError(f"H={hidden}: expected 3 Triton repetitions")
        rows.append(
            {
                "hidden_size": hidden,
                "rows": int(items[0]["rows"]),
                "dtype": str(items[0]["dtype"]),
                "torch_p50_ms_median": median([float(x["torch_p50_ms"]) for x in items]),
                "triton_p50_ms_median": median([float(x["triton_p50_ms"]) for x in items]),
                "speedup_median": median([float(x["speedup"]) for x in items]),
                "speedup_run1": float(items[0]["speedup"]),
                "speedup_run2": float(items[1]["speedup"]),
                "speedup_run3": float(items[2]["speedup"]),
                "max_abs_error_max": max(float(x["max_abs_error"]) for x in items),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def shape_labels() -> list[str]:
    return [f"B{b}/S{s}" for b, s in EXPECTED_SHAPES]


def plot_compiler(warm_rows: list[dict[str, Any]]) -> None:
    labels = shape_labels()
    by_dtype = {
        dtype: [
            next(
                row["compile_speedup_median"]
                for row in warm_rows
                if row["dtype"] == dtype and row["batch_size"] == b and row["sequence_length"] == s
            )
            for b, s in EXPECTED_SHAPES
        ]
        for dtype in ("FP16", "BF16", "FP32")
    }

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for dtype, values in by_dtype.items():
        ax.plot(x, values, marker="o", label=dtype)
    ax.axhline(1.0, linewidth=1)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("TorchInductor speedup vs eager (×)")
    ax.set_title("RTX 5070 Ti: warm-cache TorchInductor speedup")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "torchinductor_speedup_by_dtype.png", dpi=180)
    plt.close(fig)


def plot_onnx(rows: list[dict[str, Any]]) -> None:
    labels = shape_labels()
    values = [float(row["onnx_vs_eager_ratio_median"]) for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values)
    ax.axhline(1.0, linewidth=1)
    ax.set_ylabel("ONNX Runtime / eager speed ratio (×)")
    ax.set_title("RTX 5070 Ti: ONNX Runtime CUDA vs eager FP32")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "onnx_fp32_vs_eager.png", dpi=180)
    plt.close(fig)


def plot_triton(rows: list[dict[str, Any]]) -> None:
    labels = [f"H={row['hidden_size']}" for row in rows]
    values = [float(row["speedup_median"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.axhline(1.0, linewidth=1)
    ax.set_ylabel("Triton speedup vs PyTorch (×)")
    ax.set_title("RTX 5070 Ti: FP16 RMSNorm")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHARTS / "triton_rmsnorm_speedup.png", dpi=180)
    plt.close(fig)


def report(
    warm_rows: list[dict[str, Any]],
    cold_rows: list[dict[str, Any]],
    onnx_rows: list[dict[str, Any]],
    triton_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# RTX 5070 Ti final benchmark summary",
        "",
        "Final results are aggregated only from the designated final benchmark runs.",
        "Warm compiler and ONNX speed ratios are the median of three independently measured paired ratios.",
        "",
        "## TorchInductor warm-cache results",
        "",
        "| Dtype | Min speedup | Max speedup | Shapes faster than eager |",
        "|---|---:|---:|---:|",
    ]

    for dtype in ("FP16", "BF16", "FP32"):
        values = [
            float(row["compile_speedup_median"]) for row in warm_rows if row["dtype"] == dtype
        ]
        faster = sum(value > 1.0 for value in values)
        lines.append(
            f"| {dtype} | {min(values):.2f}x | {max(values):.2f}x | {faster}/{len(values)} |"
        )

    lines += [
        "",
        "The compiler comparison uses static shapes and the final max-autotune methodology.",
        "",
        "## ONNX Runtime CUDA FP32",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]

    onnx_ratios = [float(row["onnx_vs_eager_ratio_median"]) for row in onnx_rows]
    lines += [
        f"| Min ONNX/eager ratio | {min(onnx_ratios):.2f}x |",
        f"| Max ONNX/eager ratio | {max(onnx_ratios):.2f}x |",
        f"| Shapes faster than eager | {sum(x > 1.0 for x in onnx_ratios)}/{len(onnx_ratios)} |",
        f"| Worst max abs error across final runs | {max(float(r['onnx_max_abs_error_max']) for r in onnx_rows):.6f} |",
        f"| Minimum argmax agreement | {100 * min(float(r['onnx_argmax_match_min']) for r in onnx_rows):.2f}% |",
        "",
        "ONNX Runtime CUDA uses TF32 disabled to match the PyTorch FP32 precision policy.",
        "The ONNX timing scope includes NumPy host input/output around `session.run`, so this is an end-to-end runtime comparison rather than a GPU-resident I/O-binding benchmark.",
        "",
        "## Triton RMSNorm FP16",
        "",
        "| Hidden size | Median speedup | Max abs error |",
        "|---:|---:|---:|",
    ]

    for row in triton_rows:
        lines.append(
            f"| {row['hidden_size']} | {float(row['speedup_median']):.2f}x | "
            f"{float(row['max_abs_error_max']):.6f} |"
        )

    lines += [
        "",
        "## Cold compiler evidence",
        "",
        "The cold-run CSV records TorchInductor setup time, first-call time, and steady-state speedup for each dtype and shape. Cold costs are intentionally kept separate from warm-cache steady-state latency.",
        "",
        "## Profiling evidence",
        "",
        "Representative FP16 B4/S256 PyTorch profiler traces are preserved for eager and max-autotune TorchInductor execution under the RTX 5070 Ti result tree.",
        "",
        "## Generated charts",
        "",
        "- `charts/torchinductor_speedup_by_dtype.png`",
        "- `charts/onnx_fp32_vs_eager.png`",
        "- `charts/triton_rmsnorm_speedup.png`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    warm_rows, cold_rows = aggregate_compiler()
    onnx_rows = aggregate_onnx()
    triton_rows = aggregate_triton()

    write_csv(OUT / "compiler_warm_summary.csv", warm_rows)
    write_csv(OUT / "compiler_cold_summary.csv", cold_rows)
    write_csv(OUT / "onnx_fp32_summary.csv", onnx_rows)
    write_csv(OUT / "triton_rmsnorm_summary.csv", triton_rows)

    payload = {
        "aggregation": {
            "compiler_warm": "median of three paired eager/compile ratios",
            "onnx": "median of three paired eager/onnx ratios",
            "triton": "median of three reported speedups",
            "cold_compiler": "single designated cold-cache run per dtype",
        },
        "compiler_warm": warm_rows,
        "compiler_cold": cold_rows,
        "onnx_fp32": onnx_rows,
        "triton_rmsnorm": triton_rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_compiler(warm_rows)
    plot_onnx(onnx_rows)
    plot_triton(triton_rows)

    (OUT / "report.md").write_text(
        report(warm_rows, cold_rows, onnx_rows, triton_rows),
        encoding="utf-8",
    )

    print(f"Wrote final RTX 5070 Ti artifacts to {OUT}")
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
