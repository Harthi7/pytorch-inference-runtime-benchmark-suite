from __future__ import annotations

import json
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch

_TRITON_IMPORT_ERROR: Exception | None = None
try:
    import triton
    import triton.language as tl
except Exception as exc:  # Triton is intentionally optional
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        weight_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = x * inv_rms * weight
        tl.store(output_ptr + row * n_cols + offsets, output, mask=mask)


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("the Triton RMSNorm experiment requires CUDA")


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    _require_triton()
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight length must equal the final input dimension")
    n_cols = x.shape[-1]
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError("hidden dimension is too large for this tutorial kernel")
    output = torch.empty_like(x)
    rows = x.numel() // n_cols
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        output,
        n_cols=n_cols,
        eps=eps,
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


def _measure(function: Any, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def benchmark_triton_rmsnorm(
    *,
    hidden_sizes: list[int],
    rows: int,
    dtype_name: str,
    output_dir: Path,
    warmup: int = 20,
    iterations: int = 100,
    eps: float = 1e-5,
) -> dict[str, Any]:
    _require_triton()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for hidden in hidden_sizes:
        x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)
        expected = torch_rmsnorm(x, weight, eps)
        actual = triton_rmsnorm(x, weight, eps)
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
        max_error = float((actual.float() - expected.float()).abs().max().item())

        torch_samples = _measure(
            partial(torch_rmsnorm, x, weight, eps),
            warmup,
            iterations,
        )
        triton_samples = _measure(
            partial(triton_rmsnorm, x, weight, eps),
            warmup,
            iterations,
        )
        torch_p50 = statistics.median(torch_samples)
        triton_p50 = statistics.median(triton_samples)
        results.append(
            {
                "hidden_size": hidden,
                "rows": rows,
                "dtype": dtype_name,
                "torch_p50_ms": torch_p50,
                "triton_p50_ms": triton_p50,
                "speedup": torch_p50 / triton_p50,
                "max_abs_error": max_error,
            }
        )

    payload = {"results": results}
    (output_dir / "triton_rmsnorm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Triton RMSNorm benchmark",
        "",
        "| Hidden | Rows | Dtype | PyTorch p50 ms | Triton p50 ms | Speedup | Max abs error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['hidden_size']} | {result['rows']} | {result['dtype']} | "
            f"{result['torch_p50_ms']:.4f} | {result['triton_p50_ms']:.4f} | "
            f"{result['speedup']:.2f}x | {result['max_abs_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The PyTorch baseline is an explicit RMSNorm expression. Correctness is checked before timing. Results include kernel launch and synchronization overhead for each call.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
