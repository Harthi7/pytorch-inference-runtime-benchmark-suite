# RTX 5070 Ti GPU Findings

## Environment and scope

The final CUDA evidence was collected on an NVIDIA GeForce RTX 5070 Ti under
WSL2 with PyTorch `2.13.0+cu130`, ONNX Runtime `1.27.0`, and Triton `3.7.1`.

The primary model is the repository's decoder-only `TinyDecoderLM` with six
transformer blocks, hidden size 512, eight attention heads, vocabulary size
16,384, and maximum sequence length 512.

The main matrix covers batch sizes 1, 4, and 8 and sequence lengths 64, 128,
256, and 512. Results are specific to this model, hardware, software stack,
shapes, and measurement methodology.

## TorchInductor

The final compiler study uses static shapes and `max-autotune`. Warm-cache
results are aggregated from three independent repetitions by taking the median
paired eager/compiled speed ratio for each shape.

| Dtype | Minimum speedup | Maximum speedup | Shapes faster than eager |
|---|---:|---:|---:|
| FP16 | 1.42x | 4.36x | 12/12 |
| BF16 | 1.49x | 4.25x | 12/12 |
| FP32 | 1.05x | 2.17x | 12/12 |

The largest gains occur on smaller workloads where eager launch/framework
overhead is a larger fraction of total latency. FP32 improvements are smaller
and less consistent across individual repetitions than FP16/BF16, even though
the median paired result remains above 1.0 for all 12 shapes.

Cold compiler costs are recorded separately in
`results/rtx-5070-ti/final/compiler_cold_summary.csv`. They should not be
combined with the warm steady-state speedups.

## BF16 correctness investigation

Initial BF16 comparisons treated eager BF16 as exact ground truth. That caused
TorchInductor results to fail strict parity with max absolute differences on
the order of `0.025-0.032`.

A layer-isolation experiment showed:

- eager BF16 and Dynamo-only execution matched bit-for-bit;
- AOT eager also matched eager BF16;
- numerical divergence first appeared with TorchInductor.

A higher-precision check then compared eager BF16 and TorchInductor against the
same BF16-rounded weights executed in FP32. TorchInductor had lower aggregate
RMSE than eager BF16 in the isolated reproducer, although argmax agreement was
not always 100%.

The benchmark therefore reports both implementation parity and accuracy against
the FP32 reference. If reduced-precision parity fails, the candidate is
accepted only when its RMSE against the FP32 reference is no worse than eager's
RMSE against that same reference.

This is a benchmark correctness policy, not a claim that TorchInductor is
universally more accurate than eager execution.

## ONNX Runtime CUDA

The first ONNX Runtime CUDA FP32 runs showed approximately `4.2e-4` RMSE while
PyTorch eager CPU/CUDA stayed near `3e-7` relative to an independent float64
diagnostic reference.

A 2x2 provider/optimization ablation isolated the discrepancy:

- ONNX Runtime CPU with optimizations disabled: accurate;
- ONNX Runtime CPU with optimizations enabled: accurate;
- ONNX Runtime CUDA with optimizations disabled: inaccurate;
- ONNX Runtime CUDA with optimizations enabled: inaccurate.

The exported ONNX graph and ORT graph optimizer were therefore not the root
cause. The difference came from the CUDA execution path.

The CUDA Execution Provider was using TF32 while the PyTorch FP32 baseline was
not. Setting `use_tf32=0` aligned the precision policy. The strict smoke then
passed with max absolute error around `2e-6`.

Across the final three 12-shape runs:

- strict parity passed for every ONNX result;
- worst max absolute error was approximately `5e-6`;
- argmax agreement was 100%;
- ONNX/eager median speed ratio ranged from 0.42x to 0.89x;
- ONNX Runtime was slower than eager on all 12 tested shapes.

The timing scope matters: the adapter calls `session.run` with NumPy host input
and output, so CUDA measurements include host/device transfer. These results
must not be presented as GPU-resident I/O-binding or kernel-only measurements.

## Triton RMSNorm

The standalone FP16 RMSNorm kernel was measured for 256 rows over three hidden
sizes and repeated three times.

| Hidden size | Median speedup | Max absolute error |
|---:|---:|---:|
| 1024 | 2.41x | 0.0078125 |
| 2048 | 2.52x | 0.0078125 |
| 4096 | 2.94x | 0.0078125 |

The baseline is the explicit PyTorch RMSNorm expression. Correctness is checked
before timing, and the measurements include kernel launch and synchronization
overhead.

## GPU profiler evidence

A representative FP16 B=4, S=256 workload was profiled in eager and
TorchInductor `max-autotune` modes after warm-up.

Eager execution spends substantial CUDA time in separate matrix multiplication
and elementwise kernels, including CUTLASS GEMMs, `aten::mm`, FlashAttention,
copies, RMSNorm reductions, and SwiGLU elementwise work.

The compiled trace instead contains large fused Triton regions around those
operations while preserving the FlashAttention kernel. This supports the
observed steady-state speedups by showing reduced operator/kernel fragmentation
rather than attributing the improvement to one isolated operation.

Profiler operator tables are stored under:

- `results/rtx-5070-ti/profile-fp16-b4-s256-eager/top_ops.txt`
- `results/rtx-5070-ti/profile-fp16-b4-s256-compile/top_ops.txt`

## Curated artifacts

The final aggregation is under `results/rtx-5070-ti/final/`:

- `report.md`
- `summary.json`
- `compiler_warm_summary.csv`
- `compiler_cold_summary.csv`
- `onnx_fp32_summary.csv`
- `triton_rmsnorm_summary.csv`
- `charts/torchinductor_speedup_by_dtype.png`
- `charts/onnx_fp32_vs_eager.png`
- `charts/triton_rmsnorm_speedup.png`

The aggregation intentionally excludes smoke tests, isolation runs, and
superseded experimental outputs.
