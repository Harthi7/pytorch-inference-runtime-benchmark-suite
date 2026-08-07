# RTX 5070 Ti final benchmark summary

Final results are aggregated only from the designated final benchmark runs.
Warm compiler and ONNX speed ratios are the median of three independently measured paired ratios.

## TorchInductor warm-cache results

| Dtype | Min speedup | Max speedup | Shapes faster than eager |
|---|---:|---:|---:|
| FP16 | 1.42x | 4.36x | 12/12 |
| BF16 | 1.49x | 4.25x | 12/12 |
| FP32 | 1.05x | 2.17x | 12/12 |

The compiler comparison uses static shapes and the final max-autotune methodology.

## ONNX Runtime CUDA FP32

| Metric | Result |
|---|---:|
| Min ONNX/eager ratio | 0.42x |
| Max ONNX/eager ratio | 0.89x |
| Shapes faster than eager | 0/12 |
| Worst max abs error across final runs | 0.000005 |
| Minimum argmax agreement | 100.00% |

ONNX Runtime CUDA uses TF32 disabled to match the PyTorch FP32 precision policy.
The ONNX timing scope includes NumPy host input/output around `session.run`, so this is an end-to-end runtime comparison rather than a GPU-resident I/O-binding benchmark.

## Triton RMSNorm FP16

| Hidden size | Median speedup | Max abs error |
|---:|---:|---:|
| 1024 | 2.41x | 0.007812 |
| 2048 | 2.52x | 0.007812 |
| 4096 | 2.94x | 0.007812 |

## Cold compiler evidence

The cold-run CSV records TorchInductor setup time, first-call time, and steady-state speedup for each dtype and shape. Cold costs are intentionally kept separate from warm-cache steady-state latency.

## Profiling evidence

Representative FP16 B4/S256 PyTorch profiler traces are preserved for eager and max-autotune TorchInductor execution under the RTX 5070 Ti result tree.

## Generated charts

- `charts/torchinductor_speedup_by_dtype.png`
- `charts/onnx_fp32_vs_eager.png`
- `charts/triton_rmsnorm_speedup.png`
