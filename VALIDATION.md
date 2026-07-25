# Validation Status

Validated on July 24, 2026 in a Linux CPU container with Python 3.13.5 and PyTorch 2.10.0+cpu.

## Executed successfully

- 10 unit and smoke tests.
- Eager transformer inference benchmark.
- `torch.compile` / TorchInductor CPU smoke benchmark.
- TorchDynamo graph-break diagnosis and tensor-only correction.
- Dynamic-shape experiment using one compiled model across repeated sequence lengths.
- PyTorch Profiler trace and operator summary generation.
- JSON, CSV, Markdown, and chart generation.
- Editable package installation with local build isolation disabled in the restricted validation environment.

## Not executed in this environment

- ONNX export and ONNX Runtime execution, because ONNX dependencies were unavailable.
- Triton RMSNorm execution, because no CUDA device or Triton installation was available.

Those paths are capability-gated and produce explicit errors or `skipped` benchmark records rather than fabricated results. Run them on the intended CPU/GPU environment before using their metrics publicly.
