## Validation environments

### Linux CPU container

The Linux CPU container was used to validate installation, command-line behavior, configuration loading, core benchmark execution, static analysis, formatting, and automated tests across the supported Python versions.

ONNX Runtime benchmarking and Apple-specific performance measurements were not performed in this environment.

### Apple M3 Max CPU

The Apple Silicon environment was used for the complete measured benchmark suite, including:

* Eager PyTorch inference
* `torch.compile` and TorchInductor inference
* ONNX Runtime inference
* Cold-cache and warm-cache compilation experiments
* TorchDynamo graph-break diagnosis
* Dynamic-shape graph-reuse analysis
* PyTorch Profiler analysis
* Correctness comparison across runtime implementations

The corresponding reports, CSV files, JSON files, charts, profiler summary, and environment metadata are committed under `results/apple-silicon-cpu`.
