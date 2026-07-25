# PyTorch Inference Runtime and Compiler Benchmark Suite

A reproducible ML-systems project for studying how PyTorch eager execution, `torch.compile`, TorchDynamo, TorchInductor, ONNX Runtime, and Triton behave on decoder-only transformer inference workloads.

This repository is deliberately not a chatbot, RAG demo, or model-serving wrapper. It focuses on compiler behavior, runtime overhead, operator-level profiling, dynamic shapes, graph breaks, kernel correctness, and benchmark methodology.

## What this project demonstrates

- A self-contained decoder-only transformer with RMSNorm, causal scaled-dot-product attention, and SwiGLU.
- Comparable eager PyTorch, TorchInductor, and ONNX Runtime execution paths.
- Measurement of setup cost, cold-start latency, steady-state latency, p50/p95, throughput, and memory.
- TorchDynamo graph-break diagnosis with a deliberately broken example and a corrected implementation.
- Dynamic-shape experiments that distinguish first-seen shape cost from steady-state execution.
- PyTorch Profiler traces and operator tables.
- An optional fused Triton RMSNorm kernel with correctness and latency comparison against PyTorch.
- Reproducible configuration files, JSON/CSV/Markdown reports, tests, linting, and CI.

## Architecture

```text
input token IDs
      |
      v
TinyDecoderLM
  Embedding -> N x TransformerBlock -> RMSNorm -> LM head
                     |
                     +-- causal SDPA attention
                     +-- SwiGLU MLP

Execution paths:
  1. PyTorch eager
  2. torch.compile(..., backend="inductor")
  3. torch.onnx.export(..., dynamo=True) -> ONNX Runtime

Separate kernel experiment:
  PyTorch RMSNorm <-> fused Triton RMSNorm
```

See [docs/architecture.md](docs/architecture.md) for design details and [docs/methodology.md](docs/methodology.md) for benchmark rules.

## Quick start

### 1. Create an environment

Linux is recommended. Triton requires a supported accelerator environment; the core eager/compiler suite also runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,onnx,viz]"
```

For a Linux CUDA environment with ONNX Runtime GPU and Triton support:

```bash
pip install -e ".[dev,gpu]"
```

Install the PyTorch build appropriate for your CPU or accelerator before installing this project when necessary.

### 2. Run tests

```bash
make test
```

### 3. Run a smoke benchmark

```bash
ml-runtime-bench suite \
  --config configs/smoke.json \
  --output results/smoke
```

### 4. Run the CPU comparison suite

```bash
ml-runtime-bench suite \
  --config configs/cpu.json \
  --output results/cpu
```

Optional runtimes are recorded as `skipped` when their dependency is unavailable. Add `--strict` to fail immediately instead.

### 5. Run the GPU suite

```bash
ml-runtime-bench suite \
  --config configs/gpu.json \
  --output results/gpu
```

### 6. Generate profiler artifacts

```bash
ml-runtime-bench profile \
  --mode compile \
  --device auto \
  --dtype float32 \
  --batch-size 4 \
  --sequence-length 256 \
  --output results/profile
```

Open the generated Chrome trace in `chrome://tracing` or Perfetto.

### 7. Diagnose graph breaks

```bash
ml-runtime-bench diagnose \
  --device auto \
  --output results/graph-breaks
```

The report compares a data-dependent `.item()` implementation with a tensor-only correction.

### 8. Study dynamic shapes

```bash
ml-runtime-bench dynamic-shapes \
  --device auto \
  --lengths 32 64 128 64 256 128 \
  --output results/dynamic-shapes
```

This runs one compiled model across multiple sequence lengths and records first-call and repeat-call latency for each shape.

### 9. Benchmark the Triton kernel

```bash
ml-runtime-bench triton-rmsnorm \
  --hidden-sizes 256 512 1024 2048 \
  --rows 2048 \
  --dtype float16 \
  --output results/triton-rmsnorm
```

This command requires CUDA and Triton. It verifies numerical correctness before timing.

### 10. Create charts

```bash
ml-runtime-bench plot \
  --results results/gpu/results.json \
  --output results/gpu/charts
```

## Output format

Each benchmark directory contains:

```text
results.json       full machine-readable result and environment metadata
results.csv        flat result table
report.md          generated comparison report and break-even analysis
```

Profiler, graph-break, dynamic-shape, and Triton commands add their own traces or reports.

## Example engineering questions

The repository is structured to answer questions such as:

1. When does TorchInductor outperform eager execution?
2. How large is the compile/cold-start cost, and how many requests are needed to amortize it?
3. Which sequence lengths or batch sizes trigger recompilation or new specialization?
4. Which operators dominate transformer prefill latency?
5. Does ONNX Runtime outperform PyTorch for the same CPU workload?
6. How much does batching improve token throughput while increasing latency?
7. Does a fused Triton RMSNorm kernel beat the equivalent PyTorch expression for a given hidden size?
8. What code patterns cause TorchDynamo graph breaks, and how can they be rewritten?

## Fair-benchmark rules

- Compare the same model weights, input shape, dtype, device, and output semantics.
- Separate one-time setup and cold-start cost from steady-state latency.
- Synchronize accelerator work before and after timing.
- Verify output correctness before comparing performance.
- Report multiple latency statistics rather than one best run.
- Do not compare ONNX Runtime CPU results against PyTorch GPU results.
- Treat ONNX Runtime `session.run` as end-to-end host-I/O timing unless you add explicit device I/O binding.
- Record hardware and software versions with every run.

## Repository map

```text
src/ml_runtime_bench/
  benchmark.py           benchmark orchestration and metrics
  cli.py                 command-line interface
  config.py              validated experiment configuration
  diagnostics.py         TorchDynamo graph-break experiments
  dynamic_shapes.py      shape-specialization experiment
  models/                decoder-only transformer
  profiling.py           torch.profiler integration
  reporting.py           JSON/CSV/Markdown reports
  runtimes/              eager, compile, and ONNX Runtime adapters
  triton_kernels/        optional fused RMSNorm kernel

tests/                    correctness and smoke tests
configs/                  reproducible experiment definitions
docs/                     design, methodology, findings template, résumé entry
.github/workflows/        CPU CI
```

## Official technical references

- PyTorch compiler overview: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html
- `torch.compile`: https://docs.pytorch.org/docs/main/generated/torch.compile.html
- PyTorch graph breaks: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html
- PyTorch ONNX exporter: https://docs.pytorch.org/docs/main/onnx.html
- PyTorch Profiler: https://docs.pytorch.org/docs/main/profiler.html
- ONNX Runtime performance: https://onnxruntime.ai/docs/performance/tune-performance/
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

## Limitations

- The transformer is intentionally small and randomly initialized. The benchmark isolates runtime behavior; it does not evaluate model quality.
- The main workload measures full-sequence forward/prefill inference, not production-grade autoregressive decoding with a paged KV cache.
- ONNX Runtime CUDA timing through NumPy includes host/device transfer overhead. Add I/O binding before presenting it as a kernel-only comparison.
- Results are hardware-, version-, shape-, and dtype-specific. Do not generalize one device result to another.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Measured Apple Silicon CPU results

The benchmark was validated on Apple Silicon using FP32 inference with PyTorch
2.13.0 and ONNX Runtime 1.27.0.

For the larger tested workloads, TorchInductor reduced steady-state median
latency by approximately 11–21% compared with eager PyTorch.

Compiler-cache state materially affected deployment economics:

- Cold TorchInductor cache: approximately 1,992–4,460 calls to break even.
- Warm persistent cache: approximately 120–162 calls to break even.

ONNX Runtime was fastest only for the smallest tested input and was slower than
eager PyTorch for most larger workloads.

See:

- [Detailed CPU analysis](docs/cpu-findings.md)
- [Reproducible benchmark artifacts](results/apple-silicon-cpu)

These CPU measurements do not represent CUDA or Triton performance. GPU
benchmarking remains future work.
