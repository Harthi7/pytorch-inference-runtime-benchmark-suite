# Architecture

## Objective

The project isolates inference-runtime behavior without relying on remote model downloads or a serving framework. The model is intentionally simple enough to inspect but contains operations that exercise modern compiler/runtime paths.

## Model

`TinyDecoderLM` is a decoder-only transformer with:

- learned token and position embeddings,
- pre-normalization transformer blocks,
- RMSNorm,
- fused QKV projection,
- PyTorch scaled-dot-product attention with a causal mask,
- SwiGLU feed-forward layers,
- tied token-embedding and language-model-head weights.

The output shape is `[batch, sequence, vocabulary]`.

## Runtime adapter boundary

Every runtime implements two operations:

```python
prepare(model, sample_input, artifact_dir)
run(input_ids) -> logits
```

This keeps benchmark orchestration independent from runtime-specific setup.

### Eager

Runs the PyTorch module directly.

### TorchInductor

Uses `torch.compile(..., backend="inductor")`. The first invocation is measured separately because graph capture, code generation, and compilation can dominate short-lived workloads.

### ONNX Runtime

Exports with the `torch.export`-based ONNX path (`dynamo=True`), enables ONNX Runtime graph optimizations, and records the providers. The current adapter uses NumPy host input/output, so CUDA results include transfer cost.

## Measurement pipeline

1. Build one seeded model state.
2. Clone identical weights into each runtime-specific model.
3. Create one token tensor for a shape.
4. Measure runtime setup.
5. Measure the first invocation.
6. Compare output against eager PyTorch.
7. Warm up.
8. Record synchronized steady-state samples.
9. Compute mean, p50, p95, standard deviation, throughput, and memory.
10. Write JSON, CSV, and Markdown.

## Why the Triton kernel is separate

The Triton RMSNorm experiment is isolated from the ONNX comparison because a custom Triton call would not export as a standard ONNX operator. Keeping it separate preserves a valid apples-to-apples model comparison while still demonstrating kernel work.

## Extension points

Useful future work:

- add KV-cache decode and prefill/decode separation,
- add ONNX Runtime I/O binding,
- add CUDA Graph measurements,
- add FP8 or quantized execution where supported,
- use `torch.library` to register the Triton operation,
- add ExecuTorch or Qualcomm AI Engine Direct backends,
- test Hugging Face models after preserving the self-contained baseline.
