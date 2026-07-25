# Benchmark Methodology

## Primary metrics

- **Setup time:** runtime construction, compilation wrapper creation, or ONNX export/session creation.
- **Cold-start time:** first inference after setup.
- **Steady-state latency:** synchronized samples after warm-up.
- **p50 and p95:** median and tail latency.
- **Throughput:** input tokens processed per second for a full forward/prefill pass.
- **Peak memory:** CUDA allocator peak or process peak RSS where available.
- **Correctness:** maximum absolute error plus `torch.testing.assert_close`.

## Controls

Keep these fixed when comparing runtimes:

- model architecture and weights,
- random input tensor,
- batch size and sequence length,
- dtype,
- device,
- warm-up count,
- measurement count,
- correctness tolerance.

## Synchronization

Accelerator operations are asynchronous. Timing without synchronization measures queue submission rather than completed inference. The benchmark synchronizes before and after every timed call.

## Compilation amortization

A compiled runtime can have lower steady-state latency and still be worse for a short-lived process. The generated report estimates a break-even request count:

```text
additional setup/cold cost / per-request steady-state savings
```

This is an estimate, not a production capacity model.

## Dynamic shapes

The dynamic-shape command intentionally reuses one compiled model and repeats selected shapes. It records:

- first call for each occurrence,
- immediate repeat call,
- available TorchDynamo diagnostic counters.

This helps identify specialization and recompilation behavior. TorchDynamo counters are internal diagnostics and may change between versions.

## ONNX Runtime scope

`InferenceSession.run` receives NumPy input and returns NumPy output. On CPU, this is a reasonable end-to-end API measurement. On CUDA, it includes host/device copies. Use I/O binding before claiming a compute-only runtime comparison.

## Statistical limitations

The suite is a microbenchmark. It does not model:

- concurrent requests,
- scheduler behavior,
- thermal throttling over long periods,
- production token sampling,
- paged KV-cache memory management,
- network or service overhead.

Report exact commands and system metadata with every result.
