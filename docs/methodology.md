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

## Reduced-precision correctness policy

FP16 and BF16 execution can differ numerically between eager PyTorch and
TorchInductor even when both implementations are valid. The benchmark therefore
separates implementation parity from reduced-precision accuracy.

For FP16/BF16 comparisons:

1. The eager and candidate runtimes use the same reduced-precision model
   weights and input.
2. A second copy of those same rounded weights is executed in FP32 and used as
   the higher-precision accuracy reference.
3. Eager-versus-candidate `assert_close` remains the implementation-parity
   check.
4. If candidate parity fails, the candidate is accepted only when its RMSE
   against the FP32 reference is no worse than eager reduced precision against
   that same reference.
5. Argmax agreement is recorded as a diagnostic only.

This policy does not claim that the candidate is universally more accurate. It
only prevents eager reduced precision from being treated as an exact numerical
oracle when comparing alternative implementations of the same rounded model.

## Cross-runtime FP32 precision controls

A runtime labeled "FP32" may still use a lower-precision internal matrix
multiplication policy. Precision controls must therefore be aligned before
performance comparisons are treated as numerically equivalent.

For the RTX 5070 Ti ONNX Runtime CUDA study, the CUDA Execution Provider's TF32
path was disabled with `use_tf32=0` so that ONNX Runtime matched the PyTorch
FP32 baseline policy. Tolerances were not loosened to make the comparison pass.

## Repeated-run aggregation

The curated RTX 5070 Ti warm-cache compiler and ONNX results use three
independent repetitions. For each shape, the final speed ratio is the median of
the three paired per-run ratios rather than the ratio of separately aggregated
latencies.

The Triton RMSNorm summary likewise reports the median of three final
repetitions. Cold compiler results remain separate because they characterize a
fresh-cache deployment state rather than warm steady-state execution.
