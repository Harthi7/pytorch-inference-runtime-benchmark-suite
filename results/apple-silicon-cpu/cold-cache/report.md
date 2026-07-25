# Benchmark report: cpu-runtime-comparison

## Environment

- Platform: `macOS-26.5.1-arm64-arm-64bit`
- Python: `3.12.13`
- PyTorch: `2.13.0`
- Device: `arm`
- ONNX Runtime: `1.27.0`
- Triton: `None`

## Results

| Status | Mode | Batch | Sequence | Setup ms | Cold ms | p50 ms | p95 ms | Tokens/s | Peak MB | Max abs error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ok | eager | 1 | 64 | 0.00 | 3.02 | 2.04 | 2.25 | 30962 | 363.11 | 0.000000 |
| ok | compile | 1 | 64 | 672.75 | 7706.07 | 3.35 | 3.97 | 19184 | 507.97 | 0.000001 |
| ok | onnx | 1 | 64 | 1179.03 | 2.89 | 1.85 | 1.91 | 34527 | 633.44 | 0.000001 |
| ok | eager | 1 | 128 | 0.00 | 4.26 | 3.21 | 3.76 | 39115 | 676.88 | 0.000000 |
| ok | compile | 1 | 128 | 0.62 | 4133.23 | 3.20 | 3.27 | 40056 | 686.52 | 0.000001 |
| ok | onnx | 1 | 128 | 859.85 | 3.73 | 3.35 | 3.44 | 38087 | 693.39 | 0.000001 |
| ok | eager | 1 | 256 | 0.00 | 6.54 | 5.32 | 5.43 | 47996 | 706.75 | 0.000000 |
| ok | compile | 1 | 256 | 0.72 | 4070.87 | 4.41 | 4.57 | 58159 | 712.94 | 0.000001 |
| ok | onnx | 1 | 256 | 783.93 | 6.87 | 6.62 | 6.72 | 38679 | 761.09 | 0.000001 |
| ok | eager | 4 | 64 | 0.00 | 5.99 | 5.01 | 5.08 | 51189 | 769.38 | 0.000000 |
| ok | compile | 4 | 64 | 0.72 | 2046.13 | 3.99 | 4.06 | 64099 | 774.41 | 0.000001 |
| ok | onnx | 4 | 64 | 779.17 | 6.69 | 5.95 | 6.15 | 42661 | 774.41 | 0.000001 |
| ok | eager | 4 | 128 | 0.00 | 8.00 | 6.72 | 6.86 | 75754 | 847.94 | 0.000000 |
| ok | compile | 4 | 128 | 0.76 | 4153.10 | 5.38 | 5.60 | 94259 | 854.53 | 0.000001 |
| ok | onnx | 4 | 128 | 883.95 | 13.31 | 11.05 | 11.12 | 46533 | 883.47 | 0.000002 |
| ok | eager | 4 | 256 | 0.00 | 12.28 | 10.71 | 10.90 | 95263 | 1050.62 | 0.000000 |
| ok | compile | 4 | 256 | 0.62 | 4144.78 | 9.53 | 9.64 | 107345 | 1050.62 | 0.000002 |
| ok | onnx | 4 | 256 | 891.94 | 26.68 | 22.87 | 23.08 | 44773 | 1061.70 | 0.000002 |

## Comparisons

- B=1, S=64: **compile** was 0.61x eager p50; no steady-state break-even because it was not faster.
- B=1, S=64: **onnx** was 1.11x eager p50; estimated break-even after 6038 calls.
- B=1, S=128: **compile** was 1.00x eager p50; estimated break-even after 547582 calls.
- B=1, S=128: **onnx** was 0.96x eager p50; no steady-state break-even because it was not faster.
- B=1, S=256: **compile** was 1.21x eager p50; estimated break-even after 4460 calls.
- B=1, S=256: **onnx** was 0.80x eager p50; no steady-state break-even because it was not faster.
- B=4, S=64: **compile** was 1.26x eager p50; estimated break-even after 1992 calls.
- B=4, S=64: **onnx** was 0.84x eager p50; no steady-state break-even because it was not faster.
- B=4, S=128: **compile** was 1.25x eager p50; estimated break-even after 3098 calls.
- B=4, S=128: **onnx** was 0.61x eager p50; no steady-state break-even because it was not faster.
- B=4, S=256: **compile** was 1.12x eager p50; estimated break-even after 3486 calls.
- B=4, S=256: **onnx** was 0.47x eager p50; no steady-state break-even because it was not faster.

## Interpretation constraints

- Setup and cold-start costs are reported separately from steady-state latency.
- Token throughput is input tokens processed by a full forward/prefill pass, not generated tokens per second.
- ONNX Runtime timing includes NumPy host input/output for `session.run`.
- These results apply only to the recorded hardware, software versions, shapes, and dtype.
