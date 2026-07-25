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
| ok | eager | 1 | 64 | 0.00 | 3.03 | 1.99 | 2.11 | 31999 | 362.38 | 0.000000 |
| ok | compile | 1 | 64 | 655.07 | 1004.64 | 2.26 | 2.30 | 28333 | 497.73 | 0.000001 |
| ok | onnx | 1 | 64 | 1066.80 | 2.45 | 1.80 | 1.83 | 35536 | 617.62 | 0.000001 |
| ok | eager | 1 | 128 | 0.00 | 3.88 | 3.13 | 3.18 | 41065 | 651.09 | 0.000000 |
| ok | compile | 1 | 128 | 0.65 | 142.78 | 3.05 | 3.15 | 42127 | 655.55 | 0.000001 |
| ok | onnx | 1 | 128 | 845.75 | 3.63 | 3.32 | 3.38 | 38481 | 655.55 | 0.000001 |
| ok | eager | 1 | 256 | 0.00 | 5.92 | 5.35 | 5.42 | 47867 | 691.06 | 0.000000 |
| ok | compile | 1 | 256 | 0.60 | 147.82 | 4.28 | 4.36 | 59942 | 724.97 | 0.000001 |
| ok | onnx | 1 | 256 | 789.30 | 6.99 | 6.60 | 6.66 | 38799 | 757.66 | 0.000001 |
| ok | eager | 4 | 64 | 0.00 | 5.42 | 5.00 | 5.13 | 51209 | 766.28 | 0.000000 |
| ok | compile | 4 | 64 | 0.62 | 134.19 | 4.01 | 4.06 | 63979 | 766.50 | 0.000001 |
| ok | onnx | 4 | 64 | 853.20 | 6.55 | 6.13 | 6.18 | 42020 | 766.50 | 0.000001 |
| ok | eager | 4 | 128 | 0.00 | 8.67 | 6.42 | 6.54 | 79482 | 861.08 | 0.000000 |
| ok | compile | 4 | 128 | 0.56 | 149.96 | 5.55 | 5.73 | 91531 | 886.16 | 0.000001 |
| ok | onnx | 4 | 128 | 868.89 | 11.79 | 10.86 | 11.11 | 46890 | 886.16 | 0.000002 |
| ok | eager | 4 | 256 | 0.00 | 12.19 | 10.79 | 11.02 | 94683 | 997.33 | 0.000000 |
| ok | compile | 4 | 256 | 0.55 | 155.66 | 9.59 | 9.83 | 106046 | 1002.73 | 0.000002 |
| ok | onnx | 4 | 256 | 789.92 | 23.05 | 22.84 | 23.00 | 44803 | 1083.75 | 0.000002 |

## Comparisons

- B=1, S=64: **compile** was 0.88x eager p50; no steady-state break-even because it was not faster.
- B=1, S=64: **onnx** was 1.10x eager p50; estimated break-even after 5778 calls.
- B=1, S=128: **compile** was 1.03x eager p50; estimated break-even after 1752 calls.
- B=1, S=128: **onnx** was 0.94x eager p50; no steady-state break-even because it was not faster.
- B=1, S=256: **compile** was 1.25x eager p50; estimated break-even after 133 calls.
- B=1, S=256: **onnx** was 0.81x eager p50; no steady-state break-even because it was not faster.
- B=4, S=64: **compile** was 1.25x eager p50; estimated break-even after 129 calls.
- B=4, S=64: **onnx** was 0.82x eager p50; no steady-state break-even because it was not faster.
- B=4, S=128: **compile** was 1.16x eager p50; estimated break-even after 162 calls.
- B=4, S=128: **onnx** was 0.59x eager p50; no steady-state break-even because it was not faster.
- B=4, S=256: **compile** was 1.12x eager p50; estimated break-even after 120 calls.
- B=4, S=256: **onnx** was 0.47x eager p50; no steady-state break-even because it was not faster.

## Interpretation constraints

- Setup and cold-start costs are reported separately from steady-state latency.
- Token throughput is input tokens processed by a full forward/prefill pass, not generated tokens per second.
- ONNX Runtime timing includes NumPy host input/output for `session.run`.
- These results apply only to the recorded hardware, software versions, shapes, and dtype.
