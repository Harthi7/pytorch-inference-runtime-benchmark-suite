## Reproduced TorchInductor results

The following results are derived from the benchmark artifacts committed under `results/apple-silicon-cpu`. The cold-cache run used a newly created TorchInductor cache directory, while the warm-cache run reused the populated directory from the preceding run.

| Batch | Sequence | Cold-cache latency reduction | Warm-cache latency reduction |
| ----: | -------: | ---------------------------: | ---------------------------: |
|     1 |      256 |                        17.1% |                        20.1% |
|     4 |       64 |                        20.4% |                        20.0% |
|     4 |      128 |                        19.9% |                        13.6% |
|     4 |      256 |                        11.1% |                        11.1% |

TorchInductor therefore reduced steady-state median latency by approximately 11–20% for the larger tested workloads. The result remained directionally consistent across cold-cache and warm-cache execution, although the exact improvement varied by input shape.

## Compilation break-even

Compiler-cache state had a substantial effect on the number of calls required to recover compilation overhead:

* Cold cache: approximately 1,992–4,460 calls.
* Warm persistent cache: approximately 120–162 calls.

The cold-cache figures represent a newly deployed environment without existing TorchInductor artifacts. The warm-cache figures represent a later process reusing compiler artifacts already stored on disk.

These measurements should not be combined into one range because they represent materially different deployment conditions.
