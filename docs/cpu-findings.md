# CPU Runtime Benchmark Findings

## Test environment

The benchmark was executed on Apple Silicon using macOS, Python 3.12.13, PyTorch 2.13.0, and ONNX Runtime 1.27.0. All measurements used FP32 CPU inference on a four-layer decoder-only transformer.

The results are specific to this hardware, software stack, model architecture, and benchmark configuration. They should not be generalized to CUDA inference or production-scale language models.

## Summary

The experiment showed that runtime performance depends strongly on workload shape. `torch.compile` did not improve the smallest batch-one workloads, but reduced steady-state latency by approximately 10–22% for several larger batch or sequence configurations.

ONNX Runtime achieved the lowest latency for the smallest tested input, but performed worse than eager PyTorch on most larger inputs. Its setup and export costs also required a large number of executions before the initial cost could be recovered.

## TorchInductor results

At batch 1 and sequence length 64, compiled inference was slower than eager execution: 3.04 ms compared with 2.02 ms. It was also slightly slower at batch 1 and sequence length 128.

Compilation became beneficial as the workload increased:

* Batch 1, sequence 256: 4.25 ms compiled versus 5.35 ms eager, a 20.6% latency reduction.
* Batch 4, sequence 64: 4.11 ms compiled versus 5.01 ms eager, an 18.0% reduction.
* Batch 4, sequence 128: 5.41 ms compiled versus 6.90 ms eager, a 21.6% reduction.
* Batch 4, sequence 256: 9.64 ms compiled versus 10.72 ms eager, a 10.1% reduction.

These results indicate that generated-code optimization and operator fusion do not automatically compensate for compiler and dispatch overhead on small workloads. The benefit becomes more visible when the model performs more computation per invocation.

## Compilation overhead

Initial compiled execution required between approximately 2 and 10 seconds, depending on the shape and whether compiler initialization had already occurred.

For the configurations where compiled execution was faster, the estimated break-even point ranged from approximately 2,296 to 3,804 calls. Therefore, compilation would be unsuitable for some short-lived or low-traffic workloads even when its steady-state latency is lower.

## Dynamic-shape behavior

The dynamic-shape experiment compiled one reusable graph for sequence lengths 32, 64, 128, and 256.

The initial execution required approximately 6.87 seconds, while subsequent new and repeated sequence lengths executed in approximately 3–7 ms without increasing the unique graph count.

For this model and shape range, dynamic compilation avoided repeated shape-specific graphs. This result does not prove that every model or dynamic dimension will behave similarly because unsupported data-dependent behavior can still create guards, recompilations, or graph breaks.

## ONNX Runtime results

ONNX Runtime achieved the best result for batch 1 and sequence length 64, reducing median latency from 2.02 ms to 1.79 ms. However, its approximately 2.48-second setup cost required an estimated 11,027 calls to break even.

ONNX Runtime was slower than eager PyTorch on most larger workloads. At batch 4 and sequence length 256, its median latency was 22.84 ms compared with 10.72 ms for eager PyTorch.

The ONNX measurements include NumPy input and output handling through `session.run`. They therefore represent end-to-end runtime invocation rather than isolated operator or kernel execution.

## Profiler analysis

PyTorch Profiler showed that matrix multiplication dominated execution time. The largest matrix multiplication groups together represented most of the measured self CPU time, while scaled dot-product flash attention accounted for approximately 17%.

This indicates that the principal bottlenecks are the model's projection and feed-forward layers, followed by attention. Optimizing a small normalization operation alone is unlikely to materially change end-to-end CPU latency.

## TorchDynamo graph break

A deliberately introduced `Tensor.item()` operation extracted a Python scalar inside `forward`. TorchDynamo divided execution into two graphs and reported one graph break.

The corrected implementation retained the value as a tensor. It produced one graph, zero graph breaks, and succeeded with strict graph-break checking enabled.

The experiment demonstrates that graph breaks preserve correctness through fallback execution but can prevent whole-graph optimization and fusion.

## Conclusions

1. `torch.compile` was beneficial for several larger workloads but detrimental for the smallest tested inputs.
2. Compilation overhead must be evaluated separately from steady-state latency.
3. Dynamic-shape compilation reused one graph across the tested sequence lengths.
4. ONNX Runtime was competitive only for the smallest workload on this Apple Silicon CPU configuration.
5. Matrix multiplication and attention dominated inference time.
6. Python scalar extraction through `Tensor.item()` caused a reproducible TorchDynamo graph break.

## Limitations and next steps

* Results were collected from one local run and should be repeated before reporting final metrics.
* The experiment used a synthetic decoder model rather than a pretrained production model.
* CPU results do not predict NVIDIA GPU performance.
* FP16 and BF16 were not evaluated in this run.
* Triton requires a supported CUDA environment and was not tested.
* Future work should repeat each suite several times, report run-to-run variability, and evaluate TorchInductor, ONNX Runtime CUDA, and the Triton RMSNorm kernel on named NVIDIA hardware.

## Cold versus warm compiler cache

A separate experiment compared a fresh TorchInductor disk cache with a reused cache.

With a cold cache, compiled execution reduced steady-state latency by approximately 11–21% on the larger tested workloads, but required roughly 1,992–4,460 calls to recover compilation cost.

With a warm persistent cache, similar steady-state improvements remained, while estimated break-even fell to approximately 120–162 calls.

This distinction matters operationally: steady-state latency alone does not represent the cost experienced by a newly deployed or short-lived inference process.
