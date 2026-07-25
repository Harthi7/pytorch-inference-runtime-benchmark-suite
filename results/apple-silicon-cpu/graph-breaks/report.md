# TorchDynamo graph-break diagnosis

## Deliberately broken implementation

The wrapper calls `.item()` on a tensor inside `forward`, forcing a Python scalar extraction and data-dependent boundary.

- Graph count: `2`
- Graph-break count: `1`
- Strict graph-break mode: `failed`

Break reasons:

- Unsupported Tensor.item() call with capture_scalar_outputs=False
  Explanation: Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.
  Hint: Set `torch._dynamo.config.capture_scalar_outputs = True` or `export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` to include these operations in the captured graph.

  Developer debug context: call_method TensorVariable() item () {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0124.html

## Tensor-only correction

The corrected wrapper keeps the scalar as a tensor and therefore avoids the Python extraction.

- Graph count: `1`
- Graph-break count: `0`
- Strict graph-break mode: `success`

## Engineering conclusion

Graph breaks preserve correctness by falling back to Python between compiled regions, but can reduce fusion and add dispatch overhead. `torch._dynamo.error_on_graph_break(True)` is useful as a diagnostic because it converts an otherwise tolerated graph break into an actionable failure.
