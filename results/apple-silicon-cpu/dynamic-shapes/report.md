# Dynamic-shape experiment

- `torch.compile(dynamic=True)`
- Batch size: `1`
- Shape order: `[32, 64, 128, 64, 256, 128]`

| Index | Sequence | First call ms | Immediate repeat ms | Unique graphs | Calls captured |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 6867.45 | 3.36 | 1 | 396 |
| 1 | 64 | 3.91 | 3.36 | 1 | 396 |
| 2 | 128 | 5.14 | 4.29 | 1 | 396 |
| 3 | 64 | 3.24 | 3.26 | 1 | 396 |
| 4 | 256 | 7.36 | 5.04 | 1 | 396 |
| 5 | 128 | 3.41 | 3.48 | 1 | 396 |

Repeated sequence lengths are intentional. A large first-call cost for a new shape followed by a much smaller repeat call suggests compilation or specialization overhead. Internal Dynamo counters are diagnostic and may vary by PyTorch version.
