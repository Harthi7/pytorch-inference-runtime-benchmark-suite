# Contributing

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run `make test` and `make lint`.
4. Do not commit benchmark claims without the raw `results.json`, system metadata, and exact command.
5. Keep optional accelerator dependencies behind capability checks so CPU CI remains usable.

Performance changes should include correctness tolerances and explain whether timing includes setup, compilation, synchronization, or host/device transfer.
