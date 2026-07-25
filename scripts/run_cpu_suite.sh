#!/usr/bin/env bash
set -euo pipefail

python -m ml_runtime_bench suite \
  --config configs/cpu.json \
  --output results/cpu

python -m ml_runtime_bench plot \
  --results results/cpu/results.json \
  --output results/cpu/charts
