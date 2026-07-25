#!/usr/bin/env bash
set -euo pipefail

python -m ml_runtime_bench suite \
  --config configs/gpu.json \
  --output results/gpu

python -m ml_runtime_bench profile \
  --mode compile \
  --device cuda \
  --dtype float16 \
  --batch-size 4 \
  --sequence-length 256 \
  --output results/gpu/profile

python -m ml_runtime_bench dynamic-shapes \
  --device cuda \
  --dtype float16 \
  --lengths 64 128 256 128 512 256 \
  --output results/gpu/dynamic-shapes

python -m ml_runtime_bench triton-rmsnorm \
  --hidden-sizes 256 512 1024 2048 4096 \
  --rows 2048 \
  --dtype float16 \
  --output results/gpu/triton-rmsnorm

python -m ml_runtime_bench plot \
  --results results/gpu/results.json \
  --output results/gpu/charts
