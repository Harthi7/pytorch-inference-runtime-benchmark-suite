#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType

import torch

from ml_runtime_bench.config import load_suite_config
from ml_runtime_bench.models.tiny_decoder import RMSNorm, TinyDecoderLM
from ml_runtime_bench.runtimes.onnx_runtime import OnnxRuntime

CONFIG = Path("configs/rtx5070ti-onnx-fp32.json")
ARTIFACT_DIR = Path(".benchmark-cache/onnx-fp32-oracle-diagnostic")


def patch_rmsnorm_for_float64(model: TinyDecoderLM) -> None:
    """Preserve the mathematical RMSNorm operation in float64 for the diagnostic oracle."""

    def high_precision_forward(self: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight

    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = MethodType(high_precision_forward, module)


def metrics(
    name: str, actual: torch.Tensor, reference: torch.Tensor, atol: float, rtol: float
) -> None:
    actual64 = actual.detach().cpu().double()
    reference64 = reference.detach().cpu().double()
    diff = (actual64 - reference64).abs()

    allowed = atol + rtol * reference64.abs()
    outside = diff > allowed

    argmax_match = (actual64.argmax(dim=-1) == reference64.argmax(dim=-1)).double().mean().item()

    print(
        f"{name:20s} "
        f"max_abs={diff.max().item():.9f} "
        f"mean_abs={diff.mean().item():.9f} "
        f"rmse={diff.square().mean().sqrt().item():.9f} "
        f"outside_tol={outside.double().mean().item():.6%} "
        f"argmax_match={argmax_match:.6%}"
    )


def main() -> None:
    suite = load_suite_config(CONFIG)
    model_cfg = suite.model
    bench = suite.benchmark

    if bench.dtype != "float32":
        raise RuntimeError(f"Expected float32 diagnostic config, got {bench.dtype}")
    if bench.batch_sizes != [1] or bench.sequence_lengths != [64]:
        raise RuntimeError(
            "Expected the B1/S64 smoke config for this diagnostic; "
            f"got batches={bench.batch_sizes}, sequences={bench.sequence_lengths}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(bench.seed)
    torch.cuda.manual_seed_all(bench.seed)

    device = torch.device("cuda")
    base_model = TinyDecoderLM(model_cfg).to(device=device, dtype=torch.float32).eval()
    base_state = copy.deepcopy(base_model.state_dict())

    input_ids = torch.randint(
        0,
        model_cfg.vocab_size,
        (1, 64),
        device=device,
        dtype=torch.long,
    )

    eager_cuda = TinyDecoderLM(model_cfg).to(device=device, dtype=torch.float32).eval()
    eager_cuda.load_state_dict(base_state)

    onnx_model = TinyDecoderLM(model_cfg).to(device=device, dtype=torch.float32).eval()
    onnx_model.load_state_dict(base_state)

    state_cpu = {key: value.detach().cpu() for key, value in base_state.items()}

    eager_cpu = TinyDecoderLM(model_cfg).to(device="cpu", dtype=torch.float32).eval()
    eager_cpu.load_state_dict(state_cpu)

    oracle64 = TinyDecoderLM(model_cfg).to(device="cpu", dtype=torch.float64).eval()
    oracle64.load_state_dict(state_cpu)
    patch_rmsnorm_for_float64(oracle64)

    input_cpu = input_ids.detach().cpu()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ort_runtime = OnnxRuntime(dynamic=False)
    ort_runtime.prepare(onnx_model, input_ids, ARTIFACT_DIR)

    with torch.inference_mode():
        out_eager_cuda = eager_cuda(input_ids)
        out_onnx = ort_runtime.run(input_ids)
        out_eager_cpu = eager_cpu(input_cpu)
        out_oracle64 = oracle64(input_cpu)

    print("===== ENVIRONMENT =====")
    import onnxruntime as ort

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("ONNX Runtime:", ort.__version__)
    print("ORT session providers:", ort_runtime.session.get_providers())
    print("Input shape:", tuple(input_ids.shape))
    print("Tolerance: atol=", bench.correctness_atol, "rtol=", bench.correctness_rtol)
    print()

    print("===== AGAINST FLOAT64 CPU ORACLE =====")
    metrics(
        "eager_cuda_fp32",
        out_eager_cuda,
        out_oracle64,
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "onnx_cuda_fp32",
        out_onnx,
        out_oracle64,
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "eager_cpu_fp32",
        out_eager_cpu,
        out_oracle64,
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    print()

    print("===== IMPLEMENTATION PARITY =====")
    metrics(
        "onnx_vs_eager_cuda",
        out_onnx,
        out_eager_cuda,
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "cpu_vs_eager_cuda",
        out_eager_cpu,
        out_eager_cuda,
        bench.correctness_atol,
        bench.correctness_rtol,
    )


if __name__ == "__main__":
    main()
