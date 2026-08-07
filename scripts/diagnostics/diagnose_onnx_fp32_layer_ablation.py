#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from ml_runtime_bench.config import load_suite_config
from ml_runtime_bench.models.tiny_decoder import RMSNorm, TinyDecoderLM

CONFIG = Path("configs/rtx5070ti-onnx-fp32.json")
ARTIFACT_DIR = Path(".benchmark-cache/onnx-fp32-layer-ablation")
MODEL_PATH = ARTIFACT_DIR / "model.onnx"


def patch_rmsnorm_for_float64(model: TinyDecoderLM) -> None:
    """Preserve RMSNorm math in float64 for the diagnostic oracle."""

    def high_precision_forward(self: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight

    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.forward = MethodType(high_precision_forward, module)


def metrics(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual64 = actual.detach().cpu().double()
    reference64 = reference.detach().cpu().double()
    diff = (actual64 - reference64).abs()
    allowed = atol + rtol * reference64.abs()
    outside = diff > allowed
    argmax_match = (actual64.argmax(dim=-1) == reference64.argmax(dim=-1)).double().mean().item()

    print(
        f"{name:28s} "
        f"max_abs={diff.max().item():.9f} "
        f"mean_abs={diff.mean().item():.9f} "
        f"rmse={diff.square().mean().sqrt().item():.9f} "
        f"outside_tol={outside.double().mean().item():.6%} "
        f"argmax_match={argmax_match:.6%}"
    )


def make_session(
    ort,
    *,
    providers: list[str],
    optimization_level,
):
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization_level
    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=providers,
    )


def run_ort(session, input_ids: torch.Tensor) -> torch.Tensor:
    output = session.run(
        ["logits"],
        {"input_ids": input_ids.detach().cpu().numpy()},
    )[0]
    return torch.from_numpy(np.asarray(output))


def main() -> None:
    import onnxruntime as ort

    suite = load_suite_config(CONFIG)
    model_cfg = suite.model
    bench = suite.benchmark

    if bench.dtype != "float32":
        raise RuntimeError(f"Expected float32 config, got {bench.dtype}")
    if bench.batch_sizes != [1] or bench.sequence_lengths != [64]:
        raise RuntimeError(
            "Expected B1/S64 smoke config; "
            f"got batches={bench.batch_sizes}, sequences={bench.sequence_lengths}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available")

    torch.manual_seed(bench.seed)
    torch.cuda.manual_seed_all(bench.seed)

    cuda = torch.device("cuda")

    # Generate one FP32-rounded model state and one input.
    base_model = TinyDecoderLM(model_cfg).to(device=cuda, dtype=torch.float32).eval()
    base_state = copy.deepcopy(base_model.state_dict())
    state_cpu = {key: value.detach().cpu() for key, value in base_state.items()}

    input_ids_cuda = torch.randint(
        0,
        model_cfg.vocab_size,
        (1, 64),
        device=cuda,
        dtype=torch.long,
    )
    input_ids_cpu = input_ids_cuda.detach().cpu()

    eager_cuda = TinyDecoderLM(model_cfg).to(device=cuda, dtype=torch.float32).eval()
    eager_cuda.load_state_dict(base_state)

    eager_cpu = TinyDecoderLM(model_cfg).to(device="cpu", dtype=torch.float32).eval()
    eager_cpu.load_state_dict(state_cpu)

    oracle64 = TinyDecoderLM(model_cfg).to(device="cpu", dtype=torch.float64).eval()
    oracle64.load_state_dict(state_cpu)
    patch_rmsnorm_for_float64(oracle64)

    # Export exactly once so every ORT session consumes identical graph bytes.
    export_model = TinyDecoderLM(model_cfg).to(device=cuda, dtype=torch.float32).eval()
    export_model.load_state_dict(base_state)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        (input_ids_cuda,),
        MODEL_PATH,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        dynamic_shapes=None,
        verify=False,
    )

    sessions = {
        "cpu_no_opt": make_session(
            ort,
            providers=["CPUExecutionProvider"],
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ),
        "cpu_opt_all": make_session(
            ort,
            providers=["CPUExecutionProvider"],
            optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
        "cuda_no_opt": make_session(
            ort,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ),
        "cuda_opt_all": make_session(
            ort,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        ),
    }

    with torch.inference_mode():
        out_eager_cuda = eager_cuda(input_ids_cuda)
        out_eager_cpu = eager_cpu(input_ids_cpu)
        out_oracle64 = oracle64(input_ids_cpu)
        outputs = {name: run_ort(session, input_ids_cuda) for name, session in sessions.items()}

    print("===== ENVIRONMENT =====")
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", ort.get_available_providers())
    print("Input shape:", tuple(input_ids_cuda.shape))
    print("Tolerance: atol=", bench.correctness_atol, "rtol=", bench.correctness_rtol)
    print()

    print("===== SESSION PROVIDERS =====")
    for name, session in sessions.items():
        print(f"{name:28s} {session.get_providers()}")
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
        "eager_cpu_fp32",
        out_eager_cpu,
        out_oracle64,
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    for name, output in outputs.items():
        metrics(
            name,
            output,
            out_oracle64,
            bench.correctness_atol,
            bench.correctness_rtol,
        )
    print()

    print("===== AGAINST EAGER CUDA FP32 =====")
    for name, output in outputs.items():
        metrics(
            name,
            output,
            out_eager_cuda,
            bench.correctness_atol,
            bench.correctness_rtol,
        )
    print()

    print("===== ORT CROSS-COMPARISONS =====")
    metrics(
        "cuda_no_opt vs cpu_no_opt",
        outputs["cuda_no_opt"],
        outputs["cpu_no_opt"],
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "cuda_opt_all vs cpu_opt_all",
        outputs["cuda_opt_all"],
        outputs["cpu_opt_all"],
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "cpu_opt_all vs cpu_no_opt",
        outputs["cpu_opt_all"],
        outputs["cpu_no_opt"],
        bench.correctness_atol,
        bench.correctness_rtol,
    )
    metrics(
        "cuda_opt_all vs cuda_no_opt",
        outputs["cuda_opt_all"],
        outputs["cuda_no_opt"],
        bench.correctness_atol,
        bench.correctness_rtol,
    )


if __name__ == "__main__":
    main()
