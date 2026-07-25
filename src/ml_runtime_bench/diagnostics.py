from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.models import TinyDecoderLM
from ml_runtime_bench.utils import resolve_device, seed_everything


class GraphBreakWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.model(input_ids))
        python_scalar = input_ids.float().mean().item()
        return logits + python_scalar


class TensorOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = cast(torch.Tensor, self.model(input_ids))
        tensor_scalar = input_ids.float().mean()
        return logits + tensor_scalar


def _explain(model: nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(model)(input_ids)
    reasons = []
    for reason in getattr(explanation, "break_reasons", []):
        reasons.append(str(getattr(reason, "reason", reason)))
    return {
        "graph_count": getattr(explanation, "graph_count", None),
        "graph_break_count": getattr(explanation, "graph_break_count", None),
        "op_count": getattr(explanation, "op_count", None),
        "break_reasons": reasons,
    }


def _strict_graph_result(model: nn.Module, input_ids: torch.Tensor) -> dict[str, Any]:
    torch._dynamo.reset()
    try:
        with torch._dynamo.error_on_graph_break(True):
            compiled = torch.compile(model, backend="eager")
            compiled(input_ids)
        return {"status": "success", "error": None}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_graph_break_diagnostics(
    *,
    output_dir: Path,
    device_name: str = "auto",
    model_config: ModelConfig | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    config = model_config or ModelConfig(d_model=128, n_heads=4, n_layers=2, max_seq_len=128)
    seed_everything(seed)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    base_model = TinyDecoderLM(config).to(device).eval()
    broken = GraphBreakWrapper(base_model)
    fixed = TensorOnlyWrapper(base_model)

    payload = {
        "broken": {
            "explain": _explain(broken, input_ids),
            "strict_compile": _strict_graph_result(broken, input_ids),
        },
        "fixed": {
            "explain": _explain(fixed, input_ids),
            "strict_compile": _strict_graph_result(fixed, input_ids),
        },
    }
    (output_dir / "graph_breaks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    broken_reasons = payload["broken"]["explain"]["break_reasons"]
    report = [
        "# TorchDynamo graph-break diagnosis",
        "",
        "## Deliberately broken implementation",
        "",
        "The wrapper calls `.item()` on a tensor inside `forward`, forcing a Python scalar extraction and data-dependent boundary.",
        "",
        f"- Graph count: `{payload['broken']['explain']['graph_count']}`",
        f"- Graph-break count: `{payload['broken']['explain']['graph_break_count']}`",
        f"- Strict graph-break mode: `{payload['broken']['strict_compile']['status']}`",
        "",
        "Break reasons:",
        "",
    ]
    report.extend(
        [f"- {reason}" for reason in broken_reasons]
        or ["- No reason text returned by this PyTorch version."]
    )
    report.extend(
        [
            "",
            "## Tensor-only correction",
            "",
            "The corrected wrapper keeps the scalar as a tensor and therefore avoids the Python extraction.",
            "",
            f"- Graph count: `{payload['fixed']['explain']['graph_count']}`",
            f"- Graph-break count: `{payload['fixed']['explain']['graph_break_count']}`",
            f"- Strict graph-break mode: `{payload['fixed']['strict_compile']['status']}`",
            "",
            "## Engineering conclusion",
            "",
            "Graph breaks preserve correctness by falling back to Python between compiled regions, but can reduce fusion and add dispatch overhead. `torch._dynamo.error_on_graph_break(True)` is useful as a diagnostic because it converts an otherwise tolerated graph break into an actionable failure.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return payload
