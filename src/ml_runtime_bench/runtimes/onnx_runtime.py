from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.runtimes.base import RuntimeAdapter


class OnnxRuntime(RuntimeAdapter):
    name = "onnx"

    def __init__(self, *, dynamic: bool = False) -> None:
        self.dynamic = dynamic
        self.session: Any = None
        self.input_name = "input_ids"
        self.output_name = "logits"
        self.providers: list[str] = []
        self.model_path: Path | None = None
        self.device = torch.device("cpu")

    def prepare(self, model: nn.Module, sample_input: torch.Tensor, artifact_dir: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("install the 'onnx' optional dependency group") from exc

        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = artifact_dir / "model.onnx"
        self.device = sample_input.device

        dynamic_shapes = None
        if self.dynamic:
            config = getattr(model, "config", None)
            if not isinstance(config, ModelConfig):
                raise TypeError("dynamic ONNX export requires a model with ModelConfig")

            batch = torch.export.Dim("batch", min=1)
            sequence = torch.export.Dim(
                "sequence",
                min=1,
                max=config.max_seq_len,
            )
            dynamic_shapes = {"input_ids": {0: batch, 1: sequence}}

        export_result = torch.onnx.export(
            model,
            (sample_input,),
            self.model_path,
            input_names=[self.input_name],
            output_names=[self.output_name],
            opset_version=18,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            verify=False,
        )
        del export_result

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        optimized_path = artifact_dir / "model.optimized.onnx"
        options.optimized_model_filepath = str(optimized_path)

        available = ort.get_available_providers()
        if sample_input.device.type == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDA input requires ONNX Runtime CUDAExecutionProvider; "
                    "install onnxruntime-gpu and verify provider availability"
                )
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif sample_input.device.type == "cpu":
            self.providers = ["CPUExecutionProvider"]
        else:
            raise RuntimeError(
                f"ONNX Runtime comparison is not supported for device {sample_input.device.type}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=self.providers
        )

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.session is None:
            raise RuntimeError("runtime is not prepared")
        output = self.session.run(
            [self.output_name], {self.input_name: input_ids.detach().cpu().numpy()}
        )[0]
        return torch.from_numpy(np.asarray(output))

    def metadata(self) -> dict[str, Any]:
        return {
            "providers": self.providers,
            "dynamic": self.dynamic,
            "timing_scope": "session.run with NumPy host input/output",
            "model_path": str(self.model_path) if self.model_path else None,
        }
