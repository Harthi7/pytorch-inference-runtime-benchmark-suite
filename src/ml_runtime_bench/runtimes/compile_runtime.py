from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ml_runtime_bench.runtimes.base import RuntimeAdapter


class CompileRuntime(RuntimeAdapter):
    name = "compile"

    def __init__(
        self,
        *,
        compile_mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool | None = None,
    ) -> None:
        self.compile_mode = compile_mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.model: Callable[[torch.Tensor], torch.Tensor] | None = None

    def prepare(self, model: nn.Module, sample_input: torch.Tensor, artifact_dir: Path) -> None:
        del sample_input, artifact_dir
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode=self.compile_mode,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        self.model = cast(
            Callable[[torch.Tensor], torch.Tensor],
            compiled_model,
        )

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("runtime is not prepared")
        return self.model(input_ids)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "inductor",
            "compile_mode": self.compile_mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
        }
