from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from torch import nn

from ml_runtime_bench.runtimes.base import RuntimeAdapter


class EagerRuntime(RuntimeAdapter):
    name = "eager"

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def prepare(self, model: nn.Module, sample_input: torch.Tensor, artifact_dir: Path) -> None:
        del sample_input, artifact_dir
        self.model = model

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("runtime is not prepared")
        return cast(torch.Tensor, self.model(input_ids))
