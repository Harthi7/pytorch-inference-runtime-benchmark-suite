from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn


class RuntimeAdapter(ABC):
    name: str

    @abstractmethod
    def prepare(self, model: nn.Module, sample_input: torch.Tensor, artifact_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {}
