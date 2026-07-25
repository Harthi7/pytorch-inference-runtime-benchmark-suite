from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    mlp_ratio: float = 4.0
    rms_norm_eps: float = 1e-5

    def validate(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("d_model, n_heads, and n_layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4])
    sequence_lengths: list[int] = field(default_factory=lambda: [64, 128])
    modes: list[str] = field(default_factory=lambda: ["eager", "compile"])
    device: str = "auto"
    dtype: str = "float32"
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    compile_mode: str = "default"
    fullgraph: bool = False
    dynamic: bool | None = None
    correctness_atol: float = 1e-4
    correctness_rtol: float = 1e-3
    seed: int = 7

    def validate(self, model: ModelConfig) -> None:
        supported_modes = {"eager", "compile", "onnx"}
        unknown = set(self.modes) - supported_modes
        if unknown:
            raise ValueError(f"unsupported modes: {sorted(unknown)}")
        if not self.batch_sizes or any(value <= 0 for value in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if not self.sequence_lengths or any(value <= 0 for value in self.sequence_lengths):
            raise ValueError("sequence_lengths must contain positive integers")
        if max(self.sequence_lengths) > model.max_seq_len:
            raise ValueError("sequence length exceeds model.max_seq_len")
        if self.warmup_iterations < 0 or self.benchmark_iterations <= 0:
            raise ValueError("warmup_iterations must be >= 0 and benchmark_iterations > 0")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")


@dataclass(frozen=True)
class SuiteConfig:
    name: str = "benchmark"
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        self.model.validate()
        self.benchmark.validate(self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_suite_config(path: str | Path) -> SuiteConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = SuiteConfig(
        name=raw.get("name", "benchmark"),
        model=ModelConfig(**raw.get("model", {})),
        benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
    )
    config.validate()
    return config
