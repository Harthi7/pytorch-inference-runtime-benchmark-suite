import json
from pathlib import Path

import pytest

from ml_runtime_bench.config import load_suite_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "name": "test",
                "model": {"d_model": 64, "n_heads": 4, "n_layers": 1, "max_seq_len": 32},
                "benchmark": {
                    "batch_sizes": [1],
                    "sequence_lengths": [16],
                    "modes": ["eager"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_suite_config(path)
    assert config.name == "test"
    assert config.model.d_model == 64


def test_rejects_sequence_beyond_model_limit(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {"max_seq_len": 8},
                "benchmark": {"sequence_lengths": [16]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds"):
        load_suite_config(path)
