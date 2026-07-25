from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.diagnostics import GraphBreakWrapper, TensorOnlyWrapper
from ml_runtime_bench.models import TinyDecoderLM


def test_diagnostic_wrappers_preserve_output() -> None:
    import torch

    config = ModelConfig(vocab_size=128, max_seq_len=16, d_model=32, n_heads=4, n_layers=1)
    model = TinyDecoderLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    broken = GraphBreakWrapper(model)
    fixed = TensorOnlyWrapper(model)
    torch.testing.assert_close(broken(input_ids), fixed(input_ids))
