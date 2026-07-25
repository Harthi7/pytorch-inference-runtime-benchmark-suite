import torch

from ml_runtime_bench.config import ModelConfig
from ml_runtime_bench.models import TinyDecoderLM


def test_model_output_shape() -> None:
    config = ModelConfig(vocab_size=128, max_seq_len=32, d_model=32, n_heads=4, n_layers=2)
    model = TinyDecoderLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids)
    assert output.shape == (2, 16, 128)


def test_tied_embeddings() -> None:
    config = ModelConfig(vocab_size=128, max_seq_len=16, d_model=32, n_heads=4, n_layers=1)
    model = TinyDecoderLM(config)
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_model_is_deterministic_in_eval() -> None:
    torch.manual_seed(7)
    config = ModelConfig(vocab_size=128, max_seq_len=16, d_model=32, n_heads=4, n_layers=1)
    model = TinyDecoderLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    torch.testing.assert_close(model(input_ids), model(input_ids))
