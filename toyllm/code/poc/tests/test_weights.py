import torch

from toyllm.config import ModelConfig
from toyllm.weights import Weights, expected_tensors


def test_checkpoint_matches_expected_layout(model_dir):
    """Every tensor we think exists does exist, with the shape we predicted."""
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    assert w.verify() == []


def test_tensor_count(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    # 9 per block + embed_tokens + final norm; lm_head is tied, so not stored.
    assert len(w.names) == 9 * cfg.num_hidden_layers + 2 == 272
    assert len(expected_tensors(cfg)) == len(w.names)


def test_param_count_matches_architecture(model_dir):
    """Independently derive the parameter count and check it against the file."""
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    h, ffn, q, kv = cfg.hidden_size, cfg.intermediate_size, cfg.hidden_size, cfg.kv_size
    per_layer = (q * h + kv * h * 2 + h * q) + (ffn * h * 3) + h * 2
    total = cfg.vocab_size * h + per_layer * cfg.num_hidden_layers + h
    assert total == w.param_count() == 134_515_008


def test_lm_head_is_tied_to_embeddings(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    assert "lm_head.weight" not in w.names
    assert torch.equal(w.get("lm_head.weight"), w.get("model.embed_tokens.weight"))


def test_weights_load_as_float32_from_bf16_storage(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    name = "model.layers.0.self_attn.q_proj.weight"
    assert w.raw_dtype(name) == torch.bfloat16
    t = w.get(name)
    assert t.dtype == torch.float32
    assert t.shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    assert torch.isfinite(t).all()
