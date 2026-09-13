import json

import pytest

from toyllm.config import ModelConfig


def test_smollm2_135m_shapes(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    assert (cfg.hidden_size, cfg.num_hidden_layers) == (576, 30)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (9, 3)
    assert cfg.head_dim == 64          # 576 / 9
    assert cfg.n_rep == 3              # 9 / 3 query heads per kv head
    assert cfg.kv_size == 192          # 3 * 64, not hidden_size
    assert cfg.vocab_size == 49152
    assert cfg.tie_word_embeddings is True


def test_rejects_unsupported_config(tmp_path, model_dir):
    """A config field we don't implement must raise, not be silently ignored."""
    raw = json.loads((model_dir / "config.json").read_text())
    raw["hidden_act"] = "gelu"
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(NotImplementedError, match="hidden_act"):
        ModelConfig.from_json(p)


def test_rejects_indivisible_heads(tmp_path, model_dir):
    raw = json.loads((model_dir / "config.json").read_text())
    raw["num_key_value_heads"] = 4     # 9 is not a multiple of 4
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="multiple"):
        ModelConfig.from_json(p)
