"""Model config — the single source of truth for every shape in the engine.

Read from the model's own `config.json` rather than hardcoded, so that swapping
SmolLM2-135M for -360M or -1.7B is a path change and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    torch_dtype: str

    @property
    def head_dim(self) -> int:
        """Width of one attention head. 576 / 9 = 64 for SmolLM2-135M."""
        return self.hidden_size // self.num_attention_heads

    @property
    def n_rep(self) -> int:
        """How many Q heads share one KV head (GQA group size). 9 / 3 = 3."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_size(self) -> int:
        """Output width of k_proj / v_proj: 3 * 64 = 192, not hidden_size."""
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text())
        cfg = cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            rope_theta=raw["rope_theta"],
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=raw["tie_word_embeddings"],
            bos_token_id=raw["bos_token_id"],
            eos_token_id=raw["eos_token_id"],
            torch_dtype=raw.get("torch_dtype", "float32"),
        )
        cfg._check_supported(raw)
        return cfg

    def _check_supported(self, raw: dict) -> None:
        """Fail loudly on any config this toy engine does not actually implement.

        A silently-ignored field is the worst kind of bug here: the model loads,
        generates fluent-looking text, and is subtly wrong.
        """
        expected = {
            "model_type": "llama",
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_scaling": None,
            "rope_interleaved": False,
        }
        for key, want in expected.items():
            if key in raw and raw[key] != want:
                raise NotImplementedError(
                    f"config.json has {key}={raw[key]!r}, this engine only handles {want!r}"
                )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide evenly into attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    def summary(self) -> str:
        return "\n".join(
            [
                f"  vocab_size          {self.vocab_size}",
                f"  hidden_size         {self.hidden_size}",
                f"  layers              {self.num_hidden_layers}",
                f"  attention heads     {self.num_attention_heads}  (head_dim {self.head_dim})",
                f"  kv heads            {self.num_key_value_heads}  (GQA group {self.n_rep}, kv_size {self.kv_size})",
                f"  intermediate_size   {self.intermediate_size}",
                f"  rope_theta          {self.rope_theta}",
                f"  rms_norm_eps        {self.rms_norm_eps}",
                f"  max_position        {self.max_position_embeddings}",
                f"  tied embeddings     {self.tie_word_embeddings}",
                f"  stored dtype        {self.torch_dtype}",
            ]
        )
