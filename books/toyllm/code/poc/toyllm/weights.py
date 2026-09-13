"""Reading SmolLM2 weights out of a `.safetensors` file.

Step 1 of the project: no math yet, just get the tensors in hand and prove we
understand exactly which ones exist and what shape each one is.

Naming follows HuggingFace's Llama checkpoint layout, because that is what the
file on disk uses. Every `nn.Linear`-style weight is stored **(out_features,
in_features)**, i.e. already transposed relative to how it multiplies an input
row vector — a classic source of off-by-a-transpose bugs later on.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .config import ModelConfig


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


def expected_tensors(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the checkpoint should contain, mapped to its expected shape."""
    h = cfg.hidden_size
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv = cfg.kv_size
    ffn = cfg.intermediate_size

    spec: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (cfg.vocab_size, h)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        spec.update(
            {
                f"{p}.self_attn.q_proj.weight": (q_size, h),
                f"{p}.self_attn.k_proj.weight": (kv, h),
                f"{p}.self_attn.v_proj.weight": (kv, h),
                f"{p}.self_attn.o_proj.weight": (h, q_size),
                f"{p}.mlp.gate_proj.weight": (ffn, h),
                f"{p}.mlp.up_proj.weight": (ffn, h),
                f"{p}.mlp.down_proj.weight": (h, ffn),
                f"{p}.input_layernorm.weight": (h,),
                f"{p}.post_attention_layernorm.weight": (h,),
            }
        )
    spec["model.norm.weight"] = (h,)
    if not cfg.tie_word_embeddings:
        spec["lm_head.weight"] = (cfg.vocab_size, h)
    return spec


class Weights:
    """Lazy handle on a safetensors checkpoint.

    `safe_open` memory-maps the file, so constructing this is cheap and each
    tensor is only materialized when `get()` asks for it.
    """

    def __init__(self, model_dir: str | Path, cfg: ModelConfig, dtype: torch.dtype = torch.float32):
        self.model_dir = Path(model_dir)
        self.cfg = cfg
        self.dtype = dtype
        self.path = self.model_dir / "model.safetensors"
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — download the weights first (see CLAUDE.md)"
            )
        self._f = safe_open(self.path, framework="pt", device="cpu")
        self.names = list(self._f.keys())

    def get(self, name: str) -> torch.Tensor:
        """Fetch one tensor, cast to the engine's working dtype.

        The checkpoint is bfloat16; we upcast to float32 because this runs on
        CPU, where float32 is both faster and the reference for correctness.
        """
        if name == "lm_head.weight" and self.cfg.tie_word_embeddings:
            # Not stored separately — the output projection *is* the embedding matrix.
            name = "model.embed_tokens.weight"
        return self._f.get_tensor(name).to(self.dtype)

    def raw_dtype(self, name: str) -> torch.dtype:
        """On-disk dtype. safetensors reports its own string tags, not torch dtypes."""
        return _SAFETENSORS_DTYPES[self._f.get_slice(name).get_dtype()]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._f.get_slice(name).get_shape())

    def verify(self) -> list[str]:
        """Compare the file against `expected_tensors`. Returns a list of problems."""
        spec = expected_tensors(self.cfg)
        problems = []
        for name, want in spec.items():
            if name not in self.names:
                problems.append(f"missing: {name}")
            elif self.shape(name) != want:
                problems.append(f"shape mismatch: {name} is {self.shape(name)}, expected {want}")
        for name in self.names:
            if name not in spec:
                problems.append(f"unexpected extra tensor: {name}")
        return problems

    def param_count(self) -> int:
        """Distinct stored parameters. Tied lm_head is counted once, as it should be."""
        total = 0
        for name in self.names:
            n = 1
            for d in self.shape(name):
                n *= d
            total += n
        return total
