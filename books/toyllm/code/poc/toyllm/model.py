"""The whole model: 30 blocks, assembled from the pieces built in steps 3-6."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import ModelConfig
from .layers import (
    NO_CACHE,
    Attention,
    KVCache,
    NoCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    causal_mask,
)
from .sampling import GREEDY, Sampler
from .weights import Weights


class Block(nn.Module):
    """One transformer block. Pre-norm, two residual additions.

        h = x + attn(input_layernorm(x))
        y = h + mlp(post_attention_layernorm(h))

    Note *where* the norms sit: on the way into each sublayer, not on the way
    out. That is "pre-norm", and it is why the residual stream itself is never
    normalized — a value added at block 0 travels to block 29 untouched, with
    each block reading a normalized copy rather than rescaling the stream.
    Post-norm (the original 2017 arrangement) normalizes after the addition and
    needs learning-rate warmup to train deep; every modern LLM is pre-norm.

    The second norm reads `h`, the *post-attention* residual — not `x`. Feeding
    it `x` instead is a real and silent bug: shapes match, output looks fine.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, cos, sin, mask, cache=NO_CACHE, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    @classmethod
    def from_checkpoint(cls, cfg: ModelConfig, weights: Weights, i: int) -> Block:
        block = cls(cfg)
        with torch.no_grad():
            block.input_layernorm.weight.copy_(
                weights.get(f"model.layers.{i}.input_layernorm.weight")
            )
            block.post_attention_layernorm.weight.copy_(
                weights.get(f"model.layers.{i}.post_attention_layernorm.weight")
            )
        block.self_attn = Attention.from_checkpoint(cfg, weights, i)
        block.mlp = SwiGLU.from_checkpoint(cfg, weights, i)
        return block


class SmolLM2(nn.Module):
    """The full causal language model.

        ids -> embed -> 30 x Block -> final RMSNorm -> lm_head -> logits

    cos/sin and the causal mask are built once here and handed to every block:
    they depend only on positions, so recomputing them 30 times would be waste.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding.from_config(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        cache: NoCache = NO_CACHE,
    ):
        """input_ids: (B, T) -> logits (B, T, vocab).

        With `output_hidden_states`, also returns 31 tensors following HF's
        convention exactly, so the two can be diffed index by index:

            [0]      the embedding output
            [1..29]  the *input* to block i, i.e. the output of block i-1
            [30]     the output of block 29 **after** the final norm

        Note the asymmetry at the end — the last entry is normalized while the
        others are not. Comparing a raw block-29 output against HF's entry 30
        shows a huge mismatch that has nothing to do with correctness.
        """
        b, t = input_ids.shape
        past = cache.length  # 0 for NO_CACHE, so this collapses to step 7
        if positions is None:
            # Continuing from a cache means these tokens start at `past`, not 0.
            positions = torch.arange(past, past + t, device=input_ids.device)

        x = self.embed_tokens(input_ids)
        # No embedding scaling: Llama-family models feed the lookup straight in,
        # unlike the original transformer's sqrt(d_model) factor.
        cos, sin = self.rotary(positions)
        # A single new token may attend to everything cached, so the mask is all
        # zeros and can be skipped entirely.
        mask = causal_mask(t, past + t, past, x.dtype) if t > 1 else None

        collected = [] if output_hidden_states else None
        for i, block in enumerate(self.layers):
            if output_hidden_states:
                collected.append(x)
            x = block(x, cos, sin, mask, cache, i)

        cache.advance(t)  # no-op for NO_CACHE

        x = self.norm(x)
        if output_hidden_states:
            collected.append(x)

        logits = self.lm_head(x)
        return (logits, collected) if output_hidden_states else logits

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> SmolLM2:
        model_dir = Path(model_dir)
        cfg = ModelConfig.from_json(model_dir)
        weights = Weights(model_dir, cfg)
        problems = weights.verify()
        if problems:
            raise ValueError("checkpoint does not match config:\n  " + "\n  ".join(problems))

        model = cls(cfg)
        with torch.no_grad():
            model.embed_tokens.weight.copy_(weights.get("model.embed_tokens.weight"))
            model.norm.weight.copy_(weights.get("model.norm.weight"))
        model.layers = nn.ModuleList(
            Block.from_checkpoint(cfg, weights, i) for i in range(cfg.num_hidden_layers)
        )
        if cfg.tie_word_embeddings:
            # Share the Parameter object, do not copy: this is what makes the
            # 134.5M count come out right, and it halves the memory the
            # vocabulary costs.
            model.lm_head.weight = model.embed_tokens.weight
        else:
            with torch.no_grad():
                model.lm_head.weight.copy_(weights.get("lm_head.weight"))

        model.eval()
        return model

    def new_cache(self, max_seq_len: int, batch_size: int = 1) -> KVCache:
        return KVCache(
            self.cfg,
            max_seq_len=min(max_seq_len, self.cfg.max_position_embeddings),
            batch_size=batch_size,
            dtype=self.embed_tokens.weight.dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_id: int | None = None,
        cache: bool | NoCache = True,
        sampler: Sampler = GREEDY,
    ):
        """Autoregressive decoding, greedy by default.

        `sampler` decides how a logit row becomes a token; `GREEDY` is argmax.
        Pass a configured `Sampler` for temperature / top-k / top-p.

        `cache` accepts:
            True      allocate a fresh KVCache (default)
            False     no cache — recompute the whole sequence each step (step 7)
            KVCache   reuse an existing one, e.g. to continue a conversation

        The loop below is the same either way. The only thing the cache changes
        is **what gets fed in** each step: with a cache, just the one new token,
        because the history is already stored; without one, the entire sequence,
        because nothing is. That single `feed = ...` line is the whole
        difference between O(n) and O(n^2) decoding.

        Prefill (the first pass, over the whole prompt) and decode (every pass
        after) have very different performance characters — prefill is
        compute-bound on large matmuls, decode is memory-bound reading all the
        weights to advance one token. Real serving systems schedule them apart.
        """
        b, prompt_len = input_ids.shape
        if cache is True:
            cache = self.new_cache(prompt_len + max_new_tokens, b)
        elif cache is False:
            cache = NO_CACHE

        ids = input_ids
        feed = input_ids  # first pass is the prompt, cached or not
        for _ in range(max_new_tokens):
            # Only the last position's logits matter; the rest are discarded.
            next_id = sampler(self(feed, cache=cache)[:, -1])
            ids = torch.cat([ids, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
            feed = next_id if cache else ids
        return ids

    def num_parameters(self) -> int:
        """Distinct parameters — tied weights counted once, as the checkpoint stores them."""
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
