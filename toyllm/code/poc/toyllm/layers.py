"""Hand-written building blocks. One per step; RMSNorm is the first.

Everything here is checked against HuggingFace's implementation in `tests/`,
to a tolerance tight enough that a genuine formula difference would show up.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .weights import Weights


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019).

        y = x / sqrt(mean(x^2) + eps) * weight

    Compared with LayerNorm, two things are *missing* and that is the whole point:

      - no mean subtraction — LayerNorm re-centers with `(x - mean)`, RMSNorm does
        not. It only rescales. Turns out the re-centering contributes little for
        transformers, and dropping it removes a pass over the data.
      - no bias term — there is a learned per-channel gain `weight`, and nothing
        added afterwards. That is why the checkpoint has a `(576,)` tensor per
        norm and no matching bias.

    The operation is scale-invariant: feeding in `100 * x` gives the same output
    as `x`. It fixes the *magnitude* of each token vector while leaving its
    direction alone, which is what keeps activations from drifting across 30
    residual additions.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., hidden_size) -> same shape; normalization is over the last axis only,
        # so every token in the batch/sequence is scaled independently of the others.
        input_dtype = x.dtype
        # Upcast to fp32 before squaring: in bf16, x^2 for a large activation can
        # overflow or lose most of its mantissa, and the mean of 576 of them is
        # exactly the kind of sum bf16 is worst at.
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        # eps goes *inside* the sqrt, guarding the mean. Putting it outside
        # (rsqrt(var) + eps) silently changes the result for small activations.
        x = x * torch.rsqrt(variance + self.eps)
        # Cast back *before* applying the gain, matching HF exactly: the multiply
        # happens in the input dtype, not in fp32.
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SwiGLU(nn.Module):
    """The feed-forward block — a gated linear unit with a SiLU gate.

        SwiGLU(x) = down( silu(gate(x)) * up(x) )

    where `silu(z) = z * sigmoid(z)` (a.k.a. Swish), and `*` is elementwise.

    A classic transformer FFN is two matrices: `W2 @ relu(W1 @ x)`. This is
    three, and the extra one buys a *multiplicative* interaction: `x` is
    projected twice, one branch is squashed through SiLU and then used to gate
    the other. The network can therefore suppress or pass each of the 1536
    intermediate channels depending on the input, instead of only thresholding.

    Two things worth internalizing:

      - The gate is not a 0..1 probability. SiLU dips to about -0.278 around
        z = -1.28 before returning to 0, so `silu(gate)` can be negative and the
        "gate" can flip a channel's sign, not merely attenuate it.
      - Only the `gate` branch goes through SiLU. `up` is a plain linear
        projection. Applying the activation to the wrong branch, or to both,
        produces perfectly reasonable-looking numbers and a broken model.

    Three matrices at width `d_ff` cost 3*d_ff*d_hidden parameters against a
    plain FFN's 2*4*d_hidden^2, so `d_ff = (8/3) * d_hidden` keeps the budget
    identical. SmolLM2 uses exactly that: 576 * 8/3 = 1536. It is also where
    most of the model lives — 59% of all parameters are in these three matrices.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # bias=False throughout: SmolLM2 has no bias tensors anywhere.
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., hidden) -> (..., intermediate) -> (..., hidden)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @classmethod
    def from_checkpoint(cls, cfg: ModelConfig, weights: Weights, layer_idx: int) -> SwiGLU:
        """Load one block's MLP from the checkpoint.

        No transpose anywhere: `nn.Linear` stores its weight as
        (out_features, in_features) and computes `x @ W.T`, which is exactly how
        the tensors are laid out on disk. Adding a `.T` here is a tempting and
        very common mistake — the shapes happen to be compatible for gate/up
        only if you also swap the dimensions, so it fails loudly there but
        would silently transpose a square matrix elsewhere.
        """
        mlp = cls(cfg.hidden_size, cfg.intermediate_size)
        prefix = f"model.layers.{layer_idx}.mlp"
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(weights.get(f"{prefix}.gate_proj.weight"))
            mlp.up_proj.weight.copy_(weights.get(f"{prefix}.up_proj.weight"))
            mlp.down_proj.weight.copy_(weights.get(f"{prefix}.down_proj.weight"))
        return mlp


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair dimension i with i + d/2 and rotate each pair by 90 degrees.

        [a0 a1 .. a31 | b0 b1 .. b31]  ->  [-b0 -b1 .. -b31 | a0 a1 .. a31]

    This is the **half-split** (GPT-NeoX / HF) convention, selected by
    `rope_interleaved: false` in SmolLM2's config. The original RoFormer paper
    pairs *adjacent* dimensions instead — (0,1), (2,3), ... — which produces
    different numbers from the same weights. Both are self-consistent rotations,
    so a model trained with one and run with the other still emits fluent text;
    it is just quietly worse. Nothing raises.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021).

    Position is injected by *rotating* q and k rather than adding anything to
    them. Split a head's `head_dim` values into `head_dim/2` two-dimensional
    planes; at position `t`, plane `i` is rotated by angle `t * inv_freq[i]`,
    where

        inv_freq[i] = theta ** (-2i / head_dim),   i = 0 .. head_dim/2 - 1

    The payoff is that the attention score between a query at position `m` and a
    key at position `n` becomes a function of `m - n` alone: rotating both by the
    same amount leaves their inner product unchanged, so absolute position
    cancels and only the *relative* offset survives. That property is what makes
    it worth the trouble; see `steps/05_rope.py`, which demonstrates it directly.

    `theta` sets the range of wavelengths. Plane 0 turns once per 2*pi tokens;
    the slowest plane turns once per `2*pi*theta` tokens. SmolLM2 uses
    **theta = 100000**, ten times the usual 10000, stretching the slowest planes
    so positions stay distinguishable across its 8192-token context.

    cos/sin for every position are precomputed once here — they depend only on
    position, never on the data, so recomputing them per token would be pure
    waste in a decode loop.

    One consequence of precomputing, worth knowing before comparing against HF:
    the *angles* are bit-identical either way, but `torch.cos()` over an
    8192x64 table can differ from `torch.cos()` over a 9x64 slice by one ulp
    (~1.2e-7), because PyTorch dispatches large tensors to a vectorized kernel
    and small ones to a scalar path. HF computes cos/sin per forward call, so
    our results agree to ~1e-7 rather than exactly. That is float32 rounding,
    not a difference in the math — a genuine mistake here shows up around 1e0.
    """

    def __init__(self, head_dim: int, theta: float, max_position: int):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # int64 arange then float, matching HF exactly — with a large theta the
        # exponent's precision is visible in the low bits of inv_freq.
        exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        inv_freq = 1.0 / (theta**exponent)  # (head_dim/2,)

        t = torch.arange(max_position, dtype=torch.int64).float()
        freqs = torch.outer(t, inv_freq)  # (max_position, head_dim/2)
        # Duplicated, not interleaved: emb[:, i] and emb[:, i + d/2] carry the
        # same angle, because rotate_half pairs those two dimensions.
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_position, head_dim)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (T,) or (B, T) -> cos, sin of shape (..., T, head_dim)."""
        return self.cos_cached[positions], self.sin_cached[positions]

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> RotaryEmbedding:
        return cls(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, theta={self.theta}, max_position={self.cos_cached.shape[0]}"


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k in place of adding a positional encoding.

    q, k are (B, n_heads, T, head_dim); cos/sin are (T, head_dim) and broadcast
    across batch and heads. Every head uses the *same* rotation — position is a
    property of the token, not of the head.

    Note what is missing: `v` is never rotated. RoPE exists to make the q-k inner
    product relative; the values being attended to carry no position at all.
    """
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, n_kv_heads, T, D) to (B, n_kv_heads * n_rep, T, D) for GQA.

    The repetition is **interleaved**, not tiled: kv head `j` is duplicated
    `n_rep` times in a row, so query head `i` reads kv head `i // n_rep`.

        n_kv=3, n_rep=3  ->  kv order [0,0,0, 1,1,1, 2,2,2]
        the tiled mistake ->           [0,1,2, 0,1,2, 0,1,2]

    Both give a tensor of exactly the right shape and a model that runs. Getting
    it backwards just pairs every query head with the wrong keys.

    `expand` does not copy — it creates a stride-0 view — so the only real cost
    is the `reshape`, which has to materialize it. A production engine would
    skip this entirely and broadcast inside the matmul.
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


def causal_mask(
    q_len: int,
    kv_len: int | None = None,
    offset: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive mask of shape (1, 1, q_len, kv_len): 0 where attention is allowed.

    Additive, not boolean: it is summed into the scores *before* softmax, so
    blocked positions need a value that softmax maps to zero. We use
    `finfo.min` rather than `-inf` — matching HF — because a row that is entirely
    `-inf` softmaxes to NaN, while `finfo.min` degrades to a uniform row instead.

    `offset` is the number of tokens already in the KV cache, so query row `i`
    sits at absolute position `offset + i` and may attend to any key `j <= offset + i`.
    With the defaults this is the plain square lower-triangular mask.

    During single-token decode (q_len == 1, offset == n) every key is allowed,
    so the mask is all zeros and the caller may pass None instead.
    """
    kv_len = q_len if kv_len is None else kv_len
    q_pos = torch.arange(q_len).unsqueeze(1) + offset  # (q_len, 1)
    k_pos = torch.arange(kv_len).unsqueeze(0)  # (1, kv_len)
    blocked = k_pos > q_pos
    return torch.where(
        blocked, torch.finfo(dtype).min, torch.zeros((), dtype=dtype)
    ).to(dtype)[None, None]


class NoCache:
    """The null object: a cache-shaped thing that stores nothing.

    Exists so that caching is a *swappable strategy* rather than a branch.
    `Attention.forward` always calls `cache.update(...)`; with this one it gets
    back exactly what it passed in, `length` stays 0, and every position and
    mask calculation collapses to the uncached behaviour of step 7.

    Stateless, so a single shared instance is enough — see `NO_CACHE`.
    """

    length = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        return k, v

    def advance(self, n: int) -> None:
        pass

    def reset(self) -> None:
        pass

    def memory_bytes(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


NO_CACHE = NoCache()


class KVCache(NoCache):
    """Preallocated per-layer key/value storage for incremental decoding.

    Without this, generating token `n` recomputes the keys and values of all
    `n-1` earlier tokens — they are identical every step, because attention is
    causal and nothing in the past depends on the future. Caching turns an
    O(n^2) decode into O(n).

    Two details that matter:

      - Keys are stored **after** RoPE. Position is baked in at write time, so
        the history never needs re-rotating. Storing pre-rotation keys would
        force you to rotate the whole cache on every step, giving back the win.
      - Storage is per **kv head**, not per query head — 3 heads here, not 9.
        `repeat_kv` runs on the way out, after reading. This is where GQA
        actually pays off: the cache is a third the size it would otherwise be.

    Preallocated rather than grown by `torch.cat`, which would reallocate and
    copy the entire history on every single token.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        shape = (batch_size, cfg.num_key_value_heads, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.max_seq_len = max_seq_len
        self.length = 0  # tokens currently stored, shadowing NoCache's class attribute

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Append this step's k/v for one layer, return the full history so far.

        Does **not** advance `length` — every layer writes at the same offset,
        so the model calls `advance()` once after the whole stack.
        """
        t = k.shape[2]
        end = self.length + t
        if end > self.max_seq_len:
            raise ValueError(f"cache overflow: {end} > {self.max_seq_len}")
        self.k[layer_idx][:, :, self.length : end] = k
        self.v[layer_idx][:, :, self.length : end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def reset(self) -> None:
        self.length = 0

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def __bool__(self) -> bool:
        return True


class Attention(nn.Module):
    """Grouped-query causal self-attention.

    SmolLM2 has 9 query heads but only 3 key/value heads, so each kv head is
    shared by 3 queries (`n_rep = 3`). That is the entire idea of GQA: the
    expensive thing at inference time is not the projection but the KV cache,
    and this cuts it to a third. Compare the parameter counts — q_proj and
    o_proj are (576, 576) while k_proj and v_proj are only (192, 576).

    Shapes through the forward pass, for one batch of T tokens:

        x                (B, T, 576)
        q_proj(x)        (B, T, 576) -> view (B, T, 9, 64) -> (B, 9, T, 64)
        k_proj(x)        (B, T, 192) -> view (B, T, 3, 64) -> (B, 3, T, 64)
        after repeat_kv  (B, 9, T, 64)
        scores           (B, 9, T, T)
        out              (B, 9, T, 64) -> (B, T, 576) -> o_proj -> (B, T, 576)

    The `transpose(1, 2)` that moves heads in front of time is what makes the
    heads independent: every subsequent matmul batches over (B, n_heads).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        # 1/sqrt(d_k). Without it the scores grow with head_dim and softmax
        # saturates into a near one-hot distribution with vanishing gradients.
        self.scaling = cfg.head_dim**-0.5

        q_size = cfg.num_attention_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: NoCache = NO_CACHE,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE goes on q and k *after* the head split and *before* the scores,
        # so each head is rotated in its own 64-dim space. Never on v.
        q, k = apply_rope(q, k, cos, sin)

        # Cache the rotated k/v, then attend over the whole history. Order is
        # load-bearing: rotate first, store second. With NO_CACHE this returns
        # k and v untouched, so there is one code path either way.
        k, v = cache.update(layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling  # (B, H, T, T)
        if mask is not None:
            scores = scores + mask[:, :, :, : k.shape[-2]]

        # Softmax in fp32 regardless of input dtype: exponentials of scores that
        # differ by tens of units overflow bf16 long before fp32 notices.
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(weights, v)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(b, t, -1)  # heads back behind time
        return self.o_proj(out)

    @classmethod
    def from_checkpoint(cls, cfg: ModelConfig, weights: Weights, layer_idx: int) -> Attention:
        attn = cls(cfg)
        prefix = f"model.layers.{layer_idx}.self_attn"
        with torch.no_grad():
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                # Straight copy, no transpose — and note that q_proj and o_proj
                # are both (576, 576), so a stray .T here would NOT raise.
                getattr(attn, name).weight.copy_(weights.get(f"{prefix}.{name}.weight"))
        return attn
