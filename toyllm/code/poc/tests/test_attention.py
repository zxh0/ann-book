import math

import torch

from toyllm.config import ModelConfig
from toyllm.layers import Attention, RotaryEmbedding, causal_mask, repeat_kv
from toyllm.weights import Weights


def hf_config(cfg):
    from transformers.models.llama.configuration_llama import LlamaConfig

    c = LlamaConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        attention_bias=False,
        attention_dropout=0.0,
    )
    c._attn_implementation = "eager"
    return c


def test_repeat_kv_interleaves_rather_than_tiles():
    """kv head j serves query heads [j*n_rep, (j+1)*n_rep) — not a round-robin."""
    x = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1)  # 3 kv heads, ids 0,1,2
    out = repeat_kv(x, 3).flatten().tolist()
    assert out == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert out != [0, 1, 2, 0, 1, 2, 0, 1, 2]  # the tiled mistake


def test_repeat_kv_is_identity_when_n_rep_is_one():
    x = torch.randn(2, 4, 5, 8)
    assert repeat_kv(x, 1) is x


def test_causal_mask_shape_and_values():
    m = causal_mask(4)[0, 0]
    assert m.shape == (4, 4)
    blocked = torch.finfo(torch.float32).min
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert (m[upper] == blocked).all()           # strictly above the diagonal: blocked
    assert (m[~upper] == 0).all()                # diagonal and below: allowed
    assert torch.isfinite(m).all()               # finfo.min, not -inf


def test_masked_softmax_never_produces_nan():
    """Why finfo.min beats -inf: a fully-masked row must not become NaN."""
    scores = torch.randn(1, 1, 3, 3) + causal_mask(3)
    w = torch.softmax(scores, dim=-1, dtype=torch.float32)
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(1, 1, 3), atol=1e-5)


def test_scaling_is_inverse_sqrt_head_dim(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    assert math.isclose(Attention(cfg).scaling, 1.0 / math.sqrt(cfg.head_dim))
    assert math.isclose(Attention(cfg).scaling, 0.125)  # 1/sqrt(64)


def test_projection_shapes_and_no_bias(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    a = Attention(cfg)
    assert a.q_proj.weight.shape == (576, 576)
    assert a.k_proj.weight.shape == (192, 576)   # kv_size, not hidden_size
    assert a.v_proj.weight.shape == (192, 576)
    assert a.o_proj.weight.shape == (576, 576)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(a, name).bias is None


def test_is_actually_causal(model_dir):
    """The real test: changing a future token must not affect earlier outputs."""
    cfg = ModelConfig.from_json(model_dir)
    a = Attention.from_checkpoint(cfg, Weights(model_dir, cfg), 0)
    rope = RotaryEmbedding.from_config(cfg)
    torch.manual_seed(0)
    T = 8
    x = torch.randn(1, T, cfg.hidden_size)
    cos, sin = rope(torch.arange(T))
    m = causal_mask(T)

    with torch.no_grad():
        base = a(x, cos, sin, m)
        perturbed = x.clone()
        perturbed[0, 5:] = torch.randn(T - 5, cfg.hidden_size)  # rewrite the future
        after = a(perturbed, cos, sin, m)

    assert torch.allclose(base[0, :5], after[0, :5], atol=1e-6)   # past unchanged
    assert not torch.allclose(base[0, 5:], after[0, 5:], atol=1e-3)  # future did change


def test_without_a_mask_it_leaks_the_future(model_dir):
    """Sanity: the causality above comes from the mask, not from luck."""
    cfg = ModelConfig.from_json(model_dir)
    a = Attention.from_checkpoint(cfg, Weights(model_dir, cfg), 0)
    rope = RotaryEmbedding.from_config(cfg)
    torch.manual_seed(0)
    T = 8
    x = torch.randn(1, T, cfg.hidden_size)
    cos, sin = rope(torch.arange(T))
    with torch.no_grad():
        base = a(x, cos, sin, None)
        perturbed = x.clone()
        perturbed[0, 5:] = torch.randn(T - 5, cfg.hidden_size)
        after = a(perturbed, cos, sin, None)
    assert not torch.allclose(base[0, :5], after[0, :5], atol=1e-3)


def test_query_head_reads_its_own_kv_group(model_dir):
    """Zeroing kv head j must disturb exactly query heads [3j, 3j+3)."""
    cfg = ModelConfig.from_json(model_dir)
    a = Attention.from_checkpoint(cfg, Weights(model_dir, cfg), 0)
    rope = RotaryEmbedding.from_config(cfg)
    torch.manual_seed(0)
    T = 6
    x = torch.randn(1, T, cfg.hidden_size)
    cos, sin = rope(torch.arange(T))
    d, n_rep = cfg.head_dim, cfg.n_rep

    def per_head_out(model):
        """Run attention but stop before o_proj, so heads stay separable."""
        q = model.q_proj(x).view(1, T, cfg.num_attention_heads, d).transpose(1, 2)
        k = model.k_proj(x).view(1, T, cfg.num_key_value_heads, d).transpose(1, 2)
        v = model.v_proj(x).view(1, T, cfg.num_key_value_heads, d).transpose(1, 2)
        from toyllm.layers import apply_rope

        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        s = torch.matmul(q, k.transpose(2, 3)) * model.scaling + causal_mask(T)
        return torch.matmul(torch.softmax(s, -1, dtype=torch.float32), v)

    with torch.no_grad():
        base = per_head_out(a)
        import copy

        damaged = copy.deepcopy(a)
        damaged.v_proj.weight[d : 2 * d] = 0.0  # kv head 1
        after = per_head_out(damaged)

    diff = (base - after).abs().amax(dim=(0, 2, 3))  # per query head
    changed = {i for i in range(cfg.num_attention_heads) if diff[i] > 1e-6}
    assert changed == {3, 4, 5}, changed  # exactly kv head 1's group


def test_matches_huggingface(model_dir, llama_ref):
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    ours = Attention.from_checkpoint(cfg, w, 0)

    hf_cfg = hf_config(cfg)
    theirs = llama_ref.LlamaAttention(hf_cfg, layer_idx=0)
    theirs.eval()
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(theirs, name).weight.copy_(getattr(ours, name).weight)

    rope = RotaryEmbedding.from_config(cfg)
    hf_rope = llama_ref.LlamaRotaryEmbedding(hf_cfg)

    # Relative, not absolute: the disagreement is float32 rounding inherited
    # from the cos/sin table (see test_rope.py), so it scales with the output
    # magnitude — at T=128 the outputs reach ~40 and the absolute gap ~1.5e-5,
    # while the relative gap stays at ~4e-7. A wrong GQA grouping or RoPE
    # convention is ~1e0 relative, five orders of magnitude away.
    torch.manual_seed(0)
    for b, t in [(1, 1), (1, 7), (2, 12), (3, 33), (1, 128)]:
        for start in (0, 1000, 4000, 8000):
            x = torch.randn(b, t, cfg.hidden_size)
            pos = torch.arange(start, start + t)
            cos, sin = rope(pos)
            ch, sh = hf_rope(x, pos.unsqueeze(0).expand(b, -1))
            m = causal_mask(t)
            with torch.no_grad():
                o1 = ours(x, cos, sin, m)
                o2, _ = theirs(x, position_embeddings=(ch, sh), attention_mask=m)
            rel = (o1 - o2).abs().max() / o1.abs().max()
            assert rel < 1e-5, f"B={b} T={t} start={start}: relative {rel:.2e}"


def test_matches_huggingface_at_offset_positions(model_dir, llama_ref):
    """Positions that do not start at zero — what decoding with a cache looks like."""
    cfg = ModelConfig.from_json(model_dir)
    ours = Attention.from_checkpoint(cfg, Weights(model_dir, cfg), 0)
    hf_cfg = hf_config(cfg)
    theirs = llama_ref.LlamaAttention(hf_cfg, layer_idx=0)
    theirs.eval()
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(theirs, name).weight.copy_(getattr(ours, name).weight)

    rope = RotaryEmbedding.from_config(cfg)
    hf_rope = llama_ref.LlamaRotaryEmbedding(hf_cfg)
    torch.manual_seed(0)
    t, start = 9, 4000
    x = torch.randn(1, t, cfg.hidden_size)
    pos = torch.arange(start, start + t)
    cos, sin = rope(pos)
    ch, sh = hf_rope(x, pos.unsqueeze(0))
    m = causal_mask(t)
    with torch.no_grad():
        o1 = ours(x, cos, sin, m)
        o2 = theirs(x, position_embeddings=(ch, sh), attention_mask=m)[0]
    assert (o1 - o2).abs().max() / o1.abs().max() < 1e-5
