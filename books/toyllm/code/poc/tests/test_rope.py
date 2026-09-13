import math

import torch

from toyllm.config import ModelConfig
from toyllm.layers import RotaryEmbedding, apply_rope, rotate_half

D, THETA, MAXPOS = 64, 100000.0, 512


def rope():
    return RotaryEmbedding(D, THETA, MAXPOS)


def test_inv_freq_formula():
    r = rope()
    for i in range(D // 2):
        assert math.isclose(r.inv_freq[i].item(), THETA ** (-2 * i / D), rel_tol=1e-6)
    assert math.isclose(r.inv_freq[0].item(), 1.0)


def test_rotate_half_is_a_half_split_not_interleaved():
    x = torch.arange(8, dtype=torch.float32)
    # pairs (0,4) (1,5) (2,6) (3,7) — NOT (0,1) (2,3) ...
    assert torch.equal(rotate_half(x), torch.tensor([-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]))


def test_rotate_half_is_a_quarter_turn():
    """Applying it four times returns the original."""
    x = torch.randn(3, D)
    assert torch.allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x, atol=1e-6)
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)


def test_position_zero_is_identity():
    r = rope()
    q = torch.randn(1, 2, 1, D)
    cos, sin = r(torch.tensor([0]))
    out, _ = apply_rope(q, q, cos, sin)
    assert torch.allclose(out, q, atol=1e-6)


def test_rotation_preserves_norm():
    """A rotation cannot change a vector's length — a good sanity check."""
    r = rope()
    q = torch.randn(2, 3, 17, D)
    cos, sin = r(torch.arange(17))
    out, _ = apply_rope(q, q, cos, sin)
    assert torch.allclose(out.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_cos_sin_are_consistent():
    r = rope()
    cos, sin = r(torch.arange(MAXPOS))
    assert torch.allclose(cos**2 + sin**2, torch.ones_like(cos), atol=1e-5)
    # The duplicated layout: angle for dim i equals angle for dim i + D/2.
    assert torch.allclose(cos[:, : D // 2], cos[:, D // 2 :], atol=1e-6)


def test_attention_score_depends_only_on_relative_position():
    """The whole point of RoPE. Shifting both positions leaves the score alone."""
    r = rope()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def score(m, n):
        cq, sq = r(torch.tensor([m]))
        ck, sk = r(torch.tensor([n]))
        qr = q * cq + rotate_half(q) * sq
        kr = k * ck + rotate_half(k) * sk
        return (qr * kr).sum().item()

    base = score(7, 3)
    for shift in (1, 5, 40, 200):
        assert math.isclose(score(7 + shift, 3 + shift), base, rel_tol=1e-4), shift
    # ...and a different offset gives a genuinely different score.
    assert not math.isclose(score(7, 4), base, rel_tol=1e-3)


def test_theta_changes_the_result():
    """rope_theta=100000 is not interchangeable with the usual 10000."""
    q = torch.randn(1, 1, 6, D)
    pos = torch.arange(6)
    a_cos, a_sin = RotaryEmbedding(D, 100000.0, 64)(pos)
    b_cos, b_sin = RotaryEmbedding(D, 10000.0, 64)(pos)
    ra, _ = apply_rope(q, q, a_cos, a_sin)
    rb, _ = apply_rope(q, q, b_cos, b_sin)
    assert not torch.allclose(ra, rb, atol=1e-3)


def test_interleaved_convention_gives_different_numbers():
    """Both conventions are valid rotations; using the wrong one is silent."""
    def rotate_interleaved(x):  # RoFormer paper style: pairs (0,1), (2,3), ...
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    x = torch.randn(4, D)
    assert not torch.allclose(rotate_half(x), rotate_interleaved(x), atol=1e-3)
    # Both preserve norm, so norm checks cannot tell them apart.
    assert torch.allclose(rotate_half(x).norm(dim=-1), rotate_interleaved(x).norm(dim=-1), atol=1e-4)


def test_broadcasts_over_batch_and_heads():
    r = rope()
    q = torch.randn(3, 9, 5, D)
    k = torch.randn(3, 3, 5, D)  # fewer kv heads, as in GQA
    cos, sin = r(torch.arange(5))
    qo, ko = apply_rope(q, k, cos, sin)
    assert qo.shape == q.shape and ko.shape == k.shape
    # Same rotation for every head and batch element.
    single, _ = apply_rope(q[0:1, 0:1], q[0:1, 0:1], cos, sin)
    assert torch.allclose(qo[0, 0], single[0, 0], atol=1e-6)


def test_matches_huggingface(model_dir, llama_ref):
    from transformers.models.llama.configuration_llama import LlamaConfig

    cfg = ModelConfig.from_json(model_dir)
    ours = RotaryEmbedding.from_config(cfg)
    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    theirs = llama_ref.LlamaRotaryEmbedding(hf_cfg)
    assert torch.equal(ours.inv_freq, theirs.inv_freq)

    torch.manual_seed(0)
    B, T = 2, 13
    q = torch.randn(B, cfg.num_attention_heads, T, cfg.head_dim)
    k = torch.randn(B, cfg.num_key_value_heads, T, cfg.head_dim)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)

    cos_hf, sin_hf = theirs(q, pos)
    cos_our, sin_our = ours(torch.arange(T))
    # Not bit-exact: we read cos/sin from a precomputed 8192-row table while HF
    # computes them per call, and torch.cos() differs by ~1 ulp between the
    # vectorized and scalar kernels. See test_cache_differs_from_hf_by_one_ulp.
    assert torch.allclose(cos_our, cos_hf[0], atol=1e-6)
    assert torch.allclose(sin_our, sin_hf[0], atol=1e-6)

    q_hf, k_hf = llama_ref.apply_rotary_pos_emb(q, k, cos_hf, sin_hf)
    q_our, k_our = apply_rope(q, k, cos_our, sin_our)
    assert torch.allclose(q_our, q_hf, atol=1e-5)
    assert torch.allclose(k_our, k_hf, atol=1e-5)


def test_cache_differs_from_hf_by_at_most_one_ulp(model_dir, llama_ref):
    """Pin down the only source of disagreement with HF, so it can't hide a bug.

    The angles are bit-identical; `torch.cos` over the big precomputed table
    differs from `torch.cos` over a small slice by at most one float32 ulp.
    If this ever exceeds ~1e-6 the cause is something else and worth chasing.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig

    cfg = ModelConfig.from_json(model_dir)
    ours = RotaryEmbedding.from_config(cfg)
    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    theirs = llama_ref.LlamaRotaryEmbedding(hf_cfg)

    pos = torch.arange(4000, 4064)
    dummy = torch.zeros(1, 1, pos.shape[0], cfg.head_dim)
    cos_hf, _ = theirs(dummy, pos.unsqueeze(0))
    cos_our, _ = ours(pos)

    diff = (cos_our - cos_hf[0]).abs().max().item()
    assert 0 < diff <= 2 * torch.finfo(torch.float32).eps, diff


def test_matches_huggingface_at_long_positions(model_dir, llama_ref):
    """Late positions are where a wrong theta or dtype would first show up."""
    from transformers.models.llama.configuration_llama import LlamaConfig

    cfg = ModelConfig.from_json(model_dir)
    ours = RotaryEmbedding.from_config(cfg)
    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    theirs = llama_ref.LlamaRotaryEmbedding(hf_cfg)

    pos = torch.tensor([[0, 1, 1000, 4095, 8191]])
    dummy = torch.zeros(1, 1, pos.shape[1], cfg.head_dim)
    cos_hf, sin_hf = theirs(dummy, pos)
    cos_our, sin_our = ours(pos[0])
    assert torch.allclose(cos_our, cos_hf[0], atol=1e-6)
    assert torch.allclose(sin_our, sin_hf[0], atol=1e-6)
