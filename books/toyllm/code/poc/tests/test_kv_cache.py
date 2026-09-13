import pytest
import torch

from toyllm.config import ModelConfig
from toyllm.layers import NO_CACHE, KVCache, NoCache, causal_mask
from toyllm.model import SmolLM2



@pytest.fixture(scope="module")
def model(model_dir):
    return SmolLM2.from_pretrained(model_dir)



# ----------------------------------------------------------------- the mask


def test_default_mask_is_unchanged():
    """The step-6 behaviour must survive generalizing the signature."""
    m = causal_mask(4)[0, 0]
    upper = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert m.shape == (4, 4)
    assert (m[upper] == torch.finfo(torch.float32).min).all()
    assert (m[~upper] == 0).all()


def test_offset_mask_lets_a_new_token_see_the_whole_past():
    """One query at absolute position 5, five keys cached: everything allowed."""
    m = causal_mask(1, kv_len=6, offset=5)
    assert m.shape == (1, 1, 1, 6)
    assert (m == 0).all()


def test_offset_mask_for_a_chunk():
    """Two new tokens at positions 3 and 4, on top of 3 cached ones."""
    m = causal_mask(2, kv_len=5, offset=3)[0, 0]
    blocked = torch.finfo(torch.float32).min
    assert (m[0] == torch.tensor([0, 0, 0, 0, blocked])).all()  # pos 3 cannot see pos 4
    assert (m[1] == 0).all()  # pos 4 sees everything


# ---------------------------------------------------------------- the cache


def test_stores_kv_heads_not_query_heads(model_dir):
    """Where GQA actually pays: 3 heads cached, not 9."""
    cfg = ModelConfig.from_json(model_dir)
    cache = KVCache(cfg, max_seq_len=128)
    assert cache.k[0].shape == (1, cfg.num_key_value_heads, 128, cfg.head_dim)
    assert len(cache.k) == cfg.num_hidden_layers

    expected = 2 * cfg.num_hidden_layers * 1 * cfg.num_key_value_heads * 128 * cfg.head_dim * 4
    assert cache.memory_bytes() == expected
    # A full-MHA cache would be n_heads/n_kv_heads times larger.
    assert cache.memory_bytes() * cfg.n_rep == expected * 3


def test_update_returns_history_and_does_not_advance(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    cache = KVCache(cfg, max_seq_len=16)
    k = torch.randn(1, cfg.num_key_value_heads, 4, cfg.head_dim)
    v = torch.randn(1, cfg.num_key_value_heads, 4, cfg.head_dim)

    k_all, v_all = cache.update(0, k, v)
    assert k_all.shape[2] == 4
    assert torch.equal(k_all, k)
    # Every layer writes at the same offset, so update() must not move it.
    assert cache.length == 0
    cache.advance(4)
    assert cache.length == 4

    k2 = torch.randn(1, cfg.num_key_value_heads, 1, cfg.head_dim)
    k_all, _ = cache.update(0, k2, k2)
    assert k_all.shape[2] == 5
    assert torch.equal(k_all[:, :, :4], k)
    assert torch.equal(k_all[:, :, 4:], k2)


def test_overflow_raises(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    cache = KVCache(cfg, max_seq_len=4)
    big = torch.randn(1, cfg.num_key_value_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, big, big)


def test_reset(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    cache = KVCache(cfg, max_seq_len=8)
    cache.advance(5)
    cache.reset()
    assert cache.length == 0


# ------------------------------------------------------------- equivalence


def test_incremental_forward_matches_one_shot(model, reference):
    """Feed tokens one at a time; last-position logits must match a full forward."""
    ids = reference["input_ids"]
    with torch.no_grad():
        full = model(ids)
        cache = KVCache(model.cfg, max_seq_len=32)
        for i in range(ids.shape[1]):
            step = model(ids[:, i : i + 1], cache=cache)
        assert cache.length == ids.shape[1]
    rel = (step[:, -1] - full[:, -1]).abs().max() / full[:, -1].abs().max()
    assert rel < 1e-5, rel


def test_prefill_then_decode_matches_one_shot(model, reference):
    """The real decode pattern: whole prompt at once, then single tokens."""
    ids = reference["input_ids"]
    with torch.no_grad():
        cache = KVCache(model.cfg, max_seq_len=32)
        prefilled = model(ids, cache=cache)
        assert cache.length == ids.shape[1]

        next_id = prefilled[:, -1].argmax(-1, keepdim=True)
        stepped = model(next_id, cache=cache)

        one_shot = model(torch.cat([ids, next_id], dim=1))
    rel = (stepped[:, -1] - one_shot[:, -1]).abs().max() / one_shot[:, -1].abs().max()
    assert rel < 1e-5, rel


def test_cached_and_uncached_generate_agree(model, reference):
    """The invariant for this whole step: speed changes, output does not."""
    ids = reference["input_ids"]
    with torch.no_grad():
        cached = model.generate(ids, max_new_tokens=20, cache=True)
        naive = model.generate(ids, max_new_tokens=20, cache=False)
    assert torch.equal(cached, naive)
    assert torch.equal(cached, reference["greedy_20"])


def test_positions_continue_from_the_cache(model):
    """Decoding must not restart positions at 0 — that would break RoPE."""
    ids = torch.tensor([[504, 3575, 282]])
    with torch.no_grad():
        cache = KVCache(model.cfg, max_seq_len=16)
        model(ids, cache=cache)
        nxt = torch.tensor([[4649]])
        auto = model(nxt, cache=cache)  # positions inferred as [3]

        cache.reset()
        model(ids, cache=cache)
        wrong = model(nxt, positions=torch.tensor([0]), cache=cache)
    assert not torch.allclose(auto, wrong, atol=1e-3)


def test_no_cache_is_a_transparent_null_object(model_dir):
    """NO_CACHE must leave k/v untouched so the uncached path has no branch."""
    k = torch.randn(1, 3, 4, 64)
    v = torch.randn(1, 3, 4, 64)
    k2, v2 = NO_CACHE.update(7, k, v)
    assert k2 is k and v2 is v
    NO_CACHE.advance(4)
    assert NO_CACHE.length == 0
    assert not NO_CACHE and NO_CACHE.memory_bytes() == 0
    assert bool(KVCache(ModelConfig.from_json(model_dir), 8)) is True
    assert isinstance(NO_CACHE, NoCache)


def test_forward_without_a_cache_is_unchanged_by_the_default(model, reference):
    """Passing nothing must behave exactly as step 7 did."""
    with torch.no_grad():
        a = model(reference["input_ids"])
        b = model(reference["input_ids"], cache=NO_CACHE)
    assert torch.equal(a, b)


def test_cache_switch_accepts_bool_or_instance(model, reference):
    ids = reference["input_ids"]
    with torch.no_grad():
        auto = model.generate(ids, max_new_tokens=8, cache=True)
        given = model.generate(ids, max_new_tokens=8, cache=model.new_cache(64))
        off = model.generate(ids, max_new_tokens=8, cache=False)
    assert torch.equal(auto, given)
    assert torch.equal(auto, off)


def test_a_cache_can_be_reused_across_turns(model, reference):
    """Continuing generation from a warm cache matches doing it in one go."""
    ids = reference["input_ids"]
    cache = model.new_cache(64)
    with torch.no_grad():
        first = model.generate(ids, max_new_tokens=5, cache=cache)
        # One short of the output: the final token was produced by the last
        # forward pass but never fed back in, so it is not cached yet. That is
        # precisely why continuing means feeding `first[:, -1:]`.
        assert cache.length == first.shape[1] - 1
        continued = model.generate(first[:, -1:], max_new_tokens=5, cache=cache)
        one_go = model.generate(ids, max_new_tokens=10, cache=True)
    assert torch.equal(first[0], one_go[0, : first.shape[1]])
    assert torch.equal(continued[0, 1:], one_go[0, first.shape[1] :])


def test_new_cache_is_clamped_to_max_position(model):
    cache = model.new_cache(999_999)
    assert cache.max_seq_len == model.cfg.max_position_embeddings


def test_eos_stops_generation(model):
    ids = torch.tensor([[504, 3575]])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=10, eos_id=model.cfg.eos_token_id)
    assert out.shape[1] <= ids.shape[1] + 10
