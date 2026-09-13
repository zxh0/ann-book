import math

import pytest
import torch

from toyllm.sampling import (
    GREEDY,
    GreedySampler,
    RandomSampler,
    Sampler,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(2, 200) * 3


# --------------------------------------------------------------- temperature


def test_temperature_scales_gaps():
    x = torch.tensor([[1.0, 2.0, 4.0]])
    assert torch.equal(apply_temperature(x, 2.0), x / 2)
    assert torch.equal(apply_temperature(x, 1.0), x)


def test_low_temperature_sharpens_high_flattens():
    x = torch.tensor([[1.0, 2.0, 4.0]])
    sharp = apply_temperature(x, 0.5).softmax(-1)
    flat = apply_temperature(x, 5.0).softmax(-1)
    assert sharp.max() > x.softmax(-1).max() > flat.max()
    # Flattening moves toward uniform.
    assert flat.max() - flat.min() < sharp.max() - sharp.min()


def test_temperature_zero_builds_a_greedy_sampler():
    """The greedy/random choice is made once at construction, not per token."""
    s = Sampler.build(temperature=0.0)
    assert isinstance(s, GreedySampler)
    assert s.is_greedy
    x = torch.tensor([[1.0, 9.0, 3.0]])
    assert s(x).item() == 1
    assert GREEDY.is_greedy


def test_build_returns_a_random_sampler_above_zero():
    s = Sampler.build(temperature=0.7, top_k=10, top_p=0.9)
    assert isinstance(s, RandomSampler)
    assert not s.is_greedy


def test_random_sampler_rejects_zero_temperature():
    """RandomSampler has no greedy branch, so it must refuse T=0 outright."""
    with pytest.raises(ValueError, match="GreedySampler"):
        RandomSampler(temperature=0.0)


def test_both_are_samplers_and_share_the_interface():
    x = torch.randn(3, 40)
    for s in (GreedySampler(), RandomSampler(temperature=1.0, seed=0)):
        assert isinstance(s, Sampler)
        assert s(x).shape == (3, 1)


def test_a_custom_sampler_just_needs_call():
    """The point of the base class: generate() accepts anything with __call__."""

    class AlwaysToken7(Sampler):
        def __call__(self, logits):
            return torch.full((logits.shape[0], 1), 7, dtype=torch.long)

    assert AlwaysToken7()(torch.randn(2, 50)).tolist() == [[7], [7]]


def test_negative_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        Sampler.build(temperature=-1.0)


def test_invalid_top_p_rejected():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="top_p"):
            Sampler.build(top_p=bad)


# --------------------------------------------------------------------- top-k


def test_top_k_keeps_exactly_k():
    x = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    out = apply_top_k(x, 2)
    assert torch.isfinite(out).sum() == 2
    assert out[0, 0] == 5.0 and out[0, 2] == 4.0


def test_top_k_off_when_zero_or_too_large():
    x = torch.randn(1, 10)
    assert torch.equal(apply_top_k(x, 0), x)
    assert torch.equal(apply_top_k(x, 10), x)
    assert torch.equal(apply_top_k(x, 99), x)


def test_top_k_removed_tokens_get_zero_probability():
    x = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    probs = apply_top_k(x, 2).softmax(-1)
    assert probs[0, 1] == 0.0
    assert math.isclose(probs.sum().item(), 1.0, rel_tol=1e-6)


# --------------------------------------------------------------------- top-p


def test_top_p_keeps_the_boundary_token():
    """Probabilities .6/.3/.1: p=0.7 must keep two tokens, not one."""
    probs = torch.tensor([[0.6, 0.3, 0.1]])
    x = probs.log()
    kept = torch.isfinite(apply_top_p(x, 0.7)).sum()
    assert kept == 2, "the token straddling the threshold belongs in the nucleus"


def test_top_p_is_adaptive():
    """A peaked distribution keeps fewer tokens than a flat one at the same p."""
    peaked = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    flat = torch.zeros(1, 6)
    assert torch.isfinite(apply_top_p(peaked, 0.9)).sum() < torch.isfinite(
        apply_top_p(flat, 0.9)
    ).sum()


def test_top_p_never_empties_the_nucleus():
    """Even a tiny p must leave at least one token to sample."""
    x = torch.tensor([[10.0, 1.0, 1.0]])
    out = apply_top_p(x, 1e-9)
    assert torch.isfinite(out).sum() >= 1


def test_top_p_off_at_one():
    x = torch.randn(1, 10)
    assert torch.equal(apply_top_p(x, 1.0), x)


# ------------------------------------------------- exact agreement with HF


def test_filters_match_huggingface(logits, llama_ref):
    """The deterministic half is directly comparable — so compare it exactly."""
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    dummy = torch.zeros(logits.shape[0], 1, dtype=torch.long)
    for temp in (0.7, 1.0, 1.5):
        assert torch.equal(
            apply_temperature(logits, temp),
            TemperatureLogitsWarper(temp)(dummy, logits.clone()),
        )
    for k in (1, 5, 50):
        assert torch.equal(
            apply_top_k(logits, k),
            TopKLogitsWarper(k)(dummy, logits.clone()),
        ), k
    for p in (0.1, 0.5, 0.9, 0.95):
        assert torch.equal(
            apply_top_p(logits, p),
            TopPLogitsWarper(p)(dummy, logits.clone()),
        ), p


def test_combined_filter_matches_huggingface_pipeline(logits, llama_ref):
    """Order matters: temperature, then top-k, then top-p."""
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    dummy = torch.zeros(logits.shape[0], 1, dtype=torch.long)
    ours = Sampler.build(temperature=0.8, top_k=40, top_p=0.95).filter(logits)
    theirs = LogitsProcessorList(
        [TemperatureLogitsWarper(0.8), TopKLogitsWarper(40), TopPLogitsWarper(0.95)]
    )(dummy, logits.clone())
    assert torch.equal(ours, theirs)


def test_filter_order_is_not_arbitrary(logits):
    """Applying temperature last would change which tokens survive top-p."""
    temp_first = apply_top_p(apply_temperature(logits, 0.5), 0.9)
    temp_last = apply_temperature(apply_top_p(logits, 0.9), 0.5)
    assert torch.isfinite(temp_first).sum() != torch.isfinite(temp_last).sum()


# ----------------------------------------------- the random half, statistically


def test_seeded_sampling_is_reproducible():
    x = torch.randn(1, 100)
    a = [Sampler.build(temperature=1.0, seed=7)(x).item() for _ in range(20)]
    b = [Sampler.build(temperature=1.0, seed=7)(x).item() for _ in range(20)]
    assert a == b


def test_different_seeds_diverge():
    x = torch.randn(1, 1000)
    a = [Sampler.build(temperature=1.0, seed=1)(x).item() for _ in range(30)]
    b = [Sampler.build(temperature=1.0, seed=2)(x).item() for _ in range(30)]
    assert a != b


def test_empirical_frequencies_converge_to_the_distribution():
    """The draw cannot be checked exactly, so check that it is unbiased."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    target = logits.softmax(-1)[0]
    sampler = Sampler.build(temperature=1.0, seed=0)

    n = 40_000
    counts = torch.zeros(4)
    batch = logits.expand(n, -1)
    draws = sampler(batch).flatten()
    counts.scatter_add_(0, draws, torch.ones(n))
    empirical = counts / n

    assert torch.allclose(empirical, target, atol=0.01), (empirical, target)


def test_top_k_actually_restricts_what_is_drawn():
    logits = torch.arange(50, dtype=torch.float32).unsqueeze(0)
    sampler = Sampler.build(temperature=1.0, top_k=3, seed=0)
    drawn = {sampler(logits.expand(500, -1)).flatten().unique().tolist()[i] for i in range(3)}
    assert drawn == {47, 48, 49}


def test_greedy_sampler_equals_argmax_on_real_logits():
    torch.manual_seed(0)
    logits = torch.randn(4, 1000)
    assert torch.equal(GREEDY(logits), logits.argmax(-1, keepdim=True))


# ------------------------------------------------------- inside generate()


@pytest.fixture(scope="module")
def model(model_dir):
    from toyllm.model import SmolLM2

    return SmolLM2.from_pretrained(model_dir)


def test_default_generate_is_still_greedy(model, model_dir):
    """Adding a sampler must not change the default behaviour."""
    ids = torch.tensor([[504, 3575, 282, 4649, 314]])
    with torch.no_grad():
        default = model.generate(ids, max_new_tokens=10)
        explicit = model.generate(ids, max_new_tokens=10, sampler=GREEDY)
        zero_temp = model.generate(
            ids, max_new_tokens=10, sampler=Sampler.build(temperature=0.0)
        )
    assert torch.equal(default, explicit)
    assert torch.equal(default, zero_temp)


def test_seeded_generation_is_reproducible(model):
    ids = torch.tensor([[504, 3575, 282, 4649, 314]])
    with torch.no_grad():
        a = model.generate(ids, 12, sampler=Sampler.build(temperature=0.9, top_p=0.95, seed=42))
        b = model.generate(ids, 12, sampler=Sampler.build(temperature=0.9, top_p=0.95, seed=42))
    assert torch.equal(a, b)


def test_sampling_can_differ_from_greedy(model):
    """A high temperature should eventually leave the greedy path."""
    ids = torch.tensor([[504, 3575, 282, 4649, 314]])
    with torch.no_grad():
        greedy = model.generate(ids, 20)
        sampled = model.generate(ids, 20, sampler=Sampler.build(temperature=1.5, seed=0))
    assert not torch.equal(greedy, sampled)


def test_tiny_temperature_converges_to_greedy(model):
    """As T -> 0 sampling collapses onto the argmax path."""
    ids = torch.tensor([[504, 3575, 282, 4649, 314]])
    with torch.no_grad():
        greedy = model.generate(ids, 10)
        nearly = model.generate(ids, 10, sampler=Sampler.build(temperature=0.01, seed=0))
    assert torch.equal(greedy, nearly)


def test_sampler_works_with_and_without_cache(model):
    ids = torch.tensor([[504, 3575, 282]])
    with torch.no_grad():
        for use in (True, False):
            out = model.generate(
                ids, 5, cache=use, sampler=Sampler.build(temperature=0.8, top_k=20, seed=3)
            )
            assert out.shape == (1, ids.shape[1] + 5)
