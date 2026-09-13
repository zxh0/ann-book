import pytest
import torch

from toyllm.config import ModelConfig
from toyllm.layers import RMSNorm
from toyllm.weights import Weights

EPS = 1e-5
H = 576


@pytest.fixture
def norm():
    torch.manual_seed(0)
    n = RMSNorm(H, EPS)
    with torch.no_grad():
        n.weight.copy_(torch.randn(H) * 0.2 + 1.0)
    return n


def test_matches_explicit_formula(norm):
    """Check against the definition written out longhand, not a refactor of it."""
    x = torch.randn(3, 7, H)
    rms = torch.sqrt((x * x).sum(-1, keepdim=True) / H + EPS)
    expected = x / rms * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_output_rms_is_one_before_the_gain():
    x = torch.randn(4, H) * 17.0
    plain = RMSNorm(H, EPS)  # weight is all ones
    y = plain(x)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-4)


def test_scale_invariance(norm):
    x = torch.randn(4, H)
    assert torch.allclose(norm(x), norm(x * 1000.0), atol=1e-4)


def test_does_not_recenter(norm):
    """The defining difference from LayerNorm: the mean is left alone."""
    x = torch.randn(4, H) + 5.0
    assert norm(x).mean(-1).abs().min() > 0.1


def test_normalizes_last_axis_independently(norm):
    """Each token is normalized on its own; batching must not couple them."""
    x = torch.randn(5, H)
    stacked = norm(x)
    for i in range(5):
        assert torch.allclose(stacked[i], norm(x[i]), atol=1e-6)


def test_eps_is_inside_the_sqrt():
    """A near-zero input distinguishes the two placements."""
    tiny = torch.full((1, H), 1e-8)
    inside = RMSNorm(H, EPS)(tiny)
    v = tiny.pow(2).mean(-1, keepdim=True)
    outside = tiny * (torch.rsqrt(v) + EPS)
    assert not torch.allclose(inside, outside, atol=1e-3)
    assert torch.isfinite(inside).all()


def test_upcasts_bf16_internally():
    x = (torch.randn(2, H) * 40).to(torch.bfloat16)
    out = RMSNorm(H, EPS).to(torch.bfloat16)(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_matches_huggingface(model_dir, llama_ref):
    """The real check: identical to the implementation the weights were trained with."""
    llama = llama_ref
    cfg = ModelConfig.from_json(model_dir)
    real = Weights(model_dir, cfg).get("model.layers.0.input_layernorm.weight")

    ours = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    theirs = llama.LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        ours.weight.copy_(real)
        theirs.weight.copy_(real)

    torch.manual_seed(0)
    for shape in [(cfg.hidden_size,), (4, cfg.hidden_size), (2, 8, cfg.hidden_size)]:
        for scale in [1e-4, 1.0, 100.0]:
            x = torch.randn(*shape) * scale
            assert torch.equal(ours(x), theirs(x)), f"{shape} @ {scale}"
