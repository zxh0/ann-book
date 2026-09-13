import torch
import torch.nn.functional as F

from toyllm.config import ModelConfig
from toyllm.layers import SwiGLU
from toyllm.weights import Weights

H, FFN = 576, 1536


def build(seed=0):
    torch.manual_seed(seed)
    m = SwiGLU(H, FFN)
    with torch.no_grad():
        for lin in (m.gate_proj, m.up_proj, m.down_proj):
            lin.weight.normal_(0, 0.05)
    return m


def test_matches_explicit_formula():
    m = build()
    x = torch.randn(3, 7, H)
    gate = x @ m.gate_proj.weight.T
    up = x @ m.up_proj.weight.T
    expected = (F.silu(gate) * up) @ m.down_proj.weight.T
    assert torch.allclose(m(x), expected, atol=1e-5)


def test_silu_definition():
    z = torch.randn(1000)
    assert torch.allclose(F.silu(z), z * torch.sigmoid(z), atol=1e-6)


def test_gate_can_go_negative():
    """SiLU is not a 0..1 gate — it dips below zero, so it can flip a sign."""
    z = torch.linspace(-6, 6, 20001)
    s = F.silu(z)
    assert s.min() < -0.27
    assert torch.isclose(z[s.argmin()], torch.tensor(-1.278), atol=1e-2)


def test_no_bias_anywhere():
    m = build()
    assert m.gate_proj.bias is None
    assert m.up_proj.bias is None
    assert m.down_proj.bias is None


def test_only_the_gate_branch_is_activated():
    """Swapping which branch gets SiLU must change the output."""
    m = build()
    x = torch.randn(4, H)
    correct = m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
    swapped = m.down_proj(m.gate_proj(x) * F.silu(m.up_proj(x)))
    both = m.down_proj(F.silu(m.gate_proj(x)) * F.silu(m.up_proj(x)))
    assert torch.allclose(m(x), correct, atol=1e-6)
    assert not torch.allclose(correct, swapped, atol=1e-3)
    assert not torch.allclose(correct, both, atol=1e-3)


def test_shapes_and_batch_independence():
    m = build()
    x = torch.randn(5, H)
    assert m(x).shape == (5, H)
    assert m(torch.randn(2, 3, H)).shape == (2, 3, H)
    stacked = m(x)
    for i in range(5):
        assert torch.allclose(stacked[i], m(x[i]), atol=1e-5)


def test_eight_thirds_rule(model_dir):
    """SwiGLU's third matrix is paid for by narrowing d_ff from 4d to (8/3)d."""
    cfg = ModelConfig.from_json(model_dir)
    assert cfg.intermediate_size == cfg.hidden_size * 8 // 3 == 1536
    swiglu_params = 3 * cfg.hidden_size * cfg.intermediate_size
    plain_ffn_params = 2 * cfg.hidden_size * (4 * cfg.hidden_size)
    assert swiglu_params == plain_ffn_params


def test_loads_from_checkpoint_without_transpose(model_dir):
    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)
    mlp = SwiGLU.from_checkpoint(cfg, w, 0)
    assert mlp.gate_proj.weight.shape == (cfg.intermediate_size, cfg.hidden_size)
    assert mlp.down_proj.weight.shape == (cfg.hidden_size, cfg.intermediate_size)
    assert torch.equal(mlp.gate_proj.weight, w.get("model.layers.0.mlp.gate_proj.weight"))
    out = mlp(torch.randn(2, cfg.hidden_size))
    assert out.shape == (2, cfg.hidden_size) and torch.isfinite(out).all()


def test_matches_huggingface(model_dir, llama_ref):
    from transformers.models.llama.configuration_llama import LlamaConfig

    cfg = ModelConfig.from_json(model_dir)
    w = Weights(model_dir, cfg)

    ours = SwiGLU.from_checkpoint(cfg, w, 0)
    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="silu",
        mlp_bias=False,
    )
    theirs = llama_ref.LlamaMLP(hf_cfg)
    with torch.no_grad():
        theirs.gate_proj.weight.copy_(ours.gate_proj.weight)
        theirs.up_proj.weight.copy_(ours.up_proj.weight)
        theirs.down_proj.weight.copy_(ours.down_proj.weight)

    torch.manual_seed(0)
    for shape in [(cfg.hidden_size,), (4, cfg.hidden_size), (2, 8, cfg.hidden_size)]:
        for scale in [1e-3, 1.0, 10.0]:
            x = torch.randn(*shape) * scale
            with torch.no_grad():
                assert torch.equal(ours(x), theirs(x)), f"{shape} @ {scale}"
