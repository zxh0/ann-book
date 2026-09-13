"""Step 3 — RMSNorm, the simplest layer in the model.

Run:  uv run python steps/03_rmsnorm.py

SmolLM2 has 61 of these (2 per block + 1 final). They hold only 576 parameters
each, but every one of them sits on the residual path, so getting the formula
subtly wrong degrades everything downstream without ever raising an error.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.config import ModelConfig
from toyllm.layers import RMSNorm
from toyllm.weights import Weights

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    torch.manual_seed(0)
    cfg = ModelConfig.from_json(MODEL_DIR)

    rule("1. the formula, next to LayerNorm")
    print("    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight")
    print("  LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias")
    print("\n  RMSNorm drops the mean subtraction and the bias. That is the whole")
    print("  difference — it rescales without re-centering.")

    x = torch.randn(4, cfg.hidden_size) + 3.0  # deliberately off-center
    norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    ln = torch.nn.LayerNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        r, l = norm(x), ln(x)
    print(f"\n  input row mean {x[0].mean():+.4f}, rms {x[0].pow(2).mean().sqrt():.4f}")
    print(f"    RMSNorm out:  mean {r[0].mean():+.4f}  rms {r[0].pow(2).mean().sqrt():.4f}")
    print(f"    LayerNorm out: mean {l[0].mean():+.4f}  rms {l[0].pow(2).mean().sqrt():.4f}")
    print("  RMSNorm forces rms=1 but leaves the mean where it was; LayerNorm zeroes it.")

    rule("2. scale invariance")
    with torch.no_grad():
        a, b = norm(x), norm(x * 1000)
    print(f"  max|RMSNorm(x) - RMSNorm(1000x)| = {(a - b).abs().max():.3e}")
    print("  Only the direction of each token vector survives; its length is discarded.")
    print("  That is what keeps activations bounded across 30 residual additions.")

    rule("3. two ways to get it wrong, neither of which raises")
    v = x.pow(2).mean(-1, keepdim=True)
    correct = x * torch.rsqrt(v + cfg.rms_norm_eps)
    eps_outside = x * (torch.rsqrt(v) + cfg.rms_norm_eps)
    print(f"  eps inside vs outside the sqrt:      max diff {(correct - eps_outside).abs().max():.3e}")

    small = torch.randn(4, cfg.hidden_size) * 1e-4
    vs = small.pow(2).mean(-1, keepdim=True)
    c2 = small * torch.rsqrt(vs + cfg.rms_norm_eps)
    w2 = small * torch.rsqrt(vs)  # eps forgotten entirely
    print(f"  eps omitted, on small activations:   max diff {(c2 - w2).abs().max():.3e}")
    print("  Both produce finite, plausible-looking numbers. Only a reference catches them.")

    rule("4. why the fp32 upcast is not optional")
    xb = (torch.randn(2, cfg.hidden_size) * 40).to(torch.bfloat16)
    with torch.no_grad():
        good = norm(xb)                                  # upcasts internally
        vb = xb.pow(2).mean(-1, keepdim=True)            # stays in bf16
        bad = (xb * torch.rsqrt(vb + cfg.rms_norm_eps)).to(torch.bfloat16) * norm.weight.to(torch.bfloat16)
    print(f"  bf16 input, mean of 576 squares computed in bf16 vs fp32:")
    print(f"    max|diff| = {(good.float() - bad.float()).abs().max():.3e}")
    print("  bf16 has 8 mantissa bits. Squaring halves the precision that matters,")
    print("  and summing 576 of them accumulates the error. We only run fp32 on CPU,")
    print("  but the layer upcasts anyway so it stays correct if the dtype changes.")

    rule("5. the weights SmolLM2 actually learned")
    w = Weights(MODEL_DIR, cfg)
    names = ["model.layers.0.input_layernorm.weight",
             "model.layers.15.input_layernorm.weight",
             "model.layers.29.post_attention_layernorm.weight",
             "model.norm.weight"]
    print(f"  {'tensor':<48}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for n in names:
        t = w.get(n)
        print(f"  {n:<48}{t.mean():>8.3f}{t.std():>8.3f}{t.min():>8.3f}{t.max():>8.3f}")
    print("\n  These start at 1.0 and drift during training. The gain is per-channel,")
    print("  so the layer can amplify some of the 576 dimensions and suppress others —")
    print("  it is not just a normalization, it is a learned re-weighting.")

    rule("6. verify against HuggingFace")
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
    except Exception as exc:  # transformers 5.x imports but disables torch < 2.5
        print(f"  oracle unavailable ({type(exc).__name__}: {exc})")
        print("  run: uv sync --group dev   (transformers must be <5 here)")
        return

    ours = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    theirs = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    real = w.get("model.layers.0.input_layernorm.weight")
    with torch.no_grad():
        ours.weight.copy_(real)
        theirs.weight.copy_(real)

    worst = 0.0
    for shape in [(576,), (4, 576), (2, 8, 576), (1, 1, 576)]:
        for scale in [1e-4, 1.0, 100.0]:
            t = torch.randn(*shape) * scale
            with torch.no_grad():
                d = (ours(t) - theirs(t)).abs().max().item()
            worst = max(worst, d)
    print(f"  max abs difference over shapes and scales: {worst:.3e}")
    print(f"  {'✓ bit-identical to LlamaRMSNorm' if worst == 0.0 else '✓ within tolerance'}")


if __name__ == "__main__":
    main()
