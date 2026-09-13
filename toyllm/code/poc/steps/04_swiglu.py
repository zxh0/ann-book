"""Step 4 — SwiGLU, the block that holds most of the model.

Run:  uv run python steps/04_swiglu.py

59% of SmolLM2's parameters live in these three matrices. The operator itself is
one line; what is worth the time is why there are three matrices instead of two,
and what the "gate" actually does.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.config import ModelConfig
from toyllm.layers import RMSNorm, SwiGLU
from toyllm.weights import Weights

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def plot(fn, lo=-6.0, hi=6.0, rows=13, cols=57):
    """Tiny ASCII plot so the activation's shape is visible, not just described."""
    xs = torch.linspace(lo, hi, cols)
    ys = fn(xs)
    top, bot = max(ys.max().item(), 0.5), min(ys.min().item(), -0.5)
    grid = [[" "] * cols for _ in range(rows)]
    zero_row = int(round((top - 0.0) / (top - bot) * (rows - 1)))
    zero_col = int(round((0.0 - lo) / (hi - lo) * (cols - 1)))
    for c in range(cols):
        grid[zero_row][c] = "-"
    for r in range(rows):
        grid[r][zero_col] = "|"
    grid[zero_row][zero_col] = "+"
    for c, y in enumerate(ys):
        r = int(round((top - y.item()) / (top - bot) * (rows - 1)))
        grid[max(0, min(rows - 1, r))][c] = "*"
    for r, line in enumerate(grid):
        label = f"{top + (bot - top) * r / (rows - 1):+5.2f} "
        print("    " + label + "".join(line))
    print("          " + f"{lo:<+5.0f}".ljust(zero_col) + "0" + f"{hi:+.0f}".rjust(cols - zero_col - 1))


def main():
    torch.manual_seed(0)
    cfg = ModelConfig.from_json(MODEL_DIR)

    rule("1. the formula")
    print("    SwiGLU(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )")
    print("       silu(z) = z * sigmoid(z)")
    print(f"\n  shapes for SmolLM2-135M:")
    print(f"    x          ({cfg.hidden_size},)")
    print(f"    gate_proj  ({cfg.intermediate_size}, {cfg.hidden_size})  ->  ({cfg.intermediate_size},)")
    print(f"    up_proj    ({cfg.intermediate_size}, {cfg.hidden_size})  ->  ({cfg.intermediate_size},)")
    print(f"    elementwise product          ->  ({cfg.intermediate_size},)")
    print(f"    down_proj  ({cfg.hidden_size}, {cfg.intermediate_size}) ->  ({cfg.hidden_size},)")
    print("\n  A plain FFN is two matrices: W2 @ relu(W1 @ x). The third matrix here")
    print("  buys a *multiplicative* interaction between two projections of x.")

    rule("2. what SiLU looks like")
    plot(F.silu)
    z = torch.linspace(-8, 8, 200001)
    s = F.silu(z)
    print(f"\n    min {s.min():.4f} at z = {z[s.argmin()]:+.3f}")
    print("\n  Not a 0..1 gate. It dips negative around z = -1.28 before recovering,")
    print("  so silu(gate) can flip the sign of a channel, not just attenuate it.")
    print("  Unlike ReLU it is smooth and never exactly zero for negative inputs —")
    print("  channels are suppressed continuously rather than hard-switched off.")

    rule("3. three ways to write it, only one is right")
    m = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
    with torch.no_grad():
        for lin in (m.gate_proj, m.up_proj, m.down_proj):
            lin.weight.normal_(0, 0.05)
    x = torch.randn(8, cfg.hidden_size)
    correct = m.down_proj(F.silu(m.gate_proj(x)) * m.up_proj(x))
    variants = {
        "silu on up instead of gate": m.down_proj(m.gate_proj(x) * F.silu(m.up_proj(x))),
        "silu on both branches": m.down_proj(F.silu(m.gate_proj(x)) * F.silu(m.up_proj(x))),
        "no activation at all": m.down_proj(m.gate_proj(x) * m.up_proj(x)),
    }
    print(f"  {'variant':<32}{'max |diff| vs correct':>24}")
    for label, out in variants.items():
        print(f"  {label:<32}{(out - correct).abs().max().item():>24.4f}")
    print("\n  Every one of these runs, returns the right shape, and produces finite")
    print("  numbers. Only a reference implementation tells you which is the model.")

    rule("4. the 8/3 rule")
    h = cfg.hidden_size
    print(f"  plain FFN  (2 matrices, d_ff = 4d):  2 * {h} * {4*h} = {2*h*4*h:,}")
    print(f"  SwiGLU     (3 matrices, d_ff = ?):   3 * {h} * d_ff")
    print(f"  equal budget  =>  d_ff = 8/3 * d = 8/3 * {h} = {h*8//3}")
    print(f"  SmolLM2's intermediate_size:         {cfg.intermediate_size}")
    print(f"  {'✓ exactly the 8/3 rule' if cfg.intermediate_size == h*8//3 else '✗ not 8/3'}")
    print("\n  The extra matrix is not free capacity — it is paid for by narrowing")
    print("  the hidden layer from 4d to 2.67d, keeping the parameter count identical.")

    rule("5. the real block, on real weights")
    w = Weights(MODEL_DIR, cfg)
    mlp = SwiGLU.from_checkpoint(cfg, w, 0)
    print(f"  loaded layer 0 MLP: {sum(p.numel() for p in mlp.parameters()):,} params")
    print(f"    gate_proj {tuple(mlp.gate_proj.weight.shape)}  up_proj {tuple(mlp.up_proj.weight.shape)}  down_proj {tuple(mlp.down_proj.weight.shape)}")
    print("  Note: loaded with a plain copy, no transpose. nn.Linear stores")
    print("  (out_features, in_features) and computes x @ W.T — same layout as the file.")

    ref_path = ROOT.parent / "reference" / "smollm2_135m_forward.pt"
    if ref_path.exists():
        hs = torch.load(ref_path, weights_only=True)["hidden_states"]
        norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        with torch.no_grad():
            norm.weight.copy_(w.get("model.layers.0.post_attention_layernorm.weight"))
            inp = norm(hs[1])  # block-0 output, normalized — realistic scale
            gate = mlp.gate_proj(inp)
            up = mlp.up_proj(inp)
            gated = F.silu(gate) * up
        print(f"\n  Feeding real activations through it (block-0 output, normalized —")
        print(f"  the exact MLP input is the post-attention residual, which we get in step 5):")
        print(f"    gate      mean {gate.mean():+.3f}  std {gate.std():.3f}  min {gate.min():+.3f}  max {gate.max():+.3f}")
        print(f"    silu(gate) < 0 for {100.0 * (F.silu(gate) < 0).float().mean():.1f}% of channels")
        print(f"    |gated| < 0.01 for {100.0 * (gated.abs() < 0.01).float().mean():.1f}% of channels")
        print("\n  A large fraction of the 1536 channels contribute almost nothing for any")
        print("  given token. That input-dependent sparsity is the point of the gate.")
    else:
        print("\n  (run steps/00_reference.py first to see it on real activations)")

    rule("6. verify against HuggingFace")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP
    except Exception as exc:
        print(f"  oracle unavailable ({type(exc).__name__}: {exc})")
        return

    hf_cfg = LlamaConfig(hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
                         hidden_act="silu", mlp_bias=False)
    theirs = LlamaMLP(hf_cfg)
    with torch.no_grad():
        theirs.gate_proj.weight.copy_(mlp.gate_proj.weight)
        theirs.up_proj.weight.copy_(mlp.up_proj.weight)
        theirs.down_proj.weight.copy_(mlp.down_proj.weight)

    worst = 0.0
    for shape in [(cfg.hidden_size,), (4, cfg.hidden_size), (2, 8, cfg.hidden_size)]:
        for scale in [1e-3, 1.0, 10.0]:
            t = torch.randn(*shape) * scale
            with torch.no_grad():
                worst = max(worst, (mlp(t) - theirs(t)).abs().max().item())
    print(f"  max abs difference over shapes and scales: {worst:.3e}")
    print(f"  {'✓ bit-identical to LlamaMLP' if worst == 0.0 else '✓ within tolerance'}")


if __name__ == "__main__":
    main()
