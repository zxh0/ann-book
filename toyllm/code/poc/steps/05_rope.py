"""Step 5 — RoPE, the operator with the most ways to be quietly wrong.

Run:  uv run python steps/05_rope.py

Position is injected by rotating q and k, not by adding anything to them. The
reason that is worth doing is one specific property, demonstrated in section 4:
the attention score ends up depending only on the *distance* between two tokens.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.config import ModelConfig
from toyllm.layers import RotaryEmbedding, apply_rope, rotate_half

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    torch.manual_seed(0)
    cfg = ModelConfig.from_json(MODEL_DIR)
    rope = RotaryEmbedding.from_config(cfg)
    D = cfg.head_dim

    rule("1. the idea")
    print(f"  A head's {D} values are read as {D // 2} two-dimensional planes.")
    print(f"  At position t, plane i is rotated by angle  t * theta^(-2i/{D}).")
    print("\n    q' = q * cos + rotate_half(q) * sin")
    print("    k' = k * cos + rotate_half(k) * sin")
    print("\n  Nothing is added and nothing is concatenated — the vector is turned.")
    print("  Applied to q and k only. v is never rotated: RoPE exists to make the")
    print("  q-k inner product relative, and the values carry no position at all.")

    rule("2. the frequency spectrum")
    print(f"  inv_freq[i] = theta^(-2i/{D}),  wavelength = 2*pi / inv_freq[i]")
    print(f"\n  {'plane i':>8}{'inv_freq':>14}{'wavelength (tokens)':>22}")
    for i in [0, 1, 8, 16, 24, 31]:
        f = rope.inv_freq[i].item()
        print(f"  {i:>8}{f:>14.3e}{2 * torch.pi / f:>22,.0f}")
    print(f"\n  Plane 0 turns once every ~6 tokens; the slowest plane takes longer than")
    print(f"  any sequence will ever be. Fast planes encode 'a few tokens apart',")
    print(f"  slow ones encode 'far apart' — together they make position legible at")
    print(f"  every scale, which is why a single frequency would not do.")

    rule("3. theta = 100000, not the usual 10000")
    slow = RotaryEmbedding(D, 10000.0, 4096)
    print(f"  {'':>14}{'slowest wavelength':>22}   {'used by'}")
    for label, r, ctx in [("theta=10000", slow, "typical 4k-context models"),
                          ("theta=100000", rope, f"SmolLM2 ({cfg.max_position_embeddings} context)")]:
        wl = 2 * torch.pi / r.inv_freq[-1].item()
        print(f"  {label:>14}{wl:>22,.0f}   {ctx}")
    print("\n  A bigger theta stretches every wavelength, so distant positions stay")
    print("  distinguishable instead of wrapping around. Using 10000 here would run")
    print("  and produce fluent text — just with degraded long-range position sense.")

    rule("4. the property that justifies all of it")
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def score(m, n):
        cq, sq = rope(torch.tensor([m]))
        ck, sk = rope(torch.tensor([n]))
        qr = q * cq + rotate_half(q) * sq
        kr = k * ck + rotate_half(k) * sk
        return (qr * kr).sum().item()

    print("  Same distance, different absolute positions:")
    print(f"    {'query at':>10}{'key at':>10}{'distance':>10}{'q.k score':>14}")
    for m, n in [(7, 3), (8, 4), (12, 8), (107, 103), (4007, 4003)]:
        print(f"    {m:>10}{n:>10}{m - n:>10}{score(m, n):>14.6f}")
    print("\n  Identical. Absolute position cancels out; only m - n survives.")
    print("\n  Different distances, for contrast:")
    print(f"    {'query at':>10}{'key at':>10}{'distance':>10}{'q.k score':>14}")
    for m, n in [(7, 7), (7, 6), (7, 3), (7, 0)]:
        print(f"    {m:>10}{n:>10}{m - n:>10}{score(m, n):>14.6f}")

    rule("5. the two conventions, and why the wrong one is silent")
    def rotate_interleaved(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    x = torch.arange(8, dtype=torch.float32)
    print(f"  input                    {x.tolist()}")
    print(f"  rotate_half   (HF)       {rotate_half(x).tolist()}   pairs (0,4) (1,5) (2,6) (3,7)")
    print(f"  interleaved   (RoFormer) {rotate_interleaved(x).tolist()}   pairs (0,1) (2,3) (4,5) (6,7)")
    v = torch.randn(4, D)
    print(f"\n  On real vectors they differ:      max |diff| = {(rotate_half(v) - rotate_interleaved(v)).abs().max():.4f}")
    print(f"  But both preserve the norm:       max |diff| = "
          f"{(rotate_half(v).norm(dim=-1) - rotate_interleaved(v).norm(dim=-1)).abs().max():.2e}")
    print("\n  Both are genuine rotations, so norm checks, shape checks and 'does it")
    print(f"  produce text' all pass either way. SmolLM2's config says")
    print(f"  rope_interleaved: false — the half-split one. Only a reference catches this.")

    rule("6. sanity properties")
    qh = torch.randn(2, cfg.num_attention_heads, 16, D)
    cos, sin = rope(torch.arange(16))
    out, _ = apply_rope(qh, qh, cos, sin)
    print(f"  norm preserved:        max |diff| = {(out.norm(dim=-1) - qh.norm(dim=-1)).abs().max():.2e}")
    c0, s0 = rope(torch.tensor([0]))
    id_out, _ = apply_rope(qh[:, :, :1], qh[:, :, :1], c0, s0)
    print(f"  position 0 = identity: max |diff| = {(id_out - qh[:, :, :1]).abs().max():.2e}")
    print(f"  cos^2 + sin^2 = 1:     max |diff| = {(cos**2 + sin**2 - 1).abs().max():.2e}")
    print(f"  same rotation for all {cfg.num_attention_heads} heads and both batch items: "
          f"{torch.equal(out[0, 0], apply_rope(qh[0:1, 0:1], qh[0:1, 0:1], cos, sin)[0][0, 0])}")

    rule("7. verify against HuggingFace")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import (
            LlamaRotaryEmbedding,
            apply_rotary_pos_emb,
        )
    except Exception as exc:
        print(f"  oracle unavailable ({type(exc).__name__}: {exc})")
        return

    hf_cfg = LlamaConfig(hidden_size=cfg.hidden_size, num_attention_heads=cfg.num_attention_heads,
                         rope_theta=cfg.rope_theta, max_position_embeddings=cfg.max_position_embeddings)
    theirs = LlamaRotaryEmbedding(hf_cfg)
    print(f"  inv_freq bit-identical: {torch.equal(rope.inv_freq, theirs.inv_freq)}")

    worst_angle, worst_out = 0.0, 0.0
    for T in (1, 5, 9, 64):
        for start in (0, 1000, 4000, cfg.max_position_embeddings - 64):
            pos = torch.arange(start, start + T)
            qq = torch.randn(2, cfg.num_attention_heads, T, D)
            kk = torch.randn(2, cfg.num_key_value_heads, T, D)
            c_hf, s_hf = theirs(qq, pos.unsqueeze(0).expand(2, -1))
            c_our, s_our = rope(pos)
            worst_angle = max(worst_angle, (c_our - c_hf[0]).abs().max().item())
            a1, b1 = apply_rope(qq, kk, c_our, s_our)
            a2, b2 = apply_rotary_pos_emb(qq, kk, c_hf, s_hf)
            worst_out = max(worst_out, (a1 - a2).abs().max().item(), (b1 - b2).abs().max().item())

    ulp = torch.finfo(torch.float32).eps
    print(f"  max |cos - cos_hf|      {worst_angle:.3e}   ({worst_angle / ulp:.1f} ulp)")
    print(f"  max |rotated - hf|      {worst_out:.3e}")
    print("\n  Not bit-identical, and the reason is worth knowing: we read cos/sin from")
    print(f"  a precomputed {cfg.max_position_embeddings}-row table while HF computes them per call. The")
    print("  angles are bit-identical — torch.cos() itself differs by one ulp between")
    print("  its vectorized (large tensor) and scalar (small tensor) kernels.")
    print("\n  Keep the scale in mind: this is 1e-7. Using the interleaved convention")
    print("  instead, as in section 5, was off by 3.6 — seven orders of magnitude more.")


if __name__ == "__main__":
    main()
