"""Step 6 — grouped-query causal self-attention.

Run:  uv run python steps/06_attention.py

The last missing component. After this every piece of a SmolLM2 block exists and
step 7 can assemble them.
"""

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.config import ModelConfig
from toyllm.layers import (
    Attention,
    RotaryEmbedding,
    apply_rope,
    causal_mask,
    repeat_kv,
)
from toyllm.weights import Weights

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    torch.manual_seed(0)
    cfg = ModelConfig.from_json(MODEL_DIR)
    w = Weights(MODEL_DIR, cfg)
    attn = Attention.from_checkpoint(cfg, w, 0)
    rope = RotaryEmbedding.from_config(cfg)
    H, KV, D, R = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.n_rep

    rule("1. shapes, end to end")
    T = 7
    x = torch.randn(1, T, cfg.hidden_size)
    print(f"  x                     (1, {T}, {cfg.hidden_size})")
    print(f"  q_proj(x)             (1, {T}, {H * D})   -> view (1, {T}, {H}, {D}) -> (1, {H}, {T}, {D})")
    print(f"  k_proj(x)             (1, {T}, {KV * D})   -> view (1, {T}, {KV}, {D}) -> (1, {KV}, {T}, {D})")
    print(f"  after repeat_kv       (1, {H}, {T}, {D})")
    print(f"  scores = q @ k^T      (1, {H}, {T}, {T})")
    print(f"  out = w @ v           (1, {H}, {T}, {D})  -> (1, {T}, {H * D}) -> o_proj -> (1, {T}, {cfg.hidden_size})")
    print(f"\n  transpose(1,2) puts heads in front of time; every matmul after that")
    print(f"  batches over (batch, head), which is what makes the {H} heads independent.")

    rule("2. GQA: 9 query heads, 3 kv heads")
    print(f"  q_proj {tuple(attn.q_proj.weight.shape)}    o_proj {tuple(attn.o_proj.weight.shape)}")
    print(f"  k_proj {tuple(attn.k_proj.weight.shape)}    v_proj {tuple(attn.v_proj.weight.shape)}   <- a third the size")
    kv_params = 2 * cfg.kv_size * cfg.hidden_size
    mha_params = 2 * cfg.hidden_size * cfg.hidden_size
    print(f"\n  k+v projections:  {kv_params:,} params, vs {mha_params:,} for full MHA")
    print(f"  But the real saving is the KV cache at inference time:")
    for ctx in (1024, 8192):
        gqa = 2 * cfg.num_hidden_layers * ctx * cfg.kv_size * 4
        mha = 2 * cfg.num_hidden_layers * ctx * cfg.hidden_size * 4
        print(f"    {ctx:>5} tokens, fp32:  GQA {gqa / 1e6:>6.1f} MB   vs MHA {mha / 1e6:>6.1f} MB")
    print(f"\n  That is the point of GQA: the cache, not the parameter count.")

    rule("3. repeat_kv interleaves — it does not tile")
    ids = torch.arange(KV, dtype=torch.float32).view(1, KV, 1, 1)
    print(f"  kv heads          {[int(v) for v in ids.flatten()]}")
    print(f"  repeat_kv(x, {R})    {[int(v) for v in repeat_kv(ids, R).flatten()]}   <- query head i reads kv head i//{R}")
    print(f"  the tiled mistake {[i % KV for i in range(H)]}   <- query head i reads kv head i%{KV}")
    print("\n  Same shape, same dtype, model still runs. Every query head is simply")
    print("  paired with the wrong keys. Let us prove which one this code does:")

    def per_head(model, xx, cos, sin, mask):
        q = model.q_proj(xx).view(1, xx.shape[1], H, D).transpose(1, 2)
        k = model.k_proj(xx).view(1, xx.shape[1], KV, D).transpose(1, 2)
        v = model.v_proj(xx).view(1, xx.shape[1], KV, D).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k, v = repeat_kv(k, R), repeat_kv(v, R)
        s = torch.matmul(q, k.transpose(2, 3)) * model.scaling + mask
        return torch.matmul(torch.softmax(s, -1, dtype=torch.float32), v)

    cos, sin = rope(torch.arange(T))
    m = causal_mask(T)
    with torch.no_grad():
        base = per_head(attn, x, cos, sin, m)
        for kv_idx in range(KV):
            damaged = copy.deepcopy(attn)
            damaged.v_proj.weight[kv_idx * D : (kv_idx + 1) * D] = 0.0
            diff = (base - per_head(damaged, x, cos, sin, m)).abs().amax(dim=(0, 2, 3))
            hit = [i for i in range(H) if diff[i] > 1e-6]
            print(f"    zeroing kv head {kv_idx} disturbs query heads {hit}")

    rule("4. the causal mask")
    small = causal_mask(4)[0, 0]
    print("  additive, not boolean — summed into the scores before softmax:")
    for row in small:
        print("    [" + " ".join(f"{'0' if v == 0 else '-3.4e38':>8}" for v in row) + " ]")
    print(f"\n  Blocked entries use finfo.min ({torch.finfo(torch.float32).min:.3e}), not -inf.")
    print("  A row of all -inf softmaxes to NaN; finfo.min degrades to a uniform row.")

    with torch.no_grad():
        base_out = attn(x, cos, sin, m)
        future = x.clone()
        future[0, 4:] = torch.randn(T - 4, cfg.hidden_size)
        masked_out = attn(future, cos, sin, m)
        unmasked_base = attn(x, cos, sin, None)
        unmasked_out = attn(future, cos, sin, None)
    print(f"\n  Rewrite tokens 4..{T-1}, then look at how token 0..3 outputs move:")
    print(f"    with causal mask:     {(base_out[0, :4] - masked_out[0, :4]).abs().max():.2e}   <- the past cannot see the future")
    print(f"    without any mask:     {(unmasked_base[0, :4] - unmasked_out[0, :4]).abs().max():.2e}   <- it leaks")

    rule("5. attention weights on a real sentence")
    from toyllm.tokenizer import BPETokenizer

    tok = BPETokenizer.from_file(MODEL_DIR)
    text = "The capital of France is Paris"
    ids = tok.encode(text)
    labels = [t.replace("Ġ", "_") for t in tok.tokens(ids)]
    n = len(ids)
    emb = w.get("model.embed_tokens.weight")[torch.tensor(ids)].unsqueeze(0)

    from toyllm.layers import RMSNorm

    norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(w.get("model.layers.0.input_layernorm.weight"))
        h = norm(emb)
        c2, s2 = rope(torch.arange(n))
        q = attn.q_proj(h).view(1, n, H, D).transpose(1, 2)
        k = attn.k_proj(h).view(1, n, KV, D).transpose(1, 2)
        q, k = apply_rope(q, k, c2, s2)
        k = repeat_kv(k, R)
        sc = torch.matmul(q, k.transpose(2, 3)) * attn.scaling + causal_mask(n)
        weights = torch.softmax(sc, -1, dtype=torch.float32)

    print(f"  {text!r} -> {labels}")
    wide = max(len(l) for l in labels) + 1
    print(f"\n  Layer 0, head 0 — each row is one query, columns are the keys it attends to:")
    print("  " + " " * wide + "".join(f"{l:>9}" for l in labels))
    for i, label in enumerate(labels):
        row = "".join(f"{weights[0, 0, i, j]:>9.3f}" if j <= i else f"{'-':>9}" for j in range(n))
        print(f"  {label:>{wide}}{row}")
    print("\n  Upper triangle is empty — that is the mask. Rows sum to 1.")
    print(f"  Row sums: {[round(v, 4) for v in weights[0, 0].sum(-1).tolist()]}")

    rule("6. verify against HuggingFace")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
    except Exception as exc:
        print(f"  oracle unavailable ({type(exc).__name__}: {exc})")
        return

    hf_cfg = LlamaConfig(
        hidden_size=cfg.hidden_size, num_attention_heads=H, num_key_value_heads=KV,
        rope_theta=cfg.rope_theta, max_position_embeddings=cfg.max_position_embeddings,
        attention_bias=False, attention_dropout=0.0,
    )
    hf_cfg._attn_implementation = "eager"
    theirs = LlamaAttention(hf_cfg, layer_idx=0)
    theirs.eval()
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(theirs, name).weight.copy_(getattr(attn, name).weight)
    hf_rope = LlamaRotaryEmbedding(hf_cfg)

    worst_abs, worst_rel = 0.0, 0.0
    for b, t in [(1, 1), (1, 7), (2, 12), (3, 33), (1, 128)]:
        for start in (0, 4000, 8000):
            xx = torch.randn(b, t, cfg.hidden_size)
            pos = torch.arange(start, start + t)
            c3, s3 = rope(pos)
            ch, sh = hf_rope(xx, pos.unsqueeze(0).expand(b, -1))
            mm = causal_mask(t)
            with torch.no_grad():
                o1 = attn(xx, c3, s3, mm)
                o2, _ = theirs(xx, position_embeddings=(ch, sh), attention_mask=mm)
            worst_abs = max(worst_abs, (o1 - o2).abs().max().item())
            worst_rel = max(worst_rel, ((o1 - o2).abs().max() / o1.abs().max()).item())
    print(f"  max abs difference over batch/length/offset: {worst_abs:.3e}")
    print(f"  max relative difference:                     {worst_rel:.3e}")
    print("\n  Judge this relatively, not absolutely: the gap is float32 rounding")
    print("  inherited from the cos/sin table (step 5), so it grows with the output")
    print("  magnitude while the relative error stays put at a few times 1e-7.")
    print("\n  For scale: a tiled repeat_kv or the interleaved RoPE convention both")
    print("  land around 1e0 relative. There is no ambiguity about what is a bug.")


if __name__ == "__main__":
    main()
