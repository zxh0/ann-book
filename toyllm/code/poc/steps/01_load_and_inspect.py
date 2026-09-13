"""Step 1 — load SmolLM2's weights and work out exactly what its structure is.

Run:  uv run python steps/01_load_and_inspect.py

No model math here on purpose. The goal is to end this step able to answer:
which tensors exist, what shape is each one, how do they add up to "135M", and
where does the config's arithmetic show up in the file on disk.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.config import ModelConfig
from toyllm.weights import Weights, expected_tensors

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def main():
    cfg = ModelConfig.from_json(MODEL_DIR)

    rule("1. config.json")
    print(cfg.summary())

    rule("2. shape arithmetic the config implies")
    print(f"  head_dim   = hidden_size / n_heads   = {cfg.hidden_size} / {cfg.num_attention_heads} = {cfg.head_dim}")
    print(f"  q_proj out = n_heads * head_dim      = {cfg.num_attention_heads} * {cfg.head_dim} = {cfg.num_attention_heads * cfg.head_dim}")
    print(f"  k/v out    = n_kv_heads * head_dim   = {cfg.num_key_value_heads} * {cfg.head_dim} = {cfg.kv_size}")
    print(f"  GQA group  = n_heads / n_kv_heads    = {cfg.num_attention_heads} / {cfg.num_key_value_heads} = {cfg.n_rep}")
    print(f"  -> each KV head is shared by {cfg.n_rep} query heads; K/V are {cfg.hidden_size // cfg.kv_size}x smaller than Q")

    w = Weights(MODEL_DIR, cfg)

    rule("3. what is actually in model.safetensors")
    print(f"  file            {w.path.name}  ({w.path.stat().st_size / 1e6:.1f} MB)")
    print(f"  tensor count    {len(w.names)}")
    print(f"  stored dtype    {w.raw_dtype(w.names[0])}")

    print("\n  Non-layer tensors:")
    for name in w.names:
        if ".layers." not in name:
            print(f"    {name:<44} {str(w.shape(name)):<18}")

    print(f"\n  One transformer block (layer 0 of {cfg.num_hidden_layers}, all {cfg.num_hidden_layers} are identical in shape):")
    for name in w.names:
        if name.startswith("model.layers.0."):
            print(f"    {name:<44} {str(w.shape(name)):<18}")

    rule("4. verify the file against what we expected")
    problems = w.verify()
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print(f"  ✓ all {len(expected_tensors(cfg))} expected tensors present with matching shapes")

    if cfg.tie_word_embeddings:
        print("\n  Note: lm_head.weight is NOT in the file — tie_word_embeddings=true means")
        print("  the output projection reuses model.embed_tokens.weight.")
        emb = w.get("model.embed_tokens.weight")
        head = w.get("lm_head.weight")
        print(f"  Weights.get('lm_head.weight') redirects to it: identical values = "
              f"{torch.equal(emb, head)}, shape {tuple(head.shape)}")

    rule("5. where the parameters live")
    h, ffn = cfg.hidden_size, cfg.intermediate_size
    q = cfg.num_attention_heads * cfg.head_dim
    attn = q * h + cfg.kv_size * h * 2 + h * q
    mlp = ffn * h * 2 + h * ffn
    norms = h * 2
    per_layer = attn + mlp + norms
    emb = cfg.vocab_size * h
    total = emb + per_layer * cfg.num_hidden_layers + h

    print(f"  embed_tokens                      {emb:>12,}")
    print(f"  per block: attention (q,k,v,o)    {attn:>12,}")
    print(f"             mlp (gate,up,down)     {mlp:>12,}")
    print(f"             2 rmsnorms             {norms:>12,}")
    print(f"             = block total          {per_layer:>12,}")
    print(f"  x {cfg.num_hidden_layers} blocks                       {per_layer * cfg.num_hidden_layers:>12,}")
    print(f"  final norm                        {h:>12,}")
    print(f"  {'-' * 46}")
    print(f"  predicted total                   {total:>12,}")
    print(f"  actual (counted from file)        {w.param_count():>12,}")
    assert total == w.param_count(), "our model of the architecture disagrees with the file"
    print(f"  ✓ match — {total / 1e6:.1f}M parameters, which is the '135M' in the name")

    print(f"\n  Embeddings are {emb / total:.0%} of the model; the MLP is {mlp * cfg.num_hidden_layers / total:.0%},")
    print(f"  attention only {attn * cfg.num_hidden_layers / total:.0%}. A small model is mostly vocabulary and MLP.")
    print(f"\n  fp32 working memory: {total * 4 / 1e6:.0f} MB (file is bf16, half that)")

    rule("6. the structure, in one picture")
    body = [
        ("input_layernorm  RMSNorm", f"[{h}]"),
        ("self_attn:  q_proj", f"[{q}, {h}]   -> {cfg.num_attention_heads} heads x {cfg.head_dim}"),
        ("            k_proj", f"[{cfg.kv_size}, {h}]   -> {cfg.num_key_value_heads} heads x {cfg.head_dim}"),
        ("            v_proj", f"[{cfg.kv_size}, {h}]   -> {cfg.num_key_value_heads} heads x {cfg.head_dim}"),
        ("            RoPE on q,k; causal mask", f"GQA: each kv head serves {cfg.n_rep} q heads"),
        ("            o_proj", f"[{h}, {q}]"),
        ("+ residual", ""),
        ("post_attention_layernorm", f"[{h}]"),
        ("mlp:  gate_proj", f"[{ffn}, {h}]"),
        ("      up_proj", f"[{ffn}, {h}]"),
        ("      SwiGLU: silu(gate) * up", ""),
        ("      down_proj", f"[{h}, {ffn}]"),
        ("+ residual", ""),
    ]
    inner = [f"  x{cfg.num_hidden_layers} blocks:"]
    inner += [f"    {lhs:<40}{rhs}".rstrip() for lhs, rhs in body]
    width = max(len(s) for s in inner) + 2

    print()
    print(f"  token ids  (B, T)")
    print(f"      |")
    print(f"  embed_tokens                     [{cfg.vocab_size}, {h}]")
    print(f"      |")
    print(f"      .{'-' * width}.")
    for s in inner:
        print(f"      |{s.ljust(width)}|")
    print(f"      '{'-' * width}'")
    print(f"      |")
    print(f"  model.norm  RMSNorm              [{h}]")
    print(f"      |")
    print(f"  lm_head (tied to embed_tokens)   [{cfg.vocab_size}, {h}]")
    print(f"      |")
    print(f"  logits  (B, T, {cfg.vocab_size})")
    print()


if __name__ == "__main__":
    main()
