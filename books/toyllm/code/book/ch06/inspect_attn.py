"""第六章 · 看看注意力到底在看哪

`engine.py`输入「Once upon a time」，一直吐`ppo`。这个脚本把原因摊开：

1. 第0层第0个头的注意力权重：下三角，每行和为1，这就是因果掩码
2. 最后一个位置在几层里看向各个位置的权重，和真模型对比
3. 30层走完，各位置两两之间的余弦，和刚查出来的词嵌入对比
4. 出口那一步，分数前几名，以及它们在词表里那一行的模长

第2项里真模型那一行，是借第一章用过的`transformers`跑的，它带RoPE、带FFN。
只有这个脚本import了`transformers`，引擎本身一行都没用它。

运行：
    cd code
    uv run python book/ch06/inspect_attn.py
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

TEXT = "Once upon a time"
LAYERS = (0, 15, 29)


def attn_probs(model, x, layer):
    """把`Model.attn`的前半段再走一遍，停在softmax之后：(9, T, T)。"""
    T = x.shape[0]
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    q = (x @ layer["q_proj"].T).view(T, n_heads, head_dim).transpose(0, 1)
    k = (x @ layer["k_proj"].T).view(T, n_kv, head_dim).transpose(0, 1)
    k = k.repeat_interleave(n_heads // n_kv, dim=0)
    scores = q @ k.transpose(1, 2) / head_dim ** 0.5
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)


def fmt(row):
    return "  ".join(f"{float(p):.3f}" for p in row)


def cos_range(h):
    """各行两两之间余弦的最小值和最大值（不含自己和自己）。"""
    n = F.normalize(h, dim=-1)
    c = n @ n.T
    off = c[~torch.eye(len(h), dtype=torch.bool)]
    return float(off.min()), float(off.max())


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    w = model.weights
    ids = tok.encode(TEXT, add_special_tokens=False).ids
    print(f"输入   {TEXT!r}   {ids}")

    # ---- 走一遍30层，顺路把要看的几层的注意力权重存下来 ------------------------
    hidden = model.to_embeddings(ids)
    embeds = hidden
    probs = {}
    for i, layer in enumerate(w.layers):
        if i in LAYERS:
            probs[i] = attn_probs(model, model.rms_norm(hidden, layer["input_norm"]), layer)
        hidden = model.block(hidden, layer)

    print("\n1. 第0层第0个头的注意力权重（行是谁在看，列是被看的）：")
    for row in probs[0][0]:
        print("   ", fmt(row))

    print("\n2. 最后一个位置看向各个位置的权重（9个头取平均）：")
    for i in LAYERS:
        print(f"   我们的引擎  第{i:2d}层   {fmt(probs[i][:, -1].mean(0))}")
    try:
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.float32, attn_implementation="eager")
        with torch.no_grad():
            att = hf(torch.tensor([ids]), output_attentions=True).attentions
        for i in LAYERS:
            print(f"   真模型      第{i:2d}层   {fmt(att[i][0, :, -1].mean(0))}")
    except Exception as e:  # 没装transformers也不影响前后几项
        print(f"   （真模型那几行跳过了：{type(e).__name__}）")

    lo, hi = cos_range(embeds)
    print(f"\n3. 各位置两两之间的余弦")
    print(f"   刚查出来的词嵌入   {lo:6.3f} ~ {hi:6.3f}")
    lo, hi = cos_range(hidden)
    print(f"   30层走完之后       {lo:6.3f} ~ {hi:6.3f}")

    out = model.rms_norm(hidden[-1], w.norm)
    logits = model.to_logits(out)
    cos = F.cosine_similarity(out[None], w.embed_tokens)
    print("\n4. 出口：分数前5名")
    print(f"   {'词元':<12}{'分数':>8}{'模长':>8}{'按余弦排第几':>14}")
    for s, t in zip(*logits.topk(5)):
        rank = int((cos > cos[t]).sum()) + 1
        print(f"   {tok.id_to_token(int(t))!r:<12}{float(s):8.2f}"
              f"{float(w.embed_tokens[t].norm()):8.3f}{rank:>14}")
    print(f"   按余弦排第一的是 {tok.id_to_token(int(cos.argmax()))!r}")

    print("\n5. 每多追加一个预测出来的词元，出口向量变了多少")
    seq, prev = list(ids), out
    for _ in range(4):
        seq.append(int(logits.argmax()))
        h = model.to_embeddings(seq)
        for layer in w.layers:
            h = model.block(h, layer)
        cur = model.rms_norm(h[-1], w.norm)
        logits = model.to_logits(cur)
        print(f"   T={len(seq):2d}   和上一圈的余弦 {float(F.cosine_similarity(cur, prev, dim=0)):.4f}"
              f"   预测 {tok.id_to_token(int(logits.argmax()))!r}")
        prev = cur


if __name__ == "__main__":
    main()
