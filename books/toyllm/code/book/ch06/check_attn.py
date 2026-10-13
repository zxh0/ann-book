"""第六章 · 拿PyTorch自带的注意力对拍

这一章还对不上`transformers`：它的注意力里有RoPE，我们的还没有。所以换一个参照，
PyTorch自带的`F.scaled_dot_product_attention(q, k, v, is_causal=True)`，它正好就是
「没有位置信息的因果注意力」。

PyTorch 2.2的这个函数还不认识GQA（`enable_gqa`参数要到2.5才有），所以喂进去之前，
得先自己把k、v从3个头复制成9个头。投影、拆头、拼头、o_proj也都照我们的写法来，
真正交给它算的，只有中间那一段：分数、缩放、掩码、softmax、加权求和。

最后再故意把复制的顺序写错一次，看看错了会差多少。

运行：
    cd code
    uv run python book/ch06/check_attn.py
"""

import torch
import torch.nn.functional as F

from model import Model


def reference(x, layer, n_heads, n_kv_heads, repeat):
    """同样的投影和拆头，中间交给PyTorch。`repeat`决定k、v怎么从3个头变成9个头。"""
    T = x.shape[0]
    head_dim = x.shape[1] // n_heads
    group = n_heads // n_kv_heads
    q = (x @ layer["q_proj"].T).view(T, n_heads, head_dim).transpose(0, 1)
    k = (x @ layer["k_proj"].T).view(T, n_kv_heads, head_dim).transpose(0, 1)
    v = (x @ layer["v_proj"].T).view(T, n_kv_heads, head_dim).transpose(0, 1)
    k, v = repeat(k, group), repeat(v, group)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(0, 1).reshape(T, -1) @ layer["o_proj"].T


def rel_err(a, b):
    """相对误差：最大的绝对差，除以参照里最大的那个数。"""
    return float((a - b).abs().max() / b.abs().max())


def main():
    torch.manual_seed(0)
    model = Model()
    cfg = model.config
    h, kv = cfg.num_attention_heads, cfg.num_key_value_heads

    right = lambda t, g: t.repeat_interleave(g, dim=0)   # [0,0,0,1,1,1,2,2,2]
    wrong = lambda t, g: t.repeat(g, 1, 1)               # [0,1,2,0,1,2,0,1,2]

    print(f"{'层':>4}{'T':>5}   {'逐位相同':<8}{'相对误差':>10}")
    for i in (0, 15, 29):
        layer = model.weights.layers[i]
        for T in (4, 37):
            x = torch.randn(T, cfg.hidden_size)
            ours = model.attn(x, layer)
            ref = reference(x, layer, h, kv, right)
            print(f"{i:>4}{T:>5}   {str(torch.equal(ours, ref)):<8}{rel_err(ours, ref):>10.1e}")

    layer = model.weights.layers[0]
    x = torch.randn(8, cfg.hidden_size)
    bad = reference(x, layer, h, kv, wrong)
    print(f"\n复制顺序写错（repeat而不是repeat_interleave），相对误差 "
          f"{rel_err(model.attn(x, layer), bad):.2f}")


if __name__ == "__main__":
    main()
