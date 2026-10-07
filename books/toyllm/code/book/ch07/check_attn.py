"""第七章 · 对拍：我们的attn和transformers的LlamaAttention

上一章缺RoPE，只能拿PyTorch自带的无位置注意力当参照。补上RoPE以后，
终于能和真正的`LlamaAttention`对了：同样的输入、同样的四个矩阵、同样的位置。

短句子逐位相同，长了就未必。我们的cos/sin表是启动时一次算好8192行，
transformers是每次现算T行；角度逐位相同，可`torch.cos`算多大一块，走的
不一定是同一段代码，余弦偶尔差最后一位。实测T不超过130时一位不差，从131起
时有时无。这一位的差，在第29层会被放大到1e-4的量级，所以从这一章起，
看的是相对误差：

    max|我们 - 它们| / max|它们|

再故意做错两种，看看错的时候是什么量级：配对方式换成相邻两维，以及
`rope_theta`换成常见的10000。两种都是真旋转，都不报错。

和`ch05/check_rmsnorm.py`一样，transformers只在这里当参照用。

运行：
    cd code
    uv run python book/ch07/check_attn.py
"""

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from model import Model
from weights import MODEL_DIR

TEXTS = [
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog, and then it runs away "
    "into the forest where nobody can find it ever again.",
]
LONG = 1000  # 再加一条1000个词元的长输入，随机挑ID，只为了够长


def reference(model, layer_idx, x):
    """用transformers算同一层的注意力，(T, 576) → (T, 576)。"""
    cfg = LlamaConfig.from_pretrained(MODEL_DIR)
    cfg._attn_implementation = "eager"
    attn = LlamaAttention(cfg, layer_idx)
    layer = model.weights.layers[layer_idx]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(attn, name).weight.data = layer[name].clone()

    T = x.shape[0]
    positions = torch.arange(T).unsqueeze(0)                         # (1, T)
    cos, sin = LlamaRotaryEmbedding(cfg)(x, positions)               # 各(1, T, 64)
    # eager实现不会自己加因果掩码，得把它当参数传进去。
    mask = torch.full((T, T), torch.finfo(x.dtype).min).triu(1)      # (T, T)
    with torch.no_grad():
        out, _ = attn(x.unsqueeze(0), (cos, sin), mask[None, None])  # (1, T, 576)
    return out[0]


def rotate_interleaved(x):
    """错误示范：相邻两维配对，(a0,b0,a1,b1,...) → (-b0,a0,-b1,a1,...)。"""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def rel_err(ours, ref):
    return ((ours - ref).abs().max() / ref.abs().max()).item()


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()

    # 做错的两份：一份换配对方式，一份换rope_theta。
    wrong_pairing = Model()
    # 相邻配对时，第2i和第2i+1维是一对，角度要排成θ0,θ0,θ1,θ1,...
    wrong_pairing.rotate_half = rotate_interleaved
    wrong_pairing.cos = model.cos[:, :32].repeat_interleave(2, dim=-1)  # (8192, 64)
    wrong_pairing.sin = model.sin[:, :32].repeat_interleave(2, dim=-1)  # (8192, 64)
    wrong_theta = Model()
    wrong_theta.config.rope_theta = 10000
    wrong_theta.cos, wrong_theta.sin = wrong_theta.rope_table()

    inputs = [tok.encode(text, add_special_tokens=False).ids for text in TEXTS]
    inputs.append(torch.randint(0, 49152, (LONG,), generator=torch.Generator().manual_seed(0)).tolist())

    print(f"{'T':>5}{'层':>4}{'绝对误差':>11}{'相对误差':>11}{'相邻配对':>11}{'theta=10000':>13}")
    for ids in inputs:
        emb = model.to_embeddings(ids)                               # (T, 576)
        for i in (0, 15, 29):
            layer = model.weights.layers[i]
            x = model.rms_norm(emb, layer["input_norm"])             # attn真正收到的输入
            ref = reference(model, i, x)
            ours = model.attn(x, layer)
            abs_err = (ours - ref).abs().max().item()
            errs = [rel_err(m.attn(x, layer), ref) for m in (model, wrong_pairing, wrong_theta)]
            print(f"{len(ids):>5}{i:>4}{abs_err:>13.1e}{errs[0]:>13.1e}{errs[1]:>13.1e}"
                  f"{errs[2]:>13.1e}")

    # 表本身：角度逐位相同，cos差在最后一位。
    head_dim = 64
    hf = LlamaRotaryEmbedding(LlamaConfig.from_pretrained(MODEL_DIR))
    exponent = torch.arange(0, head_dim, 2).float() / head_dim
    inv_freq = 1.0 / model.config.rope_theta ** exponent
    print(f"\ninv_freq和transformers逐位相同   {torch.equal(inv_freq, hf.inv_freq)}")
    for T in (130, 131, LONG):
        cos_hf, _ = hf(torch.zeros(1), torch.arange(T).unsqueeze(0))
        n = (model.cos[:T] != cos_hf[0]).sum().item()
        diff = (model.cos[:T] - cos_hf[0]).abs().max().item()
        print(f"cos表前{T:>4}行   {n:>4}个数不同，最大差{diff:.1e}")
    print(f"（float32里1附近的最后一位是{torch.finfo(torch.float32).eps:.1e}）")


if __name__ == "__main__":
    main()
