"""第八章 · 对拍：我们的ffn和transformers的LlamaMLP

FFN不经过RoPE，没有那一位余弦的差，所以这个部件本身可以严格比：
同样的输入、同样的三个矩阵，两边的输出应该逐位相同。

输入是`ffn`真正收到的东西：词嵌入走到第i层，过完注意力子层，再过
`post_attn_norm`。1000那组是随机挑的ID，只为了够长。

再故意做错三种，看看错的时候是什么量级：

    手写silu       z * sigmoid(z)，数学上和F.silu一样，只是写法不同
    silu放错一路    silu套在up上，gate不过激活函数
    换成ReLU       门那一路用ReLU，也就是ReGLU

和`ch07/check_attn.py`一样，transformers只在这里当参照用。

运行：
    cd code
    uv run python book/ch08/check_ffn.py
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from model import Model
from weights import MODEL_DIR

TEXTS = [
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog, and then it runs away "
    "into the forest where nobody can find it ever again.",
]
LONG = 1000  # 再加一条1000个词元的长输入，随机挑ID，只为了够长
LAYERS = (0, 15, 29)


def reference(layer, x):
    """用transformers算同一层的FFN，(T, 576) → (T, 576)。"""
    mlp = LlamaMLP(LlamaConfig.from_pretrained(MODEL_DIR))
    for name in ("gate_proj", "up_proj", "down_proj"):
        getattr(mlp, name).weight.data = layer[name].clone()
    with torch.no_grad():
        return mlp(x)


def swiglu(x, layer, act_gate, act_up=lambda z: z):
    """按给定的激活函数拼一个FFN，用来造错误示范。"""
    gate = act_gate(x @ layer["gate_proj"].T)                # (T, 1536)
    up = act_up(x @ layer["up_proj"].T)                      # (T, 1536)
    return (gate * up) @ layer["down_proj"].T                # (T, 576)


def ffn_inputs(model, ids):
    """走一遍30层，把LAYERS里每一层FFN真正收到的输入留下来。"""
    x = model.to_embeddings(ids)                             # (T, 576)
    inputs = {}
    for i, layer in enumerate(model.weights.layers):
        h = x + model.attn(model.rms_norm(x, layer["input_norm"]), layer)
        if i in LAYERS:
            inputs[i] = model.rms_norm(h, layer["post_attn_norm"])
        x = model.block(x, layer)
    return inputs


def rel_err(ours, ref):
    return ((ours - ref).abs().max() / ref.abs().max()).item()


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()

    wrong = {
        "手写silu": lambda x, l: swiglu(x, l, lambda z: z * torch.sigmoid(z)),
        "silu放错一路": lambda x, l: swiglu(x, l, lambda z: z, F.silu),
        "换成ReLU": lambda x, l: swiglu(x, l, F.relu),
    }

    inputs = [tok.encode(text, add_special_tokens=False).ids for text in TEXTS]
    inputs.append(torch.randint(0, 49152, (LONG,), generator=torch.Generator().manual_seed(0)).tolist())

    print(f"{'T':>5}{'层':>4}{'逐位相同':>10}" + "".join(f"{k:>12}" for k in wrong))
    for ids in inputs:
        with torch.no_grad():
            xs = ffn_inputs(model, ids)
        for i in LAYERS:
            layer = model.weights.layers[i]
            ref = reference(layer, xs[i])
            same = torch.equal(model.ffn(xs[i], layer), ref)
            errs = [rel_err(f(xs[i], layer), ref) for f in wrong.values()]
            print(f"{len(ids):>5}{i:>4}{str(same):>12}" + "".join(f"{e:>14.1e}" for e in errs))

    # 逐词元：把第3行单独喂进去，和整段喂进去取第3行，结果应该一样。
    x = xs[15]
    layer = model.weights.layers[15]
    whole = model.ffn(x, layer)[3]
    alone = model.ffn(x[3:4], layer)[0]
    print(f"\n第3行单独算 vs 整段算再取第3行   相对误差{rel_err(alone, whole):.1e}")
    print("（FFN逐词元，别的行影响不到第3行；矩阵乘法按块切分不同，最后一位可能不同）")


if __name__ == "__main__":
    main()
