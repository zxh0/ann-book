"""第二章 · 把权重读成一个对象

`load_weights.py` 是按字节把文件拆开看，这里换个角度：把272个张量真的读进内存，
整理成后面各章直接拿来就能用的样子。

一个`Weights`对象就是三样东西：

    embed_tokens   (49152, 576)   入口的词嵌入表，出口的线性投影用的也是它
    layers         30个dict       每个dict装9个张量，按数据流的顺序排
    norm           (576,)         整个模型最后那一个RMSNorm

文件里存的是bf16，读进来统一升成fp32，所以内存占用是文件的两倍，大约538MB。

运行：
    cd code
    uv run python book/ch02/weights.py
"""

from pathlib import Path

import torch
from safetensors import safe_open

# code/book/ch02/weights.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"


def count_layers(names):
    """层数是数出来的，不是写死的：看看`model.layers.N.`里的N最大到几。

    `config.json`里当然写着30，不过那个文件我们留到第四章再打开。
    """
    ids = [int(n.split(".")[2]) for n in names if n.startswith("model.layers.")]
    return max(ids) + 1


class Weights:
    """SmolLM2的全部权重。"""

    def __init__(self, model_dir=MODEL_DIR, dtype=torch.float32):
        path = Path(model_dir) / "model.safetensors"
        with safe_open(path, framework="pt") as f:

            def get(name): # 读一个张量，顺便从bf16升成fp32。
                return f.get_tensor(name).to(dtype)

            self.embed_tokens = get("model.embed_tokens.weight")
            self.n_layers = count_layers(f.keys())

            # 每层9个张量。这里按数据流的顺序排：先归一化，再注意力，
            # 再归一化，最后FFN。文件里它们是按名字的字母序躺着的。
            p = "model.layers" # 纯粹为了书面排版，省点宽度
            self.layers = [
                {
                    "input_norm"    : get(f"{p}.{i}.input_layernorm.weight"),
                    "q_proj"        : get(f"{p}.{i}.self_attn.q_proj.weight"),
                    "k_proj"        : get(f"{p}.{i}.self_attn.k_proj.weight"),
                    "v_proj"        : get(f"{p}.{i}.self_attn.v_proj.weight"),
                    "o_proj"        : get(f"{p}.{i}.self_attn.o_proj.weight"),
                    "post_attn_norm": get(f"{p}.{i}.post_attention_layernorm.weight"),
                    "gate_proj"     : get(f"{p}.{i}.mlp.gate_proj.weight"),
                    "up_proj"       : get(f"{p}.{i}.mlp.up_proj.weight"),
                    "down_proj"     : get(f"{p}.{i}.mlp.down_proj.weight"),
                }
                for i in range(self.n_layers)
            ]

            self.norm = get("model.norm.weight")

    @property
    def n_tensors(self):
        return 2 + sum(len(layer) for layer in self.layers)

    @property
    def n_params(self):
        """参数总数。线性投影和词嵌入共用一份权重，所以只算一次。"""
        total = self.embed_tokens.numel() + self.norm.numel()
        return total + sum(t.numel() for layer in self.layers for t in layer.values())


def main():
    w = Weights()

    print(f"{'embed_tokens':<18}{str(tuple(w.embed_tokens.shape)):>14}")
    print("layers")
    for name, t in w.layers[0].items():
        print(f"  {name:<16}{str(tuple(t.shape)):>14}{t.numel():>12,}")
    print(f"{'norm':<18}{str(tuple(w.norm.shape)):>14}")

    print()
    print(f"张量           {w.n_tensors} 块")
    print(f"层数           {w.n_layers}")
    print(f"每层           {len(w.layers[0])}个张量")
    per_layer = sum(t.numel() for t in w.layers[0].values())
    print(f"一层合计       {per_layer:,}")
    print(f"{w.n_layers}层合计       {per_layer * w.n_layers:,}")
    print(f"全部参数       {w.n_params:,}")


if __name__ == "__main__":
    main()
