"""第五章 · 对拍：我们的rms_norm和transformers的LlamaRMSNorm

整条引擎现在还没法对拍（两个空位是随机数），但RMSNorm这一个算子可以。
同样的输入、同样的权重，两边的输出应该逐位相同，一个比特都不差。

输入用第四章那句话的词嵌入。为了不只在小数值上碰运气，再放大40倍试一次，
大约是真模型里前几层主干上的量级。三个位置的权重都试一遍：
第0层的两个norm，以及出口处那一个。

和`ch01/baseline.py`、`drift.py`一样，transformers只在这里当参照用。

运行：
    cd code
    uv run python book/ch05/check_rmsnorm.py
"""

import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from model import Model

IDS = [6403, 1980, 253, 655]  # 'Once upon a time'


def main():
    model = Model()
    embeddings = model.to_embeddings(IDS)  # (4, 576)

    cases = [
        ("第0层 input_norm", model.weights.layers[0]["input_norm"]),
        ("第0层 post_attn_norm", model.weights.layers[0]["post_attn_norm"]),
        ("出口 norm", model.weights.norm),
    ]
    print(f"{'权重':<22}{'输入':<12}逐位相同")
    for name, weight in cases:
        ref = LlamaRMSNorm(model.config.hidden_size, eps=model.config.rms_norm_eps)
        ref.weight.data = weight.clone()
        for label, x in [("词嵌入", embeddings), ("词嵌入×40", embeddings * 40)]:
            with torch.no_grad():
                same = torch.equal(model.rms_norm(x, weight), ref(x))
            print(f"{name:<20}{label:<10}{same}")


if __name__ == "__main__":
    main()
