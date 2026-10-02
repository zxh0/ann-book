"""第五章 · 偷看一眼：真模型里的数值是怎么漂的

我们自己的引擎到这一章还只有骨架，两个空位填的是随机数，量不出真模型的数值。
所以这里借第一章用过的transformers，把真模型跑一遍，提前看看结果。

做法是给30个Decoder块的出口各挂一个钩子（forward hook）：每过完一层，
就记下最后一个词元的那一行向量，量它的模长。出口那个norm也挂一个。

等第九章引擎完整了，我们自己的引擎也能量出同样的数。

运行：
    cd code
    uv run python book/ch05/drift.py
    uv run python book/ch05/drift.py "你想试的任意一段话"
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from weights import MODEL_DIR

SHOW = (1, 10, 20, 30)  # 打印过完哪几层之后的模长


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ids = tok(text, return_tensors="pt")["input_ids"]

    # 钩子拿到的是这一层的输出，形状(1, T, 576)，我们只留最后一个词元那一行。
    # 有的版本返回张量，有的返回元组，元组的第一个才是隐状态。
    seen = {}

    def keep(key):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            seen[key] = out[0, -1].detach()
        return hook

    for i, layer in enumerate(model.model.layers, 1):
        layer.register_forward_hook(keep(i))
    model.model.norm.register_forward_hook(keep("norm"))

    with torch.no_grad():
        embedding = model.model.embed_tokens(ids)[0, -1]
        model(ids)

    print(f"文字     {text!r}")
    print(f"ID       {ids[0].tolist()}")
    print(f"\n最后一个词元的模长：")
    print(f"  词嵌入    {embedding.norm():8.3f}")
    for i in SHOW:
        print(f"  第{i:2d}层后  {seen[i].norm():8.3f}")
    print(f"  出口norm  {seen['norm'].norm():8.3f}")

    # 模长的平方就是各维平方之和，看看最大的那一维占了多少。
    last = seen[len(model.model.layers)]
    dim = int(last.abs().argmax())
    share = float(last[dim] ** 2 / last.pow(2).sum())
    print(f"\n第{len(model.model.layers)}层后绝对值最大的是第{dim}维，"
          f"值为{float(last[dim]):.3f}，占模长平方的{share:.1%}")


if __name__ == "__main__":
    main()
