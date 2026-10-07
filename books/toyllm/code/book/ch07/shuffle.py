"""第七章 · 重做打乱实验：有了RoPE，注意力分得清词序了

上一章说注意力对词序无感。这句话要说准确：最后一个词元不动，把它前面的
词元打乱，第0层注意力在最后一个位置的输出不变。因为这一行就是对前面所有
value的加权求和，权重只看q和k像不像，加法不分先后。

为什么只看第0层、只看最后一行：有因果掩码，前面的位置各自能看见的词元
不一样，打乱以后它们的输出跟着变，到第1层就互相牵连，整条流水线的结果
本来就会变。所以要把「分不清词序」看干净，只能看第0层的最后一行。

同一个`Model`，把`rope`换成什么都不做，就是上一章的注意力；换回来，就是
这一章的。

运行：
    cd code
    uv run python book/ch07/shuffle.py
"""

from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

# 最后一个'Ġtime'不动，只打乱前三个。
ORDERS = [[0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3], [0, 2, 1, 3], [2, 0, 1, 3]]


def last_row(model, ids):
    """第0层注意力在最后一个位置的输出，(576,)。"""
    layer = model.weights.layers[0]
    x = model.rms_norm(model.to_embeddings(ids), layer["input_norm"])  # (T, 576)
    return model.attn(x, layer)[-1]


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    ids = tok.encode("Once upon a time", add_special_tokens=False).ids

    with_rope = model.rope
    no_rope = lambda x: x  # 上一章：不转

    base = {}
    print(f"{'词元顺序':<26}{'没有RoPE':>10}{'有RoPE':>11}")  # 汉字占两格，宽度扣掉
    for order in ORDERS:
        shuffled = [ids[i] for i in order]
        diffs = []
        for name, rope in [("no", no_rope), ("yes", with_rope)]:
            model.rope = rope
            out = last_row(model, shuffled)
            base.setdefault(name, out)  # 第一行是原句，拿它当基准
            diffs.append((out - base[name]).abs().max().item())
        words = " ".join(tok.id_to_token(i).lstrip("Ġ") for i in shuffled)
        print(f"{words:<30}{diffs[0]:>12.1e}{diffs[1]:>12.1e}")
    model.rope = with_rope

    print("\n表里是和原句相比，第0层最后一行最大差了多少。没有RoPE那一列只剩"
          "浮点舍入，\n加法换了顺序，最后一位会变；有RoPE那一列差在第二位小数，"
          "词序进来了。")


if __name__ == "__main__":
    main()
