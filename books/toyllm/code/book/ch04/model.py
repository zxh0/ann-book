"""第四章 · 零层的LLM

一个矩阵，两个方向：

    to_embeddings   (T,)        →  (T, 576)       入口，按行号取行
    to_logits      (..., 576)  →  (..., 49152)   出口，和词表里每一行做点积

两个方向用的是同一份权重，所以`model.safetensors`里找不到`lm_head.weight`。

把两个方向直接接起来，就是一个零层的LLM：中间那30个Decoder块一个都还没有，
所以它回答的其实只是「词表里和当前这个词最像的是谁」。

运行：
    cd code
    uv run python book/ch04/model.py
    uv run python book/ch04/model.py "你想试的任意一段话"
"""

import sys

from tokenizers import Tokenizer

from weights import MODEL_DIR, Weights


class Model:
    """SmolLM2，目前只有头和尾。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.weights = Weights(model_dir)

    def to_embeddings(self, ids):
        """入口：词元ID换成词嵌入，(T,) → (T, 576)。

        按行号取行，连矩阵乘法都不是，就是一次索引。
        `ids`收的是一串，哪怕只有一个词元，也要写成`[id]`，出来的永远是T行。
        """
        return self.weights.embed_tokens[ids]

    def to_logits(self, x):
        """出口：词嵌入换成分数，(..., 576) → (..., 49152)。

        和词表里每一行做点积，谁像谁得分高。用的还是入口那张表，转置一下而已。
        只看最后一维：给一个向量出一行分数，给T行就出T行。
        """
        return x @ self.weights.embed_tokens.T

    def forward(self, ids):
        """整个引擎，目前就这两步：(T,) → (49152,)。

        中间本该有30个Decoder块，现在一个都没有，所以这是个零层的LLM。
        只投影最后一个位置，因为要预测的下一个词元只看它。
        """
        embeddings = self.to_embeddings(ids)
        return self.to_logits(embeddings[-1])


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    ids = tok.encode(text, add_special_tokens=False).ids
    x = model.to_embeddings(ids)
    logits = model.to_logits(x)

    print(f"文字     {text!r}")
    print(f"ID       {ids}")
    print(f"形状     {(len(ids),)} → {tuple(x.shape)} → {tuple(logits.shape)}")

    # 每个位置各自预测，而不是只看最后一个：`to_logits`收多少行就出多少行。
    # 每个位置都预测出自己，因为没有任何东西把别的位置混进来，上下文还不存在。
    print("\n每个位置各自预测下一个词元：\n")
    for i, (tid, nid) in enumerate(zip(ids, logits.argmax(-1).tolist())):
        print(f"  第{i}个  {tok.id_to_token(tid)!r:<12} → {tok.id_to_token(nid)!r}")


if __name__ == "__main__":
    main()
