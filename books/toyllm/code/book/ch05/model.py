"""第五章 · Decoder块的骨架

这一章搭肋骨，器官留给后面三章：

    rms_norm   (..., 576)  →  (..., 576)   归一化，形状不变，eps来自`config.json`
    attn       第六章再填                    空位，现在拿随机数冒充
    ffn        第八章再填                    空位，现在拿随机数冒充

一个Decoder块就是两个RMSNorm、两个空位、两条残差：

    h = h + attn(rms_norm(h))
    h = h + ffn(rms_norm(h))

norm放在残差的旁路里，主干保持干净，这叫pre-norm。代价是30个块走完还得
补最后一个norm，所以全模型一共30×2+1 = 61个RMSNorm。

两个空位现在拿随机数冒充，就像第三章那个假引擎拿随机ID冒充整个LLM一样。
形状是对的，数是假的，所以输出照样是胡言乱语。没固定种子，每次跑出来都不一样，
这本身也说明了问题：真正的层是确定的，假的才会次次不同。

运行：
    cd code
    uv run python book/ch05/model.py
    uv run python book/ch05/model.py "你想试的任意一段话"
"""

import sys

import torch
from tokenizers import Tokenizer

from config import Config
from weights import MODEL_DIR, Weights


class Model:
    """SmolLM2，这一章有了骨架，但两个空位还空着。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.config = Config(model_dir)    # 规格说明，读自`config.json`
        self.weights = Weights(model_dir)  # 权重数据，读自`model.safetensors`

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

    def rms_norm(self, x, weight):
        """归一化：(..., 576) → (..., 576)，形状不变。

        除以均方根，再乘一个可学习的权重。没有减均值，也没有偏置，这正是
        RMSNorm比LayerNorm少掉的那两样。

        统计量是在每个词元自己的576个数上算的，词元和词元之间互不影响，
        所以它和序列多长、批量多大都没关系。
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        # eps加在根号里面，护着的是均方。加到根号外面结果会悄悄变，不报错。
        # 这个1e-5不是我们定的，是`config.json`里的`rms_norm_eps`。
        return x * torch.rsqrt(variance + self.config.rms_norm_eps) * weight

    def attn(self, x):
        """注意力机制，第六章再填。现在拿随机数冒充，(T, 576) → (T, 576)。

        真正的注意力会看着`x`算出一个结果，这里我们不算，直接掷骰子。形状对了，
        残差就加得上，骨架也就接通了，至于加进去的是什么，这一章不管。

        `randn_like`的`like`指的是「规格像`x`」：形状、dtype、设备全照抄`x`，
        然后填上随机数，`x`里的值一个都不看。也没有固定种子，所以每次跑出来的
        结果都不一样，这一章不需要可复现。
        """
        return torch.randn_like(x)

    def ffn(self, x):
        """前馈网络，第八章再填。和`attn`一样，现在拿随机数冒充。"""
        return torch.randn_like(x)

    def block(self, x, layer):
        """一个Decoder块：两个norm、两个空位、两条残差，(T, 576) → (T, 576)。

        注意norm在旁路里，主干上只有加法，这就是pre-norm。
        `post_attn_norm`这个名字有点坑：它说的是「注意力子层之后的那个norm」，
        指位置，不是指post-norm。它干的仍然是pre-norm的活。
        """
        x = x + self.attn(self.rms_norm(x, layer["input_norm"]))
        x = x + self.ffn(self.rms_norm(x, layer["post_attn_norm"]))
        return x

    def forward(self, ids):
        """整条流水线：(T,) → (49152,)。

            ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits

        两个空位现在拿随机数冒充，所以主干上的向量一路被搅乱，出来的分数
        也就没什么意义。只投影最后一个位置，因为要预测的下一个词元只看它。
        """
        hidden = self.to_embeddings(ids)                   # (T,) → (T, 576)
        for layer in self.weights.layers:                  # 30个Decoder块
            hidden = self.block(hidden, layer)
        hidden = self.rms_norm(hidden, self.weights.norm)  # 出口处那一个norm
        return self.to_logits(hidden[-1])                  # (576,) → (49152,)


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    ids = tok.encode(text, add_special_tokens=False).ids
    x = model.to_embeddings(ids)
    logits = model.forward(ids)

    print(f"文字     {text!r}")
    print(f"ID       {ids}")
    print(f"形状     {(len(ids),)} → {tuple(x.shape)} → {tuple(logits.shape)}")
    print(f"RMSNorm  每层2个，共{model.weights.n_layers}层，加上出口那一个，"
          f"一共{model.weights.n_layers * 2 + 1}个")

    # 主干上的向量被30层搅成什么样了？看最后一个位置的模长。
    # 注意这是重新掷了一遍骰子，和上面forward里那一遍不是同一组随机数。
    hidden = x
    print(f"\n最后一个位置的模长，每10层看一次：")
    print(f"  第 0层前  {hidden[-1].norm():8.3f}   （就是词嵌入本身）")
    for i, layer in enumerate(model.weights.layers, 1):
        hidden = model.block(hidden, layer)
        if i % 10 == 0:
            print(f"  第{i:2d}层后  {hidden[-1].norm():8.3f}")
    normed = model.rms_norm(hidden[-1], model.weights.norm)
    print(f"  出口norm  {normed.norm():8.3f}   （最后那一个norm把尺度拉了回来）")

    print(f"\n下一个词元   {tok.id_to_token(int(logits.argmax()))!r}   "
          f"（空位是假的，这个结果自然也是假的）")


if __name__ == "__main__":
    main()
