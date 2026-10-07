"""第六章 · 填上注意力

第五章搭好了骨架，留了两个空位。这一章填第一个：

    rms_norm   (..., 576)  →  (..., 576)   归一化，第五章写的
    attn       (T, 576)    →  (T, 576)     注意力，这一章写的
    ffn        第八章再填                    空位，现在返回全0

一个Decoder块还是两个RMSNorm、两条残差，只是第一个空位有了真东西：

    h = h + attn(rms_norm(h))
    h = h + ffn(rms_norm(h))     ffn返回0，这一行现在等于什么都没做

注意力是整个模型里唯一让词元之间交换信息的部件，从这一章起，前面的词元才
真正参与计算。它内部先把576维拆成9个q头、3个kv头，每头64维，算完再拼回576维。

第五章两个空位都拿随机数冒充；这一章没有随机数了，30层都是确定的计算，
同样的输入，每次跑出来都一样。但还缺位置编码（第七章）和FFN（第八章），
所以输出仍然不通顺。

运行：
    cd code
    uv run python book/ch06/model.py
    uv run python book/ch06/model.py "你想试的任意一段话"
"""

import sys

import torch
from tokenizers import Tokenizer

from config import Config
from weights import MODEL_DIR, Weights


class Model:
    """SmolLM2，这一章有了注意力，FFN那个空位还空着。"""

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

    def attn(self, x, layer):
        """注意力机制：(T, 576) → (T, 576)，形状不变。

        整个模型里只有这一步让词元之间互相看得见：第i行的输出，是前i行的
        value按分数加权求和。分数来自query和key的点积，除以√64，过因果掩码，
        再过softmax。

        9个q头，3个kv头，每3个q头共用一组kv，这就是GQA。
        还没有位置编码，第七章再补。
        """
        T = x.shape[0]
        n_heads = self.config.num_attention_heads              # 9
        n_kv_heads = self.config.num_key_value_heads           # 3
        head_dim = self.config.hidden_size // n_heads          # 576 / 9 = 64

        # 三次线性投影。权重按(out, in)存放，所以要转置。
        q = x @ layer["q_proj"].T                              # (T, 576) → (T, 576)
        k = x @ layer["k_proj"].T                              # (T, 576) → (T, 192)
        v = x @ layer["v_proj"].T                              # (T, 576) → (T, 192)

        # 拆头：先把最后一维切成(头数, 64)，再把头挪到最前面。
        q = q.view(T, n_heads, head_dim).transpose(0, 1)       # (9, T, 64)
        k = k.view(T, n_kv_heads, head_dim).transpose(0, 1)    # (3, T, 64)
        v = v.view(T, n_kv_heads, head_dim).transpose(0, 1)    # (3, T, 64)

        # GQA：每个kv头连续复制3份，排成[0,0,0,1,1,1,2,2,2]，对齐9个q头。
        # 写成k.repeat(3, 1, 1)会排成[0,1,2,0,1,2,0,1,2]，形状一样，不报错，全错。
        group = n_heads // n_kv_heads                          # 3
        k = k.repeat_interleave(group, dim=0)                  # (3, T, 64) → (9, T, 64)
        v = v.repeat_interleave(group, dim=0)                  # (3, T, 64) → (9, T, 64)

        # 每个头各算一张T×T的分数表：第i行第j列是词元i的q和词元j的k的点积。
        scores = q @ k.transpose(1, 2) / head_dim ** 0.5       # (9, T, T)

        # 因果掩码：右上角（j > i，也就是未来）置成-inf，softmax之后正好是0。
        future = torch.ones(T, T, dtype=torch.bool).triu(1)   # (T, T)
        scores = scores.masked_fill(future, float("-inf"))

        probs = torch.softmax(scores, dim=-1)                  # (9, T, T)，每行和为1
        out = probs @ v                                        # (9, T, 64)

        # 拼头：把头挪回去，9个64维首尾相接拼回576维，再过o_proj把各头混合起来。
        out = out.transpose(0, 1).reshape(T, n_heads * head_dim)  # (T, 576)
        return out @ layer["o_proj"].T                         # (T, 576) → (T, 576)

    def ffn(self, x):
        """前馈网络，第八章再填。现在返回全0，(T, 576) → (T, 576)。

        返回0，`block`里那一行就成了`x = x + 0`，这个子层等于不存在，
        看到的就只是注意力的效果。注意不能返回`x`：这里收到的是归一化之后的`x`，
        原样返回会每层往主干上多加一份。
        """
        return torch.zeros_like(x)

    def block(self, x, layer):
        """一个Decoder块：两个norm、注意力加一个空位、两条残差，(T, 576) → (T, 576)。

        注意norm在旁路里，主干上只有加法，这就是pre-norm。
        `post_attn_norm`这个名字有点坑：它说的是「注意力子层之后的那个norm」，
        指位置，不是指post-norm。它干的仍然是pre-norm的活。
        """
        x = x + self.attn(self.rms_norm(x, layer["input_norm"]), layer)
        x = x + self.ffn(self.rms_norm(x, layer["post_attn_norm"]))
        return x

    def forward(self, ids):
        """整条流水线：(T,) → (49152,)。

            ids → embed → [norm → attn → norm → □] × 30 → norm → lm_head → logits

        注意力填上了，FFN还空着，也还没有位置编码，所以出来的分数还谈不上
        靠谱。只投影最后一个位置，因为要预测的下一个词元只看它。
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

    # 主干上的向量过完30层变成什么样了？看最后一个位置的模长。
    # 这里把forward里那30层重新走一遍，没有随机数，所以和forward算的是同一个东西。
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
          f"（还缺位置编码和FFN，这个结果还当不得真）")


if __name__ == "__main__":
    main()
