"""第十一章 · 同一个模型，带上缓存

第八章填满了最后一个空位，之后模型一个字没动。这一章也不加新部件，
只让`forward`多收一个`cache`参数，从`forward`一路传到`attn`：

    forward(ids, cache)  →  block(x, layer, i, cache)  →  attn(x, layer, i, cache)

整个模型里只有注意力让词元之间互相看得见，所以也只有它需要历史。
norm、FFN、出口都是逐词元的，新词元自己算自己的就行，一行不用改。

`attn`里改了三处：

    rope(q), rope(k)     从cache.length起转，不再从0起
    k, v = cache.update  新算的k、v写进缓存，换回全部历史
    掩码                  (T, T)变成(T, 已存 + T)；T为1时整个跳过

传`NO_CACHE`的话，length是0，update原样返回，三处全部退化成第八章的样子，
一位都不差。

运行：
    cd code
    uv run python book/ch11/model.py
    uv run python book/ch11/model.py "你想试的任意一段话"
"""

import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import Config
from kv_cache import NO_CACHE, KVCache
from weights import MODEL_DIR, Weights


class Model:
    """SmolLM2，这一章所有空位都填满了。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.config = Config(model_dir)    # 规格说明，读自`config.json`
        self.weights = Weights(model_dir)  # 权重数据，读自`model.safetensors`
        self.cos, self.sin = self.rope_table()  # 各(8192, 64)，算一次，30层共用

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

    def rope_table(self):
        """预先算好RoPE要用的cos/sin表，各(8192, 64)。

        64维看成32个平面，第i个平面在位置m上转m * inv_freq[i]这么大的角度：

            inv_freq[i] = 1 / rope_theta ** (2i / 64)     i = 0 .. 31

        i越小转得越快，i越大转得越慢。表只和位置有关，和输入无关，
        所以启动时算一次，30层、每一圈都查同一张表。
        """
        head_dim = self.config.hidden_size // self.config.num_attention_heads  # 64
        theta = self.config.rope_theta                         # 100000，不是常见的10000
        # 下面的rotate_half按隔半个头配对。config要是说相邻配对，这里就算错了，
        # 而且不报错，所以先拦一下。
        assert not self.config.rope_interleaved

        # 这一行照抄transformers的写法：先用整数排好0, 2, ..., 62，再转成浮点数。
        # 写法不同，inv_freq的最后一位就可能不同。
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # (32,)

        positions = torch.arange(self.config.max_position_embeddings).float()  # (8192,)
        angles = torch.outer(positions, inv_freq)              # (8192, 32)
        # 第i维和第i+32维是一对，转同一个角度，所以32个角度抄两遍，拼成64个。
        angles = torch.cat((angles, angles), dim=-1)           # (8192, 64)
        return angles.cos(), angles.sin()

    def rotate_half(self, x):
        """把每一对(a, b)转90度，变成(-b, a)，(..., 64) → (..., 64)。

            [a0 a1 .. a31 | b0 b1 .. b31]  →  [-b0 -b1 .. -b31 | a0 a1 .. a31]

        第i维和第i+32维配成一对，叫half-split，transformers就是这么写的。
        原始论文配的是相邻两维(0,1), (2,3), ...，两种都是真旋转，都能生成通顺
        的文本，用错了不报错，只有对拍分得出来。
        """
        a, b = x.chunk(2, dim=-1)                              # 各(..., 32)
        return torch.cat((-b, a), dim=-1)                      # (..., 64)

    def rope(self, x, start=0):
        """旋转位置编码：第m个位置转m倍的角度，(heads, T, 64) → (heads, T, 64)。

        每一对(a, b)转θ角，就是二维旋转那个公式：

            a' = a·cosθ - b·sinθ
            b' = b·cosθ + a·sinθ

        写成整段向量，就是下面那一行。只转q和k，不转v：位置要影响的是
        「谁和谁匹配」，不是「匹配上了取什么内容」。

        这一章多了`start`：x的第0行不一定在第0个位置。缓存里已经存了start个，
        新来的这T个就在start .. start+T-1。写成从0起不会报错，只是RoPE按错的
        距离编码，模型悄悄变笨。
        """
        T = x.shape[-2]
        cos = self.cos[start:start + T]                        # (T, 64)
        sin = self.sin[start:start + T]                        # (T, 64)
        return x * cos + self.rotate_half(x) * sin            # 广播到每个头

    def attn(self, x, layer, i, cache=NO_CACHE):
        """注意力机制：(T, 576) → (T, 576)，形状不变。

        整个模型里只有这一步让词元之间互相看得见：第i行的输出，是前i行的
        value按分数加权求和。分数来自query和key的点积，除以√64，过因果掩码，
        再过softmax。

        9个q头，3个kv头，每3个q头共用一组kv，这就是GQA。

        这一章x不一定是整串了：prefill时是整段prompt，T个；decode时只有
        刚生成的那一个，T为1。前面的k、v从缓存里取，所以分数表从T×T变成
        T×(已存 + T)。`i`是第几层，缓存靠它找到这一层的那一份。
        """
        T = x.shape[0]
        n_heads = self.config.num_attention_heads              # 9
        n_kv_heads = self.config.num_key_value_heads           # 3
        head_dim = self.config.hidden_size // n_heads          # 576 / 9 = 64
        start = cache.length                                   # x的第0行在第几个位置

        # 三次线性投影。权重按(out, in)存放，所以要转置。
        q = x @ layer["q_proj"].T                              # (T, 576) → (T, 576)
        k = x @ layer["k_proj"].T                              # (T, 576) → (T, 192)
        v = x @ layer["v_proj"].T                              # (T, 576) → (T, 192)

        # 拆头：先把最后一维切成(头数, 64)，再把头挪到最前面。
        q = q.view(T, n_heads, head_dim).transpose(0, 1)       # (9, T, 64)
        k = k.view(T, n_kv_heads, head_dim).transpose(0, 1)    # (3, T, 64)
        v = v.view(T, n_kv_heads, head_dim).transpose(0, 1)    # (3, T, 64)

        # RoPE：q和k按位置转一下，v不动。位置从start起，不是从0起。
        q = self.rope(q, start)                                # (9, T, 64)
        k = self.rope(k, start)                                # (3, T, 64)

        # 存的是转过之后的k：位置在写进去那一刻就定了，历史永远不用再转。
        # 存的是复制之前的3个头：缓存只有9个头时的三分之一。
        k, v = cache.update(i, k, v)                           # (3, T, 64) → (3, L, 64)
        L = k.shape[1]                                         # start + T

        # GQA：每个kv头连续复制3份，排成[0,0,0,1,1,1,2,2,2]，对齐9个q头。
        # 写成k.repeat(3, 1, 1)会排成[0,1,2,0,1,2,0,1,2]，形状一样，不报错，全错。
        group = n_heads // n_kv_heads                          # 3
        k = k.repeat_interleave(group, dim=0)                  # (3, L, 64) → (9, L, 64)
        v = v.repeat_interleave(group, dim=0)                  # (3, L, 64) → (9, L, 64)

        # 每个头各算一张T×L的分数表：第r行第j列是第start+r个词元的q
        # 和第j个词元的k的点积。不带缓存时start是0，L就是T，还是那张T×T。
        scores = q @ k.transpose(1, 2) / head_dim ** 0.5       # (9, T, L)

        # 因果掩码：第r行在第start+r个位置，它右边（j > start+r）是未来。
        # T为1时那唯一一行能看见全部历史，掩码全是False，干脆不建。
        if T > 1:
            future = torch.ones(T, L, dtype=torch.bool).triu(start + 1)  # (T, L)
            scores = scores.masked_fill(future, float("-inf"))

        probs = torch.softmax(scores, dim=-1)                  # (9, T, L)，每行和为1
        out = probs @ v                                        # (9, T, 64)

        # 拼头：把头挪回去，9个64维首尾相接拼回576维，再过o_proj把各头混合起来。
        out = out.transpose(0, 1).reshape(T, n_heads * head_dim)  # (T, 576)
        return out @ layer["o_proj"].T                         # (T, 576) → (T, 576)

    def ffn(self, x, layer):
        """前馈网络，SwiGLU：(T, 576) → (T, 1536) → (T, 576)，形状不变。

        gate和up两路并行升维，gate那一路过SiLU当开关，两路逐元素相乘，
        再由down降回来：

            ffn(x) = down( silu(gate(x)) * up(x) )
            silu(z) = z * sigmoid(z)

        只有gate那一路过SiLU，up那一路不过。把激活函数放错一路，或者两路都过，
        数字看上去都很正常，不报错，模型就是坏的。

        这里每一步都只动最后一维，词元和词元之间互不相干：
        第i行的输出只取决于第i行的输入。
        """
        gate = x @ layer["gate_proj"].T                        # (T, 576) → (T, 1536)
        up = x @ layer["up_proj"].T                            # (T, 576) → (T, 1536)
        # 用F.silu而不是手写z * sigmoid(z)：两种写法数学上相同，
        # 但最后一位会差，`check_ffn.py`里能看到。用F.silu才和transformers逐位相同。
        h = F.silu(gate) * up                                  # (T, 1536)
        return h @ layer["down_proj"].T                        # (T, 1536) → (T, 576)

    def block(self, x, layer, i, cache=NO_CACHE):
        """一个Decoder块：两个norm、注意力、FFN、两条残差，(T, 576) → (T, 576)。

        注意norm在旁路里，主干上只有加法，这就是pre-norm。
        `post_attn_norm`这个名字有点坑：它说的是「注意力子层之后的那个norm」，
        指位置，不是指post-norm。它干的仍然是pre-norm的活。

        缓存只交给注意力，FFN用不着它。
        """
        x = x + self.attn(self.rms_norm(x, layer["input_norm"]), layer, i, cache)
        x = x + self.ffn(self.rms_norm(x, layer["post_attn_norm"]), layer)
        return x

    def forward(self, ids, cache=NO_CACHE):
        """整条流水线：(T,) → (49152,)。

            ids → embed → [norm → attn → norm → ffn] × 30 → norm → lm_head → logits

        只投影最后一个位置，因为要预测的下一个词元只看它。

        带缓存时，`ids`只给缓存里还没有的那几个：prefill给整段prompt，
        decode只给刚生成的那一个。30层都写在同一个偏移上，全部写完，
        最后统一`advance`一次。
        """
        hidden = self.to_embeddings(ids)                   # (T,) → (T, 576)
        for i, layer in enumerate(self.weights.layers):    # 30个Decoder块
            hidden = self.block(hidden, layer, i, cache)
        cache.advance(len(ids))                            # 30层都写完了，才挪偏移
        hidden = self.rms_norm(hidden, self.weights.norm)  # 出口处那一个norm
        return self.to_logits(hidden[-1])                  # (576,) → (49152,)


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ids = tok.encode(text, add_special_tokens=False).ids
    T = len(ids)

    print(f"文字     {text!r}")
    print(f"ID       {ids}\n")

    # ---- prefill：整段prompt一次喂进去 ----------------------------------------
    cache = KVCache(model.config, max_len=T + 3)
    logits = model.forward(ids, cache)                 # (T,) → (49152,)
    print(f"prefill  喂{T}个   x {(T, 576)}   分数表 {(9, T, T)}"
          f"   缓存里存了{cache.length}个，每层k {tuple(cache.k[0][:, :cache.length].shape)}")

    # ---- decode：每次只喂刚生成的那一个 ----------------------------------------
    out = list(ids)
    for _ in range(3):
        next_id = int(logits.argmax())
        out.append(next_id)
        L = cache.length + 1
        logits = model.forward([next_id], cache)       # (1,) → (49152,)
        print(f"decode   喂1个   x {(1, 576)}   分数表 {(9, 1, L)}"
              f"   缓存里存了{cache.length}个   {tok.id_to_token(next_id)!r}")

    # ---- 和不带缓存的比一比 ----------------------------------------------------
    # 最后一圈的logits，不带缓存的话要把整串重新喂一遍才得到。
    full = model.forward(out)                          # (T + 3,) → (49152,)
    diff = (logits - full).abs().max().item()
    print(f"\n不带缓存，整串{len(out)}个重算一遍，最后一个位置的logits：")
    print(f"  最大绝对误差 {diff:.1e}，最高分都是"
          f"{tok.id_to_token(int(logits.argmax()))!r}："
          f"{'相同' if logits.argmax() == full.argmax() else '不同'}")


if __name__ == "__main__":
    main()
