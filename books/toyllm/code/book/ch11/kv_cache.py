"""第十一章 · KV Cache

第十章的生成循环，每一圈都把整串从头forward一遍。可因果掩码保证了：在末尾追加
一个词元，前面任何一个位置的任何中间结果都不会变，它们看不见新来的这个。
所以第n圈算出来的前n-1个位置，和第n-1圈逐位相同，重算一遍纯属白费。

要存的是每一层的k和v。新词元要用自己的q去和全部历史的k匹配，再从全部历史的v里
加权取值。q只要新词元自己的，k和v却要全套历史，那就把它们存下来：

    每层一份k    (3, 已存长度, 64)
    每层一份v    (3, 已存长度, 64)

是3个kv头，不是9个。GQA省下的东西在这里兑现。

缓存做成一个开关，不做成一条分支。两个类，接口一样：

    KVCache    真存。update把新的k、v写进去，返回从头到现在的全部
    NoCache    不存。update原样返回，length永远是0

模型代码里一个`if cache is not None`都没有：传`NO_CACHE`，位置和掩码的计算自动
退化成第十章的样子。和`GreedySampler` / `RandomSampler`是同一个思路。

运行：
    cd code
    uv run python book/ch11/kv_cache.py
"""

import torch

from config import Config


class NoCache:
    """什么都不存的缓存。默认就是它，引擎的行为和第十章完全一样。"""

    length = 0  # 永远是0：位置从0数起，掩码就是T×T那张

    def update(self, i, k, v):
        """第i层的k、v原样返回，各(3, T, 64)。"""
        return k, v

    def advance(self, t):
        pass

    def __repr__(self):
        return "NoCache()"


NO_CACHE = NoCache()  # 只要一个，大家共用


class KVCache(NoCache):
    """30层的k、v，各存一份。

    空间一次分配好，`max_len`个位置，往里填。不用`torch.cat`一圈一圈地接：
    那样每来一个词元，都要把整段历史重新分配、复制一遍。
    """

    def __init__(self, config, max_len, dtype=torch.float32):
        n_kv_heads = config.num_key_value_heads                       # 3
        head_dim = config.hidden_size // config.num_attention_heads    # 64
        shape = (n_kv_heads, max_len, head_dim)                        # (3, max_len, 64)
        self.k = [torch.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.max_len = max_len
        self.length = 0  # 已经存了几个位置，盖掉NoCache那个类属性

    def update(self, i, k, v):
        """把第i层这一次的k、v写到末尾，返回从头到现在的全部。

            k, v 进   (3, T, 64)          这一次新算的T个位置
            k, v 出   (3, length + T, 64) 历史加上这一次

        这里**不**推进`length`：30层都写在同一个偏移上，要等30层全部写完，
        `Model.forward`再统一`advance`一次。在这里推进的话，后面29层就全错位了。
        """
        T = k.shape[1]
        end = self.length + T
        if end > self.max_len:
            raise ValueError(f"缓存满了：要存到第{end}个位置，只开了{self.max_len}个")
        self.k[i][:, self.length:end] = k
        self.v[i][:, self.length:end] = v
        return self.k[i][:, :end], self.v[i][:, :end]

    def advance(self, t):
        """30层都写完了，偏移整体往后挪t个位置。"""
        self.length += t

    @property
    def n_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def __repr__(self):
        return f"KVCache(length={self.length}, max_len={self.max_len})"


def main():
    config = Config()
    n_layers = config.num_hidden_layers                        # 30
    n_heads = config.num_attention_heads                       # 9
    n_kv_heads = config.num_key_value_heads                    # 3
    head_dim = config.hidden_size // n_heads                   # 64
    ctx = config.max_position_embeddings                       # 8192

    cache = KVCache(config, max_len=16)
    print(f"每层一份k、一份v，各{tuple(cache.k[0].shape)}，"
          f"也就是(kv头数, 最多存几个位置, 64)")
    print(f"一共{len(cache.k)}层 × 2 = {len(cache.k) * 2}个张量\n")

    # ---- 内存账 ---------------------------------------------------------------
    per_layer = 2 * n_kv_heads * head_dim                      # K和V
    per_token = per_layer * 4 * n_layers                       # fp32，30层
    print("内存账：")
    print(f"  每个词元每层   K + V = 2 × {n_kv_heads} × {head_dim} = {per_layer}个数")
    print(f"  fp32           {per_layer} × 4 = {per_layer * 4:,}字节")
    print(f"  {n_layers}层           {per_layer * 4:,} × {n_layers} = {per_token:,}字节")
    print(f"  {ctx}个位置     {per_token:,} × {ctx} = {per_token * ctx:,}字节\n")

    # 真分配一个8192的出来，看看和算出来的对不对得上
    full = KVCache(config, max_len=ctx)
    print(f"  真分配一个     {full.n_bytes:,}字节，"
          f"{'和算的相同' if full.n_bytes == per_token * ctx else '和算的不同'}")
    del full

    print(f"\n  {'上下文':>8}{'GQA，3个kv头':>16}{'不用GQA，9个':>16}")
    for n in (128, 1024, 8192):
        gqa = per_token * n / 1e6
        print(f"  {n:>10}{gqa:>14.1f} MB{gqa * n_heads / n_kv_heads:>14.1f} MB")
    # 权重和缓存都按fp32算，才是同一把尺子。134,515,008是第二章数出来的参数量。
    weights = 134_515_008 * 4 / 1e6
    print(f"\n  权重本身fp32是{weights:.1f} MB。上下文{ctx}时，"
          f"一条序列的缓存就有权重的{per_token * ctx / 1e6 / weights:.0%}")

    # ---- NoCache --------------------------------------------------------------
    k = v = torch.randn(n_kv_heads, 5, head_dim)
    k2, v2 = NO_CACHE.update(0, k, v)
    print(f"\nNO_CACHE.update   原样返回：{k2 is k and v2 is v}")
    NO_CACHE.advance(5)
    print(f"NO_CACHE.length   advance之后还是{NO_CACHE.length}")


if __name__ == "__main__":
    main()
