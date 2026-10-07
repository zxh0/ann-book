# 第十一章 · KV Cache

第十章接通的那个循环，每一圈都把整串从头forward一遍，生成n个词元要算1 + 2 + … + n个位置。
可因果掩码保证了：末尾追加一个词元，前面任何位置的任何中间结果都不变。重算一遍，
得到的是一模一样的数。这一章把每一层的k、v存下来，不再重算。

这一章**不新增任何部件**，总览图上一个框都不多。改的是同一条数据流的走法：
第一圈整段prompt喂进去（prefill），之后每圈只喂刚生成的那一个（decode）。

## 运行

```bash
cd code
./download_model.sh                          # 还没下权重的话，先跑这个
uv run python book/ch01/baseline.py          # 还没有第一章的基准文件的话，先跑这个
uv run python book/ch11/kv_cache.py          # 缓存长什么样，内存账
uv run python book/ch11/model.py             # prefill一次，decode三次，看形状
uv run python book/ch11/engine.py            # 默认带缓存，和第一章一字不差
uv run python book/ch11/engine.py "Once upon a time" --max-new-tokens 200
uv run python book/ch11/engine.py "Once upon a time" --max-new-tokens 200 --no-cache
uv run python book/ch11/check_cache.py       # 带缓存和不带缓存逐个ID对拍，再故意做错两种
uv run python book/ch11/check_forward.py     # 第十章的整机对拍，照样四级全过
uv run python book/ch11/speed.py             # 快了多少，prefill和decode差在哪
```

## 文件

| 文件 | 作用 |
|---|---|
| `kv_cache.py` | **这一章新增**。`KVCache`真存，`NoCache`不存，接口一样；`NO_CACHE`是默认 |
| `model.py` | **这一章的重点**。`forward` → `block` → `attn`一路多传一个`cache`和层号`i`；`rope`多收一个`start` |
| `engine.py` | `stream`里喂什么只有一句`ids[cache.length:]`；`cache`收True / False / 一个已有的`KVCache` |
| `check_cache.py` | **这一章新增**。带缓存 == 不带缓存，逐个ID；接着上一轮往下说；故意做错两种 |
| `check_forward.py` | 第十章那个，文件头改了，`block`多传一个层号 |
| `speed.py` | **这一章新增**。每个词元的耗时曲线，prefill和decode |
| `config.py` | 第十章那个，**一字未改**。缓存要的层数、头数、头宽都是现成的 |
| `sampler.py`、`detokenizer.py` | 第十章那两个，**一字未改** |
| `weights.py` | 第二章那个，**一字未改** |

## 缓存是一个开关，不是一条分支

`NoCache`的`update`把k、v原样返回，`length`永远是0。模型代码里没有一个
`if cache is not None`：

| | `NO_CACHE` | `KVCache` |
|---|---|---|
| `rope`从第几个位置起 | 0 | `cache.length` |
| `update`返回 | 这次的k、v，各(3, T, 64) | 历史加上这次，各(3, length + T, 64) |
| 分数表 | (9, T, T) | (9, T, length + T) |
| 引擎每圈喂 | `ids[0:]`，整串 | `ids[length:]`，第一圈整段，之后1个 |

传`NO_CACHE`，一切退回第十章，一位都不差，所以`check_forward.py`二、三两级的数字和第十章完全相同。

`model.py`的输出：

```
prefill  喂4个   x (4, 576)   分数表 (9, 4, 4)   缓存里存了4个，每层k (3, 4, 64)
decode   喂1个   x (1, 576)   分数表 (9, 1, 5)   缓存里存了5个   ','
decode   喂1个   x (1, 576)   分数表 (9, 1, 6)   缓存里存了6个   'Ġthere'
decode   喂1个   x (1, 576)   分数表 (9, 1, 7)   缓存里存了7个   'Ġwas'
```

## 三个坑

都不报错，只是结果悄悄错了。

1. **存RoPE之后的k。** 位置在写进去那一刻就定了，历史永远不用再转。
2. **`rope`从`cache.length`起，不是从0起。** `check_cache.py`里故意写错，第2个新词元就跑偏了。
3. **30层共用一个偏移，全写完才`advance`一次。** `update`里不推进`length`，由`Model.forward`在30层之后推一次。
   在每层里推的话，第1层写到第T个位置之后，第2层更往后，读回来的历史里全是没写过的0。

## 内存账

`kv_cache.py`：

```
  每个词元每层   K + V = 2 × 3 × 64 = 384个数
  fp32           384 × 4 = 1,536字节
  30层           1,536 × 30 = 46,080字节
  8192个位置     46,080 × 8192 = 377,487,360字节

  真分配一个     377,487,360字节，和算的相同

         8192         377.5 MB        1132.5 MB      ← 3个kv头 vs 9个
```

GQA省的不是参数，是缓存。第六章那个9个q头、3个kv头的设计，在这里才兑现。

## 引擎现状

```
ids → embed → [norm → attn → norm → ffn] × 30 → norm → lm_head → logits → sampler → id
 ^                  ↕                                                              |
 |              KV Cache                                                           |
 +---------------------------- append to，只喂新的那一个 -----------------------------+
```

剩最后一个问题：注意力那张(9, T, L)的分数表，序列一长就会撑爆内存。下一章的FlashAttention治这个。
