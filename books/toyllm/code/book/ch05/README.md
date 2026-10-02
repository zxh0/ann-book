# 第五章 · 归一化（Normalization）

第四章的引擎只有两头，中间空着。这一章把中间那 30 层的**骨架**搭起来：两个
RMSNorm、两个空位、两条残差。骨架是结构，注意力和 FFN 是塞进这个结构里的内容，
所以先有结构，后面三章才有地方放东西。

这一章真正写出来的只有 RMSNorm 一个算子，两行：

```python
variance = x.pow(2).mean(-1, keepdim=True)
return x * torch.rsqrt(variance + self.config.rms_norm_eps) * weight
```

全模型一共 **61 个** RMSNorm：30 个块各两个，加上出口那一个，30×2+1。参数量
61 × 576 = **35,136**，占全模型万分之二点六，是所有部件里最便宜的一个。

## 运行

```bash
cd code
./download_model.sh                        # 还没下权重的话，先跑这个
uv run python book/ch05/config.py          # 把 config.json 摊开看一眼
uv run python book/ch05/model.py           # 骨架走一遍，看模长怎么漂
uv run python book/ch05/model.py "任意一段话"
uv run python book/ch05/engine.py          # 接上两头转 10 圈
uv run python book/ch05/drift.py           # 借 transformers 偷看真模型的模长怎么漂
uv run python book/ch05/check_rmsnorm.py   # rms_norm 和 LlamaRMSNorm 逐位对拍
```

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | **这一章新增**。`Config` 类：把 `config.json` 读进来。第二章说的「等后面用到它时再打开」，就是现在 |
| `model.py` | `Model` 类：新增 `rms_norm`、`block`，以及 `attn` / `ffn` 两个假的空位 |
| `engine.py` | `Engine` 类和第四章**一字未改**，它只认 `model.forward`，骨架换了照样转；改的只是 `main` 里打印什么 |
| `weights.py` | 第二章那个，**一字未改** |
| `drift.py` | **这一章新增**。借第一章用过的 transformers 跑真模型，在 30 个块的出口挂钩子，量最后一个词元的模长；正文「数值会漂」那段的数字都出自这里 |
| `check_rmsnorm.py` | **这一章新增**。拿 `rms_norm` 和 transformers 的 `LlamaRMSNorm` 对拍，三处权重、两种输入量级，`torch.equal` 全为 `True` |

## 第一次打开 config.json

第二章介绍那十个文件时说过，`config.json` 是第一章讲的那份规格说明，704 字节、
二十来个字段，当时的原话是「等到后面的章节要用到它时，再一点一点介绍」。这一章
就是那个时候：RMSNorm 要用里面的 `rms_norm_eps`。

`Config` 类只给用得上的字段起名字，其余的留在 `raw` 里，后面哪章用到哪个就在
那时候补一个名字，顺便讲清楚它干什么。跑 `config.py` 会把 25 个字段全摊开：

```
字段                          值                     这本书在哪用到它
------------------------------------------------------------------------------
hidden_size                 576                   第四章，词嵌入的维度
num_hidden_layers           30                    第五章，Decoder块堆几层
vocab_size                  49152                 第三章，词表大小
tie_word_embeddings         True                  第四章，权重共享
rms_norm_eps                1e-05                 第五章，RMSNorm
num_attention_heads         9                     第六章
num_key_value_heads         3                     第六章，GQA
intermediate_size           1536                  第八章，FFN
rope_theta                  100000                第七章，RoPE
...

一共25个字段，这一章只用到`rms_norm_eps`一个。
```

分工是清楚的：**`Config` 管形状和超参数，`Weights` 管数据，`Model` 管计算。**
`Config` 只读不算，它不碰权重，也不做任何计算。

顺带一提，第二章那个 `count_layers()` 仍然是**数出来**的，没改成读
`config.json` 里的 `num_hidden_layers`。`weights.py` 从第二章一路复制过来，
一个字节都没动过，这条线索留着。

## 骨架长什么样

一个块就是两个 norm、两个空位、两条残差：

```python
def block(self, x, layer):
    x = x + self.attn(self.rms_norm(x, layer["input_norm"]))
    x = x + self.ffn(self.rms_norm(x, layer["post_attn_norm"]))
    return x
```

注意 norm 在**旁路**里，主干上只有加法，这就是 pre-norm。代价是 30 个块走完
主干没被归一化过，所以出口还得补最后一个 `model.norm`：

```
ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits
```

两个权重名来自第二章那 9 个张量：`input_norm` 是第一个 norm，`post_attn_norm`
是第二个。后者的名字有点坑，它叫 post，干的却是 pre-norm 的活，「post」说的是
位置在注意力子层后面，不是说它是 post-norm。

## 两个空位是假的

`attn` 和 `ffn` 现在不算任何东西，直接掷骰子：

```python
def attn(self, x):
    return torch.randn_like(x)

def ffn(self, x):
    return torch.randn_like(x)
```

和第三章那个假引擎是同一个路子：形状对了，残差就加得上，骨架就算接通了，至于
加进去的是什么，这一章不管。

`randn_like` 的 `like` 指的是「**规格**像 `x`」：形状、dtype、设备全照抄 `x`，
然后填上随机数，`x` 里的值一个都不看。写成 `torch.randn(x.shape)` 也能跑，但那样
dtype 和设备就不跟着 `x` 走了。

**这里没有固定种子**，和第三章那个假引擎不同，所以下面贴的输出你跑出来不会一样。
这一章不需要可复现，而且「次次不同」本身就说明了问题：真正的层是确定的函数，
假的才会每次都变。

## model.py 的输出

```
文字     'Once upon a time'
ID       [6403, 1980, 253, 655]
形状     (4,) → (4, 576) → (49152,)
RMSNorm  每层2个，共30层，加上出口那一个，一共61个

最后一个位置的模长，每10层看一次：
  第 0层前     2.894   （就是词嵌入本身）
  第10层后   111.174
  第20层后   149.435
  第30层后   187.246
  出口norm    44.099   （最后那一个norm把尺度拉了回来）

下一个词元   'Ġpreservation'   （空位是假的，这个结果自然也是假的）
```

中间那几个数你跑出来会不一样（没固定种子），但**两头那两个数是稳的**，这恰恰是
重点。跑 20 次的范围：

| | 范围 |
|---|---|
| 第 0 层前 | 2.894，纹丝不动 |
| 第 30 层后 | 175.1 ~ 198.2，次次不同 |
| 出口 norm 之后 | 42.891 ~ 44.319，几乎不动 |

这张模长表正好把这一章要讲的事演了一遍。

**一、数值真的会漂。** 词嵌入进来时模长 2.894（第四章那个数），60 次扰动之后
涨到将近 200。这里是随机游走，涨得比真模型温和，但方向是一样的：主干上的东西
只加不减，尺度必然往上走。

**二、出口那个 norm 把尺度拉了回来**，而且**拉到哪跟进来多少没关系**。进来的是
175 还是 198，出去都是 43 上下。这个 43 不是巧合：RMSNorm 先把每个分量拉到均方根
为 1，再乘上 `model.norm.weight`，而那个权重的均值是 1.791，所以出来的模长大约是
√576 × 1.791 ≈ 42.99。上面那张范围表里「出口 norm 之后几乎不动」，原因就在这儿:
**norm 的输出尺度是由权重定的，不是由输入定的**。第四章末尾说「词嵌入模长和
`model.norm.weight` 是一对耦合的旋钮」，指的就是这件事。

**三、pre-norm 的代价看得见。** 主干上一次都没被归一化过，所以才会一路涨到近 200。
post-norm 会在每层出口把它压回去，代价是主干不再透明。

## engine.py 的输出

`Engine` 类一行都没动，它只认 `model.forward`。改的是 `main`：第四章那张
「词嵌入」表对这一章没什么用了，换成第三章那种「转一圈打一行」的样子，三章
正好可以对着看。

```
转 10 圈，每圈把新词元追加到序列末尾，然后整串重算一遍：
  ids → 词嵌入 (T, 576) → 30个Block → 出口norm → 取最后一行 → 投影 → 取最高分

  第 5个词元   ID 23450   'Ġinterpre'
  ...（每次跑都不一样）
  第14个词元   ID 46733   'yroidism'

输出   'Once upon a time interpre"]: probleostasisthorneairobi’). metavaryroidism'

10个新词元里有10个互不相同。上一章这里只有1个，因为那时候中间是空的，
最后那个向量原封不动，永远问出它自己。
```

具体是哪十个词每次都变，但**「十个互不相同」这句是稳的**，这才是要看的东西。

比第四章那个「time time time」还难看，但**难看的方式变了**，这一点值得留意。

第四章是个不动点：最后那个向量没被碰过，拿自己去问「谁和我最像」，答案永远是
自己，于是无限复读。这一章 30 层往主干上灌了随机数，不动点被打破了，所以每圈
吐出来的词都不一样。**骨架确实通了，只是流过去的是垃圾。**

## 这一章丢了一条对拍

原计划里有两条检查：

1. **RMSNorm 与 `LlamaRMSNorm` 逐位相同**。这条成立，已验证 `torch.equal` 为
   `True`。这一章还能要求 bit-identical，第七章 RoPE 进来之后就只能比相对
   误差了。
2. ~~整条引擎的输出应该等于「第四章的输出再过一遍 final norm」~~。**这条作废了。**

第二条原本是唯一能验证骨架接对了的手段：残差接反、final norm 漏掉、30 层没真
串起来，这三种错都会被它抓住。换成随机数之后，这些错的输出看起来都一样（都是
噪声），错了也发现不了。

要把这条找回来，最便宜的办法是让两个空位能临时返回 0，专门跑这条对拍：`h = h + 0`
时 30 层是恒等变换，输出就该精确等于「第四章的结果再过一遍 final norm」。

## 引擎现状

```
ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits
```

30 层立起来了，残差流从头贯到尾，形状全程 `(T, 576)`。两个方框还是空的，第六、
七章填第一个，第八章填第二个。
