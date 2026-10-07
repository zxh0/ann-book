# 第十章 · 采样（Sampling）

第八章结束时，引擎能算出logits了，但只会取最高分，而且转20圈就停。这一章造采样器，
把总览图上那条`append to`虚线真正接上：生成、追加、再生成，直到EOS或者到了上限。
引擎第一次成为一个循环，到这里算是能用了。

这是整条流水线上**唯一有随机性**的地方。从词元ID到logits全程是确定性的，
模型每次说的话不一样，全部的不一样都发生在`sampler.py`里。

第九章只讲原理不落代码，所以这一章直接从第八章复制过来。

## 运行

```bash
cd code
./download_model.sh                          # 还没下权重的话，先跑这个
uv run python book/ch01/baseline.py          # 还没有第一章的基准文件的话，先跑这个
uv run python book/ch10/sampler.py           # 一个分布被温度、top-k、top-p改成什么样
uv run python book/ch10/detokenizer.py       # 逐个decode印出�，字节缓冲不会
uv run python book/ch10/engine.py            # 默认贪婪，和第八章一字不差
uv run python book/ch10/engine.py "Once upon a time" --temperature 0.7 --top-p 0.9 --seed 42
uv run python book/ch10/temperature.py       # 温度从0拧到1.5
uv run python book/ch10/check_filters.py     # 三个filter和transformers对拍
uv run python book/ch10/check_draw.py        # 抽样：可复现、频率收敛
uv run python book/ch10/check_forward.py     # 第八章的整机对拍，贪婪那条路不能变
```

## 文件

| 文件 | 作用 |
|---|---|
| `sampler.py` | **这一章的重点**。三个filter是纯函数；`GreedySampler`和`RandomSampler`两个类，`Sampler.build`在构造时选一次 |
| `detokenizer.py` | **这一章新增**。`StreamDecoder`：词元换回字节，进缓冲，凑齐了才印 |
| `engine.py` | `infer`收一个`sampler`；`stream`抽到EOS就停，最多`max_new_tokens`圈；`generate`是收齐了的`stream` |
| `config.py` | 补了一个字段：`eos_token_id`（0） |
| `model.py` | 第八章那个，**一字未改** |
| `weights.py` | 第二章那个，**一字未改** |
| `temperature.py` | **这一章新增**。同一个种子，温度取五档各跑一遍；同种子跑两次 |
| `check_filters.py` | **这一章新增**。导入transformers，三个filter和它的warper逐位对拍，再故意做错两种 |
| `check_draw.py` | **这一章新增**。抽样这一步的两条判据，不导入transformers |
| `check_forward.py` | 第八章那个，只改了文件头 |

## 采样器

```
logits (49152,) → [温度] → [top-k] → [top-p] → softmax → 概率 (49152,) → 抽一个 → ID
```

前三个是filter，logits进、logits出，只改分布的形状。砍掉的词元置成-inf，
softmax之后正好是0。真正的抽只有最后一步，`torch.multinomial`。

`sampler.py`的输出，`'Once upon a time'`之后：

```
  ','           0.760  ##############################
  ' in'         0.174  #######
  ' there'      0.027  #
  ...
                  T=0.3    T=0.7    T=1.0    T=1.5    T=3.0
  ','             0.993    0.882    0.760    0.416    0.010
  ' in'           0.007    0.108    0.174    0.156    0.006

  top_k=50         50个   0.9933
  top_p=0.9         2个   0.9341
  top_p=0.99       30个   0.9902

  50个词元，top_p=0.9：分布尖时留1个，分布平时留45个

概率0.6 / 0.3 / 0.1，top_p=0.7，留几个？  2个
```

贪婪和随机是两个类，不是一个开关：

```
Sampler.build(temperature=0)    → GreedySampler()
Sampler.build(temperature=0.7)  → RandomSampler(temperature=0.7, top_k=off, top_p=0.9)
```

T=0走`GreedySampler`，那是T趋于0的极限，不是除以0。生成循环里只有一句`sampler(logits)`，
每一圈都不用再问是哪一种。每个`RandomSampler`自带一个`torch.Generator`，
固定种子就能复现，也不受别处随机数的干扰。

## 对拍一：三个filter

`check_filters.py`，三段真logits（`'Once upon a time'`、`'The capital of France is'`、
`'def fibonacci(n):'`），温度三档、top-k三档、top-p三档，再串起来四组，
和`TemperatureLogitsWarper` / `TopKLogitsWarper` / `TopPLogitsWarper`**全部逐位相同**。

top-p留下几个词元，差别很大：`'Once upon a time'`在0.9时只留2个，
`'The capital of France is'`留406个。这就是“自适应”。

故意做错两种：

```
    0.6 / 0.3 / 0.1，top_p=0.7   transformers留2个，我们留2个，降序写法留1个
    真logits，top_p=0.9    正确留   2个，降序写法留   1个，逐位相同 False
    真logits，top_p=0.95   正确留   3个，降序写法留   2个，逐位相同 False
    T=0.5、top_p=0.9   先温度留   1个，先top-p留   2个
    T=1.5、top_p=0.9   先温度留 992个，先top-p留   2个
```

降序写法把边界词元一起砍了，候选集系统性地偏小。顺序颠倒更明显：T=1.5时，
先温度会把分布压平，top-p要992个词元才凑够0.9；先top-p的话，候选集在原始分布上
就定成了2个，温度再怎么调也只是改这2个的比例。

## 对拍二：抽这一步

`check_draw.py`：

```
一  可复现
    种子42   [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 3, 2, 0, 0]
    种子42   [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 3, 2, 0, 0]   相同
    种子43   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]   不同
    种子42，中间插了别的随机数   相同

二  抽40,000次，经验频率 vs 目标概率
    T=1.0
        词元        目标        经验        误差
           0    0.6439    0.6397    0.0042
           1    0.2369    0.2388    0.0019
           2    0.0871    0.0879    0.0007
           3    0.0321    0.0337    0.0016
```

`T=0.5`和`top_k=2`两组也都在统计误差以内，`top_k=2`砍掉的两个词元一次都没被抽到。

## 对拍三：贪婪那条路不能变

`check_forward.py`四级全过，和第八章的输出完全相同。默认采样器是`GREEDY`，
`engine.py`不带参数跑出来的还是第一章那句话：

```
Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but

20个新词元，到了上限停下，用时1.0秒，每个0.05秒
```

## 温度

`temperature.py`，种子20261006，每档40个新词元：

```
T=0  （32/40个互不相同）
  "Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but she didn't have many toys to play with. One day, her mom told her that she could"

T=0.3  （33/40个互不相同）
  'Once upon a time, there was a curious little girl named Lily. She loved to learn new things and had a special talent for remembering things. One day, she found a book about the history of the United States and'

T=0.7  （35/40个互不相同）
  'Once upon a time, there was a curious little girl named Lily. She loved to count things around her - apples, balls, and even bugs! One day she found two little sticks, each about the size of a'

T=1.0  （37/40个互不相同）
  'Once upon a time, Baron Thibaud of Edinburgh was thinking of getting married. He had stayed awake all day to try and remember the date, but no coffee apart from some jolly tea from his father’s'

T=1.5  （40/40个互不相同）
  "Once upon a time, Baron Thwaite branched off Jay Dopien Lynch Hay Haz Johnny Flat Light By Flowers strong diversity to org Shaw Hom\\'taresite Night apart Troy buried Freaked Harmonologies residing Extreme outs company"

T=1.0，种子20261006跑两次     一字不差
T=1.0，换成种子20261007     不一样
```

## 流式输出：第三章埋的雷

`detokenizer.py`：

```
      ID  词元        字节            逐个decode      先进缓冲
   40562  'ç¬'      e7 ac         '�'           ''
     122  '¬'       ac            '�'           '第'
   27991  'åį'      e5 8d         '�'           ''
     219  'ģ'       81            '�'           '十'
   ...
   24713  'ï¼ļ'     ef bc 9a      '：'           '：'
   ...

  逐个decode拼起来   '������：����'
  先进缓冲再印       '第十章：采样'
  整段decode         '第十章：采样'
```

每个汉字3个字节，被切成2 + 1两个词元，单独解哪一个都读不成字。
缓冲先扣着前两个字节，等第三个来了一起印。只有全角冒号自己凑齐了3个字节，逐个decode也没坏。

## 引擎现状

```
ids → embed → [norm → attn → norm → ffn] × 30 → norm → lm_head → logits → sampler → id
 ^                                                                                 |
 +------------------------------------ append to ----------------------------------+
```

引擎是一个完整的循环了。但每生成一个词元，都要把整串从头forward一遍，
生成n个词元要算1 + 2 + … + n个位置。下一章的KV Cache治这个。
