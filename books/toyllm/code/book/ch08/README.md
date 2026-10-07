# 第八章 · 前馈网络（FFN）

第七章的注意力已经完整，Decoder块里只剩FFN还返回全0。这一章把它填上，
30层的每个位置都满了，第二章读进来的272个张量全部用上。

FFN是SwiGLU，三个矩阵，576先升到1536，再降回576。和注意力正好相反，它逐词元：
第i行的输出只看第i行的输入。

## 运行

```bash
cd code
./download_model.sh                        # 还没下权重的话，先跑这个
uv run python book/ch01/baseline.py        # 还没有第一章的基准文件的话，先跑这个
uv run python book/ch08/config.py          # 把config.json摊开看一眼
uv run python book/ch08/model.py           # 走一遍30层，看模长怎么漂
uv run python book/ch08/model.py "任意一段话"
uv run python book/ch08/engine.py          # 接上两头转20圈，第一次说人话
uv run python book/ch08/check_ffn.py       # ffn和LlamaMLP对拍
uv run python book/ch08/check_forward.py   # 整机对拍，四级验收
```

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 补了一个字段：`intermediate_size`（1536） |
| `model.py` | **这一章的重点**。`ffn`从返回全0变成SwiGLU，多收一个`layer`参数；`block`里那一行跟着传`layer` |
| `engine.py` | `Engine`类和第四章**一字未改**；圈数从10改成20，和第一章一样 |
| `weights.py` | 第二章那个，**一字未改** |
| `check_ffn.py` | **这一章新增**。拿`ffn`和transformers的`LlamaMLP`对拍，再故意做错三种看看量级 |
| `check_forward.py` | **这一章新增**。整条流水线和第一章的基准对拍，不导入transformers |

## FFN做了什么

```
gate = x @ gate_proj.T      (T, 576)  → (T, 1536)
up   = x @ up_proj.T        (T, 576)  → (T, 1536)
h    = silu(gate) * up      (T, 1536)               逐元素相乘
out  = h @ down_proj.T      (T, 1536) → (T, 576)
```

`silu(z) = z * sigmoid(z)`，只套在gate那一路上。

`model.py`里用的是`F.silu`，不是手写的`z * sigmoid(z)`。两种写法数学上一样，
最后一位会差，下面对拍那张表里能看到。

## 对拍：ffn和LlamaMLP

输入是`ffn`真正收到的东西：词嵌入走到第i层，过完注意力子层，再过`post_attn_norm`。

```
    T   层      逐位相同      手写silu    silu放错一路      换成ReLU
    4   0        True       1.1e-07       3.5e-01       2.6e-01
    4  15        True       8.9e-08       1.2e+00       1.0e+00
    4  29        True       5.7e-09       2.4e-01       4.1e-02
   26   0        True       8.2e-08       5.9e-01       3.3e-01
   26  15        True       7.8e-08       6.3e-01       6.0e-01
   26  29        True       9.5e-09       2.6e-01       4.0e-02
 1000   0        True       1.1e-07       3.2e-01       3.5e-01
 1000  15        True       1.3e-07       4.4e-01       5.5e-01
 1000  29        True       5.2e-08       2.4e-01       4.5e-02

第3行单独算 vs 整段算再取第3行   相对误差4.5e-07
```

FFN不经过RoPE，没有那一位余弦的差，所以长短输入都**逐位相同**。上一章说
“从此改用相对误差”，指的是整条链路，单个部件能严格比就严格比。

后三列是相对误差：

- 手写silu在1e-7上下，只是舍入不同，不算错。
- silu放错一路、换成ReLU，都在1e-2到1e0之间，数字看着都正常，不报错。

最后一行是逐词元的验证：第3行单独喂进去，和整段喂进去再取第3行，只差舍入。

## 整机对拍：四级验收

```
一  参数量    我们134,515,008   第一章134,515,008   相同

二  逐层比     下标        绝对误差        相对误差
               0       0.0e+00       0.0e+00
               5       1.3e-05       2.2e-08
              10       1.7e-05       2.2e-07
              15       3.1e-05       1.5e-09
              20       3.3e-05       1.6e-09
              25       9.2e-05       4.6e-09
              29       2.4e-04       6.0e-08
              30       2.4e-05       9.4e-07
            31层里最大的相对误差   9.4e-07

三  argmax   我们[346, 253, 655, 28]   第一章[346, 253, 655, 28]   相同

四  贪婪解码
    我们     'Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but'
    第一章   'Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but'
    ID相同，文字相同
```

逐层比的下标是transformers的约定：`[0]`是词嵌入，`[1]..[29]`是第i个块的输入，
`[30]`是最后一个块的输出**而且已经过了出口norm**。`check_forward.py`里的
`hidden_states`照这个约定排，最后一份要先过norm再比。

T=4，RoPE的cos表逐位相同，但逐层比仍然不是0：第一章的transformers默认走
PyTorch的`scaled_dot_product_attention`，我们是一步一步手算的，加法顺序不同。
（让transformers改走手算的`eager`实现，它自己和自己也差这么多。）
相对误差都在1e-6以内，判据仍然是第七章那个`< 1e-5`。

argmax那一行是每个位置的预测：第0个位置预测`'upon'`之后应该是什么（346），
第3个位置预测`'time'`之后（28，也就是逗号）。

## model.py的输出

```
最后一个位置的模长，每10层看一次：
  第 0层前     2.894   （就是词嵌入本身）
  第10层后   144.387
  第20层后   209.950
  第30层后   635.043
  出口norm    40.688   （最后那一个norm把尺度拉了回来）

参数       134,515,008，272个张量全部用上
下一个词元   ','
```

第七章是222.9、684.9、1271.2。填上FFN以后，主干上的模长反而小了，第30层后只剩一半。

## engine.py的输出

```
输出   'Once upon a time, there was a little girl named Lily. She lived in a big house with her family, but'

20个新词元里有18个互不相同。
```

第六、七章一直在复读`ppo`，这一章第一次说出人话，和第一章那句话一字不差。

## 引擎现状

```
ids → embed → [norm → attn → norm → ffn] × 30 → norm → lm_head → logits
```

中间那个框里没有空位了。还没做的是采样（只会取最高分）和停止条件（圈数写死），
那是第十章。
