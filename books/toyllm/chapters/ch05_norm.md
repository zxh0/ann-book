# 归一化（Normalization）

> **本章代码**：RMSNorm，外加 Decoder Block 的**骨架**——这一章搭肋骨，器官留给后面三章填。
>
> - **造哪个框**：`norm.png` 上那三个红箭头。block 内两个 + 出口一个，全模型 30×2+1 = **61 个**
> - **形状**：`(T, 576) → (T, 576)`，逐 token 独立，不混合位置
> - **骨架**：`h = x + f(norm(x))`，其中 `f` 现在是空的
> - **引擎**：`ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits`（方框待填）
> - **跑出什么**：30 层此刻等于什么都没做，输出和上一章一样烂——**但残差流通了**，形状 `(T, 576)` 从头贯到尾。后面三章就是往两个方框里填东西
> - **对拍**：与 `LlamaRMSNorm` **逐位相同**。这一章还能要求 bit-identical，第七章起就不行了
> - **要讲**：残差连接（图上被折进框名里了，看不见，正文要补出那条旁路）；pre-norm 和 post-norm 的区别就是 norm 在旁路里还是旁路外；RMSNorm 比 LayerNorm 少了减均值这一步
> - **数字**：每个 norm 只有 576 个参数，61 个加起来 35,136，占全模型万分之三——便宜，但没它训不起来

<img src="../images/ch01/norm.png" alt="Norm" style="zoom:50%;" />

图上有三个红箭头：Decoder Block 内部两个，整个 LLM 的出口还有一个。30 层各两个，加上最后一个，全模型一共 **61 个 RMSNorm**。

上一章的引擎只有入口和出口，中间是空的。这一章开始搭中间的 30 层——但只搭**骨架**，不装器官。原因很实际：残差和归一化是 Block 的结构，注意力和 FFN 是塞进这个结构里的内容。先有结构，后面三章才有地方放东西。

要讲的几点：

- **RMSNorm 不是 LayerNorm**。LayerNorm 要先减均值再除标准差；RMSNorm 省掉减均值这一步，直接除以均方根。少一步计算，也少一组偏置参数。SmolLM2 里没有任何一个 LayerNorm。
- **位置决定了它叫 pre-norm**。看图，两个 RMS Norm 分别在 Attention 和 FFN 的**前面**。这叫 pre-norm，是现在的主流；早期 Transformer 把 norm 放在后面（post-norm），深了就训不动。
- **它便宜得离谱**。每个 RMSNorm 只有 576 个参数，61 个加起来 35,136 个，占全模型万分之三。但没有它，模型根本训不起来。



## 残差连接



### 骨架长什么样

一个子层的标准形态是这样的：

$$\mathbf{h} = \mathbf{x} + f(\mathrm{Norm}(\mathbf{x}))$$

三个部件：一条**主干**（ $\mathbf{x}$ 直通过去）、一条**旁路**（先归一化，再经过 $f$ ）、末尾一个**加法**。

$f$ 就是后面要填的器官——在第一个空位是注意力，在第二个空位是 FFN。



### 为什么要有主干

直觉上的解释：每一层不是"重新算一个 $\mathbf{x}$ "，而是"往 $\mathbf{x}$ 上**加**一点修正"。30 层就是 30 次小幅修正，而不是 30 次推倒重来。

这条主干贯穿整个模型，形状始终是 `(T, 576)` ——第一章说过，它有个通俗的叫法：**残差流**（residual stream）。每一层都在往这条流里写入自己的贡献，同时也读取前面所有层写进去的东西。

还有一个更技术的理由，但它属于训练的范畴（简单说：没有这条直通的加法通路，几十层深的网络根本训不出来）。按前言的约定，本书只取结论：**残差是必需的，不是可选的优化**。



### 图上看不见它

要提醒读者：第一章那些图上**画不出残差**。你看到的是 `Attention+Residual` 和 `FFN+Residual`——残差被折进框名里了，那条绕过去的旁路没有画出来。

而 pre-norm 和 post-norm 的区别，恰恰就是"norm 在旁路里还是旁路外"。所以这一节需要单独画一张图，把主干、旁路、加法三者分开。



### 一个 Block 有两条残差

```
h = x + Attn(norm₁(x))     ← 第六、七章填
y = h + FFN(norm₂(h))      ← 第八章填
```

对应第二章看到的两个张量：`input_layernorm` 是 norm₁，`post_attention_layernorm` 是 norm₂。

顺带说一个命名陷阱：**`post_attention_layernorm` 叫 post，干的却是 pre-norm 的活**。它的意思是"在注意力之后的那个 norm"（指位置在 attention 子层后面），不是"post-norm"。第一次读 HF 代码的人几乎都会在这里卡一下。



## 为什么需要归一化



### 数值会漂

数值在深层网络里会漂。有的维度数值特别大，有的特别小，一层层叠加下去，要么爆炸要么消失。归一化的作用就是在每一层入口把尺度拉回可控范围。

顺带回答上一章末尾那个问题：投影被向量模长带偏——归一化管的正是模长。



### LayerNorm 做两件事

$$\mathrm{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

先减均值除标准差（**归一化**），再乘一个可学习的缩放 $\boldsymbol{\gamma}$ 、加一个可学习的偏移 $\boldsymbol{\beta}$ （**还原表达力**）。

注意统计量是在**每个 token 的 576 个维度上**算的，不是跨 token 算的——这也是它叫 Layer Norm 而不是 Batch Norm 的原因。所以它逐 token 独立，和 batch 大小无关，这对推理很友好。



### post-norm：2017 年的做法

$$\mathbf{y} = \mathrm{Norm}(\mathbf{x} + f(\mathbf{x}))$$

norm 在残差**外面**，作用在相加之后。原始 Transformer 论文就是这么写的。



### pre-norm：现在的做法

$$\mathbf{y} = \mathbf{x} + f(\mathrm{Norm}(\mathbf{x}))$$

norm 在残差**里面**，只作用在旁路上。



### 为什么主流换成了 pre-norm

区别只是一个括号的位置，后果却很大。

**post-norm 的主干上有 norm。** 数据每过一层，整条主干都要被重新缩放一次。层数少的时候没事——原始 Transformer 只有 6 层——但堆到几十层，几十次缩放叠加起来，这条通路就不再"透明"了。

**pre-norm 的主干是一条干净的加法通路。** 从第一层直通最后一层，中间不经过任何缩放，每一层只是往上"加"自己的那份贡献。

至于为什么这个区别如此要命——完整的答案在训练里（关键词：梯度、warmup）。按前言的约定，本书到此打住，只取两个对我们有用的结论：**SmolLM2 用的是 pre-norm**，以及**pre-norm 欠了一笔债**。

代价是：pre-norm 每层都往主干上加东西，主干的数值方差会随层数增长。所以**最后必须再补一个 norm**，把输出拉回正常尺度再交给 lm_head。

这就是 `model.norm` 的来历，也解释了第一章那个数字：

```
61 = 30 × 2 + 1
```

多出来的那个 1，是 pre-norm 欠下的债。



## RMSNorm



### 砍掉一半

RMSNorm 把 LayerNorm 砍掉了两样东西：**不减均值，不加偏置**。

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

只除以均方根（root mean square，名字就是这么来的），再乘一个可学习的 $\boldsymbol{\gamma}$ 。



### 为什么砍得掉

LayerNorm 干了两件事：re-centering（减均值）和 re-scaling（除标准差）。RMSNorm 论文的实验结论是：**真正起作用的是 re-scaling，减均值那一步贡献很小**。既然没用，那就省掉。

偏置 $\boldsymbol{\beta}$ 同理。这也和 SmolLM2 全模型一个 bias 都没有的风格一致——第二章已经发现过。

好处是实打实的：少一趟遍历、少一组参数、更容易并行。在几十层几千次调用的规模上，这些省下来的开销很可观。



### 参数账

每个 RMSNorm 只有 **576** 个参数（就是那个 $\boldsymbol{\gamma}$ ）。61 个加起来 **35,136** 个，占全模型万分之三。

便宜到什么程度？第二章那张 `layer.png` 上，两个 norm 的条宽只有 576 / 3,540,096，**画出来不到一个像素**，所以图上根本看不见它们。

但没有它们，模型训不起来。这个反差本身就值得写一句。

`rms_norm_eps` 是 1e-5，从 config 读，不要写死。



### 一句结论

**SmolLM2 里没有一个 LayerNorm。** 这一章介绍 LayerNorm，只是为了让 RMSNorm 的那两刀有个参照——知道砍掉了什么，才知道剩下的是什么。

（章名从「LayerNorm」改成「归一化」也是这个道理。）



## 代码实现



### RMSNorm

一个 `(576,)` 的 weight，forward 三行：算均方根、除、乘 $\boldsymbol{\gamma}$ 。

有个实现细节值得说明：HF 的写法会先把输入转成 fp32 算完再转回去。我们全程 fp32，这一步看起来多余——但要知道它为什么存在（bf16 下平方和很容易溢出或损失精度），否则读 HF 源码时会困惑。

### Block 骨架

一个 `Block` 类：两个 RMSNorm、两个空位、两条残差。空位现在什么都不做。

然后堆 30 层，接上第四章的 embed 和 lm_head，中间再插进最后那个 `model.norm`：

```
ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits
```

### 跑一下

**输出还是胡言乱语，和上一章一样烂。** 这是意料之中的——30 层里两个空位都是空的，`h = x + 0`，整整 30 层等于恒等变换。

这个"白干一场"的结果恰恰是本章最好的教学点：**骨架不产生智能，器官才产生。** 后面三章每填一个空位，读者都能看到输出往人话的方向挪一点。

### 对拍

两条：

1. **RMSNorm 与 `LlamaRMSNorm` 逐位相同**。这一章还能要求 bit-identical——第七章 RoPE 进来之后就只能比相对误差了。
2. **整条引擎的输出，应该等于「第四章的输出再过一遍 final norm」**。因为中间 30 层是恒等变换。这条检查能验证骨架接对了：残差有没有接反、final norm 有没有漏、30 层有没有真的串起来。



## 本章小节

这一章搭好了 Decoder Block 的结构：

- **残差连接**：一条贯穿 30 层的主干，形状恒为 `(T, 576)`，就是残差流
- **pre-norm**：norm 放在旁路里，主干保持干净——代价是末尾要补一个 `model.norm`，于是 61 = 30×2+1
- **RMSNorm**：LayerNorm 砍掉减均值和偏置，只留下 re-scaling

引擎现在是：

```
ids → embed → [norm → □ → norm → □] × 30 → norm → lm_head → logits
```

30 层立起来了，但里面是空的，输出照旧是胡话。

下一章往第一个空位里填东西——整张图上唯一一个让 token 之间说话的部件。
