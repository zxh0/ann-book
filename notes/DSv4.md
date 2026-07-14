# 图解DeepSeek-V4(Pro)核心公式

如果你和我一样，对人工智能（AI）、深度学习（DL）、人工神经网络（ANN）、大语言模型（LLM）感兴趣，那么可能学着学着就学到了DeepSeek模型，然后就准备看它的论文，然后就被论文里的数学公式吓退了。如果你符合我描述的情况，那么这篇文章也许可以帮助到你。

本文主要介绍DeepSeek-V4论文里的数学公式，而且主要是站在“推理”的角度。这篇论文很长（共58页），里面涉及到的内容很多，我也没有全部看明白。不过，我基本上把里面的数学公式看懂了。为了帮助自己理解这些公式，我画了好多图。我觉得这些图可能对其他读者也有用，所以就写了这篇文章。

本文没有用AI润色，文字都是自己敲的，图都是自己画的，原汁原味。不过我使用AI检查了错别字，还有一些我不确定的地方，也问了AI。一些AI回答的片段，我以引用的形式贴到了文中，一眼就能看出来。由于我还在慢慢学习中，本文可能难免有错误和疏漏，如果我发现的话，会在下一版改进。



## Transformer视角

本文假设读者已经对标准的Transformer架构和注意力机制非常熟悉了，如果还不熟悉的话，可以先熟读2017年的那篇经典论文。这里我们先假设DeepSeek-V4用的就是标准的Transformer架构（Decoder-only），而且先省略大部分细节（后文会展开），它的推理过程看起来大概是这样：

<img src="../images/notes/ds4/ds11.png" alt="ds" style="zoom:50%;" />

上图中有几个重要的信息：

* DeepSeek-V4支持**1M**上下文，也就是说输入+输出，最多可以达到100万个token。

* 词嵌入和隐藏向量维度都是**7168**。假如输入序列长度是n，那么一个[n × 7168]的矩阵会流经每层，最终变成输出。

* DeepSeek-V4一共堆叠了**61层**Decoder Block。



## 标准注意力机制

这里回顾一下标准的注意力公式，也就是论文《Attention Is All You Need》中的公式（1）：

$$
\begin{aligned}
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
\end{aligned}
\tag{1}
$$

我们知道，LLM的推理过程可以分为两个阶段：Prefill和Decode。由于Prefill阶段属于计算密集型，Decode阶段属于存储密集型，所以这两个阶段往往会被调度到不同硬件上去执行，这个就叫做PD分离。

> AI：LLM 推理流程分为 Prefill 和 Decode 两个阶段：Prefill 会并行一次性处理完整输入文本，属于计算密集型负载（Compute-bound）；Decode 依托 KV Cache 逐 Token 串行生成内容，瓶颈在于 GPU 显存带宽，工程场景中常简称为存储密集型负载（Memory-bound）。
>
> 针对两类负载资源需求完全不同的特性，业界提出 PD 分离（Prefill-Decode Disaggregation）推理架构优化方案：把 Prefill、Decode 调度至相互独立的硬件资源池（既可以是单台服务器内划分不同 GPU，也支持跨集群物理隔离部署），为两类任务分别匹配适配自身负载特征的专用硬件，以此避免两类任务互相抢占资源、产生性能干扰。



### Prefill

在本文的示意图里，我们用矩形来表示**向量**或**矩阵**（如下图Q、K、V）。由于很难通过图形形状把向量或者矩阵的形状准确表达出来，所以一般都会以文字形式说明它们的形状。我们用紫色方块来表示可学习的**权重矩阵**，如下图四个W矩阵。我们用圆形来表示**标量**（圆形也用于表示计算，但是很容易区分），如下图 $QK^T$ 矩阵里的元素。

我们用圆形或椭圆来表示**计算**（如下图`softmax`），如果某个计算对应某论文中某公式，我们也会用黄色框框标识出来。为了简化示意图，我们经常会省略矩阵乘法，直接用权重矩阵本身来表示。基于这些约定，我们把上面公式（1）的计算过程大致画出来，重点展示Prefill阶段，如下图所示：

<img src="../images/notes/ds4/attn01.png" alt="attn" style="zoom:50%;" />

上图 $QK^T$ 矩阵中，有些元素是灰色的，表示加了因果掩码，这个细节这里就不展开解释了。不难看出，Q、K、V的计算、 $QK^T$ 的计算、注意力计算，以及最后输出的计算，大部分都是可以在GPU上大规模并行处理的矩阵乘法运算，所以Prefill阶段的确是**计算密集型**的，瓶颈是GPU算力。



### Decode

我们再来看Decode阶段。假如输入“好好学习”预测出的下一个字是“天”，我们会把“好好学习天”当作输入，再次预测下一个字。问题是前面4个字的K和V，完全是可以复用的，如果输入序列很长，那么重新计算就很浪费。所以我们可以把前面算好的K和V都缓存起来，下次直接拿出来用，这个就是所谓的KVCache机制。在上面的图里，我们把K和V画成了深灰色，表示它们可以被缓存起来，重复利用。基于KVCache机制，我们把Decode阶段的注意力计算也画出来，如下图所示：

<img src="../images/notes/ds4/attn02.png" alt="attn" style="zoom:50%;" />

由于KVCache的使用，Decode阶段直接变成了**存储访问密集型**，稍后会详细说明。于是，Decode阶段瓶颈不再是GPU算力，而是GPU显存（HBM）。



### Multi-Head Attention

以上描述了单个头的注意力公式，多个头的注意力，拼在一起就行，如下面公式所示：

$$
\begin{aligned}
\mathrm{MultiHead}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1,\, \dots,\, \mathrm{head}_h) W^O \\
\mathrm{where}\quad \mathrm{head}_i &= \mathrm{Attention}\left(Q W_i^Q,\, K W_i^K,\, V W_i^V\right)
\end{aligned}
\tag{2}
$$

> 🤔原论文这个公式有点让人迷糊。实际上，上面公式的第二行，Q、K、V应该换成H才更合理：
>
> $\mathrm{head}_i = \mathrm{Attention}\left(H W_i^Q,\, H W_i^K,\, H W_i^V\right)$

我们把上面这个多头计算公式画成图，如下所示：

<img src="../images/notes/ds4/attn03.png" alt="attn" style="zoom:50%;" />

我们假设每个浮点数占用`f`个字节，Transformer架构一共有`l`个decoder层。那么根据上面这个图，我们可以粗略估计，KVCache占用的空间大约是：

$$
\begin{aligned}
((n × k) + (n × v)) × h × l × f
\end{aligned}
\tag{a}
$$



### 参数&超参数

这里总结一下标准注意力机制公式中出现的参数和超参数。先看参数（可学习权重矩阵）：

| 权重矩阵 | 形状       | 说明              |
| -------- | ---------- | ----------------- |
| $W^Q$    | $[d × k]$  | 每层一套，每套h个 |
| $W^K$    | $[d × k]$  | 每层一套，每套h个 |
| $W^V$    | $[d × v]$  | 每层一套，每套h个 |
| $W^O$    | $[hv × d]$ | 每层一套，每套1个 |

超参数：

| 超参数 | 原始Transformer取值 | DeepSeek-V4(Pro)取值 | 说明         |
| ------ | ------------------- | -------------------- | ------------ |
| d      | 512                 | 7168                 | 隐藏状态维度 |
| h      | 8                   | 128                  | 注意力头数   |
| k      | 64                  |                      | d➗h          |
| v      | 64                  |                      | d➗h          |
| 层数   |                     | 61                   |              |

对于DeepSeek-V4 Pro，我们可以做一个粗略的估算。我们取n=1M，k=v=d➗h=56，f=1（fp8）。把这些值带入上面的公式（a）可得：

$$
\begin{aligned}
CacheSize &=((n × k) + (n × v)) × h × l × f \\
&= ((1M × 56) + (1M × 56)) × 128 × 61 × 1 \\
&= 1M × 56 × 2 × 128 × 61 \\
&= 874496M
\end{aligned}
\tag{b}
$$

大约是854G（这里可能估计的不准？），这显然太大了。所以DeepSeek-V4要做的优化之一，就是尽可能让KVCache缩小，后文会详细介绍。



## DeepSeekV4整体结构

现在我们切换到DeepSeek-V4视角，看看它的整体推理过程，如下图所示：

<img src="../images/notes/ds4/ds12.png" alt="ds" style="zoom:50%;" />

和最开始的Transformer视角相比，主要是两个细节发生了变化：

* DeepSeek-V4把标准的注意力机制换成了它自己的**混合注意力**机制，前两层是HCA，此后CSA和HCA交替使用。

* DeepSeek-V4也没有使用标准的FFN，而是使用了MoE。

这两点后文会详细说明。



## CSA（Compressed Sparse Attention）

先来看CSA，下面是DeepSeek-V4论文中给出的架构图。这个架构图还是比较抽象的，所以我把它展开一步一步介绍。顺便说一下，CSA其实用了DeepSeek-V3.2提出的DSA（DeepSeek Sparse Attention）。不过，看懂了CSA就没必要再去专门看DSA了。

<img src="../images/notes/ds4/v4f3.png" alt="01" style="zoom:50%;" />



### Compressed Key-Value Entries

第一步，对输入序列进行降维，得到4个序列。这是论文中公式（9）和（10）描述的计算：

$$
\begin{aligned}
C^a &= H \cdot W^{a\mathrm{KV}}, & C^b &= H \cdot W^{b\mathrm{KV}}, \\
Z^a &= H \cdot W^{a\mathrm{Z}}, & Z^b &= H \cdot W^{b\mathrm{Z}}
\end{aligned}
\tag{9, 10}
$$

这四个降维操作如下图所示。C、Z、W这些矩阵的形状，都可以从下图看到，后面也会以表格形式总结。

<img src="../images/notes/ds4/csa09.png" alt="csa" style="zoom:50%;" />

第二步，压缩。利用上面计算出的4个序列，计算出一个压缩序列，长度是原来的 $\frac{1}{m}$ 。其中`m`就是CSA的压缩率，论文中的取值是4。这一步对应论文中的公式（11）和（12）：

$$
\begin{aligned}
\left[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}\right] &= \text{Softmax}_\text{row}\left(\left[Z^a_{mi:m(i+1)-1} + B^a, Z^b_{m(i-1):mi-1} + B^b\right]\right), \\
C_i^\text{Comp} &= \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
\end{aligned}
\tag{11, 12}
$$

压缩是分组计算的，每一组m个。为了便于理解，我们只取其中的一组（m个），把这两个计算展开。哈达玛积（⊙）和累加画出来太繁琐了，所以我们就用“压缩”来表示。于是上面的两个公式就画成了下面这样：

<img src="../images/notes/ds4/csa11.png" alt="csa" style="zoom:50%;" />

最终的效果，就是输入被降维和压缩了。维度从7168降到了512，只有原来的1/14。序列长度每4个一组压缩，只有原来的1/4。后面马上要介绍的“闪电索引”，也用了同样的套路，只是维度降的更低，降到了128，只有输入的1/56。这两个过程可以用下图表示：

<img src="../images/notes/ds4/csa12.png" alt="csa" style="zoom:50%;" />



### Lightning Indexer for Sparse Selection

现在来看“闪电索引”，先看论文中的公式（13）和（14）：

$$
\begin{aligned}
\mathbf{c}_t^Q &= \mathbf{h}_t \cdot W^{DQ}, \\
\left[\mathbf{q}_{t,1}^I;\, \mathbf{q}_{t,2}^I;\, \dots;\, \mathbf{q}_{t,n_h^I}^I\right] &= \mathbf{q}_t^I = \mathbf{c}_t^Q \cdot W^{IUQ}
\end{aligned}
\tag{13, 14}
$$

这个公式是针对某一个输入（ $h_t$ ）来计算的，具体过程如下图蓝框所示。公式中出现的权重矩阵、输入输出和中间结果的形状，都画在图里了。（公式18可以暂时忽略，等会再回来看）

<img src="../images/notes/ds4/csa13.png" alt="csa" style="zoom:50%;" />

和注意力机制类似，闪电索引也是多头的，一共有 $n^I_h$ 个头（论文里这个值是64）。某个输入经过上面这套公式计算之后，就给每个闪电索引头算出来一个 $q^I$ 向量，维度是 $c^I$ 。再来看论文中的公式（15）和（16）：

$$
\begin{aligned}
\left[\mathbf{w}_{t,1}^I;\, \mathbf{w}_{t,2}^I;\, \dots;\, \mathbf{w}_{t,n_h^I}^I\right] = \mathbf{w}_t^I &= \mathbf{h}_t \cdot W^w, \\
I_{t,s} &= \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \mathrm{ReLU}\left( \mathbf{q}_{t,h}^I \cdot K_s^{\text{IComp}} \right)
\end{aligned}
\tag{15, 16}
$$

其中公式（15）也是针对某个输入来计算的，结果是 $n^I_h$ 个权重值，如下图所示：

<img src="../images/notes/ds4/csa15.png" alt="csa" style="zoom:50%;" />

公式（16）也是针对某个输入计算的，索引也用`t`表示。对于某个输入，需要一个 $q_t$ 向量，这个是公式（13）和（14）算出来的。另外，还需要一个 $K^{IComp}$ 矩阵，索引用`s`表示。这个矩阵的计算过程和前一小节压缩KV的计算基本是一样的，可以去看上一小节的最后一幅图。于是，公式（16）的整体计算细节如下图所示：

<img src="../images/notes/ds4/csa16.png" alt="csa" style="zoom:50%;" />

公式（16）看着特别复杂，但如果你忽略各种细节，其实就是针对每一个输入 $h_t$ ，算出一个打分向量 $I_t$ ，维度是n/m。有了打分之后，就可以用TopK算法去筛选稀疏KV了，也就是论文公式（17）：

$$
\begin{aligned}
C_t^{\text{SprsComp}} = \left\{ C_s^{\text{Comp}} \mid I_{t,s}\in\mathrm{TopK}(I_{t,:}) \right\}
\end{aligned}
\tag{17}
$$

最后的效果就是，输入（形状是[n × d]）经过降维和压缩，得到了 $C^{Comp}$ ，形状是[n/m × c]。对于第`t`个输入，有一个打分向量 $I_t$ ，用它进行TopK筛选，得到了 $C_t^{SprsComp}$ ，形状是[K × c]。整个过程如下图所示：

<img src="../images/notes/ds4/csa17.png" alt="csa" style="zoom:50%;" />

在DeepSeek-V4论文里，d=7168，c=512，m=4，K=1024。于是，不管输入序列有多长，对于某个输入，这么一大套算下来，最后得到的就是一个[1024 × 512]的矩阵。而 $C^{Comp}$ 矩阵，就是要放在KVCache里的数据。



### Shared Key-Value MQA

CSA最后使用的，也不是原始的注意力机制，而是[MQA](https://arxiv.org/abs/1911.02150)。和原始注意力机制相比，MQA最大的变化就是，所有的注意力头共享一组KV。前面计算的，其实是K和V。对于某个输入，我们还需要计算 $q$ 向量，这个对应论文公式（18）：

$$
\begin{aligned}
\left[\mathbf{q}_{t,1};\, \mathbf{q}_{t,2};\, \dots;\, \mathbf{q}_{t,n_h}\right] = \mathbf{q}_t = \mathbf{c}_t^Q \cdot W^{UQ}
\end{aligned}
\tag{18}
$$

这个 $q$ 向量，和公式（14）的 $q^I$ 向量，在计算过程上基本是一样的，所以我把它们画在了一起。这里重新贴一下这个图：

<img src="../images/notes/ds4/csa13.png" alt="csa" style="zoom:50%;" />

现在，所有东西都准备好了，利用这些数据进行标准注意力计算就可以了。这个对应论文公式（19）：

$$
\begin{aligned}
o_{t,i} = \mathrm{CoreAttn}\left( \mathrm{query}=\mathbf{q}_{t,i},\, \mathrm{key}=C_t^{\mathrm{SprsComp}},\, \mathrm{value}=C_t^{\mathrm{SprsComp}} \right)
\end{aligned}
\tag{19}
$$

上面公式里的`t`是输入的索引，`i`是注意力头的索引。由于使用的是MQA注意力机制，所以只有Q带着`i`下标。总之，最重要的是，不管输入序列多长，对于某个输入来说，最后进入注意力计算的，就是这个经过降维&压缩&筛选后的稀疏KV，长度固定为K=1024。到目前为止的整个过程如下图所示：

<img src="../images/notes/ds4/csa18.png" alt="csa" style="zoom:50%;" />



### Grouped Output Projection

上面这个图显示的是某个注意力头的计算，对于多个头（一共 $n_h=128$ 个），我们需要把这些 $o_{t, i}$ 都拼接起来，然后做一次线性变换，得到最终输出，如下图所示：

<img src="../images/notes/ds4/csa-gop1.png" alt="gop" style="zoom:50%;" />

在DeepSeek-V4论文里，c=512， $n_h$ = 128，所以拼接起来的 $o_t$ 向量，维度是512×128=64K。这个计算量太大了，所以DeepSeek采取了一个“分组两级计算”的优化。

首先，把 $n_h$ 个向量分成g组（论文里取值16）。也就是说，分成16组，每组8个。然后把每个组，投影成一个向量，维度是 $d_g$ （论文里取值1024）。最后把这些中间的向量拼起来，再做一次变换，得到最终需要的输出。这个计算过程如下图所示：

<img src="../images/notes/ds4/csa-gop2.png" alt="gop" style="zoom:50%;" />

这里做一个简单的比较。我们知道：若矩阵A的形状是[a,b]，矩阵B的形状是[b,c]，那么矩阵乘 AB 计算量大约为 a⋅b⋅c。于是，优化前的计算量是：

$$
\begin{aligned}
c × n_h × d = 512 × 128 × 7168 = 448M 
\end{aligned}
\tag{c}
$$

优化后，中间步骤计算量只是原来的1/7：

$$
\begin{aligned}
c × \frac{n_h}{g} × d_g × g = 512 × 128 × 1024 = 64M
\end{aligned}
\tag{c}
$$

最后一步的计算量是原来的1/4：

$$
\begin{aligned}
g × d_g × d = 16 × 1024 × 7168 = 112M
\end{aligned}
\tag{e}
$$

最终，优化后整体计算量大概是优化前的40%。



### 参数&超参数

总结一下CSA涉及到的参数和超参。下面是可学习权重矩阵和偏置：

| 权重/偏置 | 形状                     | 公式   |
| --------- | ------------------------ | ------ |
| $W^{aKV}$ | $[d × c]$                | （9）  |
| $W^{bKV}$ | $[d × c]$                | （9）  |
| $W^{aZ}$  | $[d × c]$                | （10） |
| $W^{bZ}$  | $[d × c]$                | （10） |
| $B^a$     | $[m × c]$                | （11） |
| $B^b$     | $[m × c]$                | （11） |
| $W^{DQ}$  | $[d × d_c]$              | （13） |
| $W^{IDQ}$ | $[d_c × c^I n^I_h]$      | （14） |
| $W^w$     | $[d × n^I_h]$            | （15） |
| $W^{UQ}$  | $[d_c × c n_h]$          | （18） |
| $W^{OG}$  | $[\frac{cn_h}{g} × d_g]$ |        |
| $W^{O}$   | $[d_g] × d$              |        |

超参数：

| 参数    | 取值 | 说明                        |
| ------- | ---- | --------------------------- |
| $d$     | 7168 | 隐藏向量维度                |
| $c$     | 512  | 注意力头维度                |
| $m$     | 4    | CSA压缩率                   |
| $d_c$   | 1536 | query compression dimension |
| $c^I$   | 128  | indexer head dimension      |
| $n_h^I$ | 64   | indexer query heads         |
| $n_h$   | 128  | query heads                 |
| $d_g$   | 1024 | 中间输出维度                |



## HCA（Heavily Compressed Attention）

现在来看HCA，下面是论文里给出的架构图。这个图同样也是比较抽象，所以我们还是需要展开来细看。相比CSA，HCA要简单一些（没有筛选逻辑）。如果你已经理解了CSA，那么HCA就比较好懂了，所以这一小节可能写的没那么详细。

<img src="../images/notes/ds4/v4f4.png" alt="01" style="zoom:50%;" />



### Compressed Key-Value Entries

和CSA一样，HCA也是先对隐藏状态H进行降维，这一步对应论文中的公式（20）和（21）：

$$
\begin{aligned}
C &= H \cdot W^{\text{KV}},\\
Z &= H \cdot W^{Z}
\end{aligned}
\tag{20, 21}
$$

这两个公式的计算如下图所示。其中两个W，以及C和Z的形状，都画在了图里。

<img src="../images/notes/ds4/hca20.png" alt="hca" style="zoom:50%;" />

降维之后是压缩。我们展开来看一下上图中的“压缩”操作，它对应论文中的公式（22）和（23）：

$$
\begin{aligned}
S_{m'i:m'(i+1)-1} &= \mathrm{Softmax_{row}}\left(Z_{m'i:m'(i+1)-1} + B\right), \\
C_i^{\mathrm{Comp}} &= \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j.
\end{aligned}
\tag{22, 23}
$$

这两个公式都是针对一组输入来计算的，每一组有m'（论文取值128）个向量。首先，一组Z和偏置B，算出一组S（公式22）。然后，一组S和一组C，算出一个 $C^{Comp}$ 向量（公式23）。这两个计算过程如下图所示：

<img src="../images/notes/ds4/hca22.png" alt="hca" style="zoom:50%;" />

总之，输入序列的形状是[n × d]，其中d=7168。计算出的 $C^{Comp}$ 的形状是[n/m' × c]，其中m'=128，c=512。可以看到，长度缩小为原来的1/128，降维缩小为原来的1/14，效果还是挺明显的。



### Shared Key-Value MQA

HCA没有打分和筛选环节，所以直接来到MQA计算。对于某个输入，我们还是需要计算 $q$ 向量，这个对应论文公式（24）和（25）：

$$
\begin{aligned}
\mathbf{c}_t^Q &= \mathbf{h}_t \cdot W^{DQ}, \\
\left[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \dots; \mathbf{q}_{t,n_h}\right] &= \mathbf{q}_t = \mathbf{c}_t^Q \cdot W^{UQ}
\end{aligned}
\tag{24, 25}
$$

这个计算我们也很熟悉了，如下图所示。两个W矩阵的形状，中间状态的形状等，也是直接画在图里了。

<img src="../images/notes/ds4/hca24.png" alt="hca" style="zoom:50%;" />

现在Q、K、V都准备好了，可以带入标准注意力公式进行计算了。这一步对应论文里的公式（26）：

$$
\begin{aligned}
\mathbf{o}_{t,i} = \mathrm{CoreAttn}\left(\mathrm{query} = \mathbf{q}_{t,i},\, \mathrm{key} = \mathbf{C}^{\mathrm{Comp}},\, \mathrm{value} = \mathbf{C}^{\mathrm{Comp}}\right)
\end{aligned}
\tag{26}
$$

上面公式里的`t`是输入的索引，`i`是注意力头的索引。总之，对于某个输入来说，最后进入注意力计算的，就是这个经过降维&压缩后的KV，长度只是原来的1/128。现在我们把前面这些步骤都画在一起，如下图所示：

<img src="../images/notes/ds4/hca26.png" alt="hca" style="zoom:50%;" />



### Grouped Output Projection

和CSA一样，HCA这里也是有一个分组二级投影的优化的。这个优化和CSA是完全一样的，连参数设置也一样，这里就不再重复介绍其细节了。




### 超参数&参数

总结一下HCA涉及到的参数和超参。下面是可学习权重矩阵和偏置：

| 参数     | 取值            | 说明   |
| -------- | --------------- | ------ |
| $W^{KV}$ | $[d × c]$       | （20） |
| $W^Z$    | $[d × c]$       | （21） |
| $B$      | $[m' × c]$      | （22） |
| $W^{DQ}$ | $[d × d_c]$     | （13） |
| $W^{UQ}$ | $[d_c × c n_h]$ | （13） |

超参数：

| 参数  | 取值 | 说明                        |
| ----- | ---- | --------------------------- |
| $d$   | 7168 | 隐藏向量维度                |
| $c$   | 512  | 注意力头维度                |
| $m'$  | 128  | HCA压缩率                   |
| $d_c$ | 1536 | query compression dimension |
| $n_h$ | 128  | query heads                 |
| $d_g$ | 1024 | 中间输出维度                |



## DeepSeek MoE（Mixture of Experts）

DeepSeek-V4沿用了DeepSeek-V3的MoE架构，而V3的MoE也只是在原来DeepSeekMoE基础上进行了一些微调。这个DeepSeekMoE，其实早在V2、V1里也已经在用了。这一小节，我们使用DeepSeek-V3里的公式，来讲解DeepSeekMoE。下面是DeepSeek-V3论文里的MoE架构图，比CSA和HCA好理解：

<img src="../images/notes/ds4/moe-f2.png" alt="moe" style="zoom:50%;" />



### 核心公式

MoE是按token计算的，请看论文里的公式（12）：

$$
\begin{aligned}
h_t' = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t)
\end{aligned}
\tag{12}
$$

在上面的公式里， $u_t$ 表示某个输入， $h'_t$ 表示它对应的输出。等号右边一共有三项：

* 其中第一项就是输入自己，这个就是标准的残差连接写法。不过DeepSeek-V4里已经换成了mHC了，这个后面的小节会说明。
* 第二项表示共享专家（Shared Expert），一共 $N_s$ 个（V4取值1）。共享专家是直接被激活的，它们的输出会做一个累加。
* 第三项表示路由专家（Routed Expert），一共 $N_r$ 个（V4取值384）。如果 $g_{i,t} \ne 0$ ，则该专家被激活。在V4里，每个token只会激活K个（V4取值6）路由专家。这里控制谁被激活的 $g_{i,t}$ 稍后介绍。

顺便说一下，在DeepSeek-V3里，前3层是没有使用MoE的，就是普通（Dense）FFN。但是在V4里，前三层用的也是MoE，只是用了特殊的Hash路由，这个原论文里并没有展开说，这里也不详细介绍了。上面的公式可以画成下图这样：

<img src="../images/notes/ds4/moe12.png" alt="moe" style="zoom:50%;" />

现在我们来看 $g_{i,t}$ 是怎么算出来的，稍微有点复杂，对应论文中的公式（13）、（14）和（15）：

$$
\begin{aligned}
g_{i,t} &= \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}, \\
g'_{i,t} &=
\begin{cases}
s_{i,t}, & s_{i,t} \in \mathrm{Topk}\big(\{s_{j,t} \mid 1 \le j \le N_r\}, K_r\big), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} &= \mathrm{Sigmoid}\big(\mathbf{u}_t^\top \mathbf{e}_i\big)
\end{aligned}
\tag{13, 14, 15}
$$

我们要倒着来看这三个公式。公式里的`t`是输入的索引，`i`是路由专家的索引。

* 公式（15）。某个路由专家 $\mathrm{FFN}_i^{(r)}$ ，有一个对应的向量 $e_i$ 。它和某个输入的转置 $u_t^T$ 点乘一下，就得到一个标量。这个标量在过一个激活函数，就得到了 $s_{i,t}$ 。在DeepSeek-V3里，这个激活函数是`Sigmoid(·)`。但是在V4里，换成了`Sqrt(Softplus(·))`。
* 公式（14）。对于某个输入 $u_t$ ，上面的公式一共会算出 $N_r$ 个标量（ $s_{1, t}$ ~ $s_{Nr, t}$ ）。这些标量做一个TopK筛选，选出K个来，其余都置成0，这样就得到了另外一个序列（ $g'_{1, t}$ ~ $g'_{Nr, t}$ ）。
* 公式（13）。对上面公式得到的序列，做一个简单的归一化处理。

这三个公式可以画在一起，如下图所示：

<img src="../images/notes/ds4/moe13.png" alt="moe" style="zoom:50%;" />



### 神经网络视角

前面的介绍，是站在MoE公式的角度来画图的。如果我们使用CSA和HCA那种角度，使用矩阵乘法理解MoE，也是可以的。一个MoE，其实就是一个降维再升维的两层FFN，中间的隐藏维度是3072。我们用U和V来表示这两个矩阵，于是，

<img src="../images/notes/ds4/moe-mm.png" alt="moe" style="zoom:50%;" />

在DeepSeek-V4里，每层有1个共享专家，384个路由专家，共385个专家。每个token每次只激活其中的6个路由专家，加上共享专家，一共激活7个专家。所以，就MoE而言，每次激活的专家就只有1/55。我们估计一下参数量：

* 上图中，每个紫色方块都包含7168x3072=21M个参数。
* 一共385个专家，所以是385x2x21M=16170M（1.6B）
* 一共61层，所以MoE总共是61x16170M=986370M（98B）
* 每次激活1/55的专家，986370M/55=17934M（1.8B）



## mHC（Manifold-Constrained Hyper-Connections）

这一节介绍一下mHC，这个对应DeepSeek-V4架构图里的“Residual Mixing”，如下图所示：

<img src="../images/notes/ds4/v4f2.png" alt="moe" style="zoom:50%;" />

为了便于理解，我们先从没有残差连接的版本开始说起，然后介绍残差连接，然后再介绍标准的超连接（HC），最后介绍mHC。



### 没有残差连接

我们把CSA/HCA或者MoE，都当作独立的层，抽象成一个输入函数。于是，不考虑残差连接，我们可以写出某一层的计算公式：

$$
\begin{aligned}
X_{l+1} = \mathcal{F}_l(X_l)
\end{aligned}
\tag{f}
$$

这里，我们使用DeepSeek-V4论文里的写法。其中 $X$ 表示输入输出， $l$ 表示层数， $\mathcal{F}_l$ 表示某一层的函数。这个公式是相当简单的，假如神经网络有3层，公式展开以后，画出来是下面这样：

<img src="../images/notes/ds4/mhc1.png" alt="mhc" style="zoom:50%;" />



### 残差连接（Residual Connections）

上面这种网络，层次多了以后，是很不好训练的，原因在于反向传播的梯度消失和梯度爆炸（本文不解释这个细节）。所以我们稍微调整一下结构，把输入加到函数输出里。也就是说，让网络只学习差值。下面是新的计算公式：

$$
\begin{aligned}
X_{l+1} = X_l + \mathcal{F}_l(X_l)
\end{aligned}
\tag{g}
$$

还是以3层网络为例，上面的公式展开后，画出来是下面这样：

<img src="../images/notes/ds4/mhc2.png" alt="mhc" style="zoom:50%;" />



### Standard Hyper-Connections

标准超连接（HC）对上面的残差连接进行了扩展，然后就变成了DeepSeek-V4论文里公式（1）描述的样子：

$$
\begin{aligned}
X_{l+1} = B_l X_l + C_l \mathcal{F}_l(A_l X_l)
\end{aligned}
\tag{1}
$$

这个公式还是有点难理解的，所以可以拆开一点一点理解。

* 超连接（HC）的实际效果，是把原来的残差连接，扩展 $n_{hc}$ 倍（V4论文里取值是4）。如果我们把 $n_{hc}$ 设置成1，那么HC就退化成了普通的残差连接。
* 为了达到这个效果，输入输出也要扩展 $n_{hc}$ 倍。原来的输入形状是 $[1×d]$ ，扩展后就变成了 $[n_{hc} \times d]$ 。如果是第一层的输入，直接把原始输入复制 $n_{hc}$ 份就好了。
* 然而，我们并不改变每一层的函数 $F$ 。于是，我们需要使用矩阵 $A$ 给扩展后的输入做一个线性变换，然后再送入函数 $F$ 。这就是公式里 $A_l X_l$ 的含义（下图蓝色箭头）。
* 函数的输出形状还是 $[1×d]$ ，于是我们需要矩阵 $C$ ，再做一次线性变换，把输出形状也变成 $[n_{hc} \times d]$ 。这就是公式里 $C_l \mathcal{F}_l(...)$ 的含义（下图绿色箭头）。
* 然后就是把输入加到残差值里，这里需要使用矩阵 $B$ 和输入相乘。这就是公式里 $B_l X_l$ 的含义（下图红色箭头）。

这个公式完整的计算细节，见下图（图里 $n_{hc}$ 取值3）。这里提到所有矩阵的形状，都画在图里了。

<img src="../images/notes/ds4/mhc3.png" alt="mhc" style="zoom:50%;" />



### mHC

超连接看起来很强大，但是有一个问题，就是会重蹈无残差连接网络的覆辙：梯度消失和爆炸。为了比较简单的推导出这个结论，我们把上面的公式（1）简化一下，先忽略残差连接（毕竟它只是差异），只留下输入。然后公式就变成了这样：

$$
\begin{aligned}
X_{l+1} = B_l X_l
\end{aligned}
\tag{h}
$$

你看，如果你把 $B_l$ 看成一个函数，它是不是就变成了本小节最开始的公式（f）。于是，问题就又出现了。既然问题出在矩阵 $B_l$ ，那么就解决这个问题。DeepSeek-V4想到的办法就是约束这个矩阵，让它成为一个双随机矩阵（doubly stochastic），这就是mHC。

所谓双随机矩阵，是个方阵，简单说，你单看它的任何一行、任何一列，都好像概率归一化了（每个元素都不小于0，且总和为1）。如果用数学语言表达出来，那么就是DeepSeek-V4论文里的公式（2）：

$$
\begin{aligned}
B_l \in \mathcal{M} := \{ M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^T M = \mathbf{1}_n^T,\ M \ge 0\}. 
\end{aligned}
\tag{2}
$$

双随机矩阵有个特点，就是两个双随机矩阵相乘，得到的还是一个同样形状的双随机矩阵。有了这个特点，那么前面那个简化的公式（h），无论套多少层，都不会有问题了。而标准HC带来的问题，也就解决了。



### Dynamic Parameterization

在HC原论文里，HC分为静态和动态两套。在静态HC里，A、B、C这些矩阵都是训练出来的，就是我们前面画的那样。而在动态HC里，A、B、C这些矩阵都是动态算出来的。这个计算过程，对应DeepSeek-V4论文里的公式（3）、（4）和（5）：

$$
\begin{aligned}
\tilde{A}_l &= \alpha_l^{\text{pre}} \cdot \left(\hat{X}_l W_l^{\text{pre}}\right) + S_l^{\text{pre}}, \\
\tilde{B}_l &= \alpha_l^{\text{res}} \cdot \mathrm{Mat}\left(\hat{X}_l W_l^{\text{res}}\right) + S_l^{\text{res}}, \\
\tilde{C}_l &= \alpha_l^{\text{post}} \cdot \left(\hat{X}_l W_l^{\text{post}}\right)^\top + S_l^{\text{post}}
\end{aligned}
\tag{3,4,5}
$$

这三个公式里， $W$ 属于可学习权重矩阵， $S$ 属于可学习偏置， $\alpha$ 属于可学习系数。这三个公式相对比较简单，就不画图了。具体计算细节，可以参考V4论文。



### Applying Parameter Constraints

上面算出来的动态 $\tilde{A}$ 、 $\tilde{B}$ 、 $\tilde{C}$ 矩阵，并不是直接就用的。尤其是 $\tilde{B}$ ，还不是双随机的。所以还需要进一步处理。其中 $\tilde{A}$ 和 $\tilde{C}$ 处理起来比较容易，见DeepSeek-V4论文公式（6）和（7）：

$$
\begin{aligned}
A_l &= \sigma(\tilde{A}_l), \\
C_l &= 2\sigma(\tilde{C}_l).
\end{aligned}
\tag{6,7}
$$

但是 $\tilde{B}$ 处理起来比较麻烦，要用到一种叫做”Sinkhorn-Knopp”的算法，见论文里公式（8）：

$$
\begin{aligned}
M^{(t)} &= \mathcal{T}_r\big(\mathcal{T}_c\big(M^{(t-1)}\big)\big), \\
M^{(0)} &= \mathcal{exp}(\tilde{B_l}) 
\end{aligned}
\tag{8}
$$

这是个递归公式，论文里说递归调用20次差不多就可以了： $\tilde{B}_l = M^{(20)}$ 。这几个公式也比较简单，也不画图了。具体计算细节，可以参考V4论文。



## 总结

本文主要站在推理视角，介绍了DeepSeek-V4核心公式。DeepSeek-V4论文信息量太大了，里面还有很多的细节，本文没有讲到，例如位置编码（RoPE）、Muon优化器等等。等我对这篇论文有了更深的理解，会继续写其他文章深入介绍。



## 主要参考资料

* 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* 论文：[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
* 论文：[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
* 论文：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
* 论文：[Hyper-Connections](https://arxiv.org/abs/2409.19606)
* 配置：[Hugging Face config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)

