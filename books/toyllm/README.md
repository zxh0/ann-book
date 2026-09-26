# 自己动手写LLM推理引擎

> 书名待定。

一本正在写的中文书：用 Python + PyTorch 从零手写一个 LLM 推理引擎，借此理解 LLM 到底是怎么推理的。

目标模型是 **SmolLM2-135M**，跑在**笔记本 CPU** 上。全程不使用 `transformers` 的建模代码——分词、权重加载、RMSNorm、RoPE、GQA、SwiGLU、KV Cache、采样、生成循环，全部自己写。慢，但每一个数字都能和 HuggingFace 对上。

主张很简单：

> 从训练入手，会比较难；从推理入手，就比较简单，然后再去学训练。
>
> 从 GPU 入手，会比较难；从 CPU 入手，就比较简单，然后再去学 GPU。

完整的写作动机、目标读者和阅读约定，见 [前言](chapters/_preface.md)。

## 目录

十二章，分成两段。

**跑通**——沿数据流从左走到右，每章造一个部件：

| 章 | | |
|---|---|---|
| 一 | [LLM推理引擎概览](chapters/ch01_overview.md) | 先用别人的引擎跑起来，看到终点 |
| 二 | [模型文件与权重](chapters/ch02_weights.md) | 272 个张量、134,515,008 个参数 |
| 三 | [分词（Tokenization）](chapters/ch03_tokenizer.md) | 全书唯一不碰权重的一章 |
| 四 | [词嵌入（Embedding）](chapters/ch04_embedding.md) | 一个矩阵读两遍，得到一个零层的 LLM |
| 五 | [归一化（Normalization）](chapters/ch05_norm.md) | RMSNorm，pre-norm 与 post-norm |
| 六 | [注意力机制（Attention）](chapters/ch06_attn.md) | 唯一让 token 互相交换信息的部件 |
| 七 | [位置编码（Positional Encoding）](chapters/ch07_rope.md) | 补上 RoPE，注意力才算完整 |
| 八 | [前馈网络（FFN）](chapters/ch08_ffn.md) | SwiGLU，顺带看一眼 MoE |
| 九 | [残差连接](chapters/ch09_residual.md) | 把两个子层包起来，Block 至此完整 |
| 十 | [采样（Sampling）](chapters/ch10_sampling.md) | 接上生成循环，引擎能用了 |

**跑快**——不改变数据流，改变执行方式：

| 章 | | |
|---|---|---|
| 十一 | [KV Cache](chapters/ch11_kvcache.md) | prefill 与 decode |
| 十二 | [FlashAttention](chapters/ch12_flash_attn.md) | 在线 softmax 与分块 |

## 仓库结构

| 路径 | 说明 |
|---|---|
| `chapters/` | 书稿，一章一个文件；`_front.md` 是扉页，`_preface.md` 是前言 |
| `code/` | 代码，见 [code/README.md](code/README.md) |
| `images/` | 章节配图（PNG） |
| `draw/` | 配图的 draw.io 源文件 |

## 状态

书稿在写，各章已有结构和内容草稿，尚未定稿。

随书代码（`code/book/`）写到第四章，后面八章还没动。`code/poc/` 里是最早的概念验证，已经跑通全流程并有 128 个测试，作为正确性参照保留。
