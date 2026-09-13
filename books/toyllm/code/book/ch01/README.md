# 第一章 · LLM概览

这一章不写自己的代码，用 HuggingFace transformers 跑通 SmolLM2-135M。

两个目的：先看到终点长什么样；生成一份基准输出，供后面十章对拍。

## 运行

```bash
cd code
uv run python book/ch01/five_lines.py  # 就五行，跑通
uv run python book/ch01/hello.py       # 看结构、看参数量、看 logits
uv run python book/ch01/baseline.py    # 生成全书基准，写入 ../reference/baseline.pt
```

## 文件

| 文件 | 作用 |
|---|---|
| `five_lines.py` | 只有五行有效代码，输出一行文字。整本书后面十章要手写的，就是这五行 |
| `hello.py` | 加载模型、打印结构和参数量、greedy 生成一段文字、看一眼 logits |
| `baseline.py` | 同样跑一遍，但把 input_ids、31 层 hidden_states、logits、greedy 输出存成文件 |

## 产出

`code/reference/baseline.pt`，**全书的地基**：

| 字段 | 形状 | 谁会用 |
|---|---|---|
| `input_ids` | `(1, 4)` | 第三章验证自己写的分词器 |
| `logits` | `(1, 4, 49152)` | 第九章 |
| `hidden_states` | `(31, 1, 4, 576)` | 第九章逐层定位误差 |
| `greedy` / `greedy_text` | `(1, 24)` | 第九章、第十一章 |
| `n_params` | `134515008` | 第二章 |
| `torch_version` / `transformers_version` | | 环境对不上时的第一现场 |

这个文件是 gitignore 的，也是可再生的——但**必须用同样的版本再生**。不同版本的 transformers 可能给出不同的数值，那样后面每一章的对拍都会失败。所以版本号跟着一起存了。

## 本章的输出

```
参数量     134,515,008
输入       'Once upon a time'
输出       'Once upon a time, there was a little girl named Lily. She lived in
            a big house with her family, but'
20 个 token 用了 1.23s（16.2 tok/s，greedy，fp32 CPU）
```

第九章的目标，就是让我们自己写的引擎逐字节地复现上面那句话。

## 引擎现状

还没有——这一章用的是别人的引擎。
