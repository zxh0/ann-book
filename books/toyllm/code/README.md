# code

《自己动手写LLM推理引擎》的代码。一个 uv 项目，两棵代码树，共用一个 `.venv`、一份权重、一份基准输出。

```
code/
├── models/       模型权重（SmolLM2-135M），两边共用，不入库
├── reference/    第一章生成的基准张量，两边共用，不入库
├── book/         随书代码：一章一个目录 ch01/ … ch12/
├── poc/          最初的概念验证：能跑通的完整实现，只作参照
└── download_model.sh   把权重下到 models/ 里，第一次必跑
```

## 快速开始

```bash
uv sync --group dev          # 创建 .venv，装依赖和测试用的 oracle
./download_model.sh          # 下载 SmolLM2-135M，十个文件约 272 MB
uv run pytest                # 跑测试（128 个）
```

权重放在 `models/SmolLM2-135M/`，不入库，所以第一次必须跑一遍 `download_model.sh`。已经下过的文件会跳过，中途断了再跑一遍会接着下。连不上 HuggingFace 就换镜像站：

```bash
HF_ENDPOINT=https://hf-mirror.com ./download_model.sh
```

下载权重或同步依赖时如果卡住，是代理的问题，前面加 `env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy`。详见 [CLAUDE.md](CLAUDE.md)。

## book/ —— 随书代码

每一章一个目录，从上一章复制过来再增量修改，**每一章都能单独跑**。想看第五章当时那个"骨架搭好了但还说不出人话"的状态，直接跑 `ch05/` 就行，不用 `git checkout`。

代价是十二份副本会漂移。改公共部分时记得同步，并用检查脚本确认「第 N 章里本章没改动的文件，和第 N-1 章逐字节相同」。

权重和基准输出**不要复制进章节目录**——269 MB 乘十一份没有意义，用相对路径指向 `code/models` 和 `code/reference`。

## poc/ —— 概念验证

最早写的那一版，已经跑通到采样，配套测试和 `transformers` 逐层对拍。**它不是随书代码**，不要在上面继续加东西。

它的价值在两处：一是**正确性参照**——随书代码写出来对不对，拿它和它的测试比；二是**踩坑笔记**——环境版本为什么这么钉、SmolLM2 的配置值、分词 / RoPE / KV Cache 那些不报错的静默错误，全记在 [CLAUDE.md](CLAUDE.md) 里。写新代码前先读它，别再踩一遍。

```
poc/
├── toyllm/    引擎本体（包）
├── steps/     分步里程碑脚本，每步一个可独立运行的 demo
└── tests/     pytest，重点是和参考实现的数值对拍
```

```bash
uv run python poc/steps/00_reference.py    # 用 transformers 生成基准张量
uv run python poc/steps/07_forward.py      # 完整前向，逐层和基准对拍
uv run python poc/steps/09_sampling.py     # 采样
```

`steps/` 按 00–09 编号，是**代码的构建顺序**，和书的章节顺序不同（书按数据流走）。对应关系见项目根目录的 [CLAUDE.md](../CLAUDE.md)。
