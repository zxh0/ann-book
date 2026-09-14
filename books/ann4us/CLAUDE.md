# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

《人人能懂的人工神经网络》(ANN for the Rest of Us) — an accessible introduction to artificial neural networks for general readers, still in early draft (the title page says so outright).

The premise, from `chapters/01_preface.md`: 只要掌握中学阶段的基础数学知识，普通读者也完全能读懂人工神经网络. Two structural decisions follow from it, and the manuscript must honour both:

1. **History is the thread.** The book is 一部“人工神经网络的简史” — it walks forward from the 1940s artificial neuron through the perceptron, MLP, CNN, RNN, LSTM, Seq2Seq, attention, and Transformer. Chapter order *is* the historical order; a concept is introduced where history introduced it, not where a textbook would.
2. **Training is a black box.** Backpropagation, loss functions, and gradients are deferred to 第十三章 by design (the preface's 把训练过程当黑盒 section frames each topic as a game BOSS that must not be too hard too early). Earlier chapters state results and their consequences, never derivations.

See also the repo-wide conventions in `../../CLAUDE.md` — notation, the prose/formula linters, and the Simplified-Chinese rules apply here too.

## Layout

| Path | Role |
|------|------|
| `chapters/NN_slug.md` | **the manuscript source** — one file per chapter, `NN` fixes the order, e.g. `03_ch01_neuron.md`. Edit these. |
| `chapters/00_front.md` | 扉页 — title, English subtitle, version stamp, disclaimer, cover image |
| `Book.md` | **generated** single-file manuscript. Never edit by hand. |
| `merge.py` | the generator — see below |
| `code/chNN/` | Python examples, one directory per chapter (`ch00`–`ch10`) |
| `draw/chNN_slug.drawio` | draw.io sources, one per chapter; `images/chNN/` holds the exported PNGs |
| `aigc/` | AI-generated art (the cover) |

```bash
python3 merge.py            # regenerate Book.md
python3 merge.py -o /tmp/out.md
python3 merge.py --check    # exit 1 if Book.md is out of sync with chapters/ (version stamp ignored)
```

`merge.py` concatenates `chapters/*.md` sorted by filename and, on the way through: inserts the page-break `<div>` between chapters, rewrites every relative asset path `../` → `./` (covering `../images/` and the cover's `../aigc/`), inserts the 自动生成 banner under the H1, and stamps `版本：vYYYY.MM.DD` with today's date. So **chapter files must not contain the page-break div or the banner**, and asset paths in `chapters/` are always written `../images/...` / `../aigc/...`.

## Chapters

前言, 第零章–第十四章, 后记, and appendices A–D. Chapters 0–10 are written; **11–14 are TODO stubs** — a heading and a few keywords each, nothing to preserve.

| # | Topic | | # | Topic |
|---|-------|-|---|-------|
| 00 | 基础知识 (math prerequisites) | | 08 | 编解码结构 (Seq2Seq) |
| 01 | 人工神经元 | | 09 | 注意力机制 |
| 02 | 单层神经网络 (perceptron) | | 10 | Transformer架构 |
| 03 | 多层神经网络 (MLP) | | 11 | GPT和BERT — **TODO** |
| 04 | 卷积神经网络 (CNN) | | 12 | Diffusion模型 — **TODO** |
| 05 | 循环神经网络 (RNN) | | 13 | 反向传播 — **TODO** |
| 06 | 词元和词嵌入 | | 14 | 应用 (RAG / Agent / MCP) — **TODO** |
| 07 | 长短期记忆网络 (LSTM) | | | |

Appendices: A 科学家, B 论文, C 可视化工具, D 术语列表.

Chapter file conventions:

- Heading levels start at `##` (`## 第一章：人工神经元`), since the H1 is the book title in `00_front.md`. Sections are `###`, subsections `####`.
- Every chapter ends with a `### 本章小结` section.
- **Appendix D (术语列表) is the authority on English→Chinese terminology.** Check it before introducing a new term, and add the term to it when you do.
- Diagrams exported from draw.io are embedded as `<img src="../images/chNN/x.png" alt="x" style="zoom:50%;" />`; plain screenshots (Desmos plots and the like) use ordinary `![alt](../images/chNN/x.png)`.

## Code (`code/`)

uv-managed, Python 3.12+. Dependencies are `numpy`, `gensim`, `tiktoken`; `torch`/`torchvision` are **commented out** in `pyproject.toml`, so the PyTorch examples do not run until they are uncommented and re-synced.

Run from inside `code/`, because that is how the manuscript cites them — the runnable examples carry their own invocation as a comment in the code block, e.g. `# uv run python ch02/perceptron.py`:

```bash
cd code
uv run python ch02/perceptron.py
```

The preface (公式、图表、代码) splits the examples into two kinds, and the distinction is load-bearing:

- **玩具代码** — `toy_*.py`. Short, plain-Python illustrations of a concept, meant to be read like 带有逻辑的示意图 rather than run. These are the files the `check-prose` skill's toy-code rules apply to.
- **可运行示例** — everything else (`perceptron.py`, `mlp.py`, `char_rnn.py`, `lenet5.py`, `word_lstm.py`, …). Real PyTorch that trains a small network on CPU. The book shows only key fragments and points readers at the repo for the rest. Not subject to the toy-code rules.

`check-prose` also verifies that a code block quoted in a chapter is byte-identical to the `.py` file it came from, so edit both sides together.

## Before committing a chapter

```bash
python3 ../../.claude/skills/check-formulas/scripts/check.py chapters/03_ch01_neuron.md
python3 ../../.claude/skills/check-prose/scripts/check.py    chapters/03_ch01_neuron.md
python3 merge.py
```

Both linters take files, directories, or several paths, and recurse (`chapters/`, `code/`, or the whole book). The `fix.py` next to each `check.py` auto-fixes the mechanical rules; both accept `--dry-run`. They are repo-wide and configured by `../../.claude/lint-scope.json` — see the root `CLAUDE.md`.

Note: chapters written before the current notation convention still use plain uppercase for vectors; they are being migrated one chapter at a time, so `check-prose` notation warnings in an old chapter are expected rather than new breakage.
