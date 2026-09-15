# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language book in progress, 《自己动手写LLM推理引擎》 (title still provisional) — teaching how LLM **inference** works by hand-writing an engine for it. Target model is SmolLM2-135M on PyTorch 2.2 / CPU.

The thesis, stated in `chapters/_preface.md`: 从训练入手会比较难，从推理入手就比较简单；从 GPU 入手会比较难，从 CPU 入手就比较简单. Everything follows from that — the model choice, the hardware, the chapter order.

Three standing conventions from the preface that the manuscript must honour:

1. **No backpropagation.** The target reader is 略懂 AI/DL but 害怕微积分和反向传播. Anything whose real explanation lives in gradients gets the result only, plus its consequence for inference code — never the derivation. (E.g. pre-norm vs post-norm is argued from "is there a rescale on the trunk", not from gradient flow.)
2. **Every chapter diffs against `transformers`.** Chapter 1 dumps a baseline; every later chapter compares to it. The strictness varies deliberately: `==` on ints (ch03) → bit-identical (ch05) → relative error < 1e-5 (ch07 onward, once RoPE's cos table introduces a 1-ulp gap) → back to `torch.equal` (ch10, because argmax chains amplify).
3. **Shapes are the subject.** Most bugs here are shape bugs, not math bugs, and they fail silently. Chapters spell out `(T, 576) → (T, 9, 64) → (T, 576)` at every transformation.

## Layout

| Path | Role |
|------|------|
| `chapters/_front.md` | 扉页 — title, English subtitle, version stamp, cover image |
| `chapters/todo.md` | scratch list of open decisions; **not part of the book**, to be deleted |
| `chapters/_preface.md` | 前言 — motivation, target reader, reading conventions, tech choices, TOC. Also absorbed the old `ch00.md` (准备). Sorts before `ch01` thanks to the `_` prefix. |
| `chapters/chNN_slug.md` | manuscript source, one file per chapter, `chNN` matches 第N章. Twelve chapters, `ch01`–`ch12`. |
| `images/chNN/` | exported PNGs, referenced from chapters as `../images/chNN/...` inside `<img ... style="zoom:50%;">` tags |
| `draw/chNN.drawio` | draw.io sources for those PNGs (only ch01, ch02 exist so far) |
| `code/` | one uv project holding two code trees — see `code/CLAUDE.md` |
| `code/book/` | the book's code: one directory per chapter, `ch01/`…`ch12/`. **New work goes here.** Only `ch01/` is written so far. |
| `code/poc/` | the original proof of concept. **Frozen reference, not the book's code.** |
| `code/models/`, `code/reference/` | weights and baseline tensors, shared by both trees, gitignored |

There is **no** `Book.md` and no `merge.py` here (unlike `../ann4us/`). The table of contents lives in `_preface.md` under 本书结构; chapter order is otherwise carried by the filenames.

**Read `code/CLAUDE.md` before writing any code.** It records the environment pins, the verified SmolLM2 config values, and the tokenizer / RoPE / KV-cache gotchas that were found the hard way. Reuse those findings rather than rediscovering them — but do not copy `poc/`'s file layout or API, since `book/` is deliberately a different take (one self-contained directory per chapter). Extending `poc/` is not the task.

## Chapters

Order follows the data flow of one forward pass, split into two parts.

**跑通 (1–10)** — walk the pipeline left to right, one component per chapter:

| Ch | Topic | PoC reference |
|---|---|---|
| ch01 | LLM推理引擎概览 — run it with `transformers` first, dump the baseline | `poc/steps/00_reference.py` |
| ch02 | 模型结构与权重 | `poc/toyllm/config.py`, `weights.py`, `poc/steps/01_load_and_inspect.py` |
| ch03 | 分词（Tokenizer） | `poc/toyllm/tokenizer.py`, `poc/steps/02_tokenizer.py` |
| ch04 | 词嵌入（Embedding） — one matrix read both ways | `poc/toyllm/weights.py`, `model.py` |
| ch05 | 归一化（Normalization） — RMSNorm, pre-norm vs post-norm | `layers.RMSNorm`, `poc/steps/03_rmsnorm.py` |
| ch06 | 注意力机制（Attention） — GQA + causal mask | `layers.Attention`, `poc/steps/06_attention.py` |
| ch07 | 位置编码（Positional Encoding） — classic PE, then RoPE | `layers.RotaryEmbedding`, `poc/steps/05_rope.py` |
| ch08 | 前馈网络（FFN） — SwiGLU, with a brief look at MoE | `layers.SwiGLU`, `poc/steps/04_swiglu.py` |
| ch09 | 残差连接 — wraps both sublayers; the Block is complete here | `poc/toyllm/model.py`, `poc/steps/07_forward.py` |
| ch10 | 采样（Sampling） — plus the generate loop | `poc/toyllm/sampling.py`, `poc/steps/09_sampling.py` |

**跑快 (11–12)** — same data flow, different execution:

| Ch | Topic | PoC reference |
|---|---|---|
| ch11 | KV Cache — prefill / decode | `layers.KVCache` / `NoCache`, `poc/steps/08_kv_cache.py` |
| ch12 | FlashAttention — online softmax + tiling | not implemented |

Note the chapter order **differs from the order the PoC was built in**, so a chapter's material usually lives in a differently-numbered step script. Three chapters have no single PoC counterpart: ch04 (the embedding/lm_head pair was never its own step), ch09 (the PoC never separated residuals from the Block), and ch12.

Each unfinished chapter opens with a `> **本章代码**` blockquote stating what to build, the tensor shapes, the current engine state, what to diff against, and the known traps. **This block is scaffolding, not book content** — the author deletes it once the chapter's prose is written (ch01 no longer has one). Treat it as the chapter's build contract while drafting, and never as text a reader will see.

A consequence: the diffing criteria live only in those blocks, in the code, and in this file. **The prose never mentions the baseline file or the tolerance table.** When a chapter needs to refer to the reference output, it points at something the reader has actually seen, e.g. 「和第一章那句话一字不差」 rather than 「和第一章那份基准逐字节相同」.

**When writing a chapter, run its code first and quote real output.** The numbers in ch02 (272 tensors, 269.1 MB, 134,515,008 params, 3,540,096 per block) come from `poc/steps/01_load_and_inspect.py` and must keep matching it.

## Commands

Everything runnable lives under `code/`, and must go through `uv run` (system `python3` is 3.9 with no torch):

```bash
cd code
uv sync --group dev                                # .venv + deps + pytest oracles
uv run python book/ch01/hello_5l.py              # run one chapter's code (book/ is where new work goes)
uv run python poc/steps/07_forward.py              # run one PoC milestone script
uv run pytest                                      # all tests (testpaths = poc/tests)
uv run pytest poc/tests/test_rope.py -x            # one test file
```

Downloads (weights, `uv sync`) must bypass Claude Code's proxy — prefix with
`env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy`. See `code/CLAUDE.md`.

## Writing conventions

The repo-wide rules live in `../../CLAUDE.md` — Simplified Chinese prose with English identifiers, no space between CJK and Latin/digits, full-width quotes, no em-dashes, and the notation convention (vectors **bold lowercase**, matrices plain uppercase). Only what is specific to this book is listed here.

- **Em-dash cleanup is in progress**: 171 `——` remain in ch02–ch12 and `_preface.md`, from before the author's voice pass. Clear them as each chapter gets rewritten rather than in one sweep.
- **The model is always `SmolLM2`, never `SmolLM`.** This typo has recurred several times; grep for `SmolLM[^2]` after editing.
- Bold marks a term's first formal definition, given as **中文**（English）: **词元**（Token）, **前馈网络**（Feed-Forward Network，简称FFN）. Sampling parameters keep their conventional lowercase-hyphenated spelling in prose (`top-k`, `top-p`) and snake_case in code (`top_k`, `top_p`).
- Chapter titles are 中文（English）where a standard English term exists — 分词（Tokenizer）, 归一化（RMSNorm） — and plain English only where no settled Chinese term exists (KV Cache, FlashAttention).
- Chapters end with a `## 本章小结` section (小结 = summary; 小节 would mean subsection — the whole book was corrected from that typo, so do not reintroduce it).
- Chapter files may still hold **raw pasted terminal or AI output as scratch material**, and `_preface.md` keeps its open questions in a trailing HTML comment. Do not assume everything in a chapter file is intended final text.
- The preface is the author's own voice, first person. Preserve their wording when editing it; add rather than rewrite.

### Linters

Both live at the repo root, two levels up (`../..` **is** the `ann-book` directory), and are documented in `../../CLAUDE.md`:

```bash
python3 ../../.claude/skills/check-formulas/scripts/check.py chapters/chNN_slug.md
python3 ../../.claude/skills/check-prose/scripts/check.py    chapters/chNN_slug.md
```

Both accept a directory (`chapters/`) or the whole repo, and both have a `fix.py` beside them that takes `--dry-run`. Run `check-formulas` after editing any chapter — inline `$...$` needs surrounding spaces, and it is easy to miss by hand. `check-prose` is useful here mainly for the em-dash sweep above, but note it also reports on the `> **本章代码**` scaffolding blockquotes, which are not book content — ignore those hits rather than editing the scaffolding to satisfy the linter. Its toy-code rules dispatch on the `toy_*.py` filename, so they do not touch `code/`.
