# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Chinese-language writing repo holding **two books in progress** plus a series of standalone deep-dive articles. Published at https://zxh0.github.io/ann-book/ (supports full-text search and formula rendering).

| Path | What | Its own CLAUDE.md |
|------|------|-------------------|
| `books/ann4us/` | 《人人能懂的人工神经网络》 — accessible intro to neural networks for general readers, tracing AI history from the 1940s neuron to Transformers | `books/ann4us/CLAUDE.md` |
| `books/toyllm/` | 《自己动手写LLM推理引擎》 (title provisional) — hand-write an LLM inference engine for SmolLM2-135M on CPU | `books/toyllm/CLAUDE.md`, plus `books/toyllm/code/CLAUDE.md` |
| `notes/` | standalone 图解 articles, `YYYY-MM-DD-Slug.md` | — |
| `tools/` | standalone single-file web tools, one directory each | — |

**Read the relevant book's own CLAUDE.md before working on it.** Each carries that book's layout, build commands, chapter status, and local conventions. This file holds only what is true across the whole repo.

This directory **is** the git root (`github.com/zxh0/ann-book`), and `main` is the only branch — writing is committed straight to it.

## notes/

Standalone deep-dive articles, published separately from the books. One Markdown file per article, named `YYYY-MM-DD-Slug.md` by publication date, with draw.io sources in `notes/draw/` and exported assets in `notes/images/<slug>/`. `README.md` at the repo root lists them all and is the index to keep in sync when adding one.

These are **hand-written by the author and deliberately not AI-polished**. When editing them, limit changes to what was asked (e.g. a typo fix) and preserve the author's voice.

## tools/

Self-contained front-end tools, one directory per tool, each an `index.html` with no build step and no dependencies plus a `README.md` explaining how it works. `scripts/build-site.mjs` copies every directory listed in its `TOOLS` table to `docs/public/tools/<dir>/`, so the site address is `/tools/<dir>/` and the landing page is `docs/tools.md`. The `README.md` is repo-only and is not copied to the site. **Adding a tool means dropping a directory into `tools/` and adding one entry to `TOOLS`** (title, blurb, note) — the `/tools` page and the homepage blurb read that entry from `sidebar.json`. Keep `README.md` at the repo root in sync too.

The tool pages live outside VitePress, so links to them need `withBase()` in an explicit `<a target="_blank">`: a plain Markdown link to `/tools/...` does not get the `/ann-book/` base prefix, because VitePress only rebases links that resolve to a real page.

## Notation Conventions

Repo-wide, both books and the notes:

- Scalars: lowercase — $x$, $w$, $b$, $\eta$
- Vectors: **bold** lowercase — `\mathbf{x}`, `\mathbf{h}`, `\mathbf{b}` (Greek: `\boldsymbol{\alpha}`, since `\mathbf` has no effect on Greek)
- Matrices: uppercase, never bold — $W$, $Q$, $K$, $V$
- Layer indices: superscripts

Bold is reserved for vectors, so bold uppercase (`\mathbf{W}`) and arrow vectors (`\vec{x}`) are both wrong. Enforced by the `check-prose` rule 符号记法统一.

## Content Language

Prose is **Simplified Chinese**; code, identifiers, filenames, and diagram names are English. No space between CJK and Latin/digits (`SmolLM2模型`, not `SmolLM2 模型`); full-width Chinese quotes `“”`, never `""`. The author dislikes em-dashes (`——`) — use a comma, a colon, or split the sentence.

## Custom Skills

Two linters in `.claude/skills/`, both **repo-wide**: the same prose and formula conventions govern `books/ann4us/`, `books/toyllm/`, and `notes/`. Each takes files, directories, or several paths at once, recurses into directories, and skips dot-dirs and the generated `Book.md`.

```bash
python3 .claude/skills/check-formulas/scripts/check.py .     # LaTeX formatting for GitHub rendering
python3 .claude/skills/check-prose/scripts/check.py .        # Chinese body text + toy_*.py examples
```

`check-formulas` rules: inline `$...$` needs surrounding spaces, blank lines around block formulas, `\begin{aligned}` for multi-line, `\tag{}` after `\end{aligned}`, no `\operatorname`, multi-letter function names in `\mathrm{}`.

`check-prose` rules: 符号记法统一 (notation), 尽量不用破折号 (em-dash), 公式用 `$`/代码用反引号 (math-vs-code), 玩具代码规范 (toy-code, applies to `toy_*.py` by filename), plus a semantic check that a chapter's code blocks match the `.py` file verbatim.

### Per-area exemptions

Not every rule suits every area, so the exceptions live in one file — `.claude/lint-scope.json` — read by both skills:

```json
{
  "exclude": ["CLAUDE.md", "**/CLAUDE.md", "**/todo.md", "node_modules/**", "docs/**"],
  "scopes": [{"path": "notes/**", "disable": ["符号记法"]}]
}
```

`exclude`d files are not checked at all; a file inherits the union of every matching scope's `disable` list. The notes deliberately follow each paper's own notation (`\boldsymbol`, bold uppercase matrices), which is why 符号记法 is off there — that one exemption accounts for ~166 findings. `node_modules/` and `docs/` are excluded because neither is manuscript source: the first is dependencies, the second is the VitePress build output, and scanning them buries the real findings in noise (a README's `${...}` template literals read as inline formulas). Exempted counts are always reported ("另有 N 处被 .claude/lint-scope.json 豁免"), never silently dropped.

`--all-rules` puts the *scope* exemptions back for one run; `exclude` still applies, so it never drags `node_modules/` back in.

**Adding a book or a notes folder means editing that JSON, not the scripts** — rules dispatch on file type, never on path. Shared logic lives in `.claude/skills/lintscope.py`.

Each `check.py` has a `fix.py` beside it that auto-fixes the mechanical rules. Both take `--dry-run`. `check-prose`'s fixer honours per-area exemptions; `check-formulas`'s rewrites in one pass, so it skips any file with a formula rule switched off rather than half-applying. **Always `--dry-run` first on `notes/`** — those are hand-written.

Run both after editing any chapter, before committing.

### shrink-images

A third skill, also repo-wide, compresses the manuscript's PNGs in place — for after new illustrations land in `images/chNN/` or `aigc/`, before committing:

```bash
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py --dry-run .
```

It flattens alpha onto white and strips metadata (lossless), then quantizes to an 8-bit palette only when the measured mean error says the image is line art (draw.io exports land at 0.000–0.055, AI cover art at 3.5). draw.io exports shrink by 70–90%, AI 章首图 by ~25% with pixels unchanged. It honours `lint-scope.json`'s `exclude`, so `docs/` and `node_modules/` stay out of it, and it refuses to rewrite a file for less than 10% / 1 KB, to keep already-compressed images out of git as new blobs. It needs Pillow, declared inline (PEP 723) so `uv run` builds the env — the system `python3` is 3.9 with no Pillow. Details and the calibration table are in its `SKILL.md`; run `npm run prepare:site` afterwards so `docs/public/` picks the new files up.
