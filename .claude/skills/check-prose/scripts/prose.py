#!/usr/bin/env python3
"""Shared helpers for the check-prose skill.

Provides:
  - iter_targets(paths): expand files/dirs into the files to check -- both the
    manuscript markdown and the toy_*.py examples
  - mask_lines(lines): blank out non-prose regions (code, formulas, URLs, HTML,
    frontmatter) so prose rules only ever see body text
  - iter_formulas / map_formulas: read or rewrite formula content only
  - logical_lines(lines): join bracket continuations in Python source
"""

import os
import re

SKIP_BASENAMES = {"Book.md"}
TOY_PREFIX = "toy_"

FENCE_PAT = re.compile(r'^\s*(```|~~~)')
BLOCK_FORMULA_PAT = re.compile(r'^\s*\$\$\s*$')
HTML_LINE_PAT = re.compile(r'^\s*<[a-zA-Z/!].*>\s*$')
INLINE_FORMULA_PAT = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
INLINE_CODE_PAT = re.compile(r'`[^`]*`')
# ![alt](url) / [text](url) -- only the (url) part is masked
LINK_URL_PAT = re.compile(r'(!?\[[^\]]*\])\(([^)]*)\)')


def is_toy(path):
    """True for the manuscript's toy examples, book/code/chNN/toy_*.py."""
    name = os.path.basename(path)
    return name.startswith(TOY_PREFIX) and name.endswith('.py')


def iter_targets(paths, include_generated=False):
    """Expand the given files/dirs into a sorted list of files to check.

    Picks up manuscript markdown and toy_*.py examples. Virtualenvs and other
    dot-directories are skipped, as are non-toy .py files (those are the
    runnable PyTorch examples, which this skill does not govern).
    """
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in sorted(files):
                    full = os.path.join(root, name)
                    if is_toy(name):
                        out.append(full)
                        continue
                    if not name.endswith('.md'):
                        continue
                    if not include_generated and name in SKIP_BASENAMES:
                        continue
                    out.append(full)
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _blank(text):
    """Replace every non-newline char with a space, preserving offsets."""
    return re.sub(r'[^\n]', ' ', text)


def mask_lines(lines):
    """Return a parallel list where non-prose regions are blanked to spaces.

    Line count and character offsets are preserved, so a match position in the
    masked text maps directly back to the original line/column.
    """
    masked = []
    in_fence = False
    in_formula = False
    in_frontmatter = lines and lines[0].strip() == '---'

    for i, line in enumerate(lines):
        if in_frontmatter:
            masked.append(_blank(line))
            if i > 0 and line.strip() == '---':
                in_frontmatter = False
            continue

        if FENCE_PAT.match(line):
            in_fence = not in_fence
            masked.append(_blank(line))
            continue
        if in_fence:
            masked.append(_blank(line))
            continue

        if BLOCK_FORMULA_PAT.match(line):
            in_formula = not in_formula
            masked.append(_blank(line))
            continue
        if in_formula:
            masked.append(_blank(line))
            continue

        if HTML_LINE_PAT.match(line):
            masked.append(_blank(line))
            continue

        out = line
        out = INLINE_FORMULA_PAT.sub(lambda m: _blank(m.group(0)), out)
        out = INLINE_CODE_PAT.sub(lambda m: _blank(m.group(0)), out)
        out = LINK_URL_PAT.sub(
            lambda m: m.group(1) + _blank('(' + m.group(2) + ')'), out)
        masked.append(out)

    return masked


def iter_formulas(lines):
    """Yield (lineno, content, kind) for every formula in the file.

    kind is 'inline' or 'block'. Block formulas yield one entry per line of
    their body, so every reported line number is real. Formulas inside fenced
    code blocks are skipped.
    """
    in_fence = False
    in_formula = False
    for i, line in enumerate(lines, 1):
        if FENCE_PAT.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if BLOCK_FORMULA_PAT.match(line):
            in_formula = not in_formula
            continue
        if in_formula:
            yield i, line.rstrip('\n'), 'block'
            continue

        for m in INLINE_FORMULA_PAT.finditer(INLINE_CODE_PAT.sub(
                lambda c: _blank(c.group(0)), line)):
            yield i, m.group(1), 'inline'


def map_formulas(lines, fn):
    """Return new lines with fn applied to every formula's content.

    fn takes the formula content (no `$` delimiters) and returns the
    replacement. Text outside formulas, and anything inside fenced code
    blocks, is left untouched.
    """
    out = []
    in_fence = False
    in_formula = False
    for line in lines:
        if FENCE_PAT.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        if BLOCK_FORMULA_PAT.match(line):
            in_formula = not in_formula
            out.append(line)
            continue
        if in_formula:
            body = line.rstrip('\n')
            out.append(fn(body) + line[len(body):])
            continue

        out.append(INLINE_FORMULA_PAT.sub(
            lambda m: '$' + fn(m.group(1)) + '$', line))
    return out


FENCE_OPEN_PAT = re.compile(r'^\s*(?:```|~~~)\s*([A-Za-z0-9_+-]*)\s*$')

# Opt-out marker for hand-written demos that deliberately avoid numpy in order
# to show a computation step by step. In markdown it goes in an HTML comment
# before the fence, so it does not appear in the rendered book; in a .py file
# it goes in an ordinary comment.
EXEMPT_MARKER = 'toy-exempt'
EXEMPT_LOOKBACK = 3


def has_exempt_marker(text):
    return EXEMPT_MARKER in text


def block_is_exempt(lines, fence_lineno):
    """True if a `toy-exempt` marker sits just above this code block.

    fence_lineno is 1-based and points at the block's first body line, so the
    fence itself is at index fence_lineno - 2.
    """
    top = max(0, fence_lineno - 2 - EXEMPT_LOOKBACK)
    for probe in lines[top:fence_lineno - 1]:
        if has_exempt_marker(probe):
            return True
    return False
# The runnable PyTorch examples are not governed by the toy-code rules.
RUNNABLE_MARKERS = ('torch', 'nn.', 'datasets', 'transforms', 'gensim',
                    'tiktoken')


def iter_code_blocks(lines, lang='python'):
    """Yield (start_lineno, [body lines]) for each fenced block of `lang`.

    start_lineno is the line number of the block's first body line, so a hit
    at body index i maps to start_lineno + i.
    """
    i = 0
    while i < len(lines):
        m = FENCE_OPEN_PAT.match(lines[i])
        if not m:
            i += 1
            continue
        info = m.group(1)
        body, j = [], i + 1
        while j < len(lines) and not FENCE_PAT.match(lines[j]):
            body.append(lines[j])
            j += 1
        if lang is None or info == lang:
            yield i + 2, body
        i = j + 1


def is_runnable_example(body):
    """True for the PyTorch/library examples, which the toy rules skip."""
    text = ''.join(body)
    return any(marker in text for marker in RUNNABLE_MARKERS)


def logical_lines(lines):
    """Yield (lineno, text) with bracket continuations joined into one entry.

    `np.array([[1, 2],\\n  [3, 4]])` spans three physical lines but is one
    statement; rules need to see it whole. lineno is the first physical line.
    """
    buf, start, depth = '', None, 0
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip('\n')
        stripped = line.split('#')[0] if not line.lstrip().startswith('#') else ''
        if start is None:
            start = i
        buf += (' ' if buf else '') + line.strip()
        depth += stripped.count('(') + stripped.count('[') + stripped.count('{')
        depth -= stripped.count(')') + stripped.count(']') + stripped.count('}')
        if depth <= 0:
            yield start, buf
            buf, start, depth = '', None, 0
    if buf:
        yield start, buf


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.readlines()
