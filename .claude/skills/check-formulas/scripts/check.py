#!/usr/bin/env python3
"""Check markdown files for formula rule violations.

Usage: python check.py [--all-rules] <file.md | dir> [more...]
"""

import re

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lintscope import disabled_rules, keep, excluded

SKIP_BASENAMES = {"Book.md"}


def iter_md(paths, include_generated=False):
    """Expand files/dirs into the .md files to process (dot-dirs skipped)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for name in sorted(files):
                    if not name.endswith('.md'):
                        continue
                    if not include_generated and name in SKIP_BASENAMES:
                        continue
                    out.append(os.path.join(root, name))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def mask_fences(lines):
    """Blank out the inside of ``` code blocks, keeping line numbers.

    A $ in JSON or shell output is just a character, not a formula. The ```
    lines themselves stay, so a $$ right after a code block still counts as
    having no blank line above it.
    """
    out, in_fence = [], False
    for line in lines:
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            out.append(line)
        else:
            out.append('\n' if in_fence else line)
    return out


def check(path):
    with open(path, encoding="utf-8") as f:
        lines = mask_fences(f.readlines())

    issues = []

    # Rule 1: inline formula spacing
    inline_pat = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
    for i, line in enumerate(lines, 1):
        for m in inline_pat.finditer(line):
            start, end = m.start(), m.end()
            before = line[start - 1] if start > 0 else ' '
            after = line[end] if end < len(line) else ' '
            left_ok = before in (' ', '\t', '\n') or start == 0
            right_ok = after in (' ', '\t', '\n', '') or end == len(line)
            if not left_ok or not right_ok:
                side = []
                if not left_ok:
                    side.append("左侧缺空格")
                if not right_ok:
                    side.append("右侧缺空格")
                issues.append((i, "行内公式", ", ".join(side), line.rstrip()))

    # Rule 2: block formula blank lines
    in_block = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == '$$' and not in_block:
            in_block = True
            prev_line = lines[i - 2].strip() if i >= 2 else ''
            if prev_line != '':
                issues.append((i, "块级公式", "上方缺空行", line.rstrip()))
        elif stripped == '$$' and in_block:
            in_block = False
            next_line = lines[i].strip() if i < len(lines) else ''
            if next_line != '':
                issues.append((i, "块级公式", "下方缺空行", line.rstrip()))

    # Rule 3: forbidden alignment environments
    forbidden_envs = re.compile(r'\\begin\{(align\*?|eqnarray\*?|split)\}')
    for i, line in enumerate(lines, 1):
        m = forbidden_envs.search(line)
        if m:
            issues.append((i, "对齐环境", f"禁止使用 {m.group()}，改用 \\begin{{aligned}}", line.rstrip()))

    # Rule 4: tag placement
    in_block = False
    block_lines = []
    block_start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == '$$' and not in_block:
            in_block = True
            block_start = i
            block_lines = []
        elif stripped == '$$' and in_block:
            in_block = False
            content = '\n'.join(block_lines)
            has_tag = r'\tag{' in content
            has_aligned = r'\begin{aligned}' in content
            if has_tag:
                if has_aligned:
                    aligned_inner = re.search(
                        r'\\begin\{aligned\}(.*?)\\end\{aligned\}',
                        content, re.DOTALL)
                    if aligned_inner and r'\tag{' in aligned_inner.group(1):
                        for j, bl in enumerate(block_lines):
                            if r'\tag{' in bl:
                                issues.append((block_start + 1 + j, "tag位置",
                                    r"\tag 在 \begin{aligned} 内部，需移至 \end{aligned} 之后",
                                    bl.rstrip()))
                else:
                    for j, bl in enumerate(block_lines):
                        if r'\tag{' in bl:
                            issues.append((block_start + 1 + j, "tag位置",
                                r"有 \tag 但无对齐环境，需用 \begin{aligned} 包裹",
                                bl.rstrip()))
        elif in_block:
            block_lines.append(line.rstrip())

    # Rule 5: \operatorname not supported
    for i, line in enumerate(lines, 1):
        if r'\operatorname' in line:
            issues.append((i, "operatorname", r"不支持 \operatorname，改用 \mathrm", line.rstrip()))

    # Rule 6: bare function names inside formulas
    bare_func_pat = re.compile(r'(?<!\\)(?<![a-zA-Z])([a-zA-Z]{2,})\(')
    inline_formula_pat = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
    in_block = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == '$$':
            in_block = not in_block
            continue
        if in_block:
            for m in bare_func_pat.finditer(line):
                issues.append((i, "裸函数名",
                    f"函数名 `{m.group(1)}` 未用 \\mathrm{{}} 包裹",
                    line.rstrip()))
        else:
            for fm in inline_formula_pat.finditer(line):
                for m in bare_func_pat.finditer(fm.group(1)):
                    issues.append((i, "裸函数名",
                        f"函数名 `{m.group(1)}` 未用 \\mathrm{{}} 包裹",
                        line.rstrip()))

    # Rule 7: a "[" right after \\ or \begin{...} is read as an optional argument
    for i, text in bracket_after_break(lines):
        issues.append((i, "行首方括号",
            r"[ 紧跟在 \\ 或 \begin{...} 后面，会被当成可选参数吞掉，改成 \left[...\right] 或在前面加 {}",
            text))

    return issues


# \\[2pt] and \begin{aligned}[t] are the legitimate uses of that optional argument.
_OPT_ARG_OK = re.compile(r'\[\s*(-?[\d.]+\s*(pt|em|ex|mm|cm|in|mu)|[tbc])\s*\]')
_BREAK_END = re.compile(r'(\\\\|\\begin\{[^}]*\})\s*$')
_BREAK_MID = re.compile(r'(\\\\|\\begin\{[^}]*\})\s*(?=\[)')


def bracket_after_break(lines):
    """Yield (lineno, text) for every "[" that LaTeX would take as an optional argument.

    Two shapes, both inside $$...$$ only:
      - a line starting with "[" whose previous non-blank line ends with \\ or \begin{...}
      - "\\[" or "\begin{...}[" on the same line
    A real optional argument such as \\[2pt] or \begin{aligned}[t] is left alone.
    """
    in_block = False
    prev = ''
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == '$$':
            in_block = not in_block
            prev = ''
            continue
        if not in_block or not stripped:
            continue
        hits = []
        if stripped.startswith('[') and _BREAK_END.search(prev) and not _OPT_ARG_OK.match(stripped):
            hits.append(line.rstrip())
        for m in _BREAK_MID.finditer(line):
            if not _OPT_ARG_OK.match(line, m.end()):
                hits.append(line.rstrip())
        if hits:
            yield i, hits[0]
        prev = stripped


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    all_rules = '--all-rules' in sys.argv
    include_generated = '--include-generated' in sys.argv
    if not args:
        print(f"Usage: {sys.argv[0]} [--all-rules] <file.md | dir> [more...]")
        sys.exit(1)

    total = skipped = 0
    for path in iter_md(args, include_generated):
        if excluded(path):
            continue
        issues = check(path)
        before = len(issues)
        issues = keep(issues, path, all_rules=all_rules)
        skipped += before - len(issues)
        if not issues:
            continue
        total += len(issues)
        print(f"=== {path} ===")
        for lineno, rule, problem, text in issues:
            print(f"L{lineno} [{rule}] {problem}")
            print(f"  原文: {text[:100]}")
            print()

    note = f"（另有 {skipped} 处被 .claude/lint-scope.json 豁免）" if skipped else ""
    if total == 0:
        print("✓ 未发现问题" + note)
    else:
        print(f"共发现 {total} 处违规{note}")
        sys.exit(1)


if __name__ == "__main__":
    main()
