#!/usr/bin/env python3
"""Auto-fix formula rule violations in a markdown file.

Fixes applied:
  - inline $...$ missing spaces → add space before opening $ and after closing $
  - block $$ missing blank lines → insert blank line before/after each $$ marker
  - bare function names in formulas (e.g. max() → \mathrm{max}())
  - \begin{align} / \begin{align*} / etc. → \begin{aligned}
  - \tag inside \begin{aligned} → moved after \end{aligned}, multiple tags merged
  - \tag on bare block formula → wrapped with \begin{aligned}
  - \operatorname → \mathrm

Usage: python fix.py [--dry-run] [--all-rules] <file.md | dir> [more...]

A file with any of these rules switched off in .claude/lint-scope.json is
skipped entirely: fix() applies all rules as one pass, so it cannot honour
a partial exemption. Report such files with check.py and fix them by hand.
"""

import difflib
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


tag_pat = re.compile(r'\s*\\tag\{([^}]+)\}')


def strip_tags(line):
    tags = tag_pat.findall(line)
    cleaned = tag_pat.sub('', line)
    return cleaned, tags


def _fix_line_inline_spacing(line):
    out = []
    i = 0
    while i < len(line):
        if line[i] == '$':
            if i + 1 < len(line) and line[i + 1] == '$':
                # $$ on a content line — pass through unchanged
                out.append('$$')
                i += 2
            else:
                # opening $: add space before if needed
                if out and out[-1] not in (' ', '\t'):
                    out.append(' ')
                out.append('$')
                i += 1
                # copy formula content until closing $
                while i < len(line):
                    if line[i] == '$':
                        out.append('$')
                        i += 1
                        # add space after closing $ if needed
                        if i < len(line) and line[i] not in (' ', '\t', '$'):
                            out.append(' ')
                        break
                    out.append(line[i])
                    i += 1
        else:
            out.append(line[i])
            i += 1
    return ''.join(out)


def fix_inline_spacing(content):
    lines = content.split('\n')
    result = []
    in_block = False
    for line in lines:
        if line.strip() == '$$':
            in_block = not in_block
            result.append(line)
        elif in_block:
            result.append(line)
        else:
            result.append(_fix_line_inline_spacing(line))
    return '\n'.join(result)


def fix_block_blank_lines(content):
    lines = content.split('\n')
    out = []
    in_block = False
    for i, line in enumerate(lines):
        if line.strip() == '$$':
            if not in_block:
                # opening $$: ensure blank line before it
                if out and out[-1].strip() != '':
                    out.append('')
                out.append(line)
                in_block = True
            else:
                # closing $$: ensure blank line after it
                out.append(line)
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    out.append('')
                in_block = False
        else:
            out.append(line)
    return '\n'.join(out)


_bare_func_pat = re.compile(r'(?<!\\)(?<![a-zA-Z])([a-zA-Z]{2,})\(')


def _fix_bare_funcs_in_formula(formula_content):
    return _bare_func_pat.sub(r'\\mathrm{\1}(', formula_content)


def _fix_line_bare_functions(line):
    out = []
    i = 0
    while i < len(line):
        if line[i] == '$':
            if i + 1 < len(line) and line[i + 1] == '$':
                out.append('$$')
                i += 2
            else:
                out.append('$')
                i += 1
                formula_chars = []
                while i < len(line):
                    if line[i] == '$':
                        out.append(_fix_bare_funcs_in_formula(''.join(formula_chars)))
                        out.append('$')
                        i += 1
                        break
                    formula_chars.append(line[i])
                    i += 1
        else:
            out.append(line[i])
            i += 1
    return ''.join(out)


def fix_bare_functions(content):
    lines = content.split('\n')
    result = []
    in_block = False
    for line in lines:
        if line.strip() == '$$':
            in_block = not in_block
            result.append(line)
        elif in_block:
            result.append(_fix_bare_funcs_in_formula(line))
        else:
            result.append(_fix_line_bare_functions(line))
    return '\n'.join(result)


def fix(content):
    # Fix 1: inline formula spacing
    content = fix_inline_spacing(content)

    # Fix 2: block formula blank lines
    content = fix_block_blank_lines(content)

    # Fix 3: bare function names
    content = fix_bare_functions(content)

    # Fix 4: alignment environments
    content = re.sub(r'\\begin\{align\*?\}', r'\\begin{aligned}', content)
    content = re.sub(r'\\end\{align\*?\}', r'\\end{aligned}', content)
    content = re.sub(r'\\begin\{eqnarray\*?\}', r'\\begin{aligned}', content)
    content = re.sub(r'\\end\{eqnarray\*?\}', r'\\end{aligned}', content)
    content = re.sub(r'\\begin\{split\}', r'\\begin{aligned}', content)
    content = re.sub(r'\\end\{split\}', r'\\end{aligned}', content)

    # Fix 4: \operatorname → \mathrm
    content = content.replace(r'\operatorname', r'\mathrm')

    # Fix 5: tag placement — process block by block
    lines = content.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == '$$':
            block = [line]
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                block.append(lines[i])
                i += 1
            block.append(lines[i])  # closing $$
            i += 1

            inner = block[1:-1]
            inner_text = '\n'.join(inner)
            has_tag = r'\tag{' in inner_text
            has_aligned = r'\begin{aligned}' in inner_text

            if not has_tag:
                out.extend(block)
                continue

            if has_tag and not has_aligned:
                # Wrap bare formula with \begin{aligned}, move tag after \end{aligned}
                all_tags = []
                new_inner = []
                for ln in inner:
                    cleaned, tags = strip_tags(ln)
                    cleaned = re.sub(r'(?<!&)(?<!\\)=', r'&=', cleaned, count=1)
                    all_tags.extend(tags)
                    if cleaned.strip():  # skip lines that were tag-only
                        new_inner.append(cleaned)
                merged_tag = ', '.join(all_tags)
                out.append('$$')
                out.append(r'\begin{aligned}')
                out.extend(new_inner)
                out.append(r'\end{aligned}')
                out.append(f'\\tag{{{merged_tag}}}')
                out.append('$$')
                continue

            if has_tag and has_aligned:
                # Remove tags from inside aligned, add merged tag after \end{aligned}
                all_tags = []
                new_block_lines = []
                for ln in inner:
                    if r'\tag{' in ln:
                        cleaned, tags = strip_tags(ln)
                        all_tags.extend(tags)
                        if cleaned.strip():  # skip lines that were tag-only
                            new_block_lines.append(cleaned)
                    else:
                        new_block_lines.append(ln)

                end_idx = next(
                    (j for j, ln in enumerate(new_block_lines) if r'\end{aligned}' in ln),
                    None)
                if end_idx is not None and end_idx > 0:
                    prev = new_block_lines[end_idx - 1]
                    prev = re.sub(r'\s*\\\\\s*$', '', prev)   # strip trailing \\
                    prev = re.sub(r',\s*$', '', prev)          # strip trailing comma
                    new_block_lines[end_idx - 1] = prev
                    merged_tag = ', '.join(all_tags)
                    new_block_lines.insert(end_idx + 1, f'\\tag{{{merged_tag}}}')

                out.append('$$')
                out.extend(new_block_lines)
                out.append('$$')
                continue

        out.append(line)
        i += 1

    return '\n'.join(out)


RULE_NAMES = {"行内公式", "块级公式", "对齐环境", "operatorname", "裸函数名"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    all_rules = '--all-rules' in sys.argv
    dry_run = '--dry-run' in sys.argv
    include_generated = '--include-generated' in sys.argv
    if not args:
        print(f"Usage: {sys.argv[0]} [--dry-run] [--all-rules] <file.md | dir> [more...]")
        sys.exit(1)

    changed = skipped = 0
    for path in iter_md(args, include_generated):
        if excluded(path):
            continue
        if not all_rules and (disabled_rules(path) & RULE_NAMES):
            skipped += 1
            print(f"跳过（lint-scope.json 关闭了本文件的部分公式规则，请手工处理）: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            original = f.read()
        fixed = fix(original)
        if fixed == original:
            continue
        # zip() 会把插入行之后的所有行都算成"改动"，数字虚高，用 diff 实际的增删行数
        n = sum(1 for d in difflib.ndiff(original.splitlines(), fixed.splitlines())
                if d[0] in '+-')
        if dry_run:
            sys.stdout.writelines(difflib.unified_diff(
                original.splitlines(True), fixed.splitlines(True),
                fromfile=path, tofile=path + " (fixed)"))
            print(f"would fix: {path}（{n} 行）")
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"fixed: {path}（{n} 行）")
        changed += 1

    tail = f"，跳过 {skipped} 个" if skipped else ""
    verb = "需要修改" if dry_run else "已修改"
    print(f"✓ 完成，{changed} 个文件{verb}{tail}")


if __name__ == "__main__":
    main()
