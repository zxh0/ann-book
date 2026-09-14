#!/usr/bin/env python3
"""Auto-fix manuscript body text (正文) violations, in place.

Usage:
    python3 fix.py <file.md | dir> [more...]
    python3 fix.py --dry-run <file.md>     # print a diff instead of writing

Only rules that are safely mechanical belong here; judgment calls stay in
check.py as report-only. To add a fixer:

    def fix_my_rule(lines, path):
        # return the new list of lines (same length not required)
        return [...]

    FIXERS = [fix_my_rule, ...]

A fixer must not touch masked regions (code blocks, formulas, URLs, HTML,
frontmatter) -- use prose.mask_lines to locate safe edit positions.
"""

import difflib
import re
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lintscope import disabled_rules, keep, excluded

from prose import iter_targets, map_formulas, read_lines

# --------------------------------------------------------------------------
# Fixers
# --------------------------------------------------------------------------
# 每个修复器签名 (lines, path) -> new_lines，注册到 FIXERS。

# 规则1 符号记法：只修可以机械判定的两种写法。
# 粗体大写（\mathbf{W}）不在此列——去粗体还是改成粗体小写取决于该符号是
# 矩阵还是向量，需要人工判断，check.py 只报告。
_VEC_LOWER = re.compile(r'\\vec\{\s*([a-z][a-zA-Z0-9]*)\s*\}')
_BOLDSYM_LOWER = re.compile(r'\\boldsymbol\{\s*([a-z][a-zA-Z0-9]*)\s*\}')


def fix_notation(lines, path):
    """\\vec{x} → \\mathbf{x}，\\boldsymbol{x} → \\mathbf{x}（拉丁小写）。"""
    def rewrite(content):
        content = _VEC_LOWER.sub(r'\\mathbf{\1}', content)
        content = _BOLDSYM_LOWER.sub(r'\\mathbf{\1}', content)
        return content

    return map_formulas(lines, rewrite)


fix_notation.rule = "符号记法"

FIXERS = [
    fix_notation,
]


# --------------------------------------------------------------------------


def fix_file(path, dry_run=False, all_rules=False):
    original = read_lines(path)
    lines = list(original)
    off = set() if all_rules else disabled_rules(path)
    for fixer in FIXERS:
        if getattr(fixer, 'rule', None) in off:
            continue
        lines = fixer(lines, path)

    if lines == original:
        return 0

    if dry_run:
        diff = difflib.unified_diff(original, lines,
                                    fromfile=path, tofile=path + " (fixed)")
        sys.stdout.writelines(diff)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    return sum(1 for a, b in zip(original, lines) if a != b) or 1


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    dry_run = '--dry-run' in sys.argv
    all_rules = '--all-rules' in sys.argv
    include_generated = '--include-generated' in sys.argv
    if not args:
        print(f"Usage: {sys.argv[0]} [--dry-run] [--all-rules] <file.md | dir> [more...]")
        sys.exit(1)

    if not FIXERS:
        print("（尚未定义任何正文修复规则，见 SKILL.md 的 Rules 小节）")
        return

    changed = 0
    for path in iter_targets(args, include_generated):
        if excluded(path):
            continue
        n = fix_file(path, dry_run, all_rules)
        if n:
            changed += 1
            print(f"{'would fix' if dry_run else 'fixed'}: {path}")

    print(f"✓ 完成，{changed} 个文件{'需要修改' if dry_run else '已修改'}")


if __name__ == "__main__":
    main()
