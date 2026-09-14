#!/usr/bin/env python3
"""Check manuscript body text (正文) for prose rule violations.

Usage:
    python3 check.py <file.md | dir> [more...]

Rules live in the RULES registry below. To add a rule:

    def rule_my_rule(lines, masked, path):
        issues = []
        for i, text in enumerate(masked, 1):
            if <violation in text>:
                issues.append((i, "规则名", "问题描述", lines[i - 1].rstrip()))
        return issues

    RULES = [rule_my_rule, ...]

`lines` is the raw file, `masked` is the same lines with code blocks, formulas,
URLs, HTML and frontmatter blanked out to spaces (offsets preserved). Match
against `masked`, report from `lines`.
"""

import re
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lintscope import disabled_rules, keep, excluded

from prose import (INLINE_CODE_PAT, block_is_exempt, has_exempt_marker,
                   iter_code_blocks, iter_formulas, iter_targets,
                   is_runnable_example, is_toy, logical_lines, mask_lines,
                   read_lines)

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
# 每条规则实现为一个函数，签名 (lines, masked, path)，
# 返回 [(行号, 规则名, 问题描述, 原文), ...]，然后注册到 RULES 列表中。

# 规则1 符号记法：标量=小写，向量=粗体小写，矩阵=大写
BOLD_UPPER_PAT = re.compile(r'\\(mathbf|boldsymbol|bm)\{\s*([A-Z])')
VEC_PAT = re.compile(r'\\vec\{\s*([^}]*)\}')
BOLDSYM_LATIN_PAT = re.compile(r'\\boldsymbol\{\s*([a-zA-Z][a-zA-Z0-9]*)\s*\}')


def rule_notation(lines, masked, path):
    """粗体只用于向量（粗体小写）；矩阵用非粗体大写；不用 \\vec。

    机械层只能查排版层面的硬性违规。一个不加粗的字母到底是标量、向量
    还是矩阵，需要结合上下文人工判读，见 SKILL.md 的语义层步骤。
    """
    issues = []
    seen = set()

    def add(lineno, problem):
        key = (lineno, problem)
        if key not in seen:
            seen.add(key)
            issues.append((lineno, "符号记法", problem, lines[lineno - 1].rstrip()))

    for lineno, content, _kind in iter_formulas(lines):
        for m in BOLD_UPPER_PAT.finditer(content):
            add(lineno, f"粗体大写 `\\{m.group(1)}{{{m.group(2)}...}}`："
                        f"粗体只表示向量（小写），矩阵应用非粗体大写 `{m.group(2)}`")
        for m in VEC_PAT.finditer(content):
            add(lineno, f"箭头向量 `\\vec{{{m.group(1)}}}`："
                        f"向量统一用粗体小写 `\\mathbf{{{m.group(1).lower()}}}`")
        for m in BOLDSYM_LATIN_PAT.finditer(content):
            if m.group(1)[0].islower():
                add(lineno, f"`\\boldsymbol{{{m.group(1)}}}`：拉丁字母的粗体"
                            f"统一用 `\\mathbf{{{m.group(1)}}}`"
                            f"（`\\boldsymbol` 只留给希腊字母）")

    return issues


# 规则3 少用破折号：破折号密集是典型的 AI 文风
EM_DASH_PAT = re.compile(r'\s*—+\s*')

# 规则4 公式用 $...$，代码用反引号
GREEK = 'αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ'
BACKTICK_GREEK_PAT = re.compile(r'`([^`]*[' + GREEK + r'][^`]*)`')
BACKTICK_TIMES_PAT = re.compile(r'`([^`]*×[^`]*)`')
BARE_GREEK_PAT = re.compile(r'[' + GREEK + r']')


def rule_math_vs_code(lines, masked, path):
    """公式相关用 $...$，代码相关用反引号。

    机械层只查高置信度的三类。「单个拉丁字母套反引号」看着像数学变量，
    但本书第五章讲 one-hot 时用 `a`、`b`、`z` 指字母本身，属于正当用法，
    所以那一类交给语义层人工判断，不在这里报。
    """
    issues = []
    for i, line in enumerate(lines, 1):
        for m in BACKTICK_GREEK_PAT.finditer(line):
            issues.append((i, "公式/代码", f"希腊字母 `{m.group(1)}` 用了反引号："
                                          f"数学符号应写成 $...$", line.rstrip()))
        for m in BACKTICK_TIMES_PAT.finditer(line):
            issues.append((i, "公式/代码", f"算式 `{m.group(1)}` 用了反引号："
                                          f"含 × 的算式应写成 $...$", line.rstrip()))
    # 屏蔽掉公式后仍出现的希腊字母，说明它裸露在正文里
    for i, text in enumerate(masked, 1):
        stripped = INLINE_CODE_PAT.sub('', text)
        for m in BARE_GREEK_PAT.finditer(stripped):
            issues.append((i, "公式/代码", f"正文里裸露的希腊字母 `{m.group(0)}`："
                                          f"应放进 $...$ 并用 LaTeX 命令（如 \\sigma）",
                           lines[i - 1].rstrip()))
    return issues


def rule_no_em_dash(lines, masked, path):
    """尽量不用破折号。改写为逗号、冒号、括号，或干脆断成两句。

    只报告，不自动修复：破折号承担的语义每处不同（补充说明、话锋一转、
    举例、同位语），换成什么标点取决于上下文，机械替换会写坏句子。
    """
    issues = []
    for i, text in enumerate(masked, 1):
        for m in EM_DASH_PAT.finditer(text):
            if '—' not in m.group(0):
                continue
            issues.append((i, "破折号",
                           "尽量不用破折号（AI 文风明显）："
                           "改用逗号、冒号、括号，或断成两句",
                           lines[i - 1].rstrip()))
    return issues


RULES = [
    rule_notation,
    rule_no_em_dash,
    rule_math_vs_code,
    # rule_md_toycode is appended below, after the toy-code helpers it reuses
]


# --------------------------------------------------------------------------
# Toy-code rules -- apply to book/code/**/toy_*.py only.
# Signature is (lines, path) -> [(行号, 规则名, 问题描述, 原文), ...]
# --------------------------------------------------------------------------

# 规则2 玩具代码：统一用 numpy array，向量名 vec 开头、矩阵名 mat 开头
NUMPY_IMPORT_PAT = re.compile(r'^\s*(import\s+numpy|from\s+numpy\s+import)')
LIST_ANNOT_PAT = re.compile(r'\blist\s*\[\s*(list\s*\[)?\s*(float|int)\s*\]')

# 赋值语句左侧的简单标识符
ASSIGN_PAT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?!=)(.*)$')

# 纯数字的列表字面量：一维是向量，嵌套是矩阵。要求至少含一个数字，
# 否则 `acc = []`（后续 append 矩阵的空容器）会被误判成向量。
NUM_LIST_PAT = re.compile(r'^\[[\s\d.,+\-eE\[\]]*\d[\s\d.,+\-eE\[\]]*\]$')
NESTED_LIST_PAT = re.compile(r'^\[\s*\[')

# 右侧构造出的是矩阵（二维及以上）还是向量（一维）
# Anchored at the start of the right-hand side on purpose: the assignment must
# BE the array, not merely mention one. `layer = new_fc_layer(np.random.randn(
# 100, 300), ...)` binds a closure, and `kernels = [(np.array([[..]]), b), ..]`
# binds a list of tuples -- neither is a matrix.
MATRIX_RHS = [
    re.compile(r'^np\.(array|asarray)\s*\(\s*\[\s*\['),          # np.array([[..
    re.compile(r'^np\.[\w.]+\s*\(\s*\(\s*[\w.]+\s*,'),           # np.zeros((n, m))
    re.compile(r'^np\.[\w.]+\s*\(\s*[\w.]+\s*,\s*[\w.]+\s*\)'),  # np.random.rand(n, m)
    re.compile(r'^[\w.]+\.reshape\s*\(\s*[^)]*,'),               # a.reshape(n, m)
]
VECTOR_RHS = [
    re.compile(r'^np\.(array|asarray)\s*\(\s*\[(?!\s*\[)'),      # np.array([1, 2])
    re.compile(r'^np\.[\w.]+\s*\(\s*[\w.]+\s*\)'),               # np.zeros(n)
]


def _rhs_kind(rhs):
    """'matrix' / 'vector' / None -- what this right-hand side constructs.

    Recognises both numpy constructors and bare numeric list literals (the
    latter matter because representing an array as a list is itself a rule-2
    violation). Returns None whenever the shape is not mechanically obvious;
    the rule only reports what it can prove.
    """
    rhs = rhs.strip()
    for pat in MATRIX_RHS:
        if pat.match(rhs):
            return 'matrix'
    for pat in VECTOR_RHS:
        if pat.match(rhs):
            return 'vector'
    if NUM_LIST_PAT.match(rhs):
        return 'matrix' if NESTED_LIST_PAT.match(rhs) else 'vector'
    return None


def _numpy_issues(body, offset, tag, exempt):
    """Rule 2.1 -- vectors/matrices go in numpy arrays, not Python lists.

    `exempt` suppresses this check for hand-written demos that avoid numpy on
    purpose (see the toy-exempt marker). Only representing an array as a list
    is reported; scalar-only code needs no numpy at all and is never flagged.
    """
    if exempt:
        return []
    issues = []
    for k, line in enumerate(body):
        m = LIST_ANNOT_PAT.search(line)
        if m:
            issues.append((offset + k, f"{tag}-numpy",
                           f"用 `{m.group(0)}` 表示向量／矩阵："
                           f"玩具代码统一改用 numpy 的 array",
                           line.rstrip()))
    for rel, text in logical_lines(body):
        m = ASSIGN_PAT.match(text)
        if m and NUM_LIST_PAT.match(m.group(2).strip()):
            issues.append((offset + rel - 1, f"{tag}-numpy",
                           f"`{m.group(1)}` 用 Python 列表字面量表示向量／矩阵："
                           f"改用 numpy 的 array",
                           body[rel - 1].rstrip()))
    return issues


def _naming_issues(body, offset, tag):
    """Rule 2.2 / 2.3 -- vectors named vec*, matrices named mat*.

    Applies even to toy-exempt code: the marker excuses not using numpy, it
    does not excuse a name that hides the variable's shape.
    """
    issues = []
    for rel, text in logical_lines(body):
        m = ASSIGN_PAT.match(text)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        kind = _rhs_kind(rhs)
        if kind is None:
            continue
        want = 'vec' if kind == 'vector' else 'mat'
        bare = name.lstrip('_')
        if bare.startswith(want):
            continue
        other = 'mat' if want == 'vec' else 'vec'
        label = '向量' if kind == 'vector' else '矩阵'
        hint = (f"（现在是 `{other}` 开头，但右侧构造的是{label}）"
                if bare.startswith(other) else '')
        issues.append((offset + rel - 1, f"{tag}-命名",
                       f"`{name}` 是{label}，变量名应以 `{want}` 开头{hint}",
                       body[rel - 1].rstrip()))
    return issues


def rule_toy_numpy(lines, path):
    """玩具代码统一使用 numpy 的 array，不用 Python 原生列表。"""
    exempt = has_exempt_marker(''.join(lines))
    return _numpy_issues(lines, 1, "玩具代码", exempt)


def rule_toy_naming(lines, path):
    """向量变量名统一 vec 开头，矩阵变量名统一 mat 开头。"""
    return _naming_issues(lines, 1, "玩具代码")


TOY_RULES = [
    rule_toy_numpy,
    rule_toy_naming,
]


def rule_md_toycode(lines, masked, path):
    """章节里的玩具代码块同样遵守规则2（PyTorch 示例块跳过）。"""
    issues = []
    for offset, body in iter_code_blocks(lines, 'python'):
        if is_runnable_example(body):
            continue
        exempt = block_is_exempt(lines, offset) or has_exempt_marker(''.join(body))
        issues.extend(_numpy_issues(body, offset, "正文代码", exempt))
        issues.extend(_naming_issues(body, offset, "正文代码"))
    return issues


RULES.append(rule_md_toycode)


# --------------------------------------------------------------------------


def check(path):
    lines = read_lines(path)
    issues = []
    if is_toy(path):
        for rule in TOY_RULES:
            issues.extend(rule(lines, path))
    else:
        masked = mask_lines(lines)
        for rule in RULES:
            issues.extend(rule(lines, masked, path))
    issues.sort(key=lambda x: x[0])
    return issues


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    include_generated = '--include-generated' in sys.argv
    if not args:
        print(f"Usage: {sys.argv[0]} [--all-rules] <file.md | dir> [more...]")
        sys.exit(1)

    if not RULES and not TOY_RULES:
        print("（尚未定义任何规则，见 SKILL.md 的 Rules 小节）")

    all_rules = '--all-rules' in sys.argv
    total = skipped = 0
    for path in iter_targets(args, include_generated):
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
