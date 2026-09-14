#!/usr/bin/env python3
"""Per-area rule scoping, shared by the repo's lint skills.

The repo holds two books and a folder of notes. Most conventions (formula
formatting, em-dashes, math-vs-code) are genuinely repo-wide, but a few are
not: the notes deliberately follow each paper's own notation, so the 符号记法
rule must not fire there.

Rather than teaching every skill where the books are, the areas and their
exemptions live in one file at the repo root:

    .claude/lint-scope.json

    {
      "scopes": [
        {"path": "notes/**", "disable": ["符号记法"]}
      ]
    }

Every pattern that matches a file contributes its `disable` list (they union),
so `{"path": "**", "disable": [...]}` turns a rule off everywhere and a
narrower pattern adds exemptions on top. No config, or no match, means every
rule applies -- skills degrade to their old behaviour rather than silently
checking nothing.

Patterns are matched against the file's path relative to the repo root:
`area/**` matches everything under `area/`, anything else is fnmatch.
"""

import fnmatch
import json
import os

CONFIG_RELPATH = os.path.join(".claude", "lint-scope.json")

_cache = {}


def find_root(start):
    """Walk up from `start` looking for the directory holding the config."""
    cur = os.path.abspath(start if os.path.isdir(start) else os.path.dirname(start) or ".")
    while True:
        if os.path.exists(os.path.join(cur, CONFIG_RELPATH)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _load(root):
    if root in _cache:
        return _cache[root]
    scopes, excludes = [], []
    try:
        with open(os.path.join(root, CONFIG_RELPATH), encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("scopes", []):
            pat = entry.get("path")
            if pat:
                scopes.append((pat, set(entry.get("disable", []))))
        excludes = list(data.get("exclude", []))
    except (OSError, ValueError) as exc:      # unreadable or malformed
        print(f"警告：{CONFIG_RELPATH} 读取失败（{exc}），本次全部规则生效")
    _cache[root] = (scopes, excludes)
    return _cache[root]


def _matches(rel, pattern):
    if pattern in ("**", "*"):
        return True
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return rel == prefix or rel.startswith(prefix + "/")
    return fnmatch.fnmatch(rel, pattern)


def disabled_rules(path):
    """Rule names switched off for `path`. Empty set when nothing applies."""
    root = find_root(path)
    if root is None:
        return set()
    scopes, _ = _load(root)
    if not scopes:
        return set()
    rel = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
    off = set()
    for pattern, names in scopes:
        if _matches(rel, pattern):
            off |= names
    return off


def excluded(path):
    """True when the file is not a lint target at all (agent docs, etc.)."""
    root = find_root(path)
    if root is None:
        return False
    _, patterns = _load(root)
    if not patterns:
        return False
    rel = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
    return any(_matches(rel, pat) for pat in patterns)


def keep(issues, path, rule_index=1, all_rules=False):
    """Drop issues whose rule is disabled for `path`.

    `issues` are the (lineno, rule, problem, text) tuples both skills emit;
    `rule_index` says which field holds the rule name.
    """
    if all_rules:
        return issues
    off = disabled_rules(path)
    if not off:
        return issues
    return [i for i in issues if i[rule_index] not in off]
