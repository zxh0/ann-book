---
name: check-formulas
description: This skill should be used when the user asks to "check formulas", "verify math", "check equations", "检查公式", or mentions formula errors in a markdown file. Covers the whole repo: both books and the notes.
version: 1.0.0
---

# Check Formulas in Markdown

## When This Skill Applies

- User asks to check or verify math formulas in a `.md` file
- User reports a formula looks wrong or broken
- User wants to audit all formulas in a chapter, a book, or the whole repo

## Scope

**整个仓库通用** —— 两本书和 `notes/` 面向的都是 GitHub 的 Markdown 渲染，公式格式要求完全一致。
与 `check-prose` 共用仓库根的 `.claude/lint-scope.json` 做区域豁免和排除（格式见那个技能的
SKILL.md）；目前公式规则在所有区域都全开，只排除了 `CLAUDE.md` 这类非书稿文件。

## Steps

检查和修复均有对应脚本，位于 `scripts/` 目录，可传入文件、目录或多个路径：

```bash
# 全仓库
python3 scripts/check.py .

# 单个文件 / 某一本书 / 笔记
python3 scripts/check.py books/ann4us/chapters/03_ch01_neuron.md
python3 scripts/check.py books/toyllm/ notes/

# 自动修复（覆盖原文件）。fix.py 是整体一趟改写，无法只改部分规则，
# 所以务必先用 --dry-run 看 diff，尤其是对作者手写的 notes/
python3 scripts/fix.py --dry-run notes/2026-07-05-PE.md
python3 scripts/fix.py books/ann4us/chapters/
```

目录会递归展开 `*.md`，自动跳过点目录和生成的 `Book.md`。
若某文件在 `lint-scope.json` 里关掉了任一公式规则，`fix.py` 会整体跳过它并提示手工处理。

如果需要人工处理，按以下步骤：

1. Read the target markdown file
2. Extract all formula blocks (`$...$` inline, `$$...$$` display)
3. Check each formula against the rules below

## Rules

### 行内公式空格（inline formula spacing）

行内公式 `$...$` 的左侧 `$` 前、右侧 `$` 后，必须各有一个空格（或位于行首/行尾），否则 GitHub 网页端无法正确渲染。

**违规示例：**
```
神经元的输出为$y = wx + b$，其中
```

**正确示例：**
```
神经元的输出为 $y = wx + b$ ，其中
```

检查时使用以下 grep 模式定位违规行（左侧缺空格 或 右侧缺空格）：

```bash
# 左侧缺空格：$ 前紧跟非空白、非行首的字符（排除 $$）
grep -n '[^ \t\n][$][^$]' <file>

# 右侧缺空格：$ 后紧跟非空白、非行尾的字符（排除 $$）
grep -n '[^$][$][^ \t\n$]' <file>
```

对每处违规，报告行号、原文，并给出修正后的写法。

### 块级公式空行（block formula blank lines）

块级公式 `$$...$$` 的上方和下方，必须各有一个空行，否则 GitHub 网页端无法正确渲染。

**违规示例：**
```
权重更新公式如下：
$$
w \leftarrow w - \eta \nabla L
$$
其中 $\eta$ 为学习率。
```

**正确示例：**
```
权重更新公式如下：

$$
w \leftarrow w - \eta \nabla L
$$

其中 $\eta$ 为学习率。
```

检查方法：逐行读取文件，找到 `$$` 开始行（块级公式起始），检查其前一行是否为空行；找到 `$$` 结束行，检查其后一行是否为空行。

用 grep 定位所有 `$$` 行的行号，再结合上下文逐一核查：

```bash
grep -n '^\$\$$' <file>
```

对每处违规，报告行号、原文片段，并说明是"上方缺空行"还是"下方缺空行"。

### 公式对齐环境（alignment environment）

多行对齐公式必须使用 `\begin{aligned}` / `\end{aligned}`，禁止使用其他对齐环境（如 `align`、`align*`、`eqnarray`、`split` 等）。

**违规示例：**
```
$$
\begin{align}
y &= wx + b \\
z &= \sigma(y)
\end{align}
$$
```

**正确示例：**
```
$$
\begin{aligned}
y &= wx + b \\
z &= \sigma(y)
\end{aligned}
$$
```

用 grep 定位所有违规的对齐环境：

```bash
grep -n '\\begin{\(align\*\?\|eqnarray\*\?\|split\)}' <file>
```

对每处违规，报告行号、原文，并将环境名替换为 `aligned`。

### 块级公式 tag 位置（block formula tag placement）

块级公式的 `\tag{...}` 必须写在 `\end{aligned}` 的后面（同一行或紧接的下一行），不能写在 `\begin{aligned}` / `\end{aligned}` 内部，否则 GitHub 网页端无法正确渲染。若公式内部有多个 `\tag`，需将它们合并，统一写在 `\end{aligned}` 之后。

**违规示例（tag 在内部）：**
```
$$
\begin{aligned}
y &= wx + b \tag{1} \\
z &= \sigma(y) \tag{2}
\end{aligned}
$$
```

**正确示例（tag 移至 `\end{aligned}` 后，多个合并）：**
```
$$
\begin{aligned}
y &= wx + b \\
z &= \sigma(y)
\end{aligned}
\tag{1, 2}
$$
```

检查方法：扫描每个块级公式（`$$...$$`），在其 `\begin{aligned}` 至 `\end{aligned}` 范围内查找 `\tag{...}`。有则违规。

此外，若块级公式使用了 `\tag` 但没有 `\begin{aligned}...\end{aligned}`，必须先将公式内容包入 `\begin{aligned}...\end{aligned}`，再把 `\tag` 写在 `\end{aligned}` 之后。

**违规示例（有 tag 但无对齐环境）：**
```
$$
y = wx + b \tag{1}
$$
```

**正确示例：**
```
$$
\begin{aligned}
y = wx + b
\end{aligned}
\tag{1}
$$
```

用 grep 定位所有在 aligned 内部出现的 tag，以及有 tag 但无对齐环境的块级公式：

```bash
grep -n '\\tag{' <file>
```

对每处违规，报告行号、原文，说明是以下哪种情况并给出修正写法：
1. `\tag` 在 `\begin{aligned}...\end{aligned}` 内部 → 移至 `\end{aligned}` 之后，多个合并
2. 有 `\tag` 但无对齐环境 → 用 `\begin{aligned}...\end{aligned}` 包裹公式内容，`\tag` 写在最后

### 禁用 \operatorname（unsupported operator）

GitHub 的公式渲染器不支持 `\operatorname`，必须改用 `\mathrm`。

**违规示例：**
```
\operatorname{Softmax}(x)
```

**正确示例：**
```
\mathrm{Softmax}(x)
```

用 grep 定位所有违规：

```bash
grep -n '\\operatorname' <file>
```

对每处违规，将 `\operatorname{...}` 替换为 `\mathrm{...}`。

### 公式内裸函数名（bare function names）

公式内出现的多字母函数名（如 `max`、`softmax`、`sigmoid` 等），必须用 `\mathrm{}` 包裹，使其以直立体显示，否则会被渲染为斜体变量。

**违规示例：**
```
max(x)
softmax(QK^\top)
```

**正确示例：**
```
\mathrm{max}(x)
\mathrm{softmax}(QK^\top)
```

检测方法：在公式内容中查找 2 个及以上字母直接跟着 `(` 且前面没有 `\` 的模式：

```python
re.compile(r'(?<!\\)([a-zA-Z]{2,})\(')
```

- 行内公式：仅对 `$...$` 内部的内容检测
- 块级公式：对 `$$...$$` 内每一行内容检测
- 已有 `\` 前缀的命令（`\max`、`\mathrm{...}` 等）不触发

对每处违规，将 `funcname(` 改为 `\mathrm{funcname}(`。

<!-- TODO: 后续可继续补充其他规则 -->

## Output Format

每条违规输出格式：

```
L{行号} [{规则名}] {问题描述}
  原文: {原始行内容（截取前100字符）}
```

所有规则均通过时输出：`✓ 未发现问题`
