---
name: shrink-images
description: This skill should be used when the user asks to "压缩图片", "压一下图", "图太大了", "shrink images", "compress images", or right after new illustrations are exported into a chapter's images/ directory. Covers the whole repo: both books and the notes.
version: 1.0.0
---

# Shrink Manuscript Images

把书稿插图压小，不改尺寸、不改内容。draw.io 直接导出的图通常能省 70%~90%，
AI 生成的章首图省 25% 左右且像素不变。

## When This Skill Applies

- 一章的插图刚从 draw.io 导出、放进 `images/chNN/`，提交之前
- 新的 AIGC 章首图刚放进 `aigc/`
- 作者说图太大、站点加载慢，或想知道还有哪些图没压过
- 某一章要上站之前，顺手扫一遍

## Scope

**整个仓库通用** —— 两本书的 `images/`、`aigc/` 和 `notes/images/` 都适用。只认 PNG，
仓库里也只有 PNG。

与两个 lint 技能共用仓库根的 `.claude/lint-scope.json`，其中 `exclude` 里的
`docs/**` 和 `node_modules/**` 会被跳过，点目录（`.venv` 等）也跳过。
所以直接把仓库根传进去是安全的。

**改的是仓库里的源图**（`books/*/images/`、`books/*/aigc/`、`notes/images/`），
不是 `docs/public/` 下的拷贝。压完记得重新生成站点内容：

```bash
npm run prepare:site      # 或者 npm run build，它会先跑这个
```

## Steps

```bash
# 压一章的插图（默认就地覆盖）
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py books/toyllm/images/ch04/

# 先看看能省多少，不写文件；对作者手写的 notes/ 建议先这么跑
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py --dry-run notes/images/

# 扫全仓库，看还有哪些图没压过
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py --dry-run .

# 多个路径一起传，文件和目录混着都行
uv run --no-project .claude/skills/shrink-images/scripts/shrink.py books/toyllm/images/ch04/ books/toyllm/aigc/ch04.png
```

脚本用 PEP 723 内联依赖声明（文件头的 `# /// script`），`uv run` 会自建临时环境装
pillow，既不碰 `books/toyllm/code/` 那个 uv 项目，也不往系统 python 里装东西。
系统 `python3` 是 3.9 且没有 pillow，所以**不能**直接 `python3 shrink.py`。

第一次跑要下载 pillow，和仓库里其他下载一样，得绕开 Claude Code 的代理：

```bash
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy \
  uv run --no-project .claude/skills/shrink-images/scripts/shrink.py books/toyllm/images/ch04/
```

压完**打开看一眼**（Read 工具能直接看 PNG），确认文字边缘、虚线箭头、浅色块没出问题，
再提交。

## 怎么压的

两步，第二步按实测误差决定做不做。

**第一步，无损。** 有 alpha 通道的贴到白底上（draw.io 导出的背景是透明的，而站点和
PDF 都是白底），丢掉 iCCP / eXIf / iTXt / pHYs 这些用不上的元数据块，再以
`optimize=True` 重新编码。像素一个都不变。

**第二步，转 8 位调色板。** 自适应 256 色。这一步是有损的，所以脚本先量化一遍，量出
和原图的平均绝对误差，误差够小才采纳：

| 平均误差 | 怎么办 |
|---|---|
| ≤ 0.1 | 线稿图，直接量化 |
| 0.1 ~ 0.5 | 拿不准，只做第一步，报出误差；确实要压，显式加 `--palette` |
| > 0.5 | 照片或渐变图，只做第一步 |

阈值是照仓库里这几类图实测出来的，量化到 256 色的平均误差分别是：

| 图 | 平均误差 | 前 256 色覆盖的像素 |
|---|---|---|
| draw.io 导出的插图（`images/chNN/`） | 0.000 ~ 0.034 | 99% ~ 100% |
| 手画的图表（`ann4us/images/`、`notes/images/`） | 0.035 ~ 0.055 | 约 99% |
| matplotlib 截图（`perceptron_demo.png`） | 0.237 | 94.8% |
| 带渐变的示意图（`notes/images/llm0/llm.png`） | 0.412 | 91.5% |
| 真彩可视化截图（`nn_vis_cnn_2d.png`） | 1.587 | 66.1% |
| AI 生成的章首图（`aigc/chNN.png`） | 3.519 | 11.8% |

线稿图和照片隔着两个数量级，所以 0.1 这个线画在哪都行，不必纠结。

**省得太少就不写。** 至少省 10% 且至少省 1 KB 才覆盖原文件。重写一张已经压过的图，
省几百字节，却给 git 添一个新的二进制大对象，不划算。

## 选项

| 选项 | 作用 |
|---|---|
| `--dry-run` | 只报告，不写文件 |
| `--palette` | 不管误差多大都转调色板。照片和 AI 图会有色带，**用之前先压到临时目录看一眼** |
| `--no-palette` | 只做第一步，谁都不量化 |

## Output Format

一张图两行，压了的给出前后字节数、省了多少、用的哪种办法；没压的说明为什么：

```
books/toyllm/images/ch04/proj.png
     18,775 ->     2,214  省 88.2%  调色板256色，平均误差 0.000
books/toyllm/aigc/ch04.png
    683,909 ->   499,053  省 27.0%  无损，像素不变（量化误差会有 3.519，不划算）
books/toyllm/images/ch04/progress.png
     44,501  已经压过了，重压反而大 62 字节，不动

7 张图 1,069,378 -> 584,530 字节，省 45.3%
1 张没动
```

## 注意事项

- **贴白底是不可逆的。** 仓库里 ch01~ch03 的插图本来就是这样（`type3 + PLTE`，无
  tRNS），站点和 PDF 也都是白底。站点以后若要做深色模式，这条约定得重新考虑。
- **量化是有损的。** 对线稿图肉眼无差别，对渐变会出色带。脚本的默认阈值已经挡住了
  这类图，别随手加 `--palette` 绕过去。
- **不改尺寸、不裁剪、不转 WebP。** 书稿的图还要进 PDF，改这些得作者点头。
- `notes/` 下的图是作者手写文章的配图，和正文一样**先 `--dry-run`**。
- 压完要跑 `npm run prepare:site`，`docs/public/` 下的拷贝才会更新。
