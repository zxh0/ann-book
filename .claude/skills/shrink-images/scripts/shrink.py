#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=10"]
# ///
"""Shrink the manuscript's PNG illustrations.

Usage:
    uv run --no-project shrink.py <file.png | dir> [more...] [--dry-run]

两步处理，第二步按实测误差决定做不做：

1. 无损：有 alpha 就贴到白底上（draw.io 导出的背景是透明的，站点和书都是白底），
   丢掉 iCCP / eXIf / iTXt / pHYs 这些元数据块，重新以 optimize=True 编码。
   像素不变，只是把用不上的东西去掉。
2. 调色板：转成 8 位调色板（自适应 256 色）。这一步是有损的，所以先量化一遍，
   量出平均误差，误差够小才采纳（阈值见下面 AUTO / ASK 两个常量的注释）。

线稿图（draw.io 导出、matplotlib 画的图表）省 70%~90%，照片和 AI 生成图只做第一步。
结果比原图大就不写，原样留着。
"""

import argparse
import os
import sys

from PIL import Image, ImageChops

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lintscope import excluded

WHITE = (255, 255, 255)

# 量化误差阈值，来自仓库里这几类图的实测值（见 SKILL.md 的表）：
# draw.io 插图 0.000~0.034，手画的图表 0.035~0.055，matplotlib 截图 0.237，
# 带渐变的示意图 0.412，AI 章首图 3.519。
AUTO = 0.1   # 误差在这以下：线稿图，直接量化
ASK = 0.5    # 这之间：拿不准，报出来但不动，要量化得显式加 --palette

# 省得太少就不重写，免得给 git 白添一个二进制大对象。
MIN_RATIO = 0.10
MIN_BYTES = 1024


def flatten(im):
    """有 alpha 的贴到白底上；顺带把模式统一成 RGB，元数据自然就掉了。"""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, WHITE)
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def mean_error(a, b):
    """两张图逐分量的平均绝对误差，0 表示逐像素相同。"""
    hist = ImageChops.difference(a, b).histogram()
    total = sum(hist[:768])
    if not total:
        return 0.0
    return sum(i * (hist[i] + hist[256 + i] + hist[512 + i]) for i in range(256)) / total


def encode(im):
    """编码成 PNG 字节串。Pillow 不会带上原图的元数据块，正是我们要的。"""
    import io

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def shrink(path, palette_mode):
    """返回 (新字节串或 None, 说明)。None 表示不值得动。"""
    im = Image.open(path)
    if im.format != "PNG":
        return None, f"不是 PNG（{im.format}）"

    flat = flatten(im)
    lossless = encode(flat)

    quantized = flat.convert("P", palette=Image.ADAPTIVE, colors=256)
    err = mean_error(flat, quantized.convert("RGB"))

    if palette_mode == "never":
        use_palette = False
    elif palette_mode == "always":
        use_palette = True
    else:
        use_palette = err <= AUTO

    if use_palette:
        data, how = encode(quantized), f"调色板256色，平均误差 {err:.3f}"
    elif AUTO < err <= ASK:
        data, how = lossless, f"无损（量化误差 {err:.3f} 偏大，要压得加 --palette）"
    else:
        data, how = lossless, f"无损，像素不变（量化误差会有 {err:.3f}，不划算）"

    # 压不出多少就不动它。重写一张已经压过的图，省下几百字节，却给 git
    # 添一个新的二进制大对象，不划算。
    size = os.path.getsize(path)
    saved = size - len(data)
    if saved <= 0:
        return None, f"已经压过了，重压反而大 {-saved:,} 字节，不动"
    if saved < max(size * MIN_RATIO, MIN_BYTES):
        return None, f"已经压过了，再压只省 {saved:,} 字节（{100 * saved / size:.1f}%），不动"
    return data, how


def iter_targets(paths):
    """展开成 .png 列表：目录递归，跳过点目录和 lint-scope.json 里排除的路径。"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in sorted(files):
                    if name.lower().endswith(".png"):
                        out.append(os.path.join(root, name))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p in seen or excluded(p):
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser(description="压缩书稿插图（PNG）")
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--palette", dest="palette_mode", action="store_const", const="always",
                       help="不管误差多大都转调色板（照片和 AI 图会有色带，先看过再用）")
    group.add_argument("--no-palette", dest="palette_mode", action="store_const", const="never",
                       help="只做无损那一步")
    ap.set_defaults(palette_mode="auto")
    args = ap.parse_args()

    targets = iter_targets(args.paths)
    if not targets:
        print("没有找到 .png 文件")
        return 0

    before = after = 0
    changed = skipped = 0
    for path in targets:
        size = os.path.getsize(path)
        try:
            data, how = shrink(path, args.palette_mode)
        except OSError as exc:
            print(f"{path}：读不了（{exc}）")
            continue

        if data is None:
            skipped += 1
            print(f"{path}\n  {size:>9,}  {how}")
            continue

        changed += 1
        before += size
        after += len(data)
        saved = 100 * (size - len(data)) / size
        print(f"{path}\n  {size:>9,} -> {len(data):>9,}  省 {saved:4.1f}%  {how}")
        if not args.dry_run:
            with open(path, "wb") as f:
                f.write(data)

    tail = "（--dry-run，没有写文件）" if args.dry_run else ""
    if changed:
        print(f"\n{changed} 张图 {before:,} -> {after:,} 字节，"
              f"省 {100 * (before - after) / before:.1f}%{tail}")
    if skipped:
        print(f"{skipped} 张没动")
    return 0


if __name__ == "__main__":
    sys.exit(main())
