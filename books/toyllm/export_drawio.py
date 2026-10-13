#!/usr/bin/env python3
"""把某一章的 draw/chNN.drawio 逐页导出成 images/chNN/<页名>.png。

一章一个 .drawio 文件，一页一张图，页名就是图片的文件名。导出靠 draw.io 桌面版
自带的命令行，只用标准库，系统自带的 python3 就能跑。

图上默认带着编辑器里那种网格。draw.io 的命令行导不出网格，所以分两步：
draw.io 先导出透明背景的 PNG，再让无头 Chrome 把它叠到画好网格的白底上截图。
网格的尺寸和颜色，是从 VS Code 插件里截的旧图上量出来的。

导完再调用仓库的 shrink-images 技能（.claude/skills/shrink-images/），把刚导出的图
压成8位调色板，一般能省一半多。压缩要用 uv，第一次会下载 pillow。

用法（在 books/toyllm 下）：
    python3 export_drawio.py ch06             # 也可以写成 6
    python3 export_drawio.py ch06 view        # 只导出页名是 view 的那一页
    python3 export_drawio.py ch06 view mask   # 可以一次写好几页
    python3 export_drawio.py ch06 -n          # 只列出会导出什么，不真导出
    python3 export_drawio.py ch06 --no-grid   # 纯白底，不要网格
    python3 export_drawio.py ch06 --no-shrink # 导完不压缩

注意 images/ 不在 git 里，导出会直接覆盖同名的图，覆盖了就找不回来。
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DRAWIO = "/Applications/draw.io.app/Contents/MacOS/draw.io"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ROOT = Path(__file__).resolve().parent  # books/toyllm
REPO = ROOT.parent.parent               # ann-book，仓库根
SHRINK = REPO / ".claude" / "skills" / "shrink-images" / "scripts" / "shrink.py"

SCALE = 2    # 正文用 zoom:50%，两倍图在高分屏上正好清楚
BORDER = 10  # 四周留白，单位是 draw.io 里的单位，导出后乘以 SCALE

# 网格，单位是导出后的像素。draw.io 的格子是10，乘以 SCALE 就是20；每4格一条粗线。
# 细线2像素宽、#f6f6f6，粗线1像素宽、#d0d0d0，和旧截图逐像素对得上。
# BORDER 是10的倍数，图形又都吸附在格子上，所以网格和图形的边是对齐的。
GRID = 10 * SCALE
GRID_STEPS = 4
GRID_HTML = """<!doctype html><html><head><style>
html, body {{ margin: 0; padding: 0; background: #fff; }}
body {{
  width: {w}px; height: {h}px;
  background-image:
    linear-gradient(to right,  #d0d0d0 1px, transparent 1px),
    linear-gradient(to bottom, #d0d0d0 1px, transparent 1px),
    linear-gradient(to right,  #f6f6f6 2px, transparent 2px),
    linear-gradient(to bottom, #f6f6f6 2px, transparent 2px);
  background-size: {major}px {major}px, {major}px {major}px, {minor}px {minor}px, {minor}px {minor}px;
}}
img {{ display: block; }}
</style></head><body><img src="{img}"></body></html>
"""


def chapter_name(arg):
    """`6`、`06`、`ch6`、`ch06` 都认，统一成 `ch06`。"""
    num = arg.lower().removeprefix("ch")
    if not num.isdigit():
        sys.exit(f"看不懂章节号：{arg}，应该写成 ch06 或者 6")
    return f"ch{int(num):02d}"


def page_names(drawio):
    """按顺序返回每一页的页名。draw.io 命令行的页码从1开始，比这个列表的下标大1。"""
    return [d.get("name", "") for d in ET.parse(drawio).getroot().iter("diagram")]


def is_default_name(name):
    """draw.io 新建页面时的默认页名，Page-1、Page-2……"""
    return name == "" or (name.startswith("Page-") and name[5:].isdigit())


def export_png(drawio, index, out, transparent=False):
    """用 draw.io 的命令行导出一页。页码从1开始。"""
    cmd = [DRAWIO, "-x", "-f", "png", "-p", str(index), "-s", str(SCALE),
           "-b", str(BORDER), "-o", str(out), str(drawio)]
    if transparent:
        cmd.insert(2, "-t")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        sys.exit(f"导出失败：{drawio.name} 第{index}页\n{result.stdout}{result.stderr}")


def png_size(png):
    """用 macOS 自带的 sips 读出图片的宽和高。"""
    out = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(png)],
                         capture_output=True, text=True, check=True).stdout
    values = [int(line.split()[-1]) for line in out.splitlines() if "pixel" in line]
    return values[0], values[1]


def add_grid(transparent_png, out):
    """把透明背景的图叠到网格上：写一个网页，让无头 Chrome 按原图大小截图。"""
    w, h = png_size(transparent_png)
    html = transparent_png.with_suffix(".html")
    html.write_text(GRID_HTML.format(w=w, h=h, img=transparent_png.name,
                                     minor=GRID, major=GRID * GRID_STEPS))
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-device-scale-factor=1", f"--window-size={w},{h}",
           f"--screenshot={out}", html.as_uri()]
    if out.exists():
        out.unlink()  # 免得 Chrome 失败时，留着旧图被当成成功
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists() or png_size(out) != (w, h):
        sys.exit(f"加网格失败：{out.name}\n{result.stderr[-2000:]}")


def shrink(pngs):
    """调用 shrink-images 技能，把刚导出的图压一遍。

    shrink.py 用 PEP 723 声明依赖，必须用 uv 跑，系统 python3 没有 pillow。
    --native-tls 让 uv 用系统证书下载 pillow，否则在某些网络环境下会报证书错误。
    """
    uv = shutil.which("uv")
    if uv is None:
        print("  没找到 uv，跳过压缩。可以装上 uv 以后重新导出，或者加 --no-shrink 不压缩")
        return
    print("\n压缩：", flush=True)
    cmd = [uv, "run", "--native-tls", "--no-project", str(SHRINK)] + [str(p.relative_to(REPO)) for p in pngs]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        print(f"  {line}" if line else "")
    if result.returncode != 0:
        sys.exit(f"压缩失败\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chapter", help="要导出哪一章，比如 ch06 或者 6")
    parser.add_argument("pages", nargs="*", help="只导出这几页（按页名），不写就是整章")
    parser.add_argument("-n", "--dry-run", action="store_true", help="只列出，不导出")
    parser.add_argument("--no-grid", action="store_true", help="纯白底，不要网格")
    parser.add_argument("--no-shrink", action="store_true", help="导完不压缩")
    args = parser.parse_args()

    chapter = chapter_name(args.chapter)
    drawio = ROOT / "draw" / f"{chapter}.drawio"
    out_dir = ROOT / "images" / chapter
    if not Path(DRAWIO).exists():
        sys.exit(f"找不到 draw.io 桌面版：{DRAWIO}")
    if not args.no_grid and not Path(CHROME).exists():
        sys.exit(f"找不到 Google Chrome：{CHROME}，加网格要用它。不要网格的话加 --no-grid")
    if not drawio.exists():
        sys.exit(f"找不到 {drawio.relative_to(ROOT)}")
    if drawio.stat().st_size == 0:
        sys.exit(f"{drawio.relative_to(ROOT)} 是空文件")

    names = page_names(drawio)
    dupes = sorted({n for n in names if names.count(n) > 1 and not is_default_name(n)})
    if dupes:
        sys.exit(f"{drawio.name} 里有重名的页：{', '.join(dupes)}")

    unknown = [p for p in args.pages if p not in names]
    if unknown:
        valid = ", ".join(n for n in names if not is_default_name(n))
        sys.exit(f"{drawio.name} 里没有这些页：{', '.join(unknown)}\n有的页是：{valid}")

    print(f"{drawio.relative_to(ROOT)}  {len(names)}页")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="export_drawio_"))
    exported = []
    for index, name in enumerate(names, 1):
        if args.pages and name not in args.pages:
            continue
        if is_default_name(name):
            print(f"  跳过  第{index:>2}页 {name!r}（默认页名）")
            continue
        out = out_dir / f"{name}.png"
        print(f"  {'覆盖' if out.exists() else '新增'}  第{index:>2}页 → {out.relative_to(ROOT)}", flush=True)
        if args.dry_run:
            continue
        if args.no_grid:
            export_png(drawio, index, out)
        else:
            transparent = tmp / f"{name}.png"
            export_png(drawio, index, transparent, transparent=True)
            add_grid(transparent, out)
        exported.append(out)

    if exported and not args.no_shrink:
        shrink(exported)

    # 有图没页：多半是页名和文件名对不上，或者这张图画在别处。只导几页时不必查。
    if args.pages:
        return
    orphans = sorted(p.name for p in out_dir.glob("*.png") if p.stem not in names)
    if orphans:
        print(f"  {out_dir.relative_to(ROOT)}里没有对应页的图：{', '.join(orphans)}")


if __name__ == "__main__":
    main()
