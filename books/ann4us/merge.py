#!/usr/bin/env python3
"""把 chapters/ 下的分章 Markdown 合并成完整的 Book.md。

用法：
    cd book && python3 merge.py            # 生成 Book.md
    python3 merge.py -o /tmp/out.md        # 输出到指定文件
    python3 merge.py --check               # 只检查，不写入（有差异时退出码为 1）

约定：
- chapters/ 下的文件按文件名排序（NN_slug.md）决定章节顺序；
- 每章之间自动插入分页符 <div style="page-break-after: always;"></div>；
- 分章文件里图片路径写作 ../images/xxx，合并后统一改回 ./images/xxx；
- 书名下方自动插入“自动生成”提示，分章文件里不要写这行；
- 版本号（版本：vYYYY.MM.DD）每次合并时自动更新为当天日期。
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CHAPTERS = BASE / "chapters"
DEFAULT_OUT = BASE / "Book.md"
PAGE_BREAK = '<div style="page-break-after: always;"></div>'
GENERATED_NOTICE = "<!-- 自动生成，请不要编辑！ -->"
VERSION_RE = re.compile(r"版本：v\d{4}\.\d{2}\.\d{2}")


def build() -> str:
    files = sorted(CHAPTERS.glob("*.md"))
    if not files:
        sys.exit(f"没有找到分章文件：{CHAPTERS}")

    parts = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        # 分章文件自身不应再包含分页符和“自动生成”提示，去掉以免重复
        text = text.replace(PAGE_BREAK, "").replace(GENERATED_NOTICE, "").strip()
        # 图片路径：../images/ -> ./images/
        text = text.replace("](../images/", "](./images/").replace(
            'src="../images/', 'src="./images/'
        )
        parts.append(text)

    merged = f"\n\n{PAGE_BREAK}\n\n".join(parts) + "\n"

    # 版本号更新为当天日期
    today = datetime.date.today().strftime("%Y.%m.%d")
    merged, count = VERSION_RE.subn(f"版本：v{today}", merged)
    if count == 0:
        print("警告：没有找到“版本：vYYYY.MM.DD”，跳过版本号更新", file=sys.stderr)

    # 在书名（一级标题）下方插入“自动生成”提示
    title, _, rest = merged.partition("\n")
    if title.startswith("# "):
        return f"{title}\n\n{GENERATED_NOTICE}\n\n{rest.lstrip()}"
    return f"{GENERATED_NOTICE}\n\n{merged}"


def main() -> None:
    ap = argparse.ArgumentParser(description="合并分章 Markdown 为 Book.md")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="输出文件")
    ap.add_argument(
        "--check", action="store_true", help="只检查内容是否有差异（忽略版本号），不写入"
    )
    args = ap.parse_args()

    merged = build()

    if args.check:
        old = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        # 版本号每天都变，比对时忽略，只看正文是否一致
        blank = "版本：v0000.00.00"
        if VERSION_RE.sub(blank, old) != VERSION_RE.sub(blank, merged):
            print(f"{args.output} 与 chapters/ 不一致，请重新运行 merge.py")
            sys.exit(1)
        print(f"{args.output} 已是最新")
        return

    args.output.write_text(merged, encoding="utf-8")
    version = VERSION_RE.search(merged)
    suffix = f"（{version.group()[3:]}）" if version else ""
    print(f"已合并 {len(sorted(CHAPTERS.glob('*.md')))} 个文件 -> {args.output}{suffix}")


if __name__ == "__main__":
    main()
