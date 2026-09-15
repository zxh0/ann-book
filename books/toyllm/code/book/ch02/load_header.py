"""第二章 · 把权重文件的三段拆开

model.safetensors 的结构：8 字节的头部长度 N，N 字节的 JSON 头，剩下全是裸数据。
这个脚本就干三件事：解出 N，打印 JSON 头，报出各段的字节数。

运行：
    cd code
    uv run python book/ch02/load_header.py
"""

from pathlib import Path

# code/book/ch02/load_header.py → 上三层就是 code/
PATH = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M" / "model.safetensors"

with open(PATH, "rb") as f:
    # 一、前 8 个字节是头部长度 N，小端序的无符号 64 位整数
    n = int.from_bytes(f.read(8), "little")
    print(f"N = {n}")

    # 二、接下来 N 个字节是 UTF-8 的 JSON
    print(f.read(n).decode())

# 三、剩下的都是裸数据
size = PATH.stat().st_size
print(f"头部 8 + {n} = {8 + n} 字节")
print(f"数据 {size - 8 - n} 字节")
print(f"总计 {size} 字节")
