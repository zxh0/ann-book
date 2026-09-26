"""第四章 · 词嵌入 demo

给一段文字，看它被切成哪些词元、查到哪些ID，以及每个ID取出来的那个向量。

形状上就是 (T,) → (T, 576)：进去T个整数，出来T行，每行576个浮点数。
向量太长，一行打不下，所以每行只显示头3个和尾2个数。

运行：
    cd code
    uv run python book/ch04/embed_demo.py
    uv run python book/ch04/embed_demo.py "你想试的任意一段话"
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

# code/book/ch04/embed_demo.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

TEXTS = [
    "Once upon a time",
    "Hello world",
]

HEAD, TAIL = 3, 2  # 每个向量显示头几个、尾几个数


def load_embed_tokens():
    """只取`model.embed_tokens.weight`这一个张量，顺便升成fp32。

    文件里存的是bf16，而CPU上fp32的算子更快也更全，所以加载时统一升一次。
    272个张量里我们这一章只用得到这一个，safetensors是按需读的，
    所以其余269MB根本不会进内存。
    """
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight").to(torch.float32)


def fmt(vec):
    """把一个576维向量显示成`[头3个, ..., 尾2个]`。"""
    head = ", ".join(f"{v:7.4f}" for v in vec[:HEAD].tolist())
    tail = ", ".join(f"{v:7.4f}" for v in vec[-TAIL:].tolist())
    return f"[{head}, ... , {tail}]"


def show(tok, embed, text):
    enc = tok.encode(text, add_special_tokens=False)
    ids = torch.tensor(enc.ids)
    x = embed[ids]  # 查表：按行号取行，(T,) → (T, 576)

    print(f"\n原文    {text!r}")
    print(f"词元    {enc.tokens}")
    print(f"ID      {enc.ids}")
    print(f"形状    {tuple(ids.shape)} → {tuple(x.shape)}")
    print()
    for i, (tid, token) in enumerate(zip(enc.ids, enc.tokens)):
        print(f"  第{i}行  ID {tid:>5}  {token!r:<11} {fmt(x[i])}")


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    embed = load_embed_tokens()

    print(f"词表大小     {tok.get_vocab_size()}")
    print(f"词嵌入矩阵   {tuple(embed.shape)}  {embed.dtype}  共{embed.numel():,}个参数")

    for text in sys.argv[1:] or TEXTS:
        show(tok, embed, text)


if __name__ == "__main__":
    main()
