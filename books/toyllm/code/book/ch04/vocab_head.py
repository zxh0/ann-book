"""第四章 · 词表开头的十个词元

`embed_demo.py` 是给一段文字，看它查出哪些行。这里换个方向：不给文字，直接从
词表的第0行开始，一行一行往下看。

词嵌入矩阵有49152行，和词表一一对应，第i行就是ID为i的那个词元的向量。所以「打开
词表看前十个」和「打开矩阵看前十行」是同一件事。

开头这几个ID有点特别，它们不是从语料里统计出来的词，而是手工加进去的特殊词元。

输出直接是一张Markdown表格，可以原样贴进书稿。

运行：
    cd code
    uv run python book/ch04/vocab_head.py
    uv run python book/ch04/vocab_head.py 20       # 想多看几行
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

# code/book/ch04/vocab_head.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

HEAD, TAIL = 3, 2  # 每个向量显示头几个、尾几个数


def load_embed_tokens():
    """只取`model.embed_tokens.weight`这一个张量，顺便升成fp32。"""
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt") as f:
        return f.get_tensor("model.embed_tokens.weight").to(torch.float32)


def fmt(vec):
    """把一个576维向量显示成`[头3个, ..., 尾2个]`。"""
    head = ", ".join(f"{v:7.4f}" for v in vec[:HEAD].tolist())
    tail = ", ".join(f"{v:7.4f}" for v in vec[-TAIL:].tolist())
    return f"[{head}, ... , {tail}]"


def cell(token):
    """词元放进表格单元格。

    `<|endoftext|>`里那两根竖线会把Markdown的表格从中间劈开，所以要转义成`\\|`。
    反引号挡不住竖线，这是GFM表格的规矩。
    """
    return f"`{token.replace('|', chr(92) + '|')}`"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    embed = load_embed_tokens()  # (49152, 576)

    print(f"词表大小{tok.get_vocab_size()}，词嵌入矩阵{tuple(embed.shape)}，"
          f"下面是开头{n}个词元和各自在矩阵里的那一行：\n")

    rows = [(str(tid),
             cell(tok.id_to_token(tid)),
             f"`{fmt(embed[tid])}`",
             f"{embed[tid].norm():.3f}")   # 模长：576个数平方求和再开方
            for tid in range(n)]
    header = ("ID", "词元", f"词嵌入向量（576维，只显示头{HEAD}尾{TAIL}）", "模长")

    # 每列按最宽的那格对齐。Markdown不在乎这个，但源文件好看，也好核对。
    w = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(f"{c:<{n}}" for c, n in zip(r, w)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (n + 2) for n in w) + "|")
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
