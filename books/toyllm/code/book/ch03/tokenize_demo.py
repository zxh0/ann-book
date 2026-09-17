"""第三章 · 分词 demo

给一段文字，看它被切成哪些 token。

运行：
    cd code
    uv run python book/ch03/tokenize_demo.py
    uv run python book/ch03/tokenize_demo.py "你想试的任意一段话"
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

# code/book/ch03/tokenize_demo.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

TEXTS = [
    "你好吗",
    "Once upon a time",
    "Hello world",
]


def show(tok, text):
    enc = tok.encode(text, add_special_tokens=False)

    print(f"\n原文    {text!r}")
    print(f"词元    {enc.tokens}")
    print(f"ID      {enc.ids}")
    print(f"共 {len(text)} 个字符，{len(enc.ids)} 个词元")


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    print(f"词表大小 {tok.get_vocab_size()}")

    for text in sys.argv[1:] or TEXTS:
        show(tok, text)


if __name__ == "__main__":
    main()
