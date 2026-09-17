"""第三章 · 单字节词元

把`tokenizer.json`里的单字节词元全部打印出来，每行20个。

所谓单字节词元，就是词表里那些只有一个字符的词元。ByteLevel预处理把每个字节
换成了一个可打印字符，所以“一个字符”就等于“一个字节”，它们是BPE合并的原子。
词表里ID从17到251这235个，就是它们。

只用标准库读JSON，不依赖`tokenizers`库。

运行：
    cd code
    uv run python book/ch03/byte_tokens.py
"""

import json
from pathlib import Path

# code/book/ch03/byte_tokens.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

PER_LINE = 20


def byte_to_char():
    """GPT-2那张字节到可打印字符的映射表，256个字节各对应一个字符。

    先收下本来就可打印的那些字节（ASCII可见字符，加上两段拉丁补充），
    剩下的字节（控制符、空格、0x80以后的一部分）依次映射到U+0100往后，
    避开所有不可见字符。空格0x20就是这样变成`Ġ`（U+0120）的。
    """
    keep = list(range(ord("!"), ord("~") + 1))
    keep += list(range(ord("¡"), ord("¬") + 1))
    keep += list(range(ord("®"), ord("ÿ") + 1))

    chars = list(keep)
    n = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


def main():
    vocab = json.loads((MODEL_DIR / "tokenizer.json").read_text())["model"]["vocab"]

    b2c = byte_to_char()
    c2b = {c: b for b, c in b2c.items()}

    # 单字节词元：词元字符串只有一个字符，且这个字符在字节映射表里
    singles = sorted(
        ((tid, tok) for tok, tid in vocab.items() if len(tok) == 1 and tok in c2b),
        key=lambda p: p[0],
    )

    ids = [tid for tid, _ in singles]
    print(f"词表大小 {len(vocab)}")
    print(f"单字节词元 {len(singles)} 个，ID从{ids[0]}到{ids[-1]}", end="")
    print("，连续" if ids == list(range(ids[0], ids[-1] + 1)) else "，不连续")

    print(f"\n每行{PER_LINE}个，行首是这一行第一个词元的ID：\n")
    for i in range(0, len(singles), PER_LINE):
        row = singles[i : i + PER_LINE]
        print(f"  {row[0][0]:>5}  " + " ".join(tok for _, tok in row))

    # 256个字节里，没能进词表的那些
    missing = [b for b in range(256) if b2c[b] not in vocab]
    print(f"\n256个字节里有{len(missing)}个不在词表里，它们的字节值是：\n")
    for i in range(0, len(missing), PER_LINE):
        row = missing[i : i + PER_LINE]
        print("         " + " ".join(f"{b:02X}" for b in row))
    print("\n这些字节既没有词元可查，也没有字节兜底（byte_fallback是false），")
    print("遇到它们，分词器会静默丢弃。")


if __name__ == "__main__":
    main()
