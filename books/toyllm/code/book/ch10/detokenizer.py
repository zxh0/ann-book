"""第十章 · 流式输出：第三章埋的雷

整段解码时，`tok.decode(ids)`把所有词元拼起来，换回字节，再按UTF-8读一遍，
从来不出问题。可生成是一个词元一个词元出来的，想做打字机效果，就得来一个印一个。

雷就在这里：一个汉字是3个UTF-8字节，BPE按字节切，常常把它切进两个词元。
逐个词元decode，半个汉字读不成字，就印出`�`。

办法是维护一个字节缓冲：每个词元先换回字节放进缓冲，凑成完整的字符才印，
凑不齐就留着等下一个词元。

    词元字符串 → 每个字符换回一个字节 → 字节缓冲 → 凑齐的字符印出来

第二步用的是第三章那张字节映射表，这里反过来查。

运行：
    cd code
    uv run python book/ch10/detokenizer.py
    uv run python book/ch10/detokenizer.py "你想试的任意一段话"
"""

import codecs
import sys

from tokenizers import Tokenizer

from weights import MODEL_DIR


def byte_to_char():
    """GPT-2那张字节到可打印字符的映射表，和第三章`byte_tokens.py`里的一字不差。"""
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


class StreamDecoder:
    """一个词元ID进去，能印的文字出来，可能是空串。

    缓冲用的是标准库的增量UTF-8解码器：喂给它的字节末尾要是凑不成一个字符，
    它就先扣着，下一次喂进来的字节接在后面再读。字节本身坏掉了才印`�`。
    """

    def __init__(self, tok):
        self.tok = tok
        self.char_to_byte = {c: b for b, c in byte_to_char().items()}
        self.buffer = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def to_bytes(self, i):
        """词元ID → 它代表的那几个字节。`Ġ`换回空格0x20，`Ċ`换回换行0x0A。"""
        return bytes(self.char_to_byte[c] for c in self.tok.id_to_token(i))

    def push(self, i):
        return self.buffer.decode(self.to_bytes(i))

    def flush(self):
        """生成结束时还扣着的半个字符，只能当坏字节印出来了。"""
        return self.buffer.decode(b"", final=True)


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    text = sys.argv[1] if len(sys.argv) > 1 else "第十章：采样"
    ids = tok.encode(text, add_special_tokens=False).ids

    stream = StreamDecoder(tok)
    print(f"{text!r}切成{len(ids)}个词元，一个一个印：\n")
    print(f"  {'ID':>6}  {'词元':<10}{'字节':<14}{'逐个decode':<14}{'先进缓冲'}")
    naive, buffered = "", ""
    for i in ids:
        b = stream.to_bytes(i)
        one = tok.decode([i])                              # 不缓冲：这一个词元自己解
        out = stream.push(i)                               # 缓冲：凑齐了才出
        naive += one
        buffered += out
        print(f"  {i:>6}  {tok.id_to_token(i)!r:<10}{b.hex(' '):<14}{one!r:<14}{out!r}")
    buffered += stream.flush()

    print(f"\n  逐个decode拼起来   {naive!r}")
    print(f"  先进缓冲再印       {buffered!r}")
    print(f"  整段decode         {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
