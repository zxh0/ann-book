"""第四章 · 半真的推理引擎

    文字 -> 词元 -> ID -> [ 词嵌入 -> ??? -> 线性投影 ] -> 分数 -> 新ID -> 文字

和第三章那个全假的引擎比，这一章两头是真的：入口查表、出口投影都交给`model.py`
里的`Model`，用的是同一个矩阵，只是读的方向不同。中间那30个Decoder块还不存在，
方括号里那个问号就是它们空着的位置，所以引擎虽然转得起来，说的话还是没法看。

运行：
    cd code
    uv run python book/ch04/engine.py
    uv run python book/ch04/engine.py "你想试的任意一段话"
"""

import sys

from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

ROUNDS = 10  # 转多少圈，也就是生成多少个词元

HEAD, TAIL = 3, 2  # 每个向量显示头几个、尾几个数


class Engine:
    """把四个模块凑在一起：Tokenizer、LLM、Sampler、Detokenizer。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model = Model(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def generate(self, ids):
        """一串词元ID进去，一串更长的ID出来。

        转ROUNDS圈，每圈把`infer`挑出来的新ID追加到序列末尾，然后拿整串重算
        一遍。这一章的圈数是写死的，第十章才会让调用方说停在哪。

        进出都是ID，中间不碰文字：同一段文字重新切一遍，切法未必和原来一样。
        """
        ids = list(ids)  # 复制一份，不去改调用方手里那个list
        for _ in range(ROUNDS):
            ids.append(self.infer(ids))
        return ids

    def infer(self, ids):
        """转一圈：一串词元ID进去，下一个词元ID出来。

        对应总览图中间那两个框：LLM算出分数，采样器挑一个。挑的办法这里用最
        简单的，直接选最高分，也就是贪婪采样，第十章再正式讲。

        收的是ID不是文字，因为循环要在ID上滚：每圈把新ID追加到序列末尾，而不是
        把文字拼起来重新分词。同一段文字重新切一遍，切法未必和原来一样。
        """
        logits = self.model.forward(ids)  # (T,) → (49152,)
        return int(logits.argmax())


def fmt(vec):
    """把一个576维向量显示成`[头3个, ..., 尾2个]`。"""
    head = ", ".join(f"{v:7.4f}" for v in vec[:HEAD].tolist())
    tail = ", ".join(f"{v:7.4f}" for v in vec[-TAIL:].tolist())
    return f"[{head}, ... , {tail}]"


def cell(token):
    """词元放进表格单元格。

    `<|endoftext|>`里那两根竖线会把Markdown的表格从中间劈开，所以要转义成`\|`。
    反引号挡不住竖线，这是GFM表格的规矩。生成出来的词元也可能是特殊词元，
    所以这里和`vocab_head.py`一样要转义。
    """
    return f"`{token.replace('|', chr(92) + '|')}`"


def table(header, rows):
    """打印一张Markdown表格，每列按最宽的那格对齐。

    Markdown不在乎对不对齐，但输出是拿去贴进书稿的，源文件齐整一点好核对。
    """
    w = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(f"{c:<{n}}" for c, n in zip(r, w)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (n + 2) for n in w) + "|")
    for r in rows:
        print(line(r))


def main():
    engine = Engine()
    tok, model = engine.tok, engine.model
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    enc = tok.encode(text, add_special_tokens=False)  # Tokenizer：文字到ID
    ids = list(enc.ids)

    print(f"输入   {text!r}")
    print(f"词元   {enc.tokens}")
    print(f"ID     {ids}")

    # ---- 转起来 -------------------------------------------------------------
    print(f"\n转{ROUNDS}圈，每圈把新词元追加到序列末尾，然后整串重算一遍：")
    print(f"  ids → 词嵌入 (T, 576) → 取最后一行 → 投影 ({tok.get_vocab_size()},) → 取最高分")

    out = engine.generate(ids)

    # ---- 输入和生成，摆在一张表里 -------------------------------------------
    # 生成出来的词元和输入的词元没有任何区别：都是ID，都在同一张表里查同一行。
    # 所以这里不分两段打印，`generate`出来的整串一次查表，一次列完。
    x = model.to_embeddings(out)  # (T,) → (T, 576)
    print(f"\n一共{len(out)}个词元，形状 {(len(out),)} → {tuple(x.shape)}：\n")
    table(("序号", "来源", "ID", "词元", f"词嵌入（576维，只显示头{HEAD}尾{TAIL}）"),
          [(str(i),
            "输入" if i < len(ids) else "生成",
            str(tid),
            cell(tok.id_to_token(tid)),
            f"`{fmt(x[i])}`")
           for i, tid in enumerate(out)])

    # ---- 出口：真的反查 ------------------------------------------------------
    print(f"\nID     {out}")
    print(f"词元   {[tok.id_to_token(i) for i in out]}")
    print(f"输出   {tok.decode(out)!r}")


if __name__ == "__main__":
    main()
