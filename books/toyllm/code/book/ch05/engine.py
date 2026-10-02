"""第五章 · 骨架通了的推理引擎

    文字 -> 词元 -> ID -> [ 词嵌入 -> 30个Block -> 出口norm -> 线性投影 ] -> 分数 -> 新ID -> 文字

和第四章比，方括号中间那个问号变成了30个Decoder块：每块两个RMSNorm、两条残差，
外加两个还空着的位置。空位现在拿随机数冒充，所以引擎照样转得起来，说的话也照样
没法看，只是胡说的方式和上一章不一样了。

`Engine`类和第四章一字未改，它只认`model.forward`，底下换成什么骨架都照转。
改的只是`main`里打印什么。

运行：
    cd code
    uv run python book/ch05/engine.py
    uv run python book/ch05/engine.py "你想试的任意一段话"
"""

import sys

from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

ROUNDS = 10  # 转多少圈，也就是生成多少个词元


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


def main():
    engine = Engine()
    tok = engine.tok
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"

    enc = tok.encode(text, add_special_tokens=False)  # Tokenizer：文字到ID
    ids = list(enc.ids)

    print(f"输入   {text!r}")
    print(f"词元   {enc.tokens}")
    print(f"ID     {ids}")

    # ---- 转起来 -------------------------------------------------------------
    print(f"\n转{ROUNDS}圈，每圈把新词元追加到序列末尾，然后整串重算一遍：")
    print("  ids → 词嵌入 (T, 576) → 30个Block → 出口norm → 取最后一行 → 投影 → 取最高分\n")

    out = engine.generate(ids)

    # 打印格式和第三章那个假引擎一样，方便两章对着看：同样的位置，
    # 第三章掷骰子直接得到ID，第四章真算但中间是空的，这一章中间立起来了。
    for k, i in enumerate(out[len(ids):], len(ids) + 1):
        print(f"  第{k:>2}个词元   ID {i:>5}   {tok.id_to_token(i)!r}")

    # ---- 出口：真的反查 ------------------------------------------------------
    print(f"\nID     {out}")
    print(f"词元   {[tok.id_to_token(i) for i in out]}")
    print(f"输出   {tok.decode(out)!r}")

    # 上一章这些ID会全是同一个：最后那个向量没被任何一层碰过，自己和自己最像，
    # 于是无限复读。这一章30层往主干上灌了随机数，不动点被打破了。
    new = out[len(ids):]
    print(f"\n{len(new)}个新词元里有{len(set(new))}个互不相同。上一章这里只有1个，"
          f"因为那时候中间是空的，最后那个向量原封不动，永远问出它自己。")


if __name__ == "__main__":
    main()
