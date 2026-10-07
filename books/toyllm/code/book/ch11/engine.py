"""第十一章 · 同一个循环，换一种走法

    文字 -> 词元 -> ID -> [ 词嵌入 -> 30个Block -> 出口norm -> 线性投影 ] -> 分数 -> 新ID -> 文字
                    ^                                                               |
                    +------------------------------ append to ----------------------+

总览图一个框都没多。改的是那条`append to`虚线怎么走：第十章每一圈把整串
从头喂一遍，这一章只喂缓存里还没有的那几个。

    第1圈   prefill   喂整段prompt，T个     缓存从空到T
    第2圈起 decode    只喂刚生成的那1个     缓存每圈加1

循环本身一行没多，喂什么只有一句：

    feed = ids[cache.length:]

`NO_CACHE`的length永远是0，喂的就是整串，退回第十章；`KVCache`里存了多少，
就跳过多少。没有`if`。

运行：
    cd code
    uv run python book/ch11/engine.py
    uv run python book/ch11/engine.py "你想试的任意一段话"
    uv run python book/ch11/engine.py "Once upon a time" --max-new-tokens 200
    uv run python book/ch11/engine.py "Once upon a time" --max-new-tokens 200 --no-cache
"""

import argparse
import time

from tokenizers import Tokenizer

from detokenizer import StreamDecoder
from kv_cache import NO_CACHE, KVCache
from model import Model
from sampler import GREEDY, Sampler
from weights import MODEL_DIR

MAX_NEW_TOKENS = 20  # 兜底：模型一直不说EOS，也最多转这么多圈


class Engine:
    """把四个模块凑在一起：Tokenizer、LLM、Sampler、Detokenizer。"""

    def __init__(self, model_dir=MODEL_DIR):
        self.model = Model(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.eos = self.model.config.eos_token_id          # 0，`<|endoftext|>`

    def new_cache(self, cache, max_len):
        """`cache`收三种：True新开一个，False不用缓存，或者一个已经存了东西的KVCache。

        在循环开始前定一次，循环里不再问。
        """
        if cache is True:
            return KVCache(self.model.config, max_len)
        if cache is False:
            return NO_CACHE
        return cache                                       # 接着用上一轮的

    def stream(self, ids, sampler=GREEDY, max_new_tokens=MAX_NEW_TOKENS, cache=True):
        """一串词元ID进去，新ID一个一个吐出来。

        每圈：喂缓存里还没有的那几个 → 取最后一个位置的logits → 采样 → 追加 → 再来。
        抽到EOS就停，EOS本身不吐出来。

        吐完之后，缓存里比`ids`加上吐出来的少一个：最后吐出来的那个是最后一次
        forward算出来的，还没被喂回去。所以想接着上一轮往下说，就把整串历史连同
        那个缓存一起交回来，`ids[cache.length:]`自然从漏掉的那一个喂起。
        """
        cache = self.new_cache(cache, len(ids) + max_new_tokens)
        ids = list(ids)  # 复制一份，不去改调用方手里那个list
        for _ in range(max_new_tokens):
            feed = ids[cache.length:]                      # 第1圈整段，之后每圈1个
            next_id = self.infer(feed, sampler, cache)
            if next_id == self.eos:                        # 模型自己认为说完了
                return
            ids.append(next_id)
            yield next_id

    def generate(self, ids, sampler=GREEDY, max_new_tokens=MAX_NEW_TOKENS, cache=True):
        """一串词元ID进去，一串更长的ID出来。`stream`收齐了再交出去。"""
        return list(ids) + list(self.stream(ids, sampler, max_new_tokens, cache))

    def infer(self, ids, sampler=GREEDY, cache=NO_CACHE):
        """转一圈：一串词元ID进去，下一个词元ID出来。

        对应总览图中间那两个框：LLM算出分数，采样器挑一个。
        `model.forward`只返回最后一个位置的logits：前面每个位置也都预测了
        「它的下一个词元」，但那些位置的下一个词元我们已经知道了。
        """
        logits = self.model.forward(ids, cache)            # (T,) → (49152,)
        return sampler(logits)                             # (49152,) → 一个ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="Once upon a time")
    ap.add_argument("--temperature", type=float, default=0.0, help="0就是贪婪")
    ap.add_argument("--top-k", type=int, default=0, help="0表示不截断")
    ap.add_argument("--top-p", type=float, default=1.0, help="1表示不截断")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--no-cache", action="store_true", help="退回第十章的走法")
    args = ap.parse_args()

    engine = Engine()
    tok = engine.tok
    sampler = Sampler.build(args.temperature, args.top_k, args.top_p, args.seed)

    ids = tok.encode(args.text, add_special_tokens=False).ids  # Tokenizer：文字到ID
    cache = engine.new_cache(not args.no_cache, len(ids) + args.max_new_tokens)
    print(f"输入     {args.text!r}   {ids}")
    print(f"采样器   {sampler!r}")
    print(f"缓存     {cache!r}")
    print(f"最多     {args.max_new_tokens}个新词元，抽到EOS（ID {engine.eos}）就停\n")

    # ---- 转起来，来一个印一个 -------------------------------------------------
    stream = StreamDecoder(tok)                            # Detokenizer：ID回到文字
    out = list(ids)
    start = time.perf_counter()
    print(args.text, end="", flush=True)
    for i in engine.stream(ids, sampler, args.max_new_tokens, cache):
        out.append(i)
        print(stream.push(i), end="", flush=True)
    print(stream.flush())
    seconds = time.perf_counter() - start

    new = out[len(ids):]
    why = "抽到EOS" if len(new) < args.max_new_tokens else "到了上限"
    print(f"\n{len(new)}个新词元，{why}停下，用时{seconds:.1f}秒，"
          f"每个{seconds / max(len(new), 1):.3f}秒")
    print(f"缓存     {cache!r}")
    print(f"ID       {out}")


if __name__ == "__main__":
    main()
