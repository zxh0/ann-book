"""第十一章 · 对拍：带缓存和不带缓存，一个ID都不能差

不带缓存的那条路，第十章已经和第一章那句话一字不差地对过了，它现在就是基准。
判据是逐个ID相等，不是容差：解码是一连串argmax，缓存错一个位置，不会表现成
数值微抖，而是某一步挑了另一个词，这个词再喂回去，整段从此跑偏。

    一  三段开头，各生成100个，ID逐个比
    二  logits本身：不逐位相同，但差得远不够翻转一个argmax
    三  接着上一轮往下说：两次各10个 == 一次20个
    四  故意做错两种，看文本怎么跑偏

这个脚本不导入transformers。

运行：
    cd code
    uv run python book/ch11/check_cache.py
"""

from engine import Engine
from kv_cache import KVCache
from model import Model
from sampler import Sampler

PROMPTS = ["Once upon a time", "The capital of France is", "def fibonacci(n):"]
N = 100


def first_diff(a, b):
    """两串ID从第几个开始不一样，一样就返回None。"""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def greedy(model, ids, n, cache):
    """不走Engine，手写的贪婪循环：prefill一次，之后每次喂1个。

    错法二要用它：每层都advance的话，cache.length比真实长度大了30倍，
    Engine里那句`ids[cache.length:]`会切出一个空串来。
    """
    out = list(ids)
    logits = model.forward(out, cache)
    for _ in range(n):
        out.append(int(logits.argmax()))
        logits = model.forward(out[-1:], cache)
    return out


def main():
    engine = Engine()
    model, tok = engine.model, engine.tok

    # ---- 一：ID逐个比 ---------------------------------------------------------
    print(f"一  各生成{N}个新词元，贪婪")
    for text in PROMPTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        fast = engine.generate(ids, max_new_tokens=N, cache=True)
        slow = engine.generate(ids, max_new_tokens=N, cache=False)
        print(f"    {text!r:<28}{len(fast) - len(ids):>4}个   "
              f"{'相同' if fast == slow else f'第{first_diff(fast, slow)}个起不同'}")

    ids = tok.encode(PROMPTS[0], add_special_tokens=False).ids
    fast = engine.generate(ids, Sampler.build(0.8, top_p=0.95, seed=7), N, cache=True)
    slow = engine.generate(ids, Sampler.build(0.8, top_p=0.95, seed=7), N, cache=False)
    print(f"    随机采样，T=0.8，top_p=0.95，同一个种子   "
          f"{'相同' if fast == slow else f'第{first_diff(fast, slow)}个起不同'}")

    # ---- 二：logits本身 -------------------------------------------------------
    # 把一里那段贪婪输出逐个位置重放：带缓存每次喂1个，不带缓存每次喂整串。
    out = engine.generate(ids, max_new_tokens=N)
    cache = KVCache(model.config, len(out))
    model.forward(ids[:-1], cache)                     # 先把prompt除最后一个存进去
    worst, worst_at, margin = 0.0, 0, float("inf")
    for t in range(len(ids), len(out) + 1):
        a = model.forward(out[t - 1:t], cache)         # 带缓存，喂1个
        b = model.forward(out[:t])                     # 不带缓存，整串
        rel = (a - b).abs().max().item() / b.abs().max().item()
        if rel > worst:
            worst, worst_at = rel, t
        top2 = b.topk(2).values                        # 第一名领先第二名多少
        margin = min(margin, (top2[0] - top2[1]).item())
    print(f"\n二  逐个位置比logits，共{len(out) - len(ids) + 1}个位置")
    print(f"    最大相对误差   {worst:.1e}（序列长{worst_at}时）")
    print(f"    第一名领先第二名，最少也有   {margin:.1e}")

    # ---- 三：接着说 -----------------------------------------------------------
    cache = KVCache(model.config, len(ids) + 20)
    turn1 = engine.generate(ids, max_new_tokens=10, cache=cache)
    after1 = cache.length
    turn2 = engine.generate(turn1, max_new_tokens=10, cache=cache)
    one_go = engine.generate(ids, max_new_tokens=20)
    print(f"\n三  接着上一轮往下说")
    print(f"    第一轮之后，输出{len(turn1)}个，缓存里{after1}个"
          f"（最后那个还没喂回去）")
    print(f"    两轮各10个 == 一次20个   {'相同' if turn2 == one_go else '不同'}")

    # ---- 四：故意做错 ---------------------------------------------------------
    print(f"\n四  故意做错，贪婪生成{N // 5}个")
    good = engine.generate(ids, max_new_tokens=N // 5)
    print(f"    正确         {tok.decode(good)!r}")

    # 错法一：RoPE的位置每次都从0起。decode时新词元永远以为自己在第0个位置。
    model.rope = lambda x, start=0: Model.rope(model, x)
    bad = engine.generate(ids, max_new_tokens=N // 5)
    del model.rope                                     # 摘掉，恢复原样
    print(f"    位置从0起    {tok.decode(bad)!r}")
    print(f"                 第{first_diff(good, bad) - len(ids) + 1}个新词元起不同，不报错")

    # 错法二：每层写完就advance。第1层的k写在第T个位置之后，第2层更往后……
    class EveryLayer(KVCache):
        def update(self, i, k, v):
            out = super().update(i, k, v)
            self.advance(k.shape[1])
            return out

    cfg = model.config
    room = (len(ids) + N) * (cfg.num_hidden_layers + 1)   # 每圈挪31次，要留够地方
    bad = greedy(model, ids, N // 5, EveryLayer(cfg, room))
    print(f"    每层advance  {tok.decode(bad)!r}")
    print(f"                 第{first_diff(good, bad) - len(ids) + 1}个新词元起不同，不报错")


if __name__ == "__main__":
    main()
