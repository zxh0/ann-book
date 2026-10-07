"""第十一章 · 快了多少

全书唯一一次正当地谈性能。前十章的问题都是「算得对不对」，这一章才问「算得快不快」，
而且前提是`check_cache.py`先证明了：快了，但一个ID都没变。

    一  白算了多少：不带缓存要算1 + 2 + … + n个位置，带缓存只算n个
    二  每个词元的耗时，随序列变长怎么走：一条往上爬，一条是平的
    三  prefill和decode：同样是forward，一个按词元算便宜，一个贵

数字和机器有关，书里引用的是作者那台Intel Mac，CPU，6个线程。

运行：
    cd code
    uv run python book/ch11/speed.py
"""

import time

from engine import Engine
from kv_cache import KVCache

N = 200        # 每条路生成多少个新词元
EVERY = 20     # 每隔多少个打一行
WEIGHT_BYTES = 134_515_008 * 4  # 第二章数出来的参数量，fp32


def per_token_times(engine, ids, cache):
    """生成N个，记下每一个花了多少秒。"""
    times = []
    start = time.perf_counter()
    for _ in engine.stream(ids, max_new_tokens=N, cache=cache):
        now = time.perf_counter()
        times.append(now - start)
        start = now
    return times


def main():
    engine = Engine()
    model, tok = engine.model, engine.tok
    ids = tok.encode("Once upon a time", add_special_tokens=False).ids
    p = len(ids)

    # ---- 一：白算了多少 -------------------------------------------------------
    print(f"一  开头{p}个词元，再生成n个，一共要过模型多少个位置")
    print(f"    {'n':>6}{'不带缓存':>14}{'带缓存':>10}{'倍数':>8}")
    for n in (20, 100, 1000):
        slow = sum(range(p, p + n))                    # 第k圈喂p + k个，k从0起
        fast = p + n - 1                               # prompt一次，之后每圈1个
        print(f"    {n:>6}{slow:>16,}{fast:>12,}{slow / fast:>10.1f}")

    # ---- 二：每个词元的耗时 ---------------------------------------------------
    engine.generate(ids, max_new_tokens=5)             # 热身，第一次跑总是偏慢
    fast = per_token_times(engine, ids, cache=True)
    slow = per_token_times(engine, ids, cache=False)
    n = min(len(fast), len(slow))                      # 万一中途抽到EOS

    # 单个词元的耗时抖得厉害，每EVERY个取一次平均。第1个单独算：
    # 两条路都要把prompt整段过一遍，带缓存的那一下就是prefill。
    windows = [(0, 1)] + [(k, min(k + EVERY, n)) for k in range(1, n, EVERY)]
    rows = [(a, b, sum(slow[a:b]) / (b - a), sum(fast[a:b]) / (b - a)) for a, b in windows]
    scale = max(r[2] for r in rows) / 40               # 最长那根画40格

    print(f"\n二  每生成一个词元平均花了多少毫秒，共{n}个    # 不带缓存   * 带缓存")
    for a, b, s, f in rows:
        label = f"{a + 1}" if b - a == 1 else f"{a + 1}-{b}"  # 第几个词元
        print(f"    {label:<10}{s * 1000:7.1f}  {'#' * round(s / scale)}")
        print(f"    {'':<10}{f * 1000:7.1f}  {'*' * max(1, round(f / scale))}")
    print(f"\n    合计   不带缓存{sum(slow[:n]):.1f}秒，带缓存{sum(fast[:n]):.1f}秒，"
          f"快了{sum(slow[:n]) / sum(fast[:n]):.1f}倍")

    # ---- 三：prefill和decode ---------------------------------------------------
    long = engine.generate(ids, max_new_tokens=124)    # 凑一段128个词元的真文字
    T = len(long)
    cache = KVCache(model.config, T + 1)
    start = time.perf_counter()
    logits = model.forward(long, cache)                # prefill：T个一次喂完
    prefill = time.perf_counter() - start

    rounds = 20
    start = time.perf_counter()
    for _ in range(rounds):                            # decode：每次1个
        cache.length = T                               # 退回去，每次都在同一个位置上量
        model.forward([int(logits.argmax())], cache)
    decode = (time.perf_counter() - start) / rounds

    print(f"\n三  prefill和decode，都是在{T}个词元上")
    print(f"    prefill  一次喂{T}个   {prefill * 1000:7.1f}毫秒   每个{prefill / T * 1000:5.2f}毫秒")
    print(f"    decode   一次喂1个     {decode * 1000:7.1f}毫秒   每个{decode * 1000:5.2f}毫秒")
    print(f"    同样一个词元，decode贵了{decode / (prefill / T):.0f}倍")
    print(f"\n    两种都要把{WEIGHT_BYTES / 1e6:.0f} MB的权重从内存里读一遍。prefill读一遍，"
          f"算{T}个词元；")
    print(f"    decode读一遍，只算1个。decode这一步相当于每秒搬"
          f"{WEIGHT_BYTES / decode / 1e9:.1f} GB，时间主要花在搬上。")


if __name__ == "__main__":
    main()
