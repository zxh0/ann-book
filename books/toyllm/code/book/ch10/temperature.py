"""第十章 · 把温度从0拧到1.5

同一个开头、同一个种子，温度取0 / 0.3 / 0.7 / 1.0 / 1.5各跑一遍，
看输出从复读机一路滑向胡话。不加top-k和top-p，只看温度一个旋钮。

后半段验证可复现：同一个种子跑两次一字不差，换个种子就不一样。
模型没有变，logits没有变，变的只有采样器里那个随机数生成器。

运行：
    cd code
    uv run python book/ch10/temperature.py
    uv run python book/ch10/temperature.py "你想试的任意一段话"
"""

import sys

from engine import Engine
from sampler import Sampler

SEED = 20261006   # 固定住，好让书里引用的输出能复现
N = 40            # 每档生成多少个新词元


def main():
    engine = Engine()
    tok = engine.tok
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ids = tok.encode(text, add_special_tokens=False).ids

    print(f"开头 {text!r}，种子{SEED}，每档最多{N}个新词元\n")
    for t in (0, 0.3, 0.7, 1.0, 1.5):
        out = engine.generate(ids, Sampler.build(temperature=t, seed=SEED), N)
        new = out[len(ids):]
        print(f"T={t}  （{len(set(new))}/{len(new)}个互不相同）")
        print(f"  {tok.decode(out)!r}\n")

    # ---- 可复现 --------------------------------------------------------------
    def run(seed):
        return engine.generate(ids, Sampler.build(temperature=1.0, seed=seed), N)

    a, b, c = run(SEED), run(SEED), run(SEED + 1)
    print(f"T=1.0，种子{SEED}跑两次     {'一字不差' if a == b else '不一样'}")
    print(f"T=1.0，换成种子{SEED + 1}     {'一字不差' if a == c else '不一样'}")


if __name__ == "__main__":
    main()
