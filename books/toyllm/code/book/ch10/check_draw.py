"""第十章 · 验证一个随机过程

filter能逐位对拍，抽这一步不能：它本来就该每次不一样。所以换两条判据：

    一  可复现     同一个种子抽两遍，结果完全相同；换个种子就不同
    二  分布正确   造一个已知的小分布，抽几万次，经验频率要收敛到目标概率

第二条还顺带验了filter和抽样接得对不对：top-k砍掉的词元一次都不该被抽到。

不导入transformers。

运行：
    cd code
    uv run python book/ch10/check_draw.py
"""

import torch

from sampler import RandomSampler

N = 40_000


def draw(sampler, logits, n):
    """抽n次，数每个词元被抽到几次。"""
    counts = torch.zeros(logits.shape[-1])
    for _ in range(n):
        counts[sampler(logits)] += 1
    return counts


def main():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])           # 一个只有4个词元的词表

    # ---- 一：可复现 -----------------------------------------------------------
    def seq(seed):
        s = RandomSampler(temperature=1.0, seed=seed)
        return [s(logits) for _ in range(20)]

    a, b, c = seq(42), seq(42), seq(43)
    print(f"一  可复现")
    print(f"    种子42   {a}")
    print(f"    种子42   {b}   {'相同' if a == b else '不同'}")
    print(f"    种子43   {c}   {'相同' if a == c else '不同'}")

    # 自带的生成器和全局随机数互不干扰：中间插一句torch.rand，结果照样相同。
    s = RandomSampler(temperature=1.0, seed=42)
    d = []
    for _ in range(20):
        torch.rand(100)                                    # 别处在用全局随机数
        d.append(s(logits))
    print(f"    种子42，中间插了别的随机数   {'相同' if d == a else '不同'}")

    # ---- 二：分布正确 ---------------------------------------------------------
    print(f"\n二  抽{N:,}次，经验频率 vs 目标概率")
    for label, sampler in [
        ("T=1.0", RandomSampler(temperature=1.0, seed=0)),
        ("T=0.5", RandomSampler(temperature=0.5, seed=0)),
        ("T=1.0，top_k=2", RandomSampler(temperature=1.0, top_k=2, seed=0)),
    ]:
        target = sampler.filter(logits).softmax(-1)
        freq = draw(sampler, logits, N) / N
        print(f"    {label}")
        print(f"      {'词元':>4}{'目标':>10}{'经验':>10}{'误差':>10}")
        for i in range(len(logits)):
            print(f"      {i:>6}{target[i]:>10.4f}{freq[i]:>10.4f}{abs(freq[i] - target[i]):>10.4f}")
        # 抽N次，每个频率的标准差是sqrt(p(1-p)/N)，0.5时最大，约0.0025。差到4倍还没到。
        ok = (freq - target).abs().max() < 4 * (0.25 / N) ** 0.5
        print(f"      最大误差{(freq - target).abs().max():.4f}，{'在' if ok else '不在'}统计误差以内")


if __name__ == "__main__":
    main()
