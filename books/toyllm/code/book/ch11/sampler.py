"""第十章 · 采样器

    logits (49152,) → [温度] → [top-k] → [top-p] → softmax → 概率 (49152,) → 抽一个 → ID

从词元ID到logits，整条流水线是确定性的，同样的输入永远得到同样的logits。
模型每次说的话不一样，全部的不一样都发生在这个文件里。

分两半：

  - 三个filter是纯函数，logits进、logits出，只改分布的形状，不抽。
    砍掉的词元置成-inf，softmax之后正好是0，剩下的自动归一，不用再除一遍。
  - 真正的抽只有一步，`torch.multinomial`按概率取一个。

采样器做成类，不做成`if`：贪婪和随机是两个类，选哪个在构造时定一次，
生成循环里只有一句`sampler(logits)`，每一圈都不用再问一遍。

运行：
    cd code
    uv run python book/ch10/sampler.py
    uv run python book/ch10/sampler.py "你想试的任意一段话"
"""

import sys

import torch
from tokenizers import Tokenizer

from model import Model
from weights import MODEL_DIR

FILTER_VALUE = float("-inf")  # 和因果掩码一样，置成-inf，softmax之后就是0


def apply_temperature(logits, temperature):
    """温度：logits除以T，(..., 49152) → (..., 49152)。

    T < 1，分数之间的差距被放大，分布变尖；T > 1，差距被压平，分布变平。
    T是除数，所以两边不对称：T=0.5把每个差距翻倍，T=2把每个差距减半。

    T=0不在这里处理。那是T趋于0的极限，也就是贪婪，交给`GreedySampler`，
    这里不能真的去除以0。
    """
    return logits / temperature


def apply_top_k(logits, k):
    """top-k：只留分数最高的k个，其余置-inf，(..., 49152) → (..., 49152)。

    k=0表示不截断。k是固定的，不看分布长什么样：模型很有把握时50个太多，
    模型拿不准时50个又可能太少。top-p补的就是这个短板。
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    # 第k大的那个分数就是门槛，严格低于它的都砍掉。
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, FILTER_VALUE)


def apply_top_p(logits, p):
    """top-p（核采样）：留下累计概率刚好够p的那一撮，(..., 49152) → (..., 49152)。

    照transformers的写法：**升序**排，从最小的开始砍，只要砍掉的累计概率
    还不超过1 - p就接着砍。这样跨过阈值的那个边界词元会留下来。

    更顺手的写法是降序排、累加、一超过p就截断，可那样会把边界词元一起砍掉，
    候选集系统性地偏小。不报错，生成的文字也通顺，只有对拍分得出来。
    """
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits)              # 升序
    cumulative = sorted_logits.softmax(-1).cumsum(-1)      # 从最小的累加上去
    remove = cumulative <= 1 - p                           # 这一段加起来不到1 - p，砍
    remove[..., -1] = False                                # 最大的那个永远留着，候选集不能空
    remove = remove.scatter(-1, order, remove)             # 排序后的位置换回词表里的位置
    return logits.masked_fill(remove, FILTER_VALUE)


class Sampler:
    """采样器的统一接口：logits (49152,) 进，一个词元ID（int）出。

    贪婪和随机是两个实现，不是一个开关。选哪个由`build`在构造时定一次，
    生成循环从头到尾不再问。以后要加别的策略，实现`__call__`就能插进来，
    不用动模型，也不用动循环。
    """

    def __call__(self, logits):
        raise NotImplementedError

    @staticmethod
    def build(temperature=1.0, top_k=0, top_p=1.0, seed=None):
        """按采样参数挑一个实现。贪婪还是随机，只在这里判断一次。"""
        if temperature < 0:
            raise ValueError("temperature不能是负数")
        if temperature == 0:
            return GreedySampler()  # T趋于0的极限，不是除以0
        return RandomSampler(temperature, top_k, top_p, seed)


class GreedySampler(Sampler):
    """贪婪：永远取分数最高的那个。

    确定性的，所以只有它能和transformers逐个ID对拍。第八章就是靠它。
    """

    def __call__(self, logits):
        return int(logits.argmax())

    def __repr__(self):
        return "GreedySampler()"


class RandomSampler(Sampler):
    """先过三个filter，再从剩下的里面按概率抽一个。

    顺序是温度 → top-k → top-p，和transformers一样。这个顺序不是随便排的：
    温度改的是分数之间的差距，而top-k、top-p是按差距挑人的，
    温度放到后面，改的就不只是概率，而是**哪些词元能入选**。
    """

    def __init__(self, temperature=1.0, top_k=0, top_p=1.0, seed=None):
        if temperature <= 0:
            raise ValueError("temperature必须大于0，T=0请用GreedySampler")
        if not 0 < top_p <= 1:
            raise ValueError("top_p必须在(0, 1]里")
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        # 每个采样器自带一个随机数生成器：固定种子就能复现，
        # 也不会和别处的随机数互相干扰。
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def filter(self, logits):
        """确定性的那一半：logits进、logits出。可以和transformers逐位对拍。"""
        logits = apply_temperature(logits, self.temperature)
        logits = apply_top_k(logits, self.top_k)
        return apply_top_p(logits, self.top_p)

    def __call__(self, logits):
        probs = self.filter(logits).softmax(-1)            # (49152,)，被砍掉的正好是0
        return int(torch.multinomial(probs, 1, generator=self.generator))

    def __repr__(self):
        return (f"RandomSampler(temperature={self.temperature}, "
                f"top_k={self.top_k or 'off'}, top_p={self.top_p})")


GREEDY = GreedySampler()  # 默认用它，和前面几章行为一样


def kept(logits):
    """过完filter还剩几个词元。"""
    return int(torch.isfinite(logits).sum())


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    text = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    ids = tok.encode(text, add_special_tokens=False).ids
    with torch.no_grad():
        logits = model.forward(ids)                        # (49152,)

    def show(i):
        return repr(tok.decode([int(i)]))

    # ---- 一：贪婪扔掉了什么 ---------------------------------------------------
    probs = logits.softmax(-1)
    top = probs.topk(8)
    print(f"{text!r}之后，概率最高的8个词元：\n")
    for p, i in zip(top.values, top.indices):
        print(f"  {show(i):<12}{p:7.3f}  {'#' * round(p.item() * 40)}")
    print(f"\n  这8个加起来{top.values.sum():.3f}，剩下的49144个分掉另外{1 - top.values.sum():.3f}。")

    # ---- 二：温度 ------------------------------------------------------------
    temps = (0.3, 0.7, 1.0, 1.5, 3.0)
    print(f"\n温度改变分布的尖锐程度：\n")
    print(f"  {'':<12}" + "".join(f"{f'T={t}':>9}" for t in temps))
    for i in top.indices[:5]:
        row = "".join(f"{apply_temperature(logits, t).softmax(-1)[i]:9.3f}" for t in temps)
        print(f"  {show(i):<12}{row}")

    # ---- 三：top-k固定，top-p自适应 -------------------------------------------
    print(f"\n截断长尾，还剩几个词元、多少概率：\n")
    for k in (1, 5, 50, 500):
        m = torch.isfinite(apply_top_k(logits, k))
        print(f"  top_k={k:<6}{m.sum():>7}个   {probs[m].sum():.4f}")
    for p in (0.5, 0.9, 0.95, 0.99):
        m = torch.isfinite(apply_top_p(logits, p))
        print(f"  top_p={p:<6}{m.sum():>7}个   {probs[m].sum():.4f}")

    peaked = torch.tensor([12.0] + [1.0] * 49)             # 一个词元一枝独秀
    flat = torch.zeros(50)                                 # 50个词元平分秋色
    print(f"\n  50个词元，top_p=0.9：分布尖时留{kept(apply_top_p(peaked, 0.9))}个，"
          f"分布平时留{kept(apply_top_p(flat, 0.9))}个")

    # ---- 四：边界词元 --------------------------------------------------------
    demo = torch.tensor([0.6, 0.3, 0.1]).log()
    print(f"\n概率0.6 / 0.3 / 0.1，top_p=0.7，留几个？  {kept(apply_top_p(demo, 0.7))}个")

    # ---- 五：顺序 ------------------------------------------------------------
    a = apply_top_p(apply_temperature(logits, 0.5), 0.9)
    b = apply_temperature(apply_top_p(logits, 0.9), 0.5)
    print(f"\nT=0.5、top_p=0.9，先温度后top-p留{kept(a)}个，先top-p后温度留{kept(b)}个")

    # ---- 六：构造时就选好 ------------------------------------------------------
    print(f"\nSampler.build(temperature=0)    → {Sampler.build(temperature=0)!r}")
    print(f"Sampler.build(temperature=0.7)  → {Sampler.build(temperature=0.7, top_p=0.9)!r}")


if __name__ == "__main__":
    main()
