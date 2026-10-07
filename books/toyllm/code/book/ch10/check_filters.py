"""第十章 · 三个filter和transformers对拍

采样的输出没法逐个ID比，所以拆成两半验。这个脚本验确定性的那一半：
`apply_temperature` / `apply_top_k` / `apply_top_p`是纯函数，logits进、logits出，
和transformers的三个warper逐位相同（`torch.equal`）。

    TemperatureLogitsWarper   ↔   apply_temperature
    TopKLogitsWarper          ↔   apply_top_k
    TopPLogitsWarper          ↔   apply_top_p

输入是模型对第一章那句话算出的真logits。后面再故意做错两种，看对拍能不能抓住：

    top-p降序写法   降序累加，一超过p就截断，边界词元被一起砍掉
    顺序颠倒        先top-p后温度

这个脚本导入transformers，只用来对拍，引擎本身不碰它。

运行：
    cd code
    uv run python book/ch10/check_filters.py
"""

import torch
from tokenizers import Tokenizer
from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from model import Model
from sampler import FILTER_VALUE, RandomSampler, apply_temperature, apply_top_k, apply_top_p
from weights import MODEL_DIR


def top_p_descending(logits, p):
    """错的top-p：降序累加，一超过p就截断。看着更顺手，但边界词元没了。"""
    sorted_logits, order = torch.sort(logits, descending=True)
    cumulative = sorted_logits.softmax(-1).cumsum(-1)
    remove = cumulative > p
    remove[..., 0] = False
    remove = remove.scatter(-1, order, remove)
    return logits.masked_fill(remove, FILTER_VALUE)


def hf(warper, logits):
    """transformers的warper收(batch, vocab)，我们的logits是(vocab,)，套一层再拆掉。"""
    return warper(None, logits[None])[0]


def kept(logits):
    return int(torch.isfinite(logits).sum())


def main():
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = Model()
    texts = ["Once upon a time", "The capital of France is", "def fibonacci(n):"]

    # ---- 一：三个filter单独比 -------------------------------------------------
    print(f"一  单个filter，{'逐位相同':>8}{'留下':>8}")
    for text in texts:
        with torch.no_grad():
            logits = model.forward(tok.encode(text, add_special_tokens=False).ids)
        print(f"  {text!r}")
        for t in (0.3, 0.7, 1.5):
            ours, theirs = apply_temperature(logits, t), hf(TemperatureLogitsWarper(t), logits)
            print(f"    temperature={t:<6}{str(torch.equal(ours, theirs)):>8}{kept(ours):>9}")
        for k in (1, 50, 500):
            ours, theirs = apply_top_k(logits, k), hf(TopKLogitsWarper(k), logits)
            print(f"    top_k={k:<12}{str(torch.equal(ours, theirs)):>8}{kept(ours):>9}")
        for p in (0.5, 0.9, 0.95):
            ours, theirs = apply_top_p(logits, p), hf(TopPLogitsWarper(p), logits)
            print(f"    top_p={p:<12}{str(torch.equal(ours, theirs)):>8}{kept(ours):>9}")

    # ---- 二：三个串起来比 -----------------------------------------------------
    with torch.no_grad():
        logits = model.forward(tok.encode(texts[0], add_special_tokens=False).ids)
    print(f"\n二  串起来，温度 → top-k → top-p")
    for t, k, p in [(0.7, 50, 0.9), (1.0, 0, 0.95), (1.5, 200, 0.8), (0.5, 10, 1.0)]:
        ours = RandomSampler(t, k, p).filter(logits)
        theirs = logits
        for w in [TemperatureLogitsWarper(t)] + ([TopKLogitsWarper(k)] if k else []) \
                + ([TopPLogitsWarper(p)] if p < 1 else []):
            theirs = hf(w, theirs)
        print(f"    T={t:<4} top_k={k:<4} top_p={p:<5}逐位相同 {torch.equal(ours, theirs)}"
              f"   留下{kept(ours)}个")

    # ---- 三：故意做错 ---------------------------------------------------------
    print(f"\n三  故意做错，对拍抓不抓得住")
    demo = torch.tensor([0.6, 0.3, 0.1]).log()
    theirs = hf(TopPLogitsWarper(0.7), demo)
    print(f"    0.6 / 0.3 / 0.1，top_p=0.7   "
          f"transformers留{kept(theirs)}个，我们留{kept(apply_top_p(demo, 0.7))}个，"
          f"降序写法留{kept(top_p_descending(demo, 0.7))}个")
    for p in (0.5, 0.9, 0.95):
        right, wrong = apply_top_p(logits, p), top_p_descending(logits, p)
        print(f"    真logits，top_p={p:<5}  正确留{kept(right):>4}个，降序写法留{kept(wrong):>4}个，"
              f"逐位相同 {torch.equal(right, wrong)}")
    for t in (0.5, 1.5):
        a = apply_top_p(apply_temperature(logits, t), 0.9)
        b = apply_temperature(apply_top_p(logits, 0.9), t)
        print(f"    T={t}、top_p=0.9   先温度留{kept(a):>4}个，先top-p留{kept(b):>4}个")


if __name__ == "__main__":
    main()
