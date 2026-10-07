"""第八章 · 整机对拍：四级验收

空位全部填满，第一次可以拿整条流水线和第一章transformers跑出来的结果比。
按严格程度递进，一级一级过：

    一  参数量          和第一章一个不多一个不少
    二  31层逐层比      相对误差max|diff| / max|ref|
    三  每个位置的argmax 一个都不能错
    四  贪婪解码20个词元  和第一章那句话一字不差

这个脚本不导入transformers，只读第一章`baseline.py`存下的文件。

逐层比的下标是transformers的约定，很容易搞错：

    [0]        词嵌入的输出
    [1]..[29]  第i个块的输入，也就是第i-1个块的输出
    [30]       最后一个块的输出，而且已经过了出口norm

运行：
    cd code
    uv run python book/ch01/baseline.py     # 还没有基准文件的话，先跑这个
    uv run python book/ch08/check_forward.py
"""

import torch

from engine import Engine
from weights import MODEL_DIR

BASELINE = MODEL_DIR.parent.parent / "reference" / "baseline.pt"


def hidden_states(model, ids):
    """走一遍30层，按transformers的约定留下31份中间结果，各(T, 576)。"""
    x = model.to_embeddings(ids)                       # (T, 576)
    states = [x]
    for layer in model.weights.layers:
        x = model.block(x, layer)
        states.append(x)
    states[-1] = model.rms_norm(x, model.weights.norm)  # 最后一份要过出口norm
    return states


def main():
    ref = torch.load(BASELINE)
    engine = Engine()
    model = engine.model
    ids = ref["input_ids"][0].tolist()                 # [6403, 1980, 253, 655]

    print(f"输入   {ref['prompt']!r}   {ids}")

    # ---- 一：参数量 -----------------------------------------------------------
    ours = model.weights.n_params
    print(f"\n一  参数量    我们{ours:,}   第一章{ref['n_params']:,}   "
          f"{'相同' if ours == ref['n_params'] else '不同'}")

    # ---- 二：31层逐层比 -------------------------------------------------------
    with torch.no_grad():
        states = hidden_states(model, ids)
    print(f"\n二  逐层比   {'下标':>4}{'绝对误差':>12}{'相对误差':>12}")
    worst = 0.0
    for i, (h, r) in enumerate(zip(states, ref["hidden_states"][:, 0])):
        abs_err = (h - r).abs().max().item()
        rel = abs_err / r.abs().max().item()
        worst = max(worst, rel)
        if i % 5 == 0 or i == 29:
            print(f"            {i:>4}{abs_err:>14.1e}{rel:>14.1e}")
    print(f"            31层里最大的相对误差   {worst:.1e}")

    # ---- 三：argmax ----------------------------------------------------------
    logits = model.to_logits(states[-1])               # (T, 49152)
    ours = logits.argmax(-1).tolist()
    theirs = ref["logits"][0].argmax(-1).tolist()
    print(f"\n三  argmax   我们{ours}   第一章{theirs}   "
          f"{'相同' if ours == theirs else '不同'}")

    # ---- 四：贪婪解码 ---------------------------------------------------------
    out = engine.generate(ids)
    theirs = ref["greedy"][0].tolist()
    text = engine.tok.decode(out)
    print(f"\n四  贪婪解码")
    print(f"    我们     {text!r}")
    print(f"    第一章   {ref['greedy_text']!r}")
    print(f"    ID{'相同' if out == theirs else '不同'}，"
          f"文字{'相同' if text == ref['greedy_text'] else '不同'}")


if __name__ == "__main__":
    main()
