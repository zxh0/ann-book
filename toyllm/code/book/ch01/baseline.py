"""第一章 · 生成全书的基准

用 transformers 跑一遍 SmolLM2-135M，把中间结果和最终输出存成文件。
从第二章起，我们自己写的每一个部件都要回来和这份文件对拍。

**这份文件是全书的地基。** 删了就得重跑，而且要用同样的版本重跑——
不同版本的 transformers 可能给出不同的数值，后面每一章的对拍就会全部失败。
所以文件里连版本号一起存了。

运行：
    cd code
    uv run python book/ch01/baseline.py
"""

import torch
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# code/book/ch01/baseline.py → 上三层就是 code/
CODE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = CODE_DIR / "models" / "SmolLM2-135M"
OUT_PATH = CODE_DIR / "reference" / "baseline.pt"

PROMPT = "Once upon a time"
MAX_NEW_TOKENS = 20


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    rule("1. 加载")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  参数量        {n_params:,}")

    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    print(f"\n  prompt  {PROMPT!r}")
    print(f"  ids     {ids[0].tolist()}")
    print(f"  tokens  {[tok.decode([i]) for i in ids[0]]}")

    # ---- 前向：留下每一层的中间结果 --------------------------------------------
    # output_hidden_states=True 会把每一层的输出都吐出来。
    # 这是全书最关键的一个开关：有了逐层的中间结果，第八章我们自己算错了，
    # 能一眼看出是从第几层开始跑偏的，而不是只知道"最后的答案不对"。
    rule("2. 前向，记录每一层")

    with torch.no_grad():
        res = model(ids, output_hidden_states=True)

    hidden = torch.stack(res.hidden_states)  # (层数+1, 批, 序列, 隐层)

    print(f"  logits         {tuple(res.logits.shape)}")
    print(f"  hidden_states  {tuple(hidden.shape)}   = 词嵌入输出 + 30 个 block")

    print("\n  hidden_states 的下标约定（第八章对拍时会用到，很容易搞错）：")
    print("    [0]       词嵌入层的输出")
    print("    [1]..[29] 第 i 个 block 的『输入』，也就是第 i-1 个 block 的输出")
    print("    [30]      最后一个 block 的输出，**而且已经过了 final norm**")
    print("    最后一个是特例。拿它去和第 29 个 block 的裸输出比，会看到二十几倍")
    print("    的差距——那个数字毫无意义，纯粹是因为少过了一次 norm。")

    # ---- 生成 --------------------------------------------------------------
    rule("3. greedy 生成")

    with torch.no_grad():
        greedy = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    text = tok.decode(greedy[0])
    print(f"  输出   {text!r}")
    print("\n  用 greedy 而不是采样，是因为基准必须是确定的。")
    print("  采样每次结果都不一样，没法拿来对拍。")

    # ---- 存盘 --------------------------------------------------------------
    rule("4. 存盘")

    payload = {
        "prompt": PROMPT,
        "input_ids": ids,                  # (1, T)
        "logits": res.logits,              # (1, T, 49152)
        "hidden_states": hidden,           # (31, 1, T, 576)
        "greedy": greedy,                  # (1, T + 20)
        "greedy_text": text,
        "n_params": n_params,
        # 版本一起存：换了版本数值可能对不上，到时候能立刻看出是环境问题
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    torch.save(payload, OUT_PATH)
    print(f"  已写入 {OUT_PATH.relative_to(CODE_DIR)}")

    for k, v in payload.items():
        shape = tuple(v.shape) if torch.is_tensor(v) else repr(v)
        print(f"    {k:<22} {shape}")

    print("\n  后面每一章怎么用它：")
    print("    第五章  RMSNorm 逐位相同")
    print("    第七章  注意力 + RoPE，相对误差 ~1e-7")
    print("    第九章  31 层逐层对拍，logits 的 argmax 完全一致，")
    print("            greedy 输出和上面这句话逐字节相同 ← 全书最关键的一次")
    print("    第十一章 加了 KV Cache 之后，输出必须一个 token 都不差")


if __name__ == "__main__":
    main()
