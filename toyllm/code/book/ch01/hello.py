"""第一章 · 先跑起来

用 HuggingFace transformers 加载 SmolLM2-135M 并生成一段文字。
这一章不写自己的代码——先看到终点长什么样，后面十章再一件一件自己造。

运行：
    cd code
    uv run python book/ch01/hello.py
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# code/book/ch01/hello.py → 上三层就是 code/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "SmolLM2-135M"

PROMPT = "Once upon a time"


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    rule("1. 整本书要手写的东西，用别人的库只要五行")

    print("""    tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    ids   = tok(PROMPT, return_tensors="pt")
    out   = model.generate(**ids, max_new_tokens=20, do_sample=False)
    text  = tok.decode(out[0])""")

    print("\n  这五行就是本书后面十章要做的全部事情。")
    print("  它们单独存了一份，可以直接跑：uv run python book/ch01/five_lines.py")
    print("\n  先把它跑一遍，看看终点长什么样。")

    # ---- 加载 ---------------------------------------------------------------
    # dtype=torch.float32：权重在文件里存的是 bfloat16，这里升到 float32。
    # 原因有两个：CPU 上 float32 的算子更快也更全；而且它是全书数值对拍的基准。
    rule("2. 加载模型")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()  # 推理模式：关掉 dropout 之类只在训练时起作用的东西
    load_s = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  耗时       {load_s:.1f}s")
    print(f"  架构       {model.config.architectures[0]}")
    print(f"  参数量     {n_params:,}")
    print(f"  层数       {model.config.num_hidden_layers}")
    print(f"  隐层维度   {model.config.hidden_size}")
    print(f"  词表大小   {model.config.vocab_size}")
    print("\n  记住 134,515,008 这个数字，第二章我们要自己把它算出来。")

    # ---- 结构 ---------------------------------------------------------------
    rule("3. 它长什么样")

    print(model)

    print("\n  对照第一章那张总览图看这份结构：")
    print("    embed_tokens        入口，token id → 576 维向量")
    print("    layers.0 … layers.29   30 个一模一样的 Decoder Block")
    print("      ├─ input_layernorm         ┐")
    print("      ├─ self_attn (q/k/v/o_proj) ├─ 这四行就是第五到第八章")
    print("      ├─ post_attention_layernorm │")
    print("      └─ mlp (gate/up/down_proj)  ┘")
    print("    norm                最后一个 RMSNorm")
    print("    lm_head             出口，576 维 → 49152 个词的分数")
    print("\n  注意 lm_head 的形状和 embed_tokens 一模一样——它们其实是同一个矩阵，")
    print("  第二章会发现权重文件里根本没存 lm_head，第四章解释为什么。")

    # ---- 生成 ---------------------------------------------------------------
    # do_sample=False 就是 greedy：每一步都取分数最高的词，不掷骰子。
    # 这样输出是确定的，才能当基准。第十章会把这里换成真正的采样。
    rule("4. 让它说话")

    ids = tok(PROMPT, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    gen_s = time.perf_counter() - t0

    n_new = out.shape[1] - ids["input_ids"].shape[1]
    print(f"  输入   {PROMPT!r}")
    print(f"  输出   {tok.decode(out[0])!r}")
    print(f"\n  {n_new} 个 token 用了 {gen_s:.2f}s（{n_new / gen_s:.1f} tok/s，greedy，fp32 CPU）")

    # ---- 中间那一步 ----------------------------------------------------------
    rule("5. 生成的那一刻，模型在想什么")

    with torch.no_grad():
        logits = model(**ids).logits

    print(f"  logits 形状 {tuple(logits.shape)}   (批, 序列长度, 词表大小)")
    print(f"  只有最后一个位置有用——它预测的是 {PROMPT!r} 的下一个词。\n")

    top = logits[0, -1].topk(8)
    for score, tid in zip(top.values, top.indices):
        print(f"    {tok.decode([tid])!r:<14} {score:+.3f}")

    print("\n  这 49152 个分数叫 logits，是模型的原始输出。")
    print("  取最大值就是 greedy；按概率抽一个就是采样——那是第十章。")

    print("\n  接下来：运行 baseline.py 把这次的结果存成文件。")
    print("  从第二章开始，我们自己写的每一个部件都要回来和它对。")


if __name__ == "__main__":
    main()
