"""Step 0 — the three-line version, kept as the oracle.

Run:  uv run python steps/00_reference.py

This is what the whole project would collapse to if the goal were merely to run
SmolLM2. It is here for exactly one reason: to be the ground truth every later
step is diffed against. `toyllm/` never imports transformers.

Running it writes reference tensors to `reference/` so later steps can compare
without needing transformers loaded.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models
REF_DIR = ROOT.parent / "reference"                    # code/reference

PROMPT = "The capital of France is"


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed — run: uv sync --group dev")
        return

    rule("1. everything this project does by hand, in three lines")
    print("""    tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    out   = model.generate(**tok(PROMPT, return_tensors="pt"), max_new_tokens=10)""")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"\n  loaded in {load_s:.1f}s — {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  architecture: {model.config.architectures[0]}")

    rule("2. baseline generation")
    ids = tok(PROMPT, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)
    gen_s = time.perf_counter() - t0
    text = tok.decode(out[0])
    n_new = out.shape[1] - ids["input_ids"].shape[1]
    print(f"  prompt: {PROMPT!r}")
    print(f"  output: {text!r}")
    print(f"  {n_new} tokens in {gen_s:.2f}s  ({n_new / gen_s:.1f} tok/s, greedy, fp32 CPU)")

    rule("3. saving reference tensors for later steps")
    REF_DIR.mkdir(exist_ok=True)
    with torch.no_grad():
        res = model(**ids, output_hidden_states=True)

    payload = {
        "prompt": PROMPT,
        "input_ids": ids["input_ids"],
        "logits": res.logits,
        "hidden_states": torch.stack(res.hidden_states),  # (n_layers+1, B, T, H)
        "greedy_20": out,
    }
    path = REF_DIR / "smollm2_135m_forward.pt"
    torch.save(payload, path)

    print(f"  wrote {path.relative_to(ROOT.parent)}")
    print(f"    input_ids     {tuple(ids['input_ids'].shape)}")
    print(f"    logits        {tuple(res.logits.shape)}")
    print(f"    hidden_states {tuple(payload['hidden_states'].shape)}  (embedding output + 30 blocks)")
    print(f"\n  Step 6 will assert our own forward pass reproduces `logits` to within")
    print(f"  a tight tolerance, and `hidden_states` lets us find the *first* layer")
    print(f"  where we diverge instead of only seeing that the final answer is wrong.")

    top = res.logits[0, -1].topk(5)
    print(f"\n  next-token distribution after {PROMPT!r}:")
    for score, tid in zip(top.values, top.indices):
        print(f"    {tok.decode([tid]):<12} logit {score:+.3f}")


if __name__ == "__main__":
    main()
