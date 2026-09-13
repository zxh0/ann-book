"""Step 9 — sampling: temperature, top-k, top-p.

Run:  uv run python steps/09_sampling.py

This is the first step whose output is not reproducible token-for-token against
transformers, so the verification strategy has to change. The filters are pure
functions and get compared exactly; the draw is checked statistically.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.model import SmolLM2
from toyllm.sampling import GREEDY, Sampler, apply_temperature, apply_top_k, apply_top_p
from toyllm.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def bar(p, width=28):
    return "#" * int(round(p * width))


def show(tok, tid):
    """Quoted token text — many top tokens are whitespace and would print blank."""
    return repr(tok.decode([int(tid)]))


def main():
    model = SmolLM2.from_pretrained(MODEL_DIR)
    tok = BPETokenizer.from_file(MODEL_DIR)
    prompt = "The capital of France is"
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        logits = model(ids)[0, -1]

    rule("1. the distribution greedy decoding throws away")
    probs = logits.softmax(-1)
    top = probs.topk(8)
    print(f"  after {prompt!r}:\n")
    for p, tid in zip(top.values, top.indices):
        print(f"    {show(tok, tid):<14}{p:>7.3f}  {bar(p.item())}")
    print(f"\n  Greedy always takes {show(tok, probs.argmax())}. Everything below it is discarded,")
    print("  every time — which is why greedy decoding loops so readily. Step 7's")
    print("  output repeated 'the capital of the country' twice in 20 tokens.")

    rule("2. temperature reshapes the gaps")
    print(f"  {'token':<14}" + "".join(f"{f'T={t}':>10}" for t in (0.5, 0.8, 1.0, 1.5, 3.0)))
    ids8 = top.indices
    for tid in ids8[:6]:
        row = f"  {show(tok, tid):<14}"
        for t in (0.5, 0.8, 1.0, 1.5, 3.0):
            row += f"{apply_temperature(logits, t).softmax(-1)[tid]:>10.3f}"
        print(row)
    print("\n  T is a divisor, so the effect is asymmetric: T=0.5 doubles every logit")
    print("  gap, T=2 halves it. Low T concentrates mass on the leader; high T")
    print("  spreads it toward uniform. T -> 0 is exactly greedy, which is why")
    print("  Sampler.build(temperature=0) hands back a GreedySampler instead of")
    print("  attempting a division by zero.")

    rule("3. top-k is fixed, top-p adapts")
    print(f"  {'setting':<16}{'tokens kept':>13}{'mass kept':>12}")
    for k in (1, 5, 40, 200):
        kept = torch.isfinite(apply_top_k(logits, k))
        print(f"  {f'top_k={k}':<16}{int(kept.sum()):>13}{probs[kept].sum():>12.4f}")
    for p in (0.5, 0.9, 0.95, 0.99):
        kept = torch.isfinite(apply_top_p(logits, p))
        print(f"  {f'top_p={p}':<16}{int(kept.sum()):>13}{probs[kept].sum():>12.4f}")
    print(f"\n  (vocabulary is {logits.shape[0]:,} tokens)")
    print("\n  top-k keeps a fixed count regardless of how confident the model is.")
    print("  top-p keeps a fixed *mass*, so the count adapts to the distribution:")

    flat = torch.zeros(50)
    peaked = torch.tensor([12.0] + [1.0] * 49)
    for label, l in (("peaked", peaked), ("flat", flat)):
        print(f"    {label:<8} distribution, top_p=0.9 keeps "
              f"{int(torch.isfinite(apply_top_p(l.unsqueeze(0), 0.9)).sum()):>3} of 50 tokens")

    rule("4. the boundary token, and why the reference matters")
    demo = torch.tensor([[0.6, 0.3, 0.1]]).log()
    kept = int(torch.isfinite(apply_top_p(demo, 0.7)).sum())
    print("  probabilities 0.6 / 0.3 / 0.1, top_p = 0.7")
    print(f"    tokens kept: {kept}")
    print("\n  0.6 alone does not reach 0.7, so the second token is needed and the")
    print("  nucleus is {0.6, 0.3} = 0.9 of the mass. The tempting implementation —")
    print("  sort descending, stop once the running sum exceeds p — drops that")
    print("  second token and keeps only 0.6. Both are self-consistent; only one")
    print("  matches transformers. We follow HF: sort ascending, remove while the")
    print("  cumulative mass is still below 1 - p.")

    rule("5. filter order is load-bearing")
    a = apply_top_p(apply_temperature(logits, 0.5), 0.9)
    b = apply_temperature(apply_top_p(logits, 0.9), 0.5)
    print(f"  temperature then top_p:  {int(torch.isfinite(a).sum()):>4} tokens kept")
    print(f"  top_p then temperature:  {int(torch.isfinite(b).sum()):>4} tokens kept")
    print("\n  Temperature rescales the gaps, so running it first changes which")
    print("  tokens the nucleus contains — not just how often each one wins.")
    print("  transformers applies temperature -> top-k -> top-p; so do we.")

    rule("6. verifying a random process")
    print("  The filters are pure functions of the logits, so they are compared to")
    print("  HF's warpers with torch.equal — see tests/test_sampling.py.")
    print("  The draw itself cannot be. Instead: fix a seed for reproducibility,")
    print("  and check that empirical frequencies converge to the target.\n")

    small = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    target = small.softmax(-1)[0]
    sampler = Sampler.build(temperature=1.0, seed=0)
    n = 40_000
    counts = torch.zeros(4)
    counts.scatter_add_(0, sampler(small.expand(n, -1)).flatten(), torch.ones(n))
    print(f"  {n:,} draws from softmax([2, 1, 0, -1]):")
    print(f"    {'token':>7}{'target':>10}{'empirical':>12}{'error':>10}")
    for i in range(4):
        e = (counts[i] / n).item()
        print(f"    {i:>7}{target[i]:>10.4f}{e:>12.4f}{abs(e - target[i].item()):>10.4f}")

    rule("7. what it sounds like")
    settings = [
        ("greedy", GREEDY),
        ("T=0.7, top_p=0.9", Sampler.build(temperature=0.7, top_p=0.9, seed=0)),
        ("T=1.0, top_k=50", Sampler.build(temperature=1.0, top_k=50, seed=0)),
        ("T=1.5 (no filter)", Sampler.build(temperature=1.5, seed=0)),
    ]
    for label, s in settings:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, sampler=s)
        text = tok.decode(out[0].tolist()).replace("\n", "\\n")
        print(f"  {label:<20} {text!r}")
    print("\n  Greedy loops. Moderate settings stay coherent. High temperature with")
    print("  no top-k/top-p lets the tail through and the text comes apart — that")
    print("  tail is tens of thousands of tokens, each individually unlikely but")
    print("  collectively probable enough to be picked.")

    rule("8. greedy and random are separate classes")
    print("  Greedy is not a flag on the sampler — it is its own implementation,")
    print("  so no `if greedy` branch runs once per generated token:")
    print(f"\n    Sampler.build(temperature=0.0)  -> {Sampler.build(temperature=0.0)!r}")
    print(f"    Sampler.build(temperature=0.7)  -> {Sampler.build(temperature=0.7, top_p=0.9)!r}")
    print("\n  The decision happens once, at construction. generate() just calls")
    print("  sampler(logits) and never asks which kind it has. Anything with a")
    print("  __call__(logits) -> (B, 1) works, so a custom strategy needs no")
    print("  changes to the model at all.")

    rule("9. reproducibility")
    a = model.generate(ids, 12, sampler=Sampler.build(temperature=0.9, top_p=0.95, seed=42))
    b = model.generate(ids, 12, sampler=Sampler.build(temperature=0.9, top_p=0.95, seed=42))
    c = model.generate(ids, 12, sampler=Sampler.build(temperature=0.9, top_p=0.95, seed=43))
    print(f"  same seed  -> identical: {torch.equal(a, b)}")
    print(f"  other seed -> identical: {torch.equal(a, c)}")
    print("\n  Each Sampler owns a torch.Generator, so sampling never touches global")
    print("  RNG state — a seeded run stays reproducible no matter what else in the")
    print("  process draws random numbers.")


if __name__ == "__main__":
    main()
