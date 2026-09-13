"""Step 2 — byte-level BPE, implemented by hand and checked against the real thing.

Run:  uv run python steps/02_tokenizer.py

Everything here goes through toyllm.tokenizer, our own implementation. The
`tokenizers` library appears exactly once, at the end, as the oracle we diff
against — never as the thing doing the work.
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.tokenizer import BPETokenizer, bytes_to_unicode

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    tok = BPETokenizer.from_file(MODEL_DIR)

    rule("1. the byte <-> unicode map")
    enc = bytes_to_unicode()
    print("  256 bytes, each given a printable stand-in so whitespace survives JSON:")
    for b in [0x20, 0x0A, 0x09, 0x41, 0x7F, 0xC3]:
        print(f"    byte 0x{b:02X} {repr(chr(b)):>8}  ->  {enc[b]!r}")
    print(f"\n  This is why the vocabulary is full of 'Ġ' (space) and 'Ċ' (newline).")
    print(f"  The unit is a byte, so there is no <unk> token and no OOV *word*")
    print(f"  (unk_token: null in tokenizer.json).")
    print(f"\n  But 'byte-level means anything is encodable' is a myth here:")
    missing = tok.unrepresentable_bytes
    dead = [b for b in missing if b in (0xC0, 0xC1) or b >= 0xF5]
    live = [b for b in missing if b not in dead]
    print(f"    {len(missing)} of 256 byte tokens are absent from SmolLM2's vocab.")
    print(f"    {len(dead)} are harmless — {[hex(b) for b in dead]}")
    print(f"      cannot occur in valid UTF-8 (overlong forms / beyond U+10FFFF).")
    print(f"    {len(live)} are reachable from an ordinary Python string:")
    print(f"      {[hex(b) for b in live]}")
    print(f"      control chars, plus the lead bytes of planes 4-7. They get DROPPED:")
    print(f"        'a\\x04b'  -> {tok.tokens(tok.encode('a\x04b'))}, decodes to {tok.decode(tok.encode('a\x04b'))!r}")
    s = "a" + chr(0x50000) + "b"
    print(f"        'a\\U00050000b' -> decodes to {tok.decode(tok.encode(s))!r}")
    print(f"      HF does exactly the same, silently. We match it rather than 'fix' it,")
    print(f"      because diverging would produce ids the model never saw in training.")

    rule("2. pre-tokenization: split before merging")
    for s in ["Hello world!", "  indented\n", "GPT-4 costs $20", "第 3 章"]:
        print(f"    {s!r:<24} -> {tok.pre_tokenize(s)}")
    print("\n  Note ' world' keeps its leading space as one piece ('Ġworld'), and that")
    print("  digits are always split apart — '20' can never become a single token.")

    rule("3. BPE: replaying the learned merges")
    word = " tokenizer"
    piece = tok.pre_tokenize(word)[0]
    print(f"  encoding {word!r}, byte-encoded to {piece!r}\n")
    print(f"    start                     {list(piece)}")
    for (a, b), rank, syms in tok.bpe_steps(piece):
        print(f"    merge {a!r}+{b!r} (rank {rank:>5})".ljust(46) + f"{syms}")
    print(f"\n  Merges are applied by *lowest rank first*, not left to right.")
    print(f"  The rank ordering is the trained artifact — vocab alone can't tokenize.")

    rule("4. encode / decode round-trip")
    for s in ["The capital of France is Paris.", "第三章：注意力机制", "x = [i**2 for i in range(10)]"]:
        ids = tok.encode(s)
        back = tok.decode(ids)
        print(f"    {s!r}")
        print(f"      ids    {ids}")
        print(f"      tokens {tok.tokens(ids)}")
        print(f"      decode {back!r}  {'OK' if back == s else 'MISMATCH'}\n")

    rule("5. special tokens")
    print(f"  {len(tok.special_tokens)} special tokens, ids 0-16:")
    print(f"    {list(tok.special_tokens)}")
    s = "<|im_start|>user\nhi<|im_end|>"
    print(f"\n    {s!r}")
    print(f"      with specials    {tok.encode(s)}")
    print(f"      tokens           {tok.tokens(tok.encode(s))}")
    print(f"      as literal text  {tok.encode(s, allow_special=False)}")
    print("\n  SmolLM2's post_processor is null: no BOS/EOS is added for you.")
    print(f"  When we generate, we stop on eos id {tok.special_tokens['<|endoftext|>']} ('<|endoftext|>').")

    rule("6. how much text fits in a token")
    samples = {
        "English prose": "The quick brown fox jumps over the lazy dog and keeps running.",
        "Python code": "def forward(self, x):\n    return self.w2(F.silu(self.w1(x)))\n",
        "Chinese": "注意力机制是Transformer架构的核心，它让模型能够关注序列中的任意位置。",
        "Digits": "3.14159265358979323846",
    }
    rates = {}
    for label, text in samples.items():
        ids = tok.encode(text)
        rates[label] = len(text) / len(ids)
        print(f"    {label:<15} {len(text):>4} chars -> {len(ids):>4} tokens   {rates[label]:.2f} chars/token")
    ratio = rates["English prose"] / rates["Chinese"]
    print(f"\n  Chinese costs ~{ratio:.0f}x more tokens per character than English here:")
    print("  each CJK char is 3 UTF-8 bytes, and the merges were learned mostly on")
    print("  English/code, so most CJK characters don't even survive as one token —")
    print("  look at step 4, where '第' came out as two tokens ('ç¬' + '¬').")
    print("  Digits are one token each by construction, so numeric text is fixed at 1.00.")
    print("  Practical consequence: a Chinese prompt eats context budget several times")
    print("  faster than the same content in English.")

    rule("7. verify against the reference implementation")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("  `tokenizers` not installed — skipping the differential check")
        return
    ref = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    paths = sorted(
        p
        for d in ("toyllm", "steps", "tests")
        for p in (ROOT / d).glob("*.py")
    ) + sorted(ROOT.parent.glob("*.md"))
    corpus = [p.read_text() for p in paths]
    corpus += list(samples.values()) + ["", " ", "\n\n", "½ Ⅷ 一二三", "🤖" * 5]

    total, bad = 0, 0
    for text in corpus:
        ours = tok.encode(text)
        theirs = ref.encode(text, add_special_tokens=False).ids
        total += len(ours)
        if ours != theirs or tok.decode(ours) != text:
            bad += 1
    print(f"  {len(corpus)} documents, {total:,} tokens")
    print(f"  {'✓ byte-identical to `tokenizers`, and every round-trip is exact' if bad == 0 else f'✗ {bad} documents differ'}")

    print("\n  Two Unicode traps this had to get right (both fail silently):")
    print("    '½'  category No -> IS numeric, so it splits.  Python isdigit() says False.")
    print("    '一' category Lo -> NOT numeric, no split.     Python isnumeric() says True.")
    print("    The faithful test is unicodedata.category(ch)[0] == 'N'.")
    for ch in "½一Ⅷ²":
        print(f"      {ch!r}  category={unicodedata.category(ch)}  isdigit={ch.isdigit()!s:<5} "
              f"isnumeric={ch.isnumeric()!s:<5} -> split={unicodedata.category(ch)[0] == 'N'}")


if __name__ == "__main__":
    main()
