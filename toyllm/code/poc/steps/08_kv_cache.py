"""Step 8 — the KV cache: same output, different complexity.

Run:  uv run python steps/08_kv_cache.py

The one invariant for this step: every generated token must be identical to
step 7's. Only the cost changes.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from toyllm.layers import NO_CACHE, KVCache, causal_mask
from toyllm.model import SmolLM2
from toyllm.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent          # code/poc
MODEL_DIR = ROOT.parent / "models" / "SmolLM2-135M"    # code/models


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    model = SmolLM2.from_pretrained(MODEL_DIR)
    cfg = model.cfg
    tok = BPETokenizer.from_file(MODEL_DIR)
    prompt = "The capital of France is"
    ids = torch.tensor([tok.encode(prompt)])

    rule("1. the work being repeated")
    print("  Attention is causal, so the key and value of token 3 never depend on")
    print("  tokens 4, 5, 6... They are the same tensors at every decode step.")
    print("  Step 7 recomputed all of them every time:")
    print(f"\n  {'new tokens':>12}{'token-forwards without cache':>32}{'with cache':>14}")
    for n in (20, 100, 500):
        p = ids.shape[1]
        print(f"  {n:>12}{sum(range(p, p + n)):>32,}{n:>14,}")
    print("\n  Quadratic against linear. Everything below is about that table.")

    rule("2. what gets stored")
    cache = KVCache(cfg, max_seq_len=512, batch_size=1)
    print(f"  one tensor per layer per side: {len(cache.k)} layers x (k, v)")
    print(f"  each shaped {tuple(cache.k[0].shape)}")
    print(f"              (batch, kv_heads, max_seq, head_dim)")
    print(f"\n  Note kv_heads = {cfg.num_key_value_heads}, not {cfg.num_attention_heads}. We cache before repeat_kv,")
    print(f"  so GQA makes the cache {cfg.n_rep}x smaller — that is what GQA is for.")
    print(f"\n  {'context':>9}{'GQA cache':>13}{'if MHA':>13}")
    for ctx in (512, 8192):
        c = KVCache(cfg, max_seq_len=ctx)
        print(f"  {ctx:>9}{c.memory_bytes() / 1e6:>10.1f} MB{c.memory_bytes() * cfg.n_rep / 1e6:>10.1f} MB")

    rule("3. rotate first, store second")
    print("  The cache holds keys that already went through RoPE. Position is")
    print("  baked in at write time, so the history is never re-rotated.")
    print("  Storing pre-rotation keys would mean rotating the whole cache every")
    print("  step — which is exactly the O(n) per-step work we are removing.")

    rule("4. positions and the mask during decode")
    print("  Two things must continue from the cache, not restart:")
    print(f"\n    positions:  after {ids.shape[1]} prompt tokens the next one is at "
          f"position {ids.shape[1]}, not 0")
    print("                (restart it and RoPE silently encodes the wrong distances)")
    m = causal_mask(1, kv_len=6, offset=5)
    print(f"\n    mask:       one new query, {m.shape[-1]} cached keys -> shape {tuple(m.shape)}")
    print(f"                all zeros ({bool((m == 0).all())}), because a new token may see all history")
    print("                so we skip building it entirely when T == 1")
    chunk = causal_mask(2, kv_len=5, offset=3)[0, 0]
    print(f"\n    for a 2-token chunk on top of 3 cached, the mask is still needed:")
    for i, row in enumerate(chunk):
        cells = " ".join("   0" if v == 0 else "-inf" for v in row)
        print(f"                pos {3 + i}: [{cells}]")

    rule("5. prefill and decode are two different problems")
    cache = KVCache(cfg, max_seq_len=64)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(ids, cache=cache)
    prefill_s = time.perf_counter() - t0
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(nxt, cache=cache)
    decode_s = time.perf_counter() - t0
    print(f"  {f'prefill {ids.shape[1]} tokens in one pass':<32}{prefill_s * 1000:>7.1f} ms  "
          f"({ids.shape[1] / prefill_s:>4.0f} tok/s)")
    print(f"  {'decode 1 token':<32}{decode_s * 1000:>7.1f} ms  ({1 / decode_s:>4.0f} tok/s)")
    print(f"  cache now holds {cache.length} tokens")
    print("\n  Prefill is compute-bound: big matmuls over the whole prompt at once.")
    print("  Decode is memory-bound: it reads all 134M weights to advance one token,")
    print("  and the arithmetic per weight is tiny. Serving systems schedule the two")
    print("  phases separately precisely because they bottleneck on different things.")

    rule("6. the cache is a switch, not a second code path")
    print("  NoCache is a null object with the same interface: update() hands back")
    print("  what it was given, length stays 0. So Attention has no `if cache` in it,")
    print("  and every position/mask calculation collapses to step 7's behaviour.")
    print(f"\n    NO_CACHE.update(0, k, v) returns k, v unchanged: "
          f"{NO_CACHE.update(0, ids, ids) == (ids, ids)}")
    print(f"    NO_CACHE.length = {NO_CACHE.length}, memory = {NO_CACHE.memory_bytes()} bytes")
    print("\n  The decode loop is then one loop. The only line that differs:")
    print("      feed = next_id if cache else ids")
    print("  With a cache, feed the single new token. Without, feed everything.")
    print("  That one expression is the whole difference between O(n) and O(n^2).")
    print("\n  generate(cache=...) takes True, False, or an existing KVCache —")
    print("  the last one lets you continue a conversation from a warm cache.")

    warm = model.new_cache(64)
    with torch.no_grad():
        turn1 = model.generate(ids, max_new_tokens=5, cache=warm)
        turn2 = model.generate(turn1[:, -1:], max_new_tokens=5, cache=warm)
        one_go = model.generate(ids, max_new_tokens=10, cache=True)
    print(f"\n    two turns over one warm cache == one 10-token run: "
          f"{torch.equal(torch.cat([turn1, turn2[:, 1:]], 1), one_go)}")
    print(f"    (after turn 1 the cache holds {warm.length - 5} of {turn1.shape[1]} tokens — the last")
    print("     generated token was never fed back in, so it is not stored yet)")

    rule("7. correctness: the output must not move")
    with torch.no_grad():
        cached = model.generate(ids, max_new_tokens=20, cache=True)
        naive = model.generate(ids, max_new_tokens=20, cache=False)
    print(f"  {tok.decode(cached[0].tolist())!r}")
    print(f"\n  cached == uncached (step 7):  {torch.equal(cached, naive)}")
    ref_path = ROOT.parent / "reference" / "smollm2_135m_forward.pt"
    if ref_path.exists():
        ref = torch.load(ref_path, weights_only=True)
        print(f"  cached == transformers:       {torch.equal(cached, ref['greedy_20'])}")
    print("\n  Token-for-token identical, not 'close'. Decoding is a chain of argmax")
    print("  decisions, so a real bug would derail the text, not perturb it slightly.")

    rule("8. the scaling curve")
    print(f"  {'tokens':>8}{'no cache':>12}{'cached':>12}{'no cache':>13}{'cached':>11}{'speedup':>10}")
    print(f"  {'':>8}{'(s)':>12}{'(s)':>12}{'(tok/s)':>13}{'(tok/s)':>11}")
    for n in (20, 50, 100):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=n, cache=False)
        slow = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(ids, max_new_tokens=n, cache=True)
        fast = time.perf_counter() - t0
        print(f"  {n:>8}{slow:>12.2f}{fast:>12.2f}{n / slow:>13.1f}{n / fast:>11.1f}{slow / fast:>9.1f}x")
    print("\n  Read the tok/s columns, not the speedup. Cached throughput stays flat —")
    print("  that is the linear cost. Uncached throughput falls as the sequence grows —")
    print("  that is the quadratic one. The ratio keeps widening with length; at these")
    print("  toy lengths a 135M model on CPU barely shows it.")


if __name__ == "__main__":
    main()
