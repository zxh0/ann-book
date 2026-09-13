# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **toy LLM inference engine written from scratch in Python + PyTorch**, built up one step at a time. The end goal is to load real [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2) weights and generate text with it.

This directory is a single uv project holding **two code trees** that share one venv, one `models/`, and one `reference/`:

- **`poc/`** — the original proof of concept. Working, tested, complete through sampling. **Frozen reference, not the book's code**: it is the correctness oracle and the notebook of hard-won facts recorded below. Do not extend it.
- **`book/`** — the code that ships with the book, written from scratch to the author's own structure: **one directory per chapter** (`ch01/` … `ch12/`), each copied from the previous one and extended, each runnable on its own. New work goes here.

Everything below about the environment, SmolLM2's verified config values, and the tokenizer / RoPE / KV-cache gotchas applies to both.

The point of the project is the *learning path*, not the performance. `transformers` modeling code is deliberately not used — tokenization, weight loading, RMSNorm / RoPE / GQA / SwiGLU, the KV cache, sampling, and the generate loop are all hand-written on top of plain `torch` tensor ops. Only `torch`, `safetensors`, `tokenizers`, and `huggingface-hub` are dependencies, and each exists to avoid re-implementing something that is *not* the lesson (tensor math, file format, BPE merge tables, downloads).

This project lives inside the author's personal notes repo (git root `/Users/matrix/me/github/notes`); work happens on the `dev` branch, `master` is the main branch. It is a sibling of `../../ann4us` (the manuscript of an earlier Chinese-language book on neural networks) and follows the same uv + Python 3.12 convention as `../../ann4us/code`.

## Environment Constraints — read before touching dependencies

This machine is an **Intel (x86_64) Mac**, and that dictates several pins:

- **`torch` is pinned to `<2.3` (resolves to 2.2.2).** PyTorch stopped publishing macOS x86_64 wheels with 2.3 — verified: 2.3.0/2.4.1/2.5.1/2.7.0/2.14.0 all offer `macosx_*_arm64` only. This is a *platform* limit, not a CUDA one: macOS wheels have never contained CUDA, so `--index-url .../whl/cpu` changes nothing here. Do not "upgrade" this pin; nothing newer can be installed on this machine.
- **`transformers` must stay `<5`** (resolves to 4.57.6). transformers 5.x requires torch>=2.5 and, when it doesn't find it, *silently disables its PyTorch backend* and then dies with `NameError: name 'nn' is not defined` deep inside. The `llama_ref` fixture in `poc/tests/conftest.py` skips on any exception for exactly this reason. Pinning `<5` also forces `tokenizers` down to 0.22.2 and `huggingface-hub` to 0.36.2 — harmless, since both are dev-only.
- **`numpy` is pinned to `<2`** (resolves to 1.26.4) because torch 2.2.2 was built against the numpy 1.x ABI.
- **No CUDA.** `torch.backends.mps.is_available()` does return `True` here, and MPS results are numerically correct, but it is *not* faster than CPU at toy sizes (measured: 512×512 matmul ×20 → MPS 0.031s vs CPU 0.013s). **Default to CPU**; only reach for MPS with a benchmark in hand.
- `torch.get_num_threads()` is 6.

**Bypass Claude Code's proxy for any large download.** Claude Code sets `HTTPS_PROXY=http://127.0.0.1:9000`, an allowlisting proxy that rejects huggingface.co outright (`Proxy CONNECT aborted`) and throttles even allowed hosts badly — an 11 MB wheel took over ten minutes through it and seconds without. Unsetting it works from inside the sandbox; no need to disable the sandbox. This applies to `uv` too:

```bash
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy uv sync --group dev
```

For weights:

```bash
env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy \
  curl -sSL -o models/SmolLM2-135M/model.safetensors \
  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
```

The same `env -u` prefix is needed for `hf download` or any `huggingface_hub` call. `hf-mirror.com` also works if HF is ever unreachable for real (use `/resolve/main/`, not `/raw/`, which 308s).

## Commands

```bash
uv sync                          # create .venv, install locked deps
uv sync --group dev              # + pytest

uv run python poc/steps/NN_name.py   # run one PoC milestone script
uv run pytest                        # all tests (testpaths = poc/tests)
uv run pytest poc/tests/test_rope.py::test_matches_reference -x   # a single test
```

Always go through `uv run` — the system `python3` is 3.9 and has no torch.

## Layout

| Path | Role |
|------|------|
| `models/` | downloaded weights — **shared by both trees**, gitignored, never commit (`.gitkeep` only) |
| `reference/` | reference tensors from `poc/steps/00_reference.py` — **shared**, gitignored, regenerable |
| `pyproject.toml`, `uv.lock`, `.venv/` | one uv project for everything; `pythonpath = ["poc"]`, `testpaths = ["poc/tests"]` |
| `book/` | the book's code, one directory per chapter (`ch01/` … `ch12/`) |
| `poc/toyllm/` | PoC engine package — the reusable, cumulative implementation |
| `poc/steps/` | one runnable script per PoC milestone (`NN_topic.py`), each printing something observable |
| `poc/steps/00_reference.py` | the 3-line `transformers` version — the oracle; writes `../reference/*.pt` |
| `poc/tests/` | pytest; the important ones assert numerical agreement with a reference |

`models/` and `reference/` sit **above** both trees, so code inside `poc/` reaches them with `ROOT.parent / "models"` where `ROOT` is `poc/`. Chapter directories under `book/` do the same with `Path(__file__).parent.parent.parent / "models"`. Never copy the 269 MB checkpoint into a chapter directory.

Inside `poc/`, the two-layer split matters: `steps/` scripts are the narrative (each is self-contained enough to read top-to-bottom and shows one idea working), while `toyllm/` is where the code settles once a step is done. A step script imports from `toyllm/` for everything already built and only spells out the *new* piece inline.

`book/` deliberately does **not** copy that layout — each chapter directory is a full, self-contained snapshot so a reader can run any chapter's state directly without `git checkout`. The cost is drift across twelve copies; guard it with a check that files untouched by chapter N are byte-identical to chapter N-1.

## Target Architecture: SmolLM2

SmolLM2 is a **Llama-architecture decoder-only transformer** (`LlamaForCausalLM` in HF terms), which is what makes it a good target — every component is a standard modern building block:

- pre-norm **RMSNorm** (no LayerNorm, no bias terms anywhere)
- **RoPE** rotary position embeddings
- **grouped-query attention** (fewer KV heads than Q heads)
- **SwiGLU** feed-forward (`gate_proj` / `up_proj` / `down_proj`)
- **tied** input/output embeddings on the small variants
- byte-level BPE tokenizer shipped as a single `tokenizer.json`

Sizes are 135M / 360M / 1.7B. **Start with 135M** — it is small enough to run on CPU at interactive speed and to hold entirely in memory as fp32.

Never hardcode these; `ModelConfig.from_json` reads them. Verified values for the checkpoint in `models/SmolLM2-135M/`:

| field | value | derived |
|---|---|---|
| `hidden_size` | 576 | `head_dim` = 576/9 = **64** |
| `num_hidden_layers` | 30 | |
| `num_attention_heads` | 9 | q_proj out = 9×64 = 576 |
| `num_key_value_heads` | 3 | k/v_proj out = 3×64 = **192**, GQA group = **3** |
| `intermediate_size` | 1536 | |
| `vocab_size` | 49152 | |
| `rms_norm_eps` | 1e-5 | |
| `rope_theta` | 100000 | note: not the usual 10000 |
| `max_position_embeddings` | 8192 | |
| `tie_word_embeddings` | true | `lm_head.weight` is absent from the file |
| `torch_dtype` | bfloat16 | we upcast to fp32 on load |

Total: **134,515,008** params in 272 tensors (9 per block + `embed_tokens` + `model.norm`). Note the split — embeddings 21%, MLP 59%, attention only 20%.

### Tokenizer gotchas (all found the hard way in step 2)

The pipeline is `Digits(individual_digits) -> ByteLevel(use_regex)` then BPE; no normalizer, no post-processor. Three things that fail *silently* — wrong token ids, no error:

- The `Digits` split must test **`unicodedata.category(ch)[0] == "N"`**, matching Rust's `char::is_numeric()` (Nd|Nl|No). Python's `isdigit()` is too narrow (misses `½`), `isnumeric()` too wide (accepts `一`, which is category Lo).
- **No post_processor** means no BOS/EOS is added automatically. The caller decides.
- **21 of the 256 byte tokens are missing from the vocab**, 8 of them reachable from ordinary strings (`0x04 0x06 0x13 0x14 0x16 0x1D`, plus `0xF1 0xF2` leading planes 4–7). HF drops unknown symbols silently, so `"a\x04b"` → `["a","b"]`. `BPETokenizer.encode` matches this on purpose — "fixing" it would emit ids the model never trained on. So encoding is **not** lossless in general, despite the usual byte-level BPE claim.

### RoPE conventions (step 5)

`rope_interleaved: false` selects the **half-split** (GPT-NeoX/HF) pairing — dimension `i` rotates with `i + head_dim/2`, implemented by `rotate_half`. The original RoFormer paper pairs *adjacent* dimensions `(0,1), (2,3), …` instead. Both are genuine rotations, so both preserve norms and both generate fluent text; only the reference tells you which one the weights were trained with. Same story for `rope_theta`: 100000 here, not the usual 10000.

Watch out for: `rope_theta` is 100000 (SmolLM2-1.7B uses 130000); every `.weight` is stored `(out_features, in_features)`; `attention_bias` is false so there are **no bias tensors anywhere**; `rope_interleaved` is false, meaning the GPT-NeoX/HF rotate-half convention, not the interleaved pairing from the original RoFormer paper.

## Roadmap

Planned order; adjust as the work goes, but keep the invariant that **every step ends in something runnable**.

1. ~~Weight loading + structure~~ **done** — `poc/toyllm/config.py`, `poc/toyllm/weights.py`, `poc/steps/01_load_and_inspect.py`
2. ~~Tokenizer~~ **done** — `poc/toyllm/tokenizer.py`, `poc/steps/02_tokenizer.py`; hand-written byte-level BPE, verified token-identical to `tokenizers`
3. Layers from scratch, one at a time, each with a test:
   - ~~RMSNorm~~ **done** — `poc/steps/03_rmsnorm.py`; bit-identical to `LlamaRMSNorm`
   - ~~SwiGLU~~ **done** — `poc/steps/04_swiglu.py`; bit-identical to `LlamaMLP`
   - ~~RoPE~~ **done** — `poc/steps/05_rope.py`; `inv_freq` bit-identical, cos/sin within 1 ulp (see Conventions)
4. ~~Attention~~ **done** — `poc/steps/06_attention.py`; GQA + causal mask, agrees with `LlamaAttention` to ~4e-7 relative
5. ~~Full forward + validation + greedy decode~~ **done** — `poc/toyllm/model.py`, `poc/steps/07_forward.py`; all 31 layers within 8.2e-7 relative, logits argmax identical, greedy output byte-identical to `transformers`
6. ~~KV cache~~ **done** — `KVCache` / `NoCache` in `poc/toyllm/layers.py`, `poc/steps/08_kv_cache.py`; prefill/decode split, output token-identical to the uncached path.
7. ~~Sampling~~ **done** — `poc/toyllm/sampling.py`, `poc/steps/09_sampling.py`; filters bit-identical to HF's logits warpers, draw verified statistically
8. Batching and beyond — whatever is interesting once it works

### Verifying sampling

Sampling is the first step whose output cannot be compared to HF token-for-token, so it is split in two. The **filters** (`apply_temperature`/`apply_top_k`/`apply_top_p`) are pure functions on logits and are asserted `torch.equal` against `TemperatureLogitsWarper`/`TopKLogitsWarper`/`TopPLogitsWarper`. The **draw** is checked by seeded reproducibility plus convergence of empirical frequencies to the target distribution.

Two things that are easy to get subtly wrong and match HF on purpose:

- **top-p keeps the boundary token.** HF sorts *ascending* and removes while cumulative mass is below `1 - p`. Sorting descending and cutting once the running sum exceeds `p` drops the token that straddles the threshold — a smaller nucleus, and wrong. With probabilities .6/.3/.1 and p=0.7 the correct answer is 2 tokens.
- **Filter order is temperature → top-k → top-p.** Temperature rescales the gaps, so applying it after top-p changes *which* tokens survive, not just their odds.

### Samplers are classes, not flags

`Sampler` is the base interface — `__call__(logits: (B, vocab)) -> (B, 1)`. `GreedySampler` and `RandomSampler` are separate implementations, so **no `if greedy` branch runs per generated token**; the choice is made once by `Sampler.build(temperature=...)`, which returns `GreedySampler` for `temperature=0` (the T→0 limit, not a division by zero) and `RandomSampler` otherwise. `RandomSampler` refuses `temperature=0` outright rather than special-casing it.

Same principle as `NoCache`: a strategy object instead of a runtime conditional. `generate` just calls `sampler(logits)` — anything implementing `__call__` plugs in without touching the model. `GREEDY` is the shared default instance.

Each `RandomSampler` owns its own `torch.Generator`, so seeded runs never depend on or disturb global RNG state.

### The cache is a switch, not a code path

Caching is a swappable strategy, never a branch. `NoCache` (singleton `NO_CACHE`, the default everywhere) implements the same interface and stores nothing: `update()` returns its arguments untouched and `length` stays 0, so every position and mask calculation collapses to the uncached behaviour. **Do not add `if cache is not None` anywhere** — that is what this design exists to prevent.

`generate(cache=...)` takes `True` (fresh cache), `False` (the O(n²) path), or an existing `KVCache` to continue from a warm one. The decode loop is shared; the only line that differs is `feed = next_id if cache else ids`.

After `generate` returns, the cache holds `len(output) - 1` tokens — the final token came out of the last forward pass and was never fed back in. Continuing therefore means feeding `output[:, -1:]`.

### KV cache invariants

`KVCache.update()` deliberately does **not** advance `length` — all 30 layers write at the same offset, so `SmolLM2.forward` calls `cache.advance(t)` once after the whole stack. Keys are stored **after** RoPE, so history is never re-rotated. Storage is per kv head (3), not query head (9); `repeat_kv` runs on read. When continuing from a cache, `positions` must start at `cache.length`, not 0 — restarting them makes RoPE encode wrong distances with no error. For `T == 1` the mask is all zeros and is skipped entirely.

The test that matters is `test_cached_and_uncached_generate_agree`: decoding is a chain of argmax decisions, so any cache bug derails the text rather than perturbing it. Compare with `torch.equal`, not a tolerance.

### hidden_states indexing

`SmolLM2.forward(output_hidden_states=True)` returns 31 tensors in HF's exact convention, so the two diff index by index: `[0]` is the embedding output, `[1..29]` is the *input* to block i (= output of block i-1), and `[30]` is block 29's output **after the final norm**. That last entry is the odd one out — comparing a raw block-29 output against it shows a ~27x mismatch that means nothing at all.

## Conventions

- **Compare relatively, not absolutely, once RoPE is in the path.** RMSNorm and SwiGLU are bit-identical to HF. Anything downstream of the RoPE table is not: we precompute cos/sin for all 8192 positions while HF computes them per call, and `torch.cos()` differs by one ulp between its vectorized (large-tensor) and scalar (small-tensor) kernels. The angles are bit-identical; only the cosine differs. That floor is ~1e-7 *relative* and grows in absolute terms with activation magnitude — at T=128 attention outputs reach ~40 and the absolute gap ~1.5e-5. Assert `max|diff| / max|out| < 1e-5`, never a bare `atol`. Real mistakes (tiled `repeat_kv`, interleaved RoPE) sit around 1e0 relative, so there is no grey zone.
- **Verify against a reference rather than trusting the math.** The single highest-value habit here: after each component, compare against something known-good and assert a tolerance in `poc/tests/`. `transformers` may be installed *ad hoc* as a dev-only oracle for this (`uv add --dev transformers`) — but it must never become an import in `poc/toyllm/` or `book/`.
- Keep `poc/toyllm/` readable over fast. Explicit shapes in comments/docstrings (e.g. `# (B, T, n_heads, head_dim)`) are worth more than micro-optimizations.
- Prose in `README.md` and any write-ups is **Simplified Chinese**, matching the rest of this notes repo; code, identifiers, and filenames are English.
