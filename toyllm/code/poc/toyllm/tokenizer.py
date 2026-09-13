"""Byte-level BPE tokenizer, written from scratch.

SmolLM2 ships a `tokenizer.json` in HuggingFace `tokenizers` format. We parse it
ourselves and re-implement encode/decode rather than calling the library, because
BPE *is* the lesson here — the library is kept around only as the oracle that
`tests/test_tokenizer.py` diffs against.

The pipeline that file describes, in order:

    normalizer      null            (nothing to do)
    pre_tokenizer   Sequence[ Digits(individual_digits), ByteLevel(use_regex) ]
    model           BPE             49152 vocab, 48900 merges, no unk token
    post_processor  null            (no automatic BOS/EOS — the caller adds them)
    decoder         ByteLevel

Two things make "byte-level" worth pausing on: the model never sees characters,
only the 256 possible bytes, so there is no `<unk>` token and no notion of an
out-of-vocabulary *word*; and those bytes are re-mapped into printable Unicode
(space -> 'Ġ') purely so the vocabulary is readable and whitespace survives a
round-trip through JSON.

The usual claim that byte-level BPE can therefore encode *anything* is not quite
true for this checkpoint: 21 of the 256 byte tokens are simply absent from
SmolLM2's vocabulary — they never occurred in training. Fifteen (0xC0, 0xC1,
0xF5-0xFF) are unreachable because they cannot appear in valid UTF-8, but six
control bytes (0x04, 0x06, 0x13, 0x14, 0x16, 0x1D) are perfectly legal in a
Python string. HuggingFace's implementation drops them silently — `"a\x04b"`
encodes to just `["a", "b"]` — so encoding is *not* lossless in general. We match
that behavior deliberately; see `unrepresentable_bytes`.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex

# The GPT-2 pre-tokenization regex, which `tokenizers`' ByteLevel uses when
# use_regex=true. It keeps a leading space attached to the following word
# (" the" is one piece), which is why most vocab entries start with 'Ġ'.
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map each of the 256 byte values to a distinct printable Unicode char.

    Bytes that are already printable ASCII/Latin-1 map to themselves; the rest
    (control chars, space, DEL, ...) are shifted up into U+0100.. so they get a
    visible stand-in. Space (0x20) becomes 'Ġ', newline (0x0A) becomes 'Ċ'.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = list(printable)
    chars = list(printable)
    shift = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            chars.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(byte_values, chars)}


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: dict[str, int],
    ):
        self.vocab = vocab
        self.inv_vocab = {i: s for s, i in vocab.items()}
        # Merge priority: the pair listed first in the file wins. This ordering
        # *is* the trained model — the vocab alone is not enough to tokenize.
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}

        self._split_re = regex.compile(GPT2_SPLIT_PATTERN)
        # Longest-first so "<|im_start|>" wins over any shorter prefix.
        specials = sorted(special_tokens, key=len, reverse=True)
        self._special_re = regex.compile("(" + "|".join(regex.escape(s) for s in specials) + ")")
        self._bpe_cache: dict[str, list[str]] = {}

        # Byte values with no token in the vocabulary. Text containing these
        # loses those bytes on encode — silently, matching HF.
        self.unrepresentable_bytes = sorted(
            b for b, c in self.byte_encoder.items() if c not in vocab
        )

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_file(cls, path: str | Path) -> BPETokenizer:
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        spec = json.loads(path.read_text())

        if spec.get("normalizer") is not None:
            raise NotImplementedError("this tokenizer has a normalizer; we assume none")
        if spec["model"]["type"] != "BPE":
            raise NotImplementedError(f"unsupported model type {spec['model']['type']!r}")

        raw_merges = spec["model"]["merges"]
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in raw_merges]
        specials = {t["content"]: t["id"] for t in spec.get("added_tokens", [])}
        return cls(spec["model"]["vocab"], merges, specials)

    # --------------------------------------------------------- pre-tokenizing

    def pre_tokenize(self, text: str) -> list[str]:
        """Split text into byte-encoded chunks, each BPE'd independently.

        Digits first (`individual_digits`), so "2026" can never merge into a
        single token — the model sees "2","0","2","6". That is a deliberate
        choice by SmolLM2's authors to make arithmetic tractable, and it is the
        one place this differs from plain GPT-2.
        """
        pieces = []
        for chunk in _split_digits(text):
            for piece in self._split_re.findall(chunk):
                pieces.append("".join(self.byte_encoder[b] for b in piece.encode("utf-8")))
        return pieces

    # ------------------------------------------------------------------- BPE

    def bpe(self, token: str) -> list[str]:
        """Greedily apply the highest-priority merge until none applies.

        Starts from single characters and repeatedly fuses the adjacent pair
        with the lowest rank. Note it is *lowest rank overall*, not left-to-right:
        the merge order learned during training is what gets replayed here.
        """
        if token in self._bpe_cache:
            return self._bpe_cache[token]

        symbols = list(token)
        while len(symbols) > 1:
            pairs = zip(symbols[:-1], symbols[1:])
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            merged = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        self._bpe_cache[token] = symbols
        return symbols

    def bpe_steps(self, token: str) -> list[tuple[tuple[str, str], int, list[str]]]:
        """Replay `bpe()` recording each merge — (pair, rank, symbols after). For teaching."""
        symbols = list(token)
        steps = []
        while len(symbols) > 1:
            best = min(zip(symbols[:-1], symbols[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            merged, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            steps.append((best, self.ranks[best], list(symbols)))
        return steps

    # -------------------------------------------------------------- encode/decode

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        ids: list[int] = []
        parts = self._special_re.split(text) if (allow_special and self.special_tokens) else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for piece in self.pre_tokenize(part):
                for sym in self.bpe(piece):
                    # A symbol can be missing: 21 byte tokens are absent from the
                    # vocabulary (see module docstring). HF drops them silently,
                    # so we do too — diverging here would change ids on real text.
                    if sym in self.vocab:
                        ids.append(self.vocab[sym])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Concatenate the token strings, then undo the byte mapping.

        Decoding must go through bytes, not per-token strings: a single UTF-8
        character is often split across several tokens, so decoding tokens
        individually would produce mojibake.
        """
        text = "".join(self.inv_vocab[i] for i in ids)
        raw = bytes(self.byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors="replace")

    def tokens(self, ids: list[int]) -> list[str]:
        """The raw vocabulary strings, for inspection ('Ġthe', not ' the')."""
        return [self.inv_vocab[i] for i in ids]

    def __len__(self) -> int:
        return len(self.vocab)


def _is_numeric(ch: str) -> bool:
    """Exactly Rust's `char::is_numeric()`: Unicode general category Nd, Nl or No.

    Neither Python built-in matches it, and both failures are silent — they just
    yield different token ids for some inputs:
      - `isdigit()`  is too narrow: '½' (No) is False, but Rust says numeric.
      - `isnumeric()` is too wide:  '一' is True (it has Numeric_Type), but its
        category is Lo, so Rust says *not* numeric.
    Checking the general category directly is the only faithful test.
    """
    return unicodedata.category(ch)[0] == "N"


def _split_digits(text: str) -> list[str]:
    """Isolate every numeric character as its own chunk, keeping other runs intact."""
    out: list[str] = []
    buf = ""
    for ch in text:
        if _is_numeric(ch):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out
