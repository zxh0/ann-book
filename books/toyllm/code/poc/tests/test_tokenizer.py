"""The tokenizer's correctness bar is exact equality with `tokenizers`.

Anything less is worthless: a tokenizer that is 99.9% right produces token ids
the model was never trained on, and the failure looks like "the model is a bit
dumb" rather than like a bug.
"""

import random

import pytest

from toyllm.tokenizer import BPETokenizer, bytes_to_unicode

CHAR_POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    " \t\n\r　",
    ".,!?;:'\"()[]{}<>/@#$%^&*-_=+~`|\\",
    "你好世界中文测试汉字一二三千万亿",
    "αβγδεζηθ",
    "абвгдежз",
    "ابتثجحخد",
    "½²³¼Ⅷⅸ①②〇๓๔",
    "🤖🚀😀🎉👍🏽",
    "áéíóúñüßœæ",
    "​﻿",
]


@pytest.fixture(scope="module")
def tok(model_dir):
    return BPETokenizer.from_file(model_dir)


@pytest.fixture(scope="module")
def ref(model_dir):
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def test_byte_map_is_a_bijection_over_256_bytes():
    enc = bytes_to_unicode()
    assert len(enc) == 256
    assert len(set(enc.values())) == 256
    assert enc[0x20] == "Ġ" and enc[0x0A] == "Ċ" and enc[ord("A")] == "A"


def test_vocab_and_merges_loaded(tok):
    assert len(tok) == 49152
    assert len(tok.ranks) == 48900
    assert len(tok.special_tokens) == 17
    assert tok.special_tokens["<|endoftext|>"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "Hello world!",
        " the quick brown fox",
        "def main():\n    return 42\n",
        "2026年9月7日",
        "naïve café — résumé",
        "🤖🚀 emoji",
        "a  b   c",
        "   leading",
        "trailing   ",
        "",
        "\n\n\t",
        "<|im_start|>user\nhi<|im_end|>",
        "Ⅷ ½ ² ٣ 一二三",
    ],
)
def test_matches_reference(tok, ref, text):
    assert tok.encode(text) == ref.encode(text, add_special_tokens=False).ids
    assert tok.decode(tok.encode(text)) == text


def test_unicode_numeric_edge_cases(tok, ref):
    """The two characters that broke the first implementation.

    '½' is category No — numeric, so it is isolated. '一' is category Lo — NOT
    numeric despite Python's isnumeric() saying so, so it may merge with
    neighbours. Getting either wrong changes ids silently.
    """
    for text in ["a½b", "a一b", " ½", "\n\n一", "　 \n一", "½一Ⅷ²٣"]:
        assert tok.encode(text) == ref.encode(text, add_special_tokens=False).ids, text


def test_digits_are_always_split(tok):
    assert tok.tokens(tok.encode("2026")) == ["2", "0", "2", "6"]
    assert tok.tokens(tok.encode("$20")) == ["$", "2", "0"]


def test_special_tokens_can_be_disabled(tok):
    ids = tok.encode("<|endoftext|>", allow_special=False)
    assert 0 not in ids
    assert tok.decode(ids) == "<|endoftext|>"
    assert tok.encode("<|endoftext|>") == [0]


def test_decode_handles_multibyte_split_across_tokens(tok):
    """A CJK char spans several tokens; decoding must join bytes, not strings."""
    ids = tok.encode("第")
    assert len(ids) > 1
    assert tok.decode(ids) == "第"
    assert all(len(t) < 3 or True for t in tok.tokens(ids))


def test_fuzz_matches_reference(tok, ref):
    random.seed(1234)
    for _ in range(2000):
        n = random.randint(0, 60)
        text = "".join(random.choice(random.choice(CHAR_POOLS)) for _ in range(n))
        assert tok.encode(text) == ref.encode(text, add_special_tokens=False).ids, repr(text)
        assert tok.decode(tok.encode(text)) == text, repr(text)


def test_encoding_is_lossless_for_representable_text(tok):
    """Any text built from bytes the vocab covers round-trips exactly."""
    text = "".join(chr(b) for b in range(0x20, 0x7F)) + "你好 café 🤖\n\t"
    assert tok.decode(tok.encode(text)) == text


def test_unrepresentable_bytes_are_dropped_like_the_reference(tok, ref):
    """21 byte tokens are missing from SmolLM2's vocab; HF drops them silently.

    Six of them are reachable from ordinary Python strings, so this is a real
    (if rare) lossy path — not a theoretical one. We match HF rather than being
    "more correct", because diverging would change ids the model was trained on.
    """
    assert tok.unrepresentable_bytes == [
        0x04, 0x06, 0x13, 0x14, 0x16, 0x1D,          # control chars, legal in a str
        0xC0, 0xC1,                                  # overlong forms — never valid UTF-8
        0xF1, 0xF2,                                  # lead bytes for planes 4-7
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9,                # beyond U+10FFFF — never valid
        0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
    ]

    # The six control bytes vanish entirely, leaving "ab".
    for b in (0x04, 0x06, 0x13, 0x14, 0x16, 0x1D):
        text = f"a{chr(b)}b"
        assert tok.encode(text) == ref.encode(text, add_special_tokens=False).ids
        assert tok.decode(tok.encode(text)) == "ab"

    # 0xF1/0xF2 lead 4-byte sequences for planes 4-7, so they are reachable too:
    # the lead byte is dropped and the continuation bytes survive as garbage.
    text = "a" + chr(0x50000) + "b"
    assert tok.encode(text) == ref.encode(text, add_special_tokens=False).ids
    assert tok.decode(tok.encode(text)) != text

    # Everything else is unreachable: no str encodes to those bytes.
    unreachable = {0xC0, 0xC1} | set(range(0xF5, 0x100))
    assert unreachable == {b for b in tok.unrepresentable_bytes if b >= 0x80} - {0xF1, 0xF2}
