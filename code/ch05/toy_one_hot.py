import os
import ssl
import urllib.request
import numpy as np

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare → {DATA_FILE} ...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(DATA_URL, context=ctx) as r, open(DATA_FILE, "wb") as f:
            f.write(r.read())
    text = open(DATA_FILE, encoding="utf-8").read()
    print(f"Dataset: {len(text):,} characters")
    return text


def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return chars, c2i, i2c


def one_hot(indices, vocab_size):
    """indices: (seq_len,) → one_hot: (seq_len, vocab_size)"""
    oh = np.zeros((len(indices), vocab_size), dtype=np.float32)
    oh[np.arange(len(indices)), indices] = 1.0
    return oh


def make_sequences(text, c2i, seq_len):
    """Encode the full text then yield (input, target) index array pairs."""
    data = np.array([c2i[c] for c in text], dtype=np.int32)
    total = (len(data) - 1) // seq_len * seq_len
    data = data[:total + 1]
    n_seqs = total // seq_len

    for i in range(n_seqs):
        x = data[i * seq_len:(i + 1) * seq_len]
        y = data[i * seq_len + 1:(i + 1) * seq_len + 1]
        yield x, y


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    text = load_data()
    chars, c2i, i2c = build_vocab(text)
    print('chars:', chars)
    print('c2i:', c2i)
    print('i2c:', i2c)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    # show one-hot encoding for the first 5 characters
    sample = text[:5]
    indices = np.array([c2i[c] for c in sample], dtype=np.int32)
    oh = one_hot(indices, vocab_size)
    print(f"\nSample: {repr(sample)}")
    print(f"Indices: {indices}")
    print(f"One-hot shape: {oh.shape}")
    for ch, row in zip(sample, oh):
        print(f"  {repr(ch)} → index {c2i[ch]:2d}, one-hot nonzero at {np.argmax(row)}")
