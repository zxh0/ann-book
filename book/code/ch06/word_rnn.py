"""
Word-level RNN on Tiny Shakespeare dataset.
Uses Word2Vec embeddings (via gensim) and a vanilla RNN implemented in PyTorch.

Usage:
    python word_rnn.py              # train word2vec + RNN, then generate
    python word_rnn.py --generate   # load checkpoint and generate only
"""

import os
import re
import urllib.request
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"
CKPT_FILE = "word_rnn_ckpt.pt"

EMBED_SIZE    = 128     # word2vec / embedding dimension
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
SEQ_LEN       = 50      # tokens per training chunk
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 0.003
CLIP_GRAD     = 5.0
PRINT_EVERY   = 50      # steps
W2V_WINDOW    = 5
W2V_MIN_COUNT = 1
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare → {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    text = open(DATA_FILE, encoding="utf-8").read()
    print(f"Dataset: {len(text):,} characters")
    return text


# Split into word and punctuation tokens; newlines become <nl>.
def tokenize(text):
    raw = re.findall(r"\n|[A-Za-z']+|[^A-Za-z'\s]", text)
    return ["<nl>" if t == "\n" else t.lower() for t in raw]


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    t2i = {t: i for i, t in enumerate(vocab)}
    i2t = {i: t for t, i in t2i.items()}
    return vocab, t2i, i2t


def train_word2vec(sentences, embed_size):
    print("Training Word2Vec embeddings ...")
    w2v = Word2Vec(sentences, vector_size=embed_size, window=W2V_WINDOW,
                   min_count=W2V_MIN_COUNT, workers=4, epochs=10)
    print(f"Word2Vec vocab size: {len(w2v.wv)}")
    return w2v


# Build an embedding matrix aligned with our vocab index.
def build_embedding_matrix(vocab, w2v_model, embed_size):
    matrix = (np.random.randn(len(vocab), embed_size) * 0.01).astype(np.float32)
    hits = 0
    for i, token in enumerate(vocab):
        if token in w2v_model.wv:
            matrix[i] = w2v_model.wv[token]
            hits += 1
    print(f"Embedding coverage: {hits}/{len(vocab)} tokens ({100*hits/len(vocab):.1f}%)")
    return matrix


# Yield (input, target) token-index tensor pairs.
def make_batches(indices, seq_len, batch_size):
    data = torch.tensor(indices, dtype=torch.long)
    total = (len(data) - 1) // (seq_len * batch_size) * (seq_len * batch_size)
    data = data[:total + 1]
    n_batches = total // (seq_len * batch_size)

    inputs  = data[:-1].view(batch_size, -1)   # (batch, total_steps)
    targets = data[1: ].view(batch_size, -1)

    for i in range(n_batches):
        x = inputs [:, i * seq_len:(i + 1) * seq_len].T.contiguous()  # (seq, batch)
        y = targets[:, i * seq_len:(i + 1) * seq_len].T.contiguous()
        yield x.to(DEVICE), y.to(DEVICE)


# ── Model ──────────────────────────────────────────────────────────────────────
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 pretrained_embeddings=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn    = nn.RNN(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    # x: (seq_len, batch) — token indices
    def forward(self, x, hidden=None):
        emb = self.embedding(x)              # (seq, batch, embed)
        out, hidden = self.rnn(emb, hidden)  # out: (seq, batch, hidden)
        logits = self.linear(out)            # (seq, batch, vocab)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=DEVICE)


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, indices, vocab_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        hidden = model.init_hidden(BATCH_SIZE)
        total_loss, step = 0.0, 0

        for x, y in make_batches(indices, SEQ_LEN, BATCH_SIZE):
            hidden = hidden.detach()
            logits, hidden = model(x, hidden)

            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % PRINT_EVERY == 0:
                print(f"  Epoch {epoch}/{EPOCHS}  step {step:5d}  loss {total_loss/step:.4f}")

        print(f"Epoch {epoch}/{EPOCHS} done — avg loss {total_loss/step:.4f}")

    torch.save({"state_dict": model.state_dict(),
                "hidden_size": model.hidden_size,
                "num_layers":  model.num_layers}, CKPT_FILE)
    print(f"Checkpoint saved → {CKPT_FILE}")


# ── Generation ─────────────────────────────────────────────────────────────────
# Apply temperature, top-k, and top-p filtering, then sample one index.
def sample_logits(logits, temperature=0.8, top_k=0, top_p=0.0):
    logits = logits / temperature

    # top-k: keep only the k highest-logit tokens
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = -float("inf")

    # top-p (nucleus): keep the smallest set whose cumulative prob >= p
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=0), dim=0)
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].clone()  # shift right: always keep the top token
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")

    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, 1).item()


# Reconstruct readable text from a token list.
def detokenize(tokens):
    parts = []
    for tok in tokens:
        if tok == "<nl>":
            parts.append("\n")
        elif parts and parts[-1] not in ("", "\n") and tok.isalpha():
            parts.append(" " + tok)
        else:
            parts.append(tok)
    return "".join(parts)


def generate(model, t2i, i2t, vocab_size, seed="long long ago", length=200,
             temperature=0.8, top_k=0, top_p=0.0):
    model.eval()
    hidden = model.init_hidden(1)
    seed_tokens = tokenize(seed)

    # warm up on the seed
    for tok in seed_tokens[:-1]:
        idx = torch.tensor([[t2i.get(tok, 0)]], device=DEVICE)  # (1, 1)
        _, hidden = model(idx, hidden)

    result = list(seed_tokens)
    tok = seed_tokens[-1]

    for _ in range(length):
        idx = torch.tensor([[t2i.get(tok, 0)]], device=DEVICE)
        logits, hidden = model(idx, hidden)      # logits: (1, 1, vocab)
        next_idx = sample_logits(logits[0, 0], temperature, top_k, top_p)
        tok = i2t[next_idx]
        result.append(tok)

    return detokenize(result)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="skip training, load checkpoint")
    parser.add_argument("--seed", default="long long ago", help="generation seed text")
    parser.add_argument("--length", type=int, default=200, help="number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.0, help="top-p nucleus sampling (0.0 = disabled)")
    args = parser.parse_args()

    text   = load_data()
    tokens = tokenize(text)
    print(f"Total tokens: {len(tokens):,}")

    vocab, t2i, i2t = build_vocab(tokens)
    vocab_size = len(vocab)
    print(f"Vocab size (words): {vocab_size}")

    indices = [t2i[t] for t in tokens]

    embed_matrix = None
    if not args.generate:
        sentences = [tokens[i:i + SEQ_LEN]
                     for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN)]
        w2v = train_word2vec(sentences, EMBED_SIZE)
        embed_matrix = build_embedding_matrix(vocab, w2v, EMBED_SIZE)

    model = WordRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                    pretrained_embeddings=embed_matrix).to(DEVICE)

    if args.generate:
        if not os.path.exists(CKPT_FILE):
            raise FileNotFoundError(
                f"No checkpoint found at {CKPT_FILE}. Train first.")
        ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {CKPT_FILE}")
    else:
        train(model, indices, vocab_size)

    print("\n" + "=" * 60)
    print(f"Seed: \"{args.seed}\"")
    print("=" * 60)
    print(generate(model, t2i, i2t, vocab_size,
                   seed=args.seed, length=args.length,
                   temperature=args.temp,
                   top_k=args.top_k,
                   top_p=args.top_p))
    print("=" * 60)


if __name__ == "__main__":
    main()
