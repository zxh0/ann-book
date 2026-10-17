"""
Character-level RNN on Tiny Shakespeare dataset.
Uses one-hot encoding and a vanilla RNN implemented in PyTorch.

Usage:
    python char_rnn.py              # train + generate
    python char_rnn.py --generate   # load checkpoint and generate only
"""

import os
import urllib.request
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# ── Hyperparameters ────────────────────────────────────────────────────────────
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"
CKPT_FILE = "char_rnn_ckpt.pt"

HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
SEQ_LEN      = 100      # characters per training chunk
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 0.003
CLIP_GRAD    = 5.0
PRINT_EVERY  = 10       # steps
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading Tiny Shakespeare → {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    text = open(DATA_FILE, encoding="utf-8").read()
    print(f"Dataset: {len(text):,} characters")
    return text


def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return chars, c2i, i2c


# indices: (seq_len, batch) → one_hot: (seq_len, batch, vocab_size)
def one_hot(indices, vocab_size):
    seq_len, batch = indices.shape
    oh = torch.zeros(seq_len, batch, vocab_size, device=DEVICE)
    oh.scatter_(2, indices.unsqueeze(2), 1.0)
    return oh


# Encode the full text then yield (input, target) tensor pairs.
def make_batches(text, c2i, seq_len, batch_size):
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    # trim to exact multiple of (seq_len * batch_size)
    total = (len(data) - 1) // (seq_len * batch_size) * (seq_len * batch_size)
    data = data[:total + 1]
    n_batches = total // (seq_len * batch_size)

    inputs  = data[:-1].view(batch_size, -1)   # (batch, total_steps)
    targets = data[1:].view(batch_size, -1)

    for i in range(n_batches):
        x = inputs [:, i * seq_len:(i + 1) * seq_len].T.contiguous()  # (seq, batch)
        y = targets[:, i * seq_len:(i + 1) * seq_len].T.contiguous()
        yield x.to(DEVICE), y.to(DEVICE)


# ── Model ──────────────────────────────────────────────────────────────────────
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn    = nn.RNN(vocab_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    # x_oh: (seq_len, batch, vocab_size)
    def forward(self, x_oh, hidden=None):
        out, hidden = self.rnn(x_oh, hidden) # out: (seq, batch, hidden)
        logits = self.linear(out)            # (seq, batch, vocab)
        return logits, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, text, c2i, vocab_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        hidden = model.init_hidden(BATCH_SIZE)
        total_loss, step = 0.0, 0

        for x, y in make_batches(text, c2i, SEQ_LEN, BATCH_SIZE):
            # detach hidden state from previous batch graph
            hidden = hidden.detach()

            x_oh = one_hot(x, vocab_size)        # (seq, batch, vocab)
            logits, hidden = model(x_oh, hidden) # logits: (seq, batch, vocab)

            # flatten for cross-entropy
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
        # remove tokens that push cumulative prob over the threshold
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].clone()  # shift right: always keep the top token
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")

    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, 1).item()


def generate(model, c2i, i2c, vocab_size, seed="long long ago", length=500,
             temperature=0.8, top_k=0, top_p=0.0):
    model.eval()
    hidden = model.init_hidden(1)

    # warm up on the seed
    for ch in seed[:-1]:
        idx = torch.tensor([[c2i.get(ch, 0)]], device=DEVICE) # (1,1)
        x_oh = one_hot(idx.T, vocab_size)                     # (1,1,vocab)
        _, hidden = model(x_oh, hidden)

    result = list(seed)
    ch = seed[-1]

    for _ in range(length):
        idx = torch.tensor([[c2i.get(ch, 0)]], device=DEVICE)
        x_oh = one_hot(idx.T, vocab_size)
        logits, hidden = model(x_oh, hidden) # logits: (1,1,vocab)

        next_idx = sample_logits(logits[0, 0], temperature, top_k, top_p)
        ch = i2c[next_idx]
        result.append(ch)

    return "".join(result)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="skip training, load checkpoint")
    parser.add_argument("--seed", default="long long ago", help="generation seed text")
    parser.add_argument("--length", type=int, default=500, help="number of chars to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.0, help="top-p nucleus sampling (0.0 = disabled)")
    args = parser.parse_args()

    text = load_data()
    chars, c2i, i2c = build_vocab(text)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    one_hot_table = torch.eye(vocab_size, dtype=torch.int)
    for i, ch in enumerate(chars):
        print(f"{repr(ch)}: {one_hot_table[i].tolist()}")

    model = CharRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    if args.generate:
        if not os.path.exists(CKPT_FILE):
            raise FileNotFoundError(f"No checkpoint found at {CKPT_FILE}. Train first.")
        ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {CKPT_FILE}")
    else:
        train(model, text, c2i, vocab_size)

    print("\n" + "=" * 60)
    print(f"Seed: \"{args.seed}\"")
    print("=" * 60)
    print(generate(model, c2i, i2c, vocab_size,
                   seed=args.seed, length=args.length, 
				   temperature=args.temp,
                   top_k=args.top_k,
				   top_p=args.top_p))
    print("=" * 60)


if __name__ == "__main__":
    main()
