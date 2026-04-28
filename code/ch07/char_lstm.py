"""
Character-level LSTM on Tiny Shakespeare dataset.
Uses one-hot encoding and an LSTM implemented in PyTorch.

Usage:
    python char_rnn_batch.py              # train + generate
    python char_rnn_batch.py --generate   # load checkpoint and generate only
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
CKPT_FILE = "char_lstm_ckpt.pt"

HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
SEQ_LEN      = 100      # characters per training chunk
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 0.003
CLIP_GRAD    = 5.0
PRINT_EVERY  = 200      # steps
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


def one_hot(indices, vocab_size):
    """indices: (batch, seq_len) → one_hot: (seq_len, batch, vocab_size)"""
    seq_len, batch = indices.shape
    oh = torch.zeros(seq_len, batch, vocab_size, device=DEVICE)
    oh.scatter_(2, indices.unsqueeze(2), 1.0)
    return oh


def make_batches(text, c2i, seq_len, batch_size):
    """Encode the full text then yield (input, target) tensor pairs."""
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
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # LSTM 有两个状态：隐藏状态 h 和记忆单元 c
        self.lstm   = nn.LSTM(vocab_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_oh, hidden=None):
        """x_oh: (seq_len, batch, vocab_size)
        hidden: tuple (h, c)，每个形状为 (num_layers, batch, hidden_size)
        """
        out, hidden = self.lstm(x_oh, hidden)         # out: (seq, batch, hidden)
        logits = self.linear(out)                      # (seq, batch, vocab)
        return logits, hidden

    def init_hidden(self, batch_size):
        # LSTM 需要同时初始化 h 和 c
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        return (zeros, zeros.clone())


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, text, c2i, vocab_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        hidden = model.init_hidden(BATCH_SIZE)
        total_loss, step = 0.0, 0

        for x, y in make_batches(text, c2i, SEQ_LEN, BATCH_SIZE):
            # detach (h, c) from previous batch graph，避免跨 batch 反向传播
            hidden = (hidden[0].detach(), hidden[1].detach())

            x_oh   = one_hot(x, vocab_size)          # (seq, batch, vocab)
            logits, hidden = model(x_oh, hidden)     # logits: (seq, batch, vocab)

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
def generate(model, c2i, i2c, vocab_size, seed="long long ago", length=500, temperature=0.8):
    model.eval()
    hidden = model.init_hidden(1)

    # warm up on the seed
    for ch in seed[:-1]:
        idx = torch.tensor([[c2i.get(ch, 0)]], device=DEVICE)   # (1,1)
        x_oh = one_hot(idx.T, vocab_size)                        # (1,1,vocab)
        _, hidden = model(x_oh, hidden)

    result = list(seed)
    ch = seed[-1]

    for _ in range(length):
        idx = torch.tensor([[c2i.get(ch, 0)]], device=DEVICE)
        x_oh = one_hot(idx.T, vocab_size)
        logits, hidden = model(x_oh, hidden)          # logits: (1,1,vocab)

        # sample from distribution
        probs = torch.softmax(logits[0, 0] / temperature, dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        ch = i2c[next_idx]
        result.append(ch)

    return "".join(result)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="skip training, load checkpoint")
    parser.add_argument("--seed",  default="long long ago", help="generation seed text")
    parser.add_argument("--length", type=int, default=500,  help="number of chars to generate")
    parser.add_argument("--temp",   type=float, default=0.8, help="sampling temperature")
    args = parser.parse_args()

    text = load_data()
    chars, c2i, i2c = build_vocab(text)
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    model = CharLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    if args.generate:
        if not os.path.exists(CKPT_FILE):
            raise FileNotFoundError(f"No checkpoint found at {CKPT_FILE}. Train first.")
        ckpt = torch.load(CKPT_FILE, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {CKPT_FILE}")
    else:
        train(model, text, c2i, vocab_size)

    print("\n" + "=" * 60)
    print(f"Seed: \"{args.seed}\"")
    print("=" * 60)
    print(generate(model, c2i, i2c, vocab_size,
                   seed=args.seed, length=args.length, temperature=args.temp))
    print("=" * 60)


if __name__ == "__main__":
    main()
