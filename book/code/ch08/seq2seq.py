"""
词级 Seq2Seq（英译法，eng-fra.txt）
使用 gensim Word2Vec 初始化嵌入层，Encoder/Decoder 均为 LSTM；无注意力、无 Transformer。

架构：
  Encoder  ──逐词读取英文──→  context (h, c)
  Decoder  ──从 context 逐词生成法文──→  logits

数据集：PyTorch 官方教程配套数据 eng-fra.txt
  - 英法句对，已过滤为短句（≤ MAX_WORDS 词）
  - 格式：每行 "English sentence\\tFrench sentence"

Usage:
    python seq2seq_char.py                        # 训练 Word2Vec + 模型 + 演示
    python seq2seq_char.py --load                 # 加载 checkpoint + 演示
    python seq2seq_char.py --translate "i am ok"  # 翻译单句
"""

import os
import random
import argparse
import urllib.request
import zipfile
import unicodedata
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec

# ── 数据下载 ─────────────────────────────────────────────────────────────────
DATA_URL  = "https://download.pytorch.org/tutorial/data.zip"
DATA_ZIP  = "data.zip"
DATA_FILE = os.path.join("data", "eng-fra.txt")
CKPT_FILE = "seq2seq_word_ckpt.pt"

MAX_WORDS = 10  # 每句最多词数（与原版一致）

# ── Word2Vec / 嵌入（与 ch07 对齐）──────────────────────────────────────────
EMBED_SIZE    = 128
W2V_WINDOW    = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS    = 10

# ── 特殊符号（词表中的显式 token）────────────────────────────────────────────
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

# ── 超参数 ───────────────────────────────────────────────────────────────────
HIDDEN_SIZE   = 256
NUM_LAYERS    = 1
BATCH_SIZE    = 64
EPOCHS        = 15
LR            = 0.001
CLIP_GRAD     = 5.0
PRINT_EVERY   = 200
TEACHER_FORCE = 0.5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_data():
    if not os.path.exists(DATA_FILE):
        if not os.path.exists(DATA_ZIP):
            print(f"Downloading {DATA_URL} ...")
            urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
        print("Extracting ...")
        with zipfile.ZipFile(DATA_ZIP) as zf:
            zf.extractall(".")
    print(f"Data ready: {DATA_FILE}")


# ── 文本规范化 ────────────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    """转小写，去掉重音符号，标点与单词之间插入空格。"""
    s = s.lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    result = []
    for c in s:
        if c.isalpha() or c in " '":
            result.append(c)
        elif c in ".,!?":
            result.append(" " + c + " ")   # 标点独立成 token
    return " ".join("".join(result).split())


def tokenize(s: str) -> list[str]:
    """词级切分（规范化后按空白分词）。"""
    return s.split() if s else []


def load_pairs() -> list[tuple[str, str]]:
    pairs = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng = normalize(parts[0])
            fra = normalize(parts[1])
            te, tf = tokenize(eng), tokenize(fra)
            if (len(te) <= MAX_WORDS and len(tf) <= MAX_WORDS
                    and len(te) > 0 and len(tf) > 0):
                pairs.append((eng, fra))
    print(f"Loaded {len(pairs):,} sentence pairs")
    return pairs


def build_vocab(pairs: list[tuple[str, str]]):
    """统一词表（英/法所有词 + 特殊符号）。"""
    tokens = set()
    for eng, fra in pairs:
        tokens.update(tokenize(eng))
        tokens.update(tokenize(fra))
    specials = [SOS, EOS, PAD, UNK]
    rest = sorted(tokens)
    vocab = specials + rest
    t2i = {t: i for i, t in enumerate(vocab)}
    i2t = {i: t for t, i in t2i.items()}
    return vocab, t2i, i2t


def encode_tokens(tokens: list[str], t2i: dict) -> list[int]:
    unk = t2i[UNK]
    return [t2i.get(w, unk) for w in tokens]


def train_word2vec(sentences: list[list[str]], embed_size: int) -> Word2Vec:
    print("Training Word2Vec ...")
    w2v = Word2Vec(
        sentences,
        vector_size=embed_size,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        epochs=W2V_EPOCHS,
    )
    print(f"Word2Vec vocab size: {len(w2v.wv)}")
    return w2v


def build_embedding_matrix(vocab: list[str], w2v: Word2Vec, embed_size: int) -> np.ndarray:
    matrix = (np.random.randn(len(vocab), embed_size) * 0.01).astype(np.float32)
    hits = 0
    for i, token in enumerate(vocab):
        if token in w2v.wv:
            matrix[i] = w2v.wv[token]
            hits += 1
    print(f"Embedding coverage: {hits}/{len(vocab)} tokens ({100 * hits / len(vocab):.1f}%)")
    return matrix


def make_batch(
    pairs: list,
    t2i: dict,
    batch_size: int,
):
    batch_pairs = random.sample(pairs, batch_size)
    pad_idx = t2i[PAD]
    eos_idx = t2i[EOS]

    src_enc = [encode_tokens(tokenize(eng), t2i) for eng, _ in batch_pairs]
    tgt_enc = [encode_tokens(tokenize(fra), t2i) + [eos_idx] for _, fra in batch_pairs]

    max_src = max(len(s) for s in src_enc)
    max_tgt = max(len(t) for t in tgt_enc)

    src_pad = torch.full((max_src, batch_size), pad_idx, dtype=torch.long, device=DEVICE)
    tgt_pad = torch.full((max_tgt, batch_size), pad_idx, dtype=torch.long, device=DEVICE)

    for b, (s, t) in enumerate(zip(src_enc, tgt_enc)):
        src_pad[: len(s), b] = torch.tensor(s, dtype=torch.long)
        tgt_pad[: len(t), b] = torch.tensor(t, dtype=torch.long)

    return src_pad, tgt_pad


# ── 模型 ─────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)

    def forward(self, emb):
        """emb: (src_len, batch, embed_dim) → hidden (h, c)"""
        _, hidden = self.lstm(emb)
        return hidden


class Decoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, emb_step, hidden):
        """emb_step: (1, batch, embed_dim)"""
        out, hidden = self.lstm(emb_step, hidden)
        logits = self.linear(out.squeeze(0))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        sos_idx: int,
        pretrained_embeddings: np.ndarray | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.encoder = Encoder(embed_dim, hidden_size, num_layers)
        self.decoder = Decoder(embed_dim, hidden_size, num_layers, vocab_size)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sos_idx = sos_idx

    def forward(self, src_indices, tgt_indices, teacher_forcing_ratio: float = TEACHER_FORCE):
        batch_size = src_indices.size(1)
        tgt_len = tgt_indices.size(0)

        emb_src = self.embedding(src_indices)
        hidden = self.encoder(emb_src)

        sos = torch.full((batch_size,), self.sos_idx, dtype=torch.long, device=DEVICE)
        dec_in = self.embedding(sos).unsqueeze(0)

        logits_all = []
        for t in range(tgt_len):
            logits, hidden = self.decoder(dec_in, hidden)
            logits_all.append(logits)

            if self.training and random.random() < teacher_forcing_ratio:
                nxt = tgt_indices[t]
            else:
                nxt = logits.argmax(dim=1)

            dec_in = self.embedding(nxt).unsqueeze(0)

        return torch.stack(logits_all, dim=0)


def train(model: Seq2Seq, pairs: list, t2i: dict, vocab_size: int):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=t2i[PAD])
    model.train()

    steps_per_epoch = len(pairs) // BATCH_SIZE

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for step in range(1, steps_per_epoch + 1):
            src, tgt = make_batch(pairs, t2i, BATCH_SIZE)
            logits = model(src, tgt)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            if step % PRINT_EVERY == 0:
                print(
                    f"  Epoch {epoch}/{EPOCHS}  step {step:5d}/{steps_per_epoch}"
                    f"  loss {total_loss/step:.4f}"
                )

        print(f"Epoch {epoch}/{EPOCHS} done — avg loss {total_loss/steps_per_epoch:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "t2i": t2i,
            "i2t": {i: t for t, i in t2i.items()},
            "vocab_size": vocab_size,
            "embed_size": model.embed_dim,
        },
        CKPT_FILE,
    )
    print(f"Checkpoint saved → {CKPT_FILE}")


@torch.no_grad()
def translate(model: Seq2Seq, src_str: str, t2i: dict, i2t: dict, max_len: int = 40) -> str:
    model.eval()
    src_str = normalize(src_str)
    idxs = encode_tokens(tokenize(src_str), t2i)
    src = torch.tensor(idxs, dtype=torch.long, device=DEVICE).unsqueeze(1)
    emb = model.embedding(src)
    hidden = model.encoder(emb)

    sos = torch.tensor([t2i[SOS]], dtype=torch.long, device=DEVICE)
    dec_in = model.embedding(sos).unsqueeze(0)

    out_tokens = []
    for _ in range(max_len):
        logits, hidden = model.decoder(dec_in, hidden)
        nxt = logits.argmax(dim=1).item()
        tok = i2t[nxt]
        if tok == EOS:
            break
        if tok not in (SOS, PAD, UNK):
            out_tokens.append(tok)
        dec_in = model.embedding(
            torch.tensor([nxt], dtype=torch.long, device=DEVICE)
        ).unsqueeze(0)

    return " ".join(out_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="加载 checkpoint，跳过训练")
    parser.add_argument("--translate", default=None, help="翻译单句英文")
    args = parser.parse_args()

    download_data()
    pairs = load_pairs()

    need_ckpt = args.load or args.translate is not None

    if need_ckpt:
        if not os.path.exists(CKPT_FILE):
            raise FileNotFoundError(f"找不到 {CKPT_FILE}，请先训练。")
        ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
        t2i = ckpt["t2i"]
        i2t = ckpt["i2t"]
        vocab_size = ckpt["vocab_size"]
        embed_size = ckpt.get("embed_size", EMBED_SIZE)
        model = Seq2Seq(vocab_size, embed_size, HIDDEN_SIZE, NUM_LAYERS, sos_idx=t2i[SOS]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint  vocab_size={vocab_size}  embed={embed_size}  device={DEVICE}")
    else:
        _, t2i, i2t = build_vocab(pairs)
        vocab_size = len(t2i)
        print(f"Vocab size: {vocab_size}  Device: {DEVICE}")

        sentences = []
        for eng, fra in pairs:
            sentences.append(tokenize(eng))
            sentences.append(tokenize(fra))
        w2v = train_word2vec(sentences, EMBED_SIZE)
        vocab_order = [tok for tok, _ in sorted(t2i.items(), key=lambda kv: kv[1])]
        embed_matrix = build_embedding_matrix(vocab_order, w2v, EMBED_SIZE)

        model = Seq2Seq(
            vocab_size,
            EMBED_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            sos_idx=t2i[SOS],
            pretrained_embeddings=embed_matrix,
        ).to(DEVICE)
        train(model, pairs, t2i, vocab_size)

    if args.translate:
        result = translate(model, args.translate, t2i, i2t)
        print(f"\n  EN: {args.translate}")
        print(f"  FR: {result}")
        return

    samples = random.sample(pairs, 10)
    print("\n" + "=" * 70)
    print(f"{'English':<35} {'Expected French':<35}")
    print(f"{'':35} {'Got':<35}")
    print("=" * 70)
    for eng, fra in samples:
        got = translate(model, eng, t2i, i2t)
        print(f"{eng:<35} {fra}")
        exp_join = " ".join(tokenize(fra))
        match = "✓" if got == exp_join else " "
        print(f"{'':<35} {got}  {match}")
        print("-" * 70)


if __name__ == "__main__":
    main()
