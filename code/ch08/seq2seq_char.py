"""
字符级 Seq2Seq：英译法（eng-fra.txt）
使用 one-hot 编码，不使用嵌入层、注意力机制或 Transformer。

架构：
  Encoder  ──逐字符读取英文句子──→  隐藏状态（context vector）
  Decoder  ──从隐藏状态逐字符生成法文句子──→  一个字符一个字符地输出

数据集：PyTorch 官方教程配套数据 eng-fra.txt
  - 英法句对，已过滤为短句（≤ 10 词），约 13.5 万条
  - 格式：每行 "English sentence\tFrench sentence"

Usage:
    python seq2seq_char.py                        # 训练 + 演示
    python seq2seq_char.py --load                 # 加载已有 checkpoint + 演示
    python seq2seq_char.py --translate "i am ok"  # 翻译单句
"""

import os
import unicodedata
import random
import argparse
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim

# ── 数据下载 ─────────────────────────────────────────────────────────────────
DATA_URL  = "https://download.pytorch.org/tutorial/data.zip"
DATA_ZIP  = "data.zip"
DATA_FILE = os.path.join("data", "eng-fra.txt")
CKPT_FILE = "seq2seq_char_ckpt.pt"

MAX_WORDS   = 10     # 过滤掉词数超过此值的句对
MAX_CHAR_LEN = 80    # 过滤掉字符数超过此值的句子（保险起见）

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
    """转小写，去掉重音符号，保留字母/空格/基本标点。"""
    s = s.lower().strip()
    # NFD 分解，去掉组合用发音符（Mn 类）
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    # 只保留字母、空格和几个常见标点
    result = []
    for c in s:
        if c.isalpha() or c in " .,!?'":
            result.append(c)
    return "".join(result).strip()


def load_pairs() -> list[tuple[str, str]]:
    """读取并过滤句对，返回 [(eng, fra), ...]。"""
    pairs = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng = normalize(parts[0])
            fra = normalize(parts[1])
            # 只保留短句
            if (len(eng.split()) <= MAX_WORDS and len(fra.split()) <= MAX_WORDS
                    and len(eng) <= MAX_CHAR_LEN and len(fra) <= MAX_CHAR_LEN
                    and len(eng) > 0 and len(fra) > 0):
                pairs.append((eng, fra))
    print(f"Loaded {len(pairs):,} sentence pairs")
    return pairs


# ── 词表 ─────────────────────────────────────────────────────────────────────
SOS = "\x00"   # Start-Of-Sequence
EOS = "\x01"   # End-Of-Sequence

def build_vocab(pairs: list[tuple[str, str]]):
    """从所有句对中收集出现的字符，构建统一词表。"""
    chars = set()
    for eng, fra in pairs:
        chars.update(eng)
        chars.update(fra)
    chars = sorted(chars)
    all_chars = [SOS, EOS] + chars
    c2i = {c: i for i, c in enumerate(all_chars)}
    i2c = {i: c for c, i in c2i.items()}
    return all_chars, c2i, i2c


# ── 超参数 ───────────────────────────────────────────────────────────────────
HIDDEN_SIZE    = 256
NUM_LAYERS     = 1
BATCH_SIZE     = 64
EPOCHS         = 15
LR             = 0.001
CLIP_GRAD      = 5.0
PRINT_EVERY    = 200    # steps
TEACHER_FORCE  = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 编码 & one-hot ────────────────────────────────────────────────────────────
def encode(s: str, c2i: dict) -> list[int]:
    return [c2i[c] for c in s if c in c2i]


def one_hot(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    indices: LongTensor (seq_len, batch)
    return : FloatTensor (seq_len, batch, vocab_size)
    """
    seq_len, batch = indices.shape
    oh = torch.zeros(seq_len, batch, vocab_size, device=DEVICE)
    oh.scatter_(2, indices.unsqueeze(2), 1.0)
    return oh


def make_batch(pairs: list, c2i: dict, vocab_size: int, batch_size: int):
    """
    随机采样一批句对，padding 对齐后返回张量。
      src: (src_len,  batch)
      tgt: (tgt_len,  batch)  含末尾 EOS
    """
    batch_pairs = random.sample(pairs, batch_size)

    eos_idx = c2i[EOS]
    src_encoded = [encode(eng, c2i)         for eng, _   in batch_pairs]
    tgt_encoded = [encode(fra, c2i) + [eos_idx] for _,   fra in batch_pairs]

    max_src = max(len(s) for s in src_encoded)
    max_tgt = max(len(t) for t in tgt_encoded)

    src_pad = torch.full((max_src, batch_size), eos_idx, dtype=torch.long, device=DEVICE)
    tgt_pad = torch.full((max_tgt, batch_size), eos_idx, dtype=torch.long, device=DEVICE)

    for b, (s, t) in enumerate(zip(src_encoded, tgt_encoded)):
        src_pad[:len(s), b] = torch.tensor(s, dtype=torch.long)
        tgt_pad[:len(t), b] = torch.tensor(t, dtype=torch.long)

    return src_pad, tgt_pad


# ── 模型 ─────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """逐字符读取英文句子，最终隐藏状态 = context vector。"""

    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers)

    def forward(self, src_oh):
        """
        src_oh: (src_len, batch, vocab_size)
        return: (h, c)，形状均为 (num_layers, batch, hidden_size)
        """
        _, hidden = self.lstm(src_oh)
        return hidden


class Decoder(nn.Module):
    """每步接收一个字符的 one-hot，输出下一个字符的 logits。"""

    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.lstm   = nn.LSTM(vocab_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_oh, hidden):
        """
        x_oh  : (1, batch, vocab_size)
        return: logits (batch, vocab_size), new_hidden
        """
        out, hidden = self.lstm(x_oh, hidden)
        logits = self.linear(out.squeeze(0))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.encoder    = Encoder(vocab_size, hidden_size, num_layers)
        self.decoder    = Decoder(vocab_size, hidden_size, num_layers)
        self.vocab_size = vocab_size

    def forward(self, src_oh, tgt_indices, c2i, teacher_forcing_ratio=TEACHER_FORCE):
        """
        src_oh      : (src_len, batch, vocab_size)
        tgt_indices : (tgt_len, batch)  —— 含末尾 EOS 的目标序列
        return      : logits (tgt_len, batch, vocab_size)

        Teacher forcing：训练时以概率 teacher_forcing_ratio 把"正确答案"
        而非模型自身的预测作为下一步输入，有助于稳定早期训练。
        """
        batch_size = src_oh.size(1)
        tgt_len    = tgt_indices.size(0)

        # 1. 编码：把整个英文句子压缩成 context vector
        hidden = self.encoder(src_oh)

        # 2. 解码：以 SOS 开始，逐字符生成法文
        sos_idx   = torch.full((batch_size,), c2i[SOS], dtype=torch.long, device=DEVICE)
        dec_input = one_hot(sos_idx.unsqueeze(0), self.vocab_size)  # (1, batch, vocab)

        all_logits = []
        for t in range(tgt_len):
            logits, hidden = self.decoder(dec_input, hidden)  # (batch, vocab)
            all_logits.append(logits)

            if self.training and random.random() < teacher_forcing_ratio:
                next_idx = tgt_indices[t]        # 正确答案
            else:
                next_idx = logits.argmax(dim=1)  # 模型预测

            dec_input = one_hot(next_idx.unsqueeze(0), self.vocab_size)

        return torch.stack(all_logits, dim=0)    # (tgt_len, batch, vocab)


# ── 训练 ─────────────────────────────────────────────────────────────────────
def train(model, pairs, c2i, vocab_size):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=c2i[EOS])  # 忽略 padding 位置
    model.train()

    steps_per_epoch = len(pairs) // BATCH_SIZE

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(pairs)
        total_loss = 0.0

        for step in range(1, steps_per_epoch + 1):
            src, tgt = make_batch(pairs, c2i, vocab_size, BATCH_SIZE)
            src_oh   = one_hot(src, vocab_size)

            logits = model(src_oh, tgt, c2i)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            total_loss += loss.item()
            if step % PRINT_EVERY == 0:
                print(f"  Epoch {epoch}/{EPOCHS}  step {step:5d}/{steps_per_epoch}"
                      f"  loss {total_loss/step:.4f}")

        print(f"Epoch {epoch}/{EPOCHS} done — avg loss {total_loss/steps_per_epoch:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "c2i": c2i,
        "i2c": {i: c for c, i in c2i.items()},
        "vocab_size": vocab_size,
    }, CKPT_FILE)
    print(f"Checkpoint saved → {CKPT_FILE}")


# ── 推理 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def translate(model, src_str: str, c2i: dict, i2c: dict, max_len: int = 120) -> str:
    """贪心解码：每步取概率最高的字符，直到生成 EOS。"""
    model.eval()
    src_str = normalize(src_str)
    src     = torch.tensor(encode(src_str, c2i), dtype=torch.long, device=DEVICE).unsqueeze(1)
    src_oh  = one_hot(src, model.vocab_size)
    hidden  = model.encoder(src_oh)

    sos_idx   = torch.tensor([c2i[SOS]], dtype=torch.long, device=DEVICE)
    dec_input = one_hot(sos_idx.unsqueeze(0), model.vocab_size)  # (1,1,vocab)

    result = []
    for _ in range(max_len):
        logits, hidden = model.decoder(dec_input, hidden)
        next_idx = logits.argmax(dim=1).item()
        ch = i2c[next_idx]
        if ch == EOS:
            break
        result.append(ch)
        dec_input = one_hot(
            torch.tensor([next_idx], dtype=torch.long, device=DEVICE).unsqueeze(0),
            model.vocab_size
        )

    return "".join(result)


# ── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",      action="store_true", help="加载 checkpoint，跳过训练")
    parser.add_argument("--translate", default=None,        help="翻译单句英文")
    args = parser.parse_args()

    download_data()
    pairs = load_pairs()

    if args.load or not args.translate is None:
        # 推理时直接从 checkpoint 读取词表
        if not os.path.exists(CKPT_FILE):
            raise FileNotFoundError(f"找不到 {CKPT_FILE}，请先训练。")
        ckpt      = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
        c2i       = ckpt["c2i"]
        i2c       = ckpt["i2c"]
        vocab_size = ckpt["vocab_size"]
        model     = Seq2Seq(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint  vocab_size={vocab_size}  device={DEVICE}")
    else:
        _, c2i, i2c = build_vocab(pairs)
        vocab_size  = len(c2i)
        print(f"Vocab size: {vocab_size}  Device: {DEVICE}")
        model = Seq2Seq(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
        train(model, pairs, c2i, vocab_size)

    # 演示
    if args.translate:
        result = translate(model, args.translate, c2i, i2c)
        print(f"\n  EN: {args.translate}")
        print(f"  FR: {result}")
        return

    # 随机抽取几条句对展示效果
    samples = random.sample(pairs, 10)
    print("\n" + "=" * 70)
    print(f"{'English':<35} {'Expected French':<35}")
    print(f"{'':35} {'Got':<35}")
    print("=" * 70)
    for eng, fra in samples:
        got = translate(model, eng, c2i, i2c)
        print(f"{eng:<35} {fra}")
        match = "✓" if got == fra else " "
        print(f"{'':<35} {got}  {match}")
        print("-" * 70)


if __name__ == "__main__":
    main()
