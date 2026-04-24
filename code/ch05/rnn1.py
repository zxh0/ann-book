import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CharRNN(
    vocab_size=vocab_size,
    embedding_dim=16,
    hidden_size=32
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = X.to(device)
y = y.to(device)

# =========================
# 5. 训练模型
# =========================
epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs, _ = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =========================
# 6. 文本生成函数
# =========================
def generate_text(model, start_text, length=50):
    model.eval()

    generated = start_text

    input_seq = [char_to_idx[ch] for ch in start_text]

    hidden = None

    for _ in range(length):
        x = torch.tensor([input_seq[-seq_length:]], dtype=torch.long).to(device)

        # 如果长度不足 seq_length，则左侧补 0
        if x.shape[1] < seq_length:
            padding = torch.zeros((1, seq_length - x.shape[1]), dtype=torch.long).to(device)
            x = torch.cat([padding, x], dim=1)

        output, hidden = model(x, hidden)

        probs = torch.softmax(output[0], dim=0)
        next_idx = torch.argmax(probs).item()
        next_char = idx_to_char[next_idx]

        generated += next_char
        input_seq.append(next_idx)

    return generated

# =========================
# 7. 测试生成
# =========================
print("\n生成结果:")
print(generate_text(model, "hell", length=30))
print(generate_text(model, "how ", length=30))