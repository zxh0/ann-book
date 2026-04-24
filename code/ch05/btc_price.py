import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. 获取 BTC 最近一年的真实数据 ---
# 自动抓取截至今天的最新日线数据
df = yf.download("BTC-USD", period="1y", interval="1d")
prices = df['Close'].values.reshape(-1, 1)

# 归一化：将价格缩放到 [0, 1]，这是神经网络处理价格波动的标配
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(prices)

# --- 2. 制作滑动窗口数据集 ---
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

seq_length = 7  # 用过去 7 天预测第 8 天
X, Y = create_sequences(scaled_data, seq_length)

# --- 3. 定义最简单的 RNN 模型 ---
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size=1 (价格), hidden_size=32
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        # out 包含所有时间步的隐藏状态，我们只取最后一个 [-1]
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = SimpleRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4. 训练模型 ---
epochs = 100
for epoch in range(epochs):
    model.train()
    output = model(X)
    loss = criterion(output, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# --- 5. 预测未来 ---
model.eval()
with torch.no_grad():
    # 使用最后 7 天的数据作为起点
    last_seq = torch.tensor(scaled_data[-seq_length:].reshape(1, seq_length, 1), dtype=torch.float32)
    prediction_scaled = model(last_seq)
    # 逆归一化回真实价格
    prediction = scaler.inverse_transform(prediction_scaled.numpy())

print(f"\n预测明天的 BTC 价格约为: ${prediction[0][0]:.2f}")

# --- 6. 绘图对比 ---
all_predictions = model(X).detach().numpy()
plt.plot(scaler.inverse_transform(scaled_data[seq_length:]), label="Actual Price")
plt.plot(scaler.inverse_transform(all_predictions), label="RNN Predict")
plt.title("BTC Price Prediction (Last 1 Year)")
plt.legend()
plt.show()