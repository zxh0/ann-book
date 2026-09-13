import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 1. 超参数
# =========================
batch_size = 64
learning_rate = 0.01
epochs = 5

# =========================
# 2. 加载数据
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# 3. 定义神经网络（感知机）
# =========================
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(28 * 28, 10) # 全连接层：784 -> 10

    def forward(self, x):
        x = x.view(-1, 28 * 28) # 展平
        x = self.fc(x)
        return x

# =========================
# 4. 初始化模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Perceptron().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# =========================
# 5. 训练模型
# =========================
for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    
    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# =========================
# 6. 测试模型
# =========================
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
