import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 1. 超参数
# =========================
batch_size = 64
learning_rate = 0.001
epochs = 5

# =========================
# 2. 数据集 (28x28 → 16x16)
# =========================
transform = transforms.Compose([
    transforms.Resize((16,16)),
    transforms.ToTensor()
])

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
# 3. 定义神经网络 (LeCun 1989)
# =========================
class CNN1989(nn.Module):

    def __init__(self):
        super().__init__()

        # 输入 1×16×16
        # 输出 12×8×8
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=12,
            kernel_size=5,
            padding=2,
            stride=2
        )

        # 输入 12×8×8
        # 输出 12×4×4
        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=5,
            padding=2,
            stride=2
        )

        # 全连接层
        self.fc1 = nn.Linear(12 * 4 * 4, 30)
        self.fc2 = nn.Linear(30, 10)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 4. 初始化模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1989().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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
