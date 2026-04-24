import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =========================
# 1 CNN (LeCun 1989)
# =========================
class CNN1989(nn.Module):

    def __init__(self):
        super().__init__()

        # 输入 1×16×16
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=12,
            kernel_size=5,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=12,
            kernel_size=5,
            stride=2
        )

        self.tanh = nn.Tanh()

        # 计算 flatten size
        # 16x16
        # conv1 (k=5,s=2) -> 6x6
        # conv2 (k=5,s=2) -> 1x1

        self.fc1 = nn.Linear(12 * 1 * 1, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):

        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = self.tanh(self.fc1(x))
        x = self.fc2(x)

        return x


# =========================
# 2 数据集 (28x28 → 16x16)
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# =========================
# 3 初始化
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1989().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# =========================
# 4 训练
# =========================
epochs = 5

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss {total_loss/len(train_loader):.4f}")


# =========================
# 5 测试
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


print("Test Accuracy:", 100 * correct / total)