import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================
# 1 定义 CNN 网络
# ==========================
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        # Conv layer 1
        # 1×28×28 -> 6×28×28
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            padding=2
        )

        # MaxPool
        # 6×28×28 -> 6×14×14
        self.pool = nn.MaxPool2d(2, 2)

        # Conv layer 2
        # 6×14×14 -> 16×10×10
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        # Fully connected layers
        # 16×5×5 = 400
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Conv1
        x = self.relu(self.conv1(x))

        # Pool1
        x = self.pool(x)

        # Conv2
        x = self.relu(self.conv2(x))

        # Pool2
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 16 * 5 * 5)

        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# ==========================
# 2 加载 MNIST 数据
# ==========================
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# ==========================
# 3 初始化模型
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet5().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==========================
# 4 训练模型
# ==========================
epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


# ==========================
# 5 测试准确率
# ==========================
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

print(f"\nTest Accuracy: {accuracy:.2f}%")