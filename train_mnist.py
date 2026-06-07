import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from bp_network import BPNetwork
from torch.utils.data import DataLoader
from metrics import evaluate_classification

batch_size=64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型
model = BPNetwork(
    input_size=784,
    output_size=10
).to(device)

train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss:.4f}")

train(model, train_loader, criterion, optimizer, epochs=10)

# 保存模型
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "./weights/mnist_bp.pth")
print("Model saved!")
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 使用sklearn的评估函数
    metrics = evaluate_classification(
        np.array(all_labels),
        np.array(all_preds),
        average="macro"
    )

    return metrics

# 评估模型

evaluate(model, test_loader)