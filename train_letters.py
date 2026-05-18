import os
import torch
import torch.nn as nn
import torch.optim as optim

from bp_network import BPNetwork
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size=64

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = datasets.EMNIST(
        root='./data',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )

test_dataset = datasets.EMNIST(
        root='./data',
        split='letters',
        train=False,
        download=True,
        transform=transform
    )

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


# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型
model = BPNetwork(
    input_size=784,
    hidden_size=256,
    output_size=26
).to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.view(-1, 784).to(device)

        # labels范围 1~26
        labels = (labels - 1).to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")


# 测试
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.view(-1, 784).to(device)

        labels = (labels - 1).to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print(f"Letters Accuracy: {100 * correct / total:.2f}%")

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "./weights/letters_bp.pth")