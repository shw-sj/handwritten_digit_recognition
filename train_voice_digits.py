import torch
import torch.nn as nn
import torch.optim as optim

from bp_network import BPNetwork
from voice_dataset import get_voice_loader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 数据
train_loader = get_voice_loader(
    "./data/voice_digits"
)

# 输入大小
input_size = 40 * 32

# 模型
model = BPNetwork(
    input_size=input_size,
    output_size=10,
).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20
best_acc = 0
for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0
    total = 0


    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {total_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "./weights/voice_digits_bp.pth")


print("模型已保存")