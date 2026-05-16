import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from bp_network import BPNet


# -------------------------
# 1. 数据预处理
# -------------------------

transform = transforms.Compose([
    transforms.ToTensor(),   # 转 Tensor
])


# 下载 MNIST 数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# -------------------------
# 2. 创建网络
# -------------------------

model = BPNet()


# -------------------------
# 3. 损失函数
# -------------------------

criterion = nn.CrossEntropyLoss()


# -------------------------
# 4. 优化器
# -------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# 5. 开始训练
# -------------------------

epochs = 5

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # BP反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {running_loss:.4f}")

print("训练完成！")

torch.save(model.state_dict(), "mnist_bp.pth")

print("模型保存成功！")