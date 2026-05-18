# cnn_model.py
# 成员C：CNN 模型定义 (PyTorch)
# 支持手写数字(10类)和手写字母(26类)

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

class LeNet5(nn.Module):
    """
    LeNet-5 变体，适用于 28x28 灰度图像输入
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 卷积层1: 输入1通道，输出6通道，卷积核5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 输出28x28
        # 池化层: 2x2 平均池化
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)        # 输出14x14
        # 卷积层2: 输入6通道，输出16通道，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 输出10x10 (无padding)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                    # 经过第二次池化后为5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))   # -> (batch, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (batch, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)             # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                        # 不加 softmax，因为训练时用 CrossEntropyLoss
        return x

# 可选：更简单的 2 层卷积网络，适合小数据集
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28->14
        x = self.pool(F.relu(self.conv2(x)))   # 14->7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 测试模型结构
    model = LeNet5(num_classes=10)
    dummy = torch.randn(1, 1, 28, 28)
    out = model(dummy)
    print(f"LeNet5 output shape: {out.shape}")  # (1,10)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")