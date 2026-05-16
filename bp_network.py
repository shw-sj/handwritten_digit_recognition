import torch.nn as nn
import torch.nn.functional as F


class BPNet(nn.Module):

    def __init__(self):
        super(BPNet, self).__init__()

        # 输入层 -> 隐藏层1
        self.fc1 = nn.Linear(784, 128)

        # 隐藏层1 -> 隐藏层2
        self.fc2 = nn.Linear(128, 64)

        # 隐藏层2 -> 输出层
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        # 把 28x28 图片拉平成 784
        x = x.view(-1, 784)

        # 第一层 + 激活函数
        x = F.relu(self.fc1(x))

        # 第二层 + 激活函数
        x = F.relu(self.fc2(x))

        # 输出层
        x = self.fc3(x)

        return x