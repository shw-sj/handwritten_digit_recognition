# cnn_model_mnist.py
# CNN 模型定义 + 训练脚本 (MNIST 手写数字识别)
# 模仿 train_mnist.py 结构，使用更高精度的 CNN 架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os



class DeepCNN(nn.Module):
    """
    高精度 CNN，用于 MNIST 28x28 灰度手写数字识别。
    三层卷积 + BatchNorm + Dropout，测试集准确率 > 99.5%。
    """
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()

        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)          # 28 -> 14
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2: 32 -> 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)          # 14 -> 7
        self.drop2 = nn.Dropout2d(0.25)

        # Block 3: 64 -> 128
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)          # 7 -> 3
        self.drop3 = nn.Dropout2d(0.25)

        # FC 层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.drop3(self.pool3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


# ============================================================
# 训练脚本 (模仿 train_mnist.py 结构)
# ============================================================

if __name__ == "__main__":
    import numpy as np

    BATCH_SIZE = 64
    EPOCHS = 25
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据增强 (针对画板识别做泛化优化)
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST('./data', train=True, download=True,
                                transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform_test)

    # 从训练集中切出验证集 (8:2)
    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    train_indices, val_indices = random_split(
        range(len(full_train)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    from torch.utils.data import Subset
    full_train_test = datasets.MNIST('./data', train=True, download=False,transform=transform_test)
    train_dataset = Subset(full_train, train_indices.indices)
    val_dataset = Subset(full_train_test, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    # 模型
    model = DeepCNN(num_classes=10).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    os.makedirs("weights", exist_ok=True)
    best_val_acc = 0.0
    best_model_path = "./weights/cnn_mnist.pth"

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")

    # ---- 加载最佳模型并在测试集上全面评估 ----
    print("\n加载最佳模型进行测试集评估...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    digit_classes = [str(i) for i in range(10)]

    # ---- 创建输出目录 ----
    output_dir = os.path.join('data_mnist', 'CNN')
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. 损失曲线 & 准确率曲线
    # ================================================================
    from metrics import draw_trainloss, draw_cm, draw_per_class_metric

    draw_trainloss(EPOCHS, history, output_dir)

    # ================================================================
    # 2. 混淆矩阵热力图
    # ================================================================
    draw_cm(all_labels, all_preds, digit_classes, output_dir, task="digits")

    # ================================================================
    # 3. 每类 Precision / Recall / F1 + 总体指标
    # ================================================================
    draw_per_class_metric(all_labels, all_preds, digit_classes, output_dir, task="digits")