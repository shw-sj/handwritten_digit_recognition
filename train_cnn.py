# train_cnn.py 适配 EMNIST 混合数据集版本
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from cnn_model import LeNet5, SimpleCNN  # 保持原有模型导入

# -----------------------------
# 配置参数
# -----------------------------
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# 适配 prepare_data.py 的数据保存路径
EMNIST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "emnist")

def load_data(mode='mixed'):
    """
    加载 prepare_data.py 预处理后的 EMNIST 数据
    mode: 
      - 'mixed': 数字0-9 + 字母A-Z（47类，默认）
      - 'digits': 仅数字0-9（从混合数据中筛选）
      - 'letters': 仅字母A-Z（从混合数据中筛选）
    返回: (train_dataset, val_dataset, test_dataset, num_classes)
         train_dataset: TensorDataset，图像 shape (N, 1, 28, 28)，值范围 [0,1]
    """
    # 1. 加载 prepare_data.py 保存的原始数据
    X_train = np.load(os.path.join(EMNIST_DATA_DIR, "X_train.npy"))  # (N, 28, 28) uint8
    y_train = np.load(os.path.join(EMNIST_DATA_DIR, "y_train.npy"))  # (N,) int64
    X_test = np.load(os.path.join(EMNIST_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(EMNIST_DATA_DIR, "y_test.npy"))

    # 2. 根据模式筛选数据（EMNIST balanced 标签规则：0-9=数字，10-35=大写字母，36-46=小写字母）
    if mode == 'digits':
        # 仅保留数字 0-9（标签0-9）
        train_mask = y_train <= 9
        test_mask = y_test <= 9
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        num_classes = 10
    elif mode == 'letters':
        # 仅保留字母 A-Z（合并大小写，统一映射为0-25）
        # 大写字母标签10-35 → 映射为0-25；小写字母36-46舍弃（或按需保留）
        train_mask = (y_train >= 10) & (y_train <= 35)
        test_mask = (y_test >= 10) & (y_test <= 35)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask] - 10  # 10→0, 11→1 ... 35→25
        X_test = X_test[test_mask]
        y_test = y_test[test_mask] - 10
        num_classes = 26
    elif mode == 'mixed':
        # 保留全部 47 类（数字0-9 + 大小写字母）
        num_classes = 47
    else:
        raise ValueError("mode must be 'mixed', 'digits' or 'letters'")

    # 3. 从训练集中拆分验证集（8:2 分割）
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 4. 数据归一化 + 维度调整
    # 归一化到 [0,1]（uint8→float32）
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # 增加通道维度: (N, 28, 28) → (N, 1, 28, 28)（适配CNN输入）
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    # 5. 转换为PyTorch Dataset
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long())
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())

    # 打印数据信息
    print(f"Loaded {mode} data:")
    print(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples | Test: {len(test_dataset)} samples")
    print(f"  Number of classes: {num_classes}")
    return train_dataset, val_dataset, test_dataset, num_classes

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_save_path):
    """
    训练模型并保存最佳模型（基于验证集精度）
    保留原有训练逻辑，无修改
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算本轮指标
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total

        print(f'Epoch [{epoch+1}/{epochs}] | '
              f'Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_path)
            print(f'Best model updated! Saved to {model_save_path} (Val Acc: {best_val_acc:.2f}%)')

    # 加载最佳模型并返回
    model.load_state_dict(best_model_state)
    return model

def main():
    # 1. 选择训练模式：mixed（47类）/ digits（10类）/ letters（26类）
    train_mode = 'mixed'  # 核心修改：默认训练数字+字母混合任务

    # 2. 加载数据（适配 prepare_data.py 的输出）
    train_dataset, val_dataset, test_dataset, num_classes = load_data(mode=train_mode)

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型（适配类别数）
    model = LeNet5(num_classes=num_classes)  # 替换为 SimpleCNN(num_classes) 也可

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 定义模型保存路径（区分模式）
    model_name = f'lenet5_{train_mode}.pth'  # 如 lenet5_mixed.pth / lenet5_digits.pth
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)

    # 7. 训练模型
    print(f"\nStart training {model.__class__.__name__} for {train_mode} (num_classes={num_classes})...")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                                EPOCHS, DEVICE, model_save_path)

    # 8. 测试最佳模型
    trained_model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total
    print(f'\n✅ Final Test Accuracy (best model): {test_acc:.2f}%')
    print(f'📌 Model saved to: {model_save_path}')

if __name__ == '__main__':
    main()