# cnn_model_letters.py
# CNN 模型定义 + 训练脚本 (EMNIST Letters 手写字母识别)
# 已加入与评估一致的方向修正（rot90 + flip）
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class DeepCNN(nn.Module):
    """高精度 CNN，用于 28x28 灰度手写字母识别 (A-Z, 26类)。"""
    def __init__(self, num_classes=26):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)

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
# 方向修正函数（与 compare.py 完全一致）
# ============================================================
def correct_emnist_orientation(x):
    """修正 EMNIST 图像的旋转和镜像，使其变为正常朝向。

    EMNIST 图像存储时被顺时针旋转90°并水平镜像。
    修正步骤：逆时针旋转90° + 水平翻转 = 恢复人类可读朝向。
    """
    return torch.rot90(x, k=-1, dims=[1, 2]).flip(dims=[2])


# ============================================================
# 训练脚本
# ============================================================
if __name__ == "__main__":
    import numpy as np
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )

    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 数据准备 ----
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(correct_emnist_orientation),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(correct_emnist_orientation),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.EMNIST(
        root='./data', split='letters', train=True,
        download=True, transform=transform_train
    )
    test_dataset = datasets.EMNIST(
        root='./data', split='letters', train=False,
        download=True, transform=transform_eval
    )

    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")

    model = DeepCNN(num_classes=26).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    os.makedirs("weights", exist_ok=True)
    best_val_acc = 0.0
    best_model_path = "./weights/cnn_letters.pth"

    # ---- 记录训练历史 ----
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels - 1

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

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels - 1
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        history['train_loss'].append(total_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
              f"Loss: {total_loss:.4f}  "
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
            labels = labels - 1
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    letter_classes = [chr(ord('A') + i) for i in range(26)]

    # ---- 创建输出目录 ----
    output_dir = os.path.join('data_1', 'cnn_letters')
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. 损失曲线 & 准确率曲线
    # ================================================================
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-', linewidth=1.5, label='训练 Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history['train_acc'], 'b-', linewidth=1.5, label='训练准确率')
    ax2.plot(epochs_range, history['val_acc'], 'r-', linewidth=1.5, label='验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练 & 验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 损失曲线 → {loss_curve_path}")

    # ================================================================
    # 2. 混淆矩阵热力图
    # ================================================================
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(26))
    ax.set_yticks(range(26))
    ax.set_xticklabels(letter_classes, fontsize=8)
    ax.set_yticklabels(letter_classes, fontsize=8)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('CNN 字母识别 — 混淆矩阵 (归一化)')

    # 在格子中标注数值（仅对 > 0.05 的格子标注，避免太密）
    for i in range(26):
        for j in range(26):
            val = cm_normalized[i, j]
            if val > 0.05:
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('比例')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 混淆矩阵 → {cm_path}")

    # ================================================================
    # 3. 每类 Precision / Recall / F1 + 总体指标
    # ================================================================
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # 每类指标柱状图
    x = np.arange(26)
    width = 0.25

    fig, ax = plt.subplots(figsize=(18, 8))
    bars1 = ax.bar(x - width, per_class_precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, per_class_recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, per_class_f1, width, label='F1 Score', color='#e74c3c')

    ax.set_xlabel('字母类别')
    ax.set_ylabel('分数')
    ax.set_title('每类分类指标 (Precision / Recall / F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(letter_classes, fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    # 标注总体指标
    summary_text = (f'Overall Accuracy: {overall_accuracy:.4f}\n'
                    f'Macro Precision:  {macro_precision:.4f}\n'
                    f'Macro Recall:     {macro_recall:.4f}\n'
                    f'Macro F1:         {macro_f1:.4f}')
    ax.text(0.98, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'per_class_metrics.png')
    fig.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[保存] 每类指标图 → {metrics_path}")

    # ================================================================
    # 4. 保存数据文件
    # ================================================================
    # 训练历史
    np.savez(
        os.path.join(output_dir, 'training_history.npz'),
        train_loss=np.array(history['train_loss']),
        train_acc=np.array(history['train_acc']),
        val_acc=np.array(history['val_acc'])
    )
    # 混淆矩阵
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)

    # 每类指标 JSON
    per_class_metrics = {}
    for i, cls in enumerate(letter_classes):
        per_class_metrics[cls] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i])
        }
    metrics_json = {
        'overall_accuracy': float(overall_accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class': per_class_metrics
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"测试集评估结果")
    print(f"{'='*60}")
    print(f"Overall Accuracy : {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Macro Precision  : {macro_precision:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"\n模型已保存至: {best_model_path}")
    print(f"所有指标/图像已保存至: {output_dir}/")
    print(f"  - loss_curve.png         训练损失 & 准确率曲线")
    print(f"  - confusion_matrix.png   归一化混淆矩阵热力图")
    print(f"  - per_class_metrics.png  每类 Precision/Recall/F1")
    print(f"  - training_history.npz   训练历史数据")
    print(f"  - confusion_matrix.npy   混淆矩阵原始数据")
    print(f"  - metrics_summary.json   指标汇总 JSON")