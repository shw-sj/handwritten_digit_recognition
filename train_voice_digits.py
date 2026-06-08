import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bp_network import BPNetwork
from voice_dataset import get_voice_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# 输入大小
input_size = 40 * 32

# 数据
train_loader, test_loader = get_voice_loader("./data/voice_digits",batch_size=BATCH_SIZE)

# 模型
model = BPNetwork(input_size=input_size,output_size=10).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

os.makedirs("weights", exist_ok=True)
best_model_path = "./weights/bp_voice_digits.pth"

def train(model, train_loader, criterion, optimizer, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        train_correct = 0
        train_total = 0
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

            train_total += labels.size(0)

            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        history['train_loss'].append(total_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch + 1:d}/{EPOCHS}]  "
              f"Loss: {total_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")

    return history

history = train(model, train_loader, criterion, optimizer, epochs=EPOCHS)

def evaluate(model, test_loader, best_model_path=None):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    digit_classes = [str(i) for i in range(10)]

    return all_preds, all_labels, digit_classes

all_preds, all_labels, digit_classes = evaluate(model, test_loader, best_model_path)
# ---- 创建输出目录 ----
output_dir = os.path.join('data_mnist', 'BP_voice')
os.makedirs(output_dir, exist_ok=True)


from metrics import draw_trainloss, draw_cm, draw_per_class_metric
# ================================================================
# 1. 损失曲线 & 准确率曲线
# ================================================================
draw_trainloss(EPOCHS, history, output_dir)

# ================================================================
# 2. 混淆矩阵热力图
# ================================================================
draw_cm(all_labels, all_preds, digit_classes, output_dir, task="digits")

# ================================================================
# 3. 每类 Precision / Recall / F1 + 总体指标
# ================================================================
draw_per_class_metric(all_labels, all_preds, digit_classes, output_dir, task="digits")