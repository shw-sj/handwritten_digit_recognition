import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets,transforms
from bp_network import BPNetwork
from torch.utils.data import DataLoader


BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),
    transforms.Lambda(lambda x: torch.flip(x, [2])),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型
model = BPNetwork(input_size=784,output_size=26,task="letters").to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

os.makedirs("weights", exist_ok=True)
best_model_path = "./weights/bp_letters.pth"

def train(model, train_loader, criterion, optimizer, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.view(-1, 784).to(device)
            labels = (labels-1).to(device)

            # 前向传播
            outputs = model(images)

            loss = criterion(outputs, labels)

            # 反向传播
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
            for images, labels in test_loader:
                images, labels = images.view(-1, 784).to(device), (labels-1).to(device)
                outputs = model(images)
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
        for images, labels in test_loader:
            images, labels = images.view(-1, 784).to(device), (labels-1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    letter_classes = [chr(ord('A') + i) for i in range(26)]

    return all_preds, all_labels, letter_classes

all_preds, all_labels, digit_classes = evaluate(model, test_loader, best_model_path)
# ---- 创建输出目录 ----
output_dir = os.path.join('data_letters', 'BP')
os.makedirs(output_dir, exist_ok=True)

from metrics import draw_trainloss, draw_cm, draw_per_class_metric
# ================================================================
# 1. 损失曲线 & 准确率曲线
# ================================================================
draw_trainloss(EPOCHS, history, output_dir)

# ================================================================
# 2. 混淆矩阵热力图
# ================================================================
draw_cm(all_labels, all_preds, digit_classes, output_dir, task="letters")

# ================================================================
# 3. 每类 Precision / Recall / F1 + 总体指标
# ================================================================
draw_per_class_metric(all_labels, all_preds, digit_classes, output_dir, task="letters")