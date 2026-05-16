import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from bp_network import BPNet


# -------------------------
# 1. 加载测试集
# -------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------
# 2. 加载模型
# -------------------------

model = BPNet()

# 如果你保存过模型：
model.load_state_dict(torch.load("mnist_bp.pth"))

# 切换到测试模式
model.eval()


# -------------------------
# 3. 开始测试
# -------------------------

all_labels = []
all_preds = []


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        # 预测类别
        _, predicted = torch.max(outputs, 1)

        # 保存真实标签
        all_labels.extend(labels.numpy())

        # 保存预测结果
        all_preds.extend(predicted.numpy())


# -------------------------
# 4. 计算指标
# -------------------------

acc = accuracy_score(all_labels, all_preds)

precision = precision_score(
    all_labels,
    all_preds,
    average='macro'
)

recall = recall_score(
    all_labels,
    all_preds,
    average='macro'
)

f1 = f1_score(
    all_labels,
    all_preds,
    average='macro'
)


# -------------------------
# 5. 输出结果
# -------------------------

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# 每个类别详细指标
print("\n分类报告：")
print(classification_report(all_labels, all_preds))