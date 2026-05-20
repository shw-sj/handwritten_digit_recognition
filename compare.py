import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

# 正常导入模型（无循环导入）
from cnn_model_mnist import DeepCNN as DeepCNN_MNIST
from cnn_model_letters import DeepCNN as DeepCNN_Letters
from bp_network import BPNetwork
from matplotlib.widgets import Button
# 工具函数：计算参数量
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

# 🔥 终极评估函数（无bug+全修复）
def evaluate_model(model, test_loader, task, device):
    model.eval()
    correct = 0
    total = 0
    inference_time = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            # 核心修复：字母标签 1~26 → 0~25
            if task == "letters":
                target = target - 1

            data, target = data.to(device), target.to(device)

            # 核心修复：BP模型展平输入
            if isinstance(model, BPNetwork):
                data = data.view(-1, 28 * 28)

            # 推理
            start = time.time()
            output = model(data)
            inference_time += time.time() - start

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_time = (inference_time / total) * 1000
    return accuracy, avg_time

# 🔥 关键修复：和训练代码完全一致的预处理（含EMNIST图像方向修正）
def load_datasets(batch_size=64):
    # ✅ MNIST数字预处理（不变）
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ✅ EMNIST字母预处理（必须旋转+翻转！和训练代码一致）
    letters_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1,2])),  # 旋转90度
        transforms.Lambda(lambda x: torch.flip(x, dims=[2])),         # 水平翻转
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    mnist_test = MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    letters_test = EMNIST(root='./data', split='letters', train=False, download=True, transform=letters_transform)
    letters_loader = DataLoader(letters_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return mnist_loader, letters_loader

# 主函数（无报错版）
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    mnist_loader, letters_loader = load_datasets()

    print("\n=== 加载模型 ===")
    # 1. 数字模型
    mnist_cnn = DeepCNN_MNIST().to(device)
    mnist_cnn.load_state_dict(torch.load("./weights/cnn_mnist.pth", map_location=device))

    mnist_bp = BPNetwork(input_size=784, output_size=10, task="digits").to(device)
    mnist_bp.load_state_dict(torch.load("./weights/mnist_bp.pth", map_location=device))

    # 2. 字母模型
    letters_cnn = DeepCNN_Letters().to(device)
    letters_cnn.load_state_dict(torch.load("./weights/cnn_letters.pth", map_location=device))

    letters_bp = BPNetwork(input_size=784, output_size=26, task="letters").to(device)
    letters_bp.load_state_dict(torch.load("./weights/letters_bp.pth", map_location=device))

    # 评估模型
    print("\n=== 开始评估 ===")
    mnist_cnn_acc, mnist_cnn_time = evaluate_model(mnist_cnn, mnist_loader, "digits", device)
    mnist_bp_acc, mnist_bp_time = evaluate_model(mnist_bp, mnist_loader, "digits", device)

    letters_cnn_acc, letters_cnn_time = evaluate_model(letters_cnn, letters_loader, "letters", device)
    letters_bp_acc, letters_bp_time = evaluate_model(letters_bp, letters_loader, "letters", device)

    # 打印结果
    print("\n===== MNIST 手写数字识别结果 =====")
    print(f"CNN 模型 | 准确率: {mnist_cnn_acc:.2f}%")
    print(f"BP 模型  | 准确率: {mnist_bp_acc:.2f}%")

    print("\n===== EMNIST 手写字母识别结果 =====")
    print(f"CNN 模型 | 准确率: {letters_cnn_acc:.2f}%")
    print(f"BP 模型  | 准确率: {letters_bp_acc:.2f}%")

    # 可视化
       # ========== 可视化（按钮切换） ==========
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    models = ['数字CNN', '数字BP', '字母CNN', '字母BP']
    accuracies = [mnist_cnn_acc, mnist_bp_acc, letters_cnn_acc, letters_bp_acc]
    params = [count_params(mnist_cnn), count_params(mnist_bp), count_params(letters_cnn), count_params(letters_bp)]
    times = [mnist_cnn_time, mnist_bp_time, letters_cnn_time, letters_bp_time]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # 初始显示准确率
    ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_title('模型准确率对比 (%)')

    def show_accuracy(event):
        ax.clear()
        ax.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('模型准确率对比 (%)')
        plt.draw()

    def show_params(event):
        ax.clear()
        ax.bar(models, params, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('模型参数量对比 (M)')
        plt.draw()

    def show_time(event):
        ax.clear()
        ax.bar(models, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title('推理速度对比 (ms/样本)')
        plt.draw()

    # 三个按钮
    ax_btn1 = plt.axes([0.15, 0.05, 0.2, 0.075])
    ax_btn2 = plt.axes([0.40, 0.05, 0.2, 0.075])
    ax_btn3 = plt.axes([0.65, 0.05, 0.2, 0.075])

    btn_acc = Button(ax_btn1, '准确率')
    btn_params = Button(ax_btn2, '训练参数量')
    btn_time = Button(ax_btn3, '推理速度')

    btn_acc.on_clicked(show_accuracy)
    btn_params.on_clicked(show_params)
    btn_time.on_clicked(show_time)

    plt.show()

if __name__ == '__main__':
    main()