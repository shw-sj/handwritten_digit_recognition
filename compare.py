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

# ========== 工具函数 ==========
def count_params(model):
    """计算模型参数量（M）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def evaluate_model(model, test_loader, task, device):
    """评估模型准确率和推理速度"""
    model.eval()
    correct = 0
    total = 0
    inference_time = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            if task == "letters":
                target = target - 1   # EMNIST letters: 1~26 -> 0~25

            data, target = data.to(device), target.to(device)

            # BP网络需要展平输入
            if isinstance(model, BPNetwork):
                data = data.view(-1, 28 * 28)

            start = time.time()
            output = model(data)
            inference_time += time.time() - start

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_time = (inference_time / total) * 1000  # ms per sample
    return accuracy, avg_time

# ========== 数据加载（分别处理CNN和BP的归一化）==========
def load_datasets(batch_size=64):
    """返回三个DataLoader: MNIST测试集, 字母CNN测试集, 字母BP测试集"""
    # 数字测试集（CNN和BP统一使用MNIST标准归一化）
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 方向修正函数（与EMNIST letters训练时一致）
    def correct_orientation(x):
        # x是Tensor (C,H,W)，执行旋转和翻转
        return torch.rot90(x, k=1, dims=[1, 2]).flip(dims=[2])

    # 字母CNN测试集：方向修正 + MNIST归一化（与CNN训练时一致）
    letters_transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(correct_orientation),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 字母BP测试集：方向修正 + (0.5,0.5)归一化（与BP训练时一致）
    letters_transform_bp = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(correct_orientation),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数字测试集
    mnist_test = MNIST(root='./data', train=False, download=True, transform=mnist_transform)
    mnist_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # 加载字母测试集（同一个数据集，两种预处理）
    letters_test_cnn = EMNIST(root='./data', split='letters', train=False, download=True,
                               transform=letters_transform_cnn)
    letters_test_bp = EMNIST(root='./data', split='letters', train=False, download=True,
                              transform=letters_transform_bp)

    letters_loader_cnn = DataLoader(letters_test_cnn, batch_size=batch_size, shuffle=False, num_workers=0)
    letters_loader_bp = DataLoader(letters_test_bp, batch_size=batch_size, shuffle=False, num_workers=0)

    return mnist_loader, letters_loader_cnn, letters_loader_bp

# ========== 主函数 ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据（三个加载器）
    mnist_loader, letters_loader_cnn, letters_loader_bp = load_datasets(batch_size=64)

    print("\n=== 加载模型 ===")
    # 数字模型
    mnist_cnn = DeepCNN_MNIST().to(device)
    mnist_cnn.load_state_dict(torch.load("./weights/cnn_mnist.pth", map_location=device))

    mnist_bp = BPNetwork(input_size=784, output_size=10, task="digits").to(device)
    mnist_bp.load_state_dict(torch.load("./weights/mnist_bp.pth", map_location=device))

    # 字母模型
    letters_cnn = DeepCNN_Letters().to(device)
    letters_cnn.load_state_dict(torch.load("./weights/cnn_letters.pth", map_location=device))

    letters_bp = BPNetwork(input_size=784, output_size=26, task="letters").to(device)
    letters_bp.load_state_dict(torch.load("./weights/letters_bp.pth", map_location=device))

    # 评估模型
    print("\n=== 开始评估 ===")
    # 数字
    mnist_cnn_acc, mnist_cnn_time = evaluate_model(mnist_cnn, mnist_loader, "digits", device)
    mnist_bp_acc, mnist_bp_time = evaluate_model(mnist_bp, mnist_loader, "digits", device)

    # 字母CNN（使用CNN专用加载器）
    letters_cnn_acc, letters_cnn_time = evaluate_model(letters_cnn, letters_loader_cnn, "letters", device)
    # 字母BP（使用BP专用加载器）
    letters_bp_acc, letters_bp_time = evaluate_model(letters_bp, letters_loader_bp, "letters", device)

    # 打印结果
    print("\n===== MNIST 手写数字识别结果 =====")
    print(f"CNN 模型 | 准确率: {mnist_cnn_acc:.2f}%")
    print(f"BP 模型  | 准确率: {mnist_bp_acc:.2f}%")

    print("\n===== EMNIST 手写字母识别结果 =====")
    print(f"CNN 模型 | 准确率: {letters_cnn_acc:.2f}%")
    print(f"BP 模型  | 准确率: {letters_bp_acc:.2f}%")

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