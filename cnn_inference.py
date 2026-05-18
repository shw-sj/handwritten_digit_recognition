import torch
import torch.nn as nn
import numpy as np
import os  # 新增：导入os处理路径

class CNNInference:
    def __init__(self, mode='digits', weight_path=None):
        """
        初始化CNN推理模型
        :param mode: 识别模式 - 'digits'（数字，10类）/ 'letters'（字母，26类）
        :param weight_path: 权重文件路径（优先使用传入的路径）
        """
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 权重路径优先级：传入路径 > 原有默认路径
        if weight_path is not None and os.path.exists(weight_path):
            self.weight_path = weight_path
            print(f"[INFO] 使用自定义权重：{self.weight_path}")
        else:
            # 保留原有默认路径（可选，防止权重文件缺失时兜底）
            self.weight_path = f"lenet5_{mode}.pth"
            print(f"[WARNING] 自定义权重不存在，使用默认：{self.weight_path}")
        
        # 2. 构建模型（需与训练lenet5_mixed.pth时的结构一致）
        self.model = self._build_model()
        
        # 3. 加载权重
        try:
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            print(f"[SUCCESS] 加载CNN权重：{self.weight_path} (mode: {mode})")
        except Exception as e:
            raise RuntimeError(f"加载权重失败！请检查模型结构是否匹配：{e}")
        
        # 4. 设置推理模式（禁用Dropout/BatchNorm）
        self.model.eval()

    def _build_model(self):
        """构建LeNet5模型（需与训练时的结构完全一致！）"""
        # 根据模式确定分类数（数字=10类，字母=26类；若混合训练则改为36类）
        if self.mode == 'digits':
            num_classes = 10
        elif self.mode == 'letters':
            num_classes = 26
        # 若你的lenet5_mixed.pth是「数字+字母混合训练（36类）」，取消下面注释：
        # else:
        #     num_classes = 36

        # 标准LeNet5结构（适配MNIST 28x28单通道输入）
        class LeNet5(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 输入1通道，输出6通道
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # 池化后尺寸14x14
                
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 输出16通道，尺寸10x10
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # 池化后尺寸5x5
                
                self.fc1 = nn.Linear(16 * 5 * 5, 120)                   # 全连接层1
                self.relu3 = nn.ReLU()
                self.fc2 = nn.Linear(120, 84)                           # 全连接层2
                self.relu4 = nn.ReLU()
                self.fc3 = nn.Linear(84, num_classes)                   # 输出层（分类数匹配）

            def forward(self, x):
                # 输入shape: (batch, 1, 28, 28)
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = x.view(x.size(0), -1)  # 展平为一维：(batch, 16*5*5)
                x = self.relu3(self.fc1(x))
                x = self.relu4(self.fc2(x))
                x = self.fc3(x)            # 输出logits（未经过softmax）
                return x

        # 初始化模型并移至指定设备（CPU/GPU）
        model = LeNet5(num_classes).to(self.device)
        return model

    def predict(self, img_array):
        """
        推理单张图像
        :param img_array: 28x28的numpy数组，值域0~1（float32）
        :return: 预测标签（str）、类别概率数组（numpy）
        """
        # 转换为模型输入格式：(batch, channel, height, width)
        x = torch.from_numpy(img_array).float()
        x = x.unsqueeze(0).unsqueeze(0)  # 增加batch和channel维度 → (1,1,28,28)
        x = x.to(self.device)

        # 无梯度推理（提升速度+避免显存占用）
        with torch.no_grad():
            logits = self.model(x)                  # 模型输出logits
            probs = torch.softmax(logits, dim=1)    # 转换为概率（0~1）
            probs = probs.cpu().numpy()[0]          # 转为numpy数组

        # 解析预测结果
        pred_idx = np.argmax(probs)
        if self.mode == 'digits':
            pred = str(pred_idx)                    # 数字：0-9
        elif self.mode == 'letters':
            pred = chr(ord('A') + pred_idx)         # 字母：A-Z
        # 若混合训练（36类），需补充映射逻辑：
        # elif self.mode == 'mixed':
        #     pred = str(pred_idx) if pred_idx <10 else chr(ord('A') + pred_idx-10)
        
        return pred, probs