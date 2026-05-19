import torch.nn as nn

class BPNetwork(nn.Module):

    def __init__(self, input_size, output_size, task='digits'):
        super(BPNetwork, self).__init__()

        self.task = task

        # 共享的基础层（所有任务都用）
        self.base = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 任务特定的扩展层
        if task == 'letters':
            # 字母识别（26类）：添加额外的隐藏层
            self.task_specific = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128, output_size)
            )

        elif task == 'digits':
            # 数字识别（10类）：简单直接输出
            self.task_specific = nn.Sequential(
                nn.Linear(256, output_size)
            )

        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x):
        x = self.base(x)  # 通过基础层
        x = self.task_specific(x)  # 通过任务特定层
        return x