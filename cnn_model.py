# 加载数据集
from image_loader import load_torchvision_data

# 构建CNN模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return model


def evaluate(model, test_loader):
    device = torch.cuda.device_count() if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100.0 * correct / total:.2f}%')


def numpy_to_loader(X, y, batch_size=64, shuffle=True):
    """将 numpy 数组转为 DataLoader"""
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,H,W) -> (N,1,H,W)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    digits = load_torchvision_data('mnist', verbose=False)
    print('数字数据集:')
    print(f'  训练: {digits["X_train"].shape}, 标签: {digits["y_train"].shape}')
    print(f'  验证: {digits["X_val"].shape},   标签: {digits["y_val"].shape}')
    print(f'  测试: {digits["X_test"].shape},  标签: {digits["y_test"].shape}')
    print(f'  类别: {digits["classes"]}')

    train_loader = numpy_to_loader(digits['X_train'], digits['y_train'], shuffle=True)
    val_loader = numpy_to_loader(digits['X_val'], digits['y_val'], shuffle=False)
    test_loader = numpy_to_loader(digits['X_test'], digits['y_test'], shuffle=False)

    model = CNNModel(num_classes=len(digits['classes']))
    model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    print()
    evaluate(model, test_loader)

    torch.save(model.state_dict(), 'cnn_model.pth')
    print(f"\n模型已保存到 cnn_model.pth")


if __name__ == "__main__":
    main()
