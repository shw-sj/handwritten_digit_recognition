"""
图像数据加载与预处理模块 [成员A]

提供统一接口供成员B调用：
  - load_images()        加载原始图像并预处理
  - split_dataset()      按比例划分训练/验证/测试集
  - preprocess_single()  单张图像预处理（GUI画板实时识别用）
  - load_mnist()         加载MNIST数据集作为补充

预处理流水线：原始扫描图
    ↓
灰度化
    ↓
二值化去噪
    ↓
提取数字区域
    ↓
缩放到 20×20
    ↓
抗锯齿平滑
    ↓
质心居中
    ↓
嵌入 28×28
    ↓
归一化
    ↓
输入神经网络
"""

import os
import cv2
import struct
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# 预处理核心函数
# ============================================================
def to_uint8(img):
    """统一转 uint8"""
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.max() <= 1.0:
        img = (img * 255)

    return img.astype(np.uint8)


def binarize(img, thresh=30):
    """二值化（保留黑底白字结构）"""
    _, th = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return th


def crop_digit(img):
    """裁剪数字区域"""
    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def resize_keep_ratio(img, size=20):
    """等比例缩放到 size×size 内"""
    h, w = img.shape

    if h > w:
        new_h = size
        new_w = max(1, int(w * size / h))
    else:
        new_w = size
        new_h = max(1, int(h * size / w))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_28x28(img_20x20ish):
    """放入 28x28 画布并居中"""
    canvas = np.zeros((28, 28), dtype=np.uint8)

    h, w = img_20x20ish.shape

    x_offset = (28 - w) // 2
    y_offset = (28 - h) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img_20x20ish
    return canvas


def center_by_mass(img):
    """质心居中（MNIST关键步骤）"""
    moments = cv2.moments(img)

    if moments["m00"] == 0:
        return img

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    shift_x = 14 - cx
    shift_y = 14 - cy

    M = np.float32([[1, 0, shift_x],
                    [0, 1, shift_y]])

    return cv2.warpAffine(img, M, (28, 28))

def preprocess_single(img):
    """
        完整 MNIST 风格预处理
        输入：numpy / PIL（黑底白字）
        输出：28x28 uint8 图像
        前景白、背景黑
        自动裁剪数字区域
        保持比例缩放到 20×20
        放到 28×28 中央
        使用质心居中
        像素归一化
        """
    # 1. 类型统一
    img = to_uint8(img)

    # 2. 二值化
    img = binarize(img)

    # 3. 裁剪
    cropped = crop_digit(img)
    if cropped is None:
        return np.zeros((28, 28), dtype=np.uint8)

    # 4. 缩放
    resized = resize_keep_ratio(cropped, size=20)

    # 5. 放入 28x28
    canvas = center_28x28(resized)

    # 6. 质心居中
    canvas = center_by_mass(canvas)

    return canvas


def preprocess_letter(img):
    """
    字母识别专用预处理 —— 保留灰度信息，不做二值化。

    与 preprocess_single 的关键区别：
      - 不二值化 → 保留抗锯齿灰度过渡
      - 其余步骤一致：裁剪 → 等比缩放 → 居中

    输入：numpy / PIL（黑底白字）
    输出：28×28 float32，范围 [0, 1]
    """
    img = to_uint8(img)

    # 用非零像素定位内容区域
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    pad_x = max(1, int(w * 0.15))
    pad_y = max(1, int(h * 0.15))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img.shape[1], x + w + pad_x)
    y2 = min(img.shape[0], y + h + pad_y)
    cropped = img[y1:y2, x1:x2]

    # 等比缩放到 20×20 以内
    h_c, w_c = cropped.shape
    scale = 20.0 / max(h_c, w_c)
    new_h = max(1, int(h_c * scale))
    new_w = max(1, int(w_c * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 居中放入 28×28 画布
    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized.astype(np.float32) / 255.0

    return canvas


# ============================================================
# 批量数据加载与数据集划分
# ============================================================

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pgm', '.ppm'}


def load_images(data_dir, target_size=28, verbose=True):
    """
    从目录加载图像并按类别标签组织。

    目录结构要求:
        data_dir/
            class_0/    # 标签 0 的图像
            class_1/    # 标签 1 的图像
            ...
    或:
        data_dir/
            0_xxx.png    # 文件名以标签开头
            1_xxx.png
            ...

    参数:
        data_dir:    数据目录路径
        target_size: 预处理后的尺寸
        verbose:     是否打印进度

    返回:
        images:  np.ndarray, shape (N, target_size, target_size), uint8
        labels:  np.ndarray, shape (N,), int (0-based class index)
        classes: list[str], 类别名称列表
    """
    images, labels = [], []
    class_dirs = {}

    # 检查是否为子目录结构（每类一个文件夹）
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            subdir = os.path.join(data_dir, name)
            if os.path.isdir(subdir):
                class_dirs[name] = subdir

    if class_dirs:
        # 子目录模式
        class_names = sorted(class_dirs.keys())
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = class_dirs[cls_name]
            cls_idx = class_to_idx[cls_name]
            files = [f for f in os.listdir(cls_dir)
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
            if verbose:
                print(f"  加载类别 '{cls_name}' ({cls_idx}): {len(files)} 张图像")

            for fname in files:
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath)
                    processed = preprocess_single(img)
                    images.append(processed)
                    labels.append(cls_idx)
                except Exception as e:
                    print(f"  [警告] 跳过 {fpath}: {e}")

        classes = class_names

    else:
        # 扁平目录模式：文件名必须包含标签信息
        files = [f for f in os.listdir(data_dir)
                 if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        label_set = set()
        for fname in files:
            # 尝试从文件名解析标签：取第一个非字母部分前的数字
            base = os.path.splitext(fname)[0]
            parts = base.replace('_', ' ').replace('-', ' ').split()
            if parts and parts[0].isdigit():
                label_set.add(int(parts[0]))
            else:
                label_set.add(base[0])

        classes = sorted(list(label_set), key=lambda x: str(x))
        #主要支持子目录模式

    if verbose:
        print(f"  总计: {len(images)} 张图像, {len(classes)} 个类别")

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32), classes


def split_dataset(images, labels, train_ratio=0.6, val_ratio=0.2,
                  test_ratio=0.2, stratify=True, random_seed=42):
    """
    划分训练集、验证集、测试集。

    参数:
        images:      np.ndarray (N, H, W)
        labels:      np.ndarray (N,)
        train_ratio: 训练集比例
        val_ratio:   验证集比例
        test_ratio:  测试集比例
        stratify:    是否按类别分层采样
        random_seed: 随机种子

    返回:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为 1.0"

    strat = labels if stratify else None

    if test_ratio > 0:
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_ratio,
            stratify=strat, random_state=random_seed
        )
    else:
        X_temp, y_temp = images, labels
        X_test, y_test = np.array([]), np.array([])

    if val_ratio > 0:
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        strat_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted,
            stratify=strat_temp, random_state=random_seed
        )
    else:
        X_train, y_train = X_temp, y_temp
        X_val, y_val = np.array([]), np.array([])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ============================================================
# MNIST 数据集加载（IDX 格式）
# ============================================================


def load_mnist(data_dir, target_size=28):
    """
    加载 MNIST IDX 格式数据集（train-images-idx3-ubyte 等）。

    参数:
        data_dir:    MNIST 数据文件所在目录
        target_size: 目标尺寸（MNIST 原生 28×28）

    返回:
        (X_train, y_train), (X_test, y_test)
        每个 X: (N, target_size, target_size) uint8
        每个 y: (N,) int32
    """
    def _read_idx(filename):
        with open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            if magic == 2051:  # 图像文件
                rows, cols = struct.unpack('>II', f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(num, rows, cols)
            elif magic == 2049:  # 标签文件
                return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
            else:
                raise ValueError(f"Unknown magic number: {magic}")

    train_images = _read_idx(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = _read_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = _read_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = _read_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    # 对 MNIST 做 OTSU 二值化预处理
    train_processed = np.zeros_like(train_images)
    test_processed = np.zeros_like(test_images)

    for i in range(len(train_images)):
        train_processed[i] = preprocess_single(train_images[i])
    for i in range(len(test_images)):
        test_processed[i] = preprocess_single(test_images[i])

    return (train_processed, train_labels), (test_processed, test_labels)


# ============================================================
# 对外统一接口（供成员B调用）
# ============================================================


def get_dataset(config):
    """
    统一数据获取接口 —— 成员B/C通过此函数获取所有数据。

    参数:
        config: dict, 配置字典，支持以下键:
            'digits_dir':        手写数字目录路径 (可选)
            'letters_dir':       手写字母目录路径 (可选)
            'mnist_dir':         MNIST数据目录路径 (可选)
            'use_torchvision':   是否使用 torchvision 在线加载 (默认 False)
                                设为 True 时自动忽略 digits_dir/mnist_dir/letters_dir
            'torchvision_datasets': 要加载的数据集列表, 如 ['mnist', 'letters']
                                   仅在 use_torchvision=True 时生效
            'target_size':       预处理目标尺寸, 默认 28
            'train_ratio':       训练集比例, 默认 0.6
            'val_ratio':         验证集比例, 默认 0.2
            'test_ratio':        测试集比例, 默认 0.2
            'random_seed':       随机种子, 默认 42
            'verbose':           是否打印日志, 默认 True

    返回:
        datasets: dict, 包含各数据集的划分结果:
            {
                'digits': {   # 仅当提供数字数据源时存在
                    'X_train', 'y_train',
                    'X_val',   'y_val',
                    'X_test',  'y_test',
                    'classes'
                },
                'letters': {  # 仅当提供字母数据源时存在
                    'X_train', 'y_train',
                    'X_val',   'y_val',
                    'X_test',  'y_test',
                    'classes'
                }
            }
    """
    target_size = config.get('target_size', 28)
    train_r = config.get('train_ratio', 0.6)
    val_r = config.get('val_ratio', 0.2)
    test_r = config.get('test_ratio', 0.2)
    seed = config.get('random_seed', 42)
    verbose = config.get('verbose', True)

    # ★ torchvision 模式：通过 PyTorch 内置数据集在线加载 ★
    if config.get('use_torchvision'):
        tv_datasets = config.get('torchvision_datasets', ['mnist'])
        tv_kwargs = config.get('torchvision_kwargs', {})
        datasets = {}
        for ds_name in tv_datasets:
            data = load_torchvision_data(
                ds_name,
                split_val=(val_r > 0),
                train_ratio=train_r,
                val_ratio=val_r,
                random_seed=seed,
                verbose=verbose,
                **tv_kwargs
            )
            # 统一 key 名称
            key = 'digits' if ds_name == 'mnist' else 'letters'
            datasets[key] = data
        return datasets

    datasets = {}

    # --- 手写数字（本地文件模式）---
    if config.get('digits_dir') or config.get('mnist_dir'):
        if verbose:
            print("[image_loader] 加载手写数字数据...")

        all_images, all_labels = [], []
        classes = None

        if config.get('mnist_dir'):
            (tr_img, tr_lbl), (te_img, te_lbl) = load_mnist(
                config['mnist_dir'], target_size=target_size
            )
            all_images.append(tr_img)
            all_labels.append(tr_lbl)
            all_images.append(te_img)
            all_labels.append(te_lbl)
            classes = [str(i) for i in range(10)]

        if config.get('digits_dir'):
            imgs, lbls, classes = load_images(
                config['digits_dir'], target_size=target_size, verbose=verbose
            )
            all_images.append(imgs)
            all_labels.append(lbls)

        images = np.concatenate(all_images, axis=0) if len(all_images) > 1 else all_images[0]
        labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 1 else all_labels[0]

        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_dataset(
            images, labels, train_r, val_r, test_r, random_seed=seed
        )

        datasets['digits'] = {
            'X_train': X_tr, 'y_train': y_tr,
            'X_val': X_va, 'y_val': y_va,
            'X_test': X_te, 'y_test': y_te,
            'classes': classes or [str(i) for i in range(10)]
        }

        if verbose:
            print(f"  数字数据集: 训练 {len(y_tr)} / 验证 {len(y_va)} / 测试 {len(y_te)}")

    # --- 手写字母（本地文件模式）---
    if config.get('letters_dir'):
        if verbose:
            print("[image_loader] 加载手写字母数据...")

        images, labels, classes = load_images(
            config['letters_dir'], target_size=target_size, verbose=verbose
        )
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_dataset(
            images, labels, train_r, val_r, test_r, random_seed=seed
        )

        datasets['letters'] = {
            'X_train': X_tr, 'y_train': y_tr,
            'X_val': X_va, 'y_val': y_va,
            'X_test': X_te, 'y_test': y_te,
            'classes': classes
        }

        if verbose:
            print(f"  字母数据集: 训练 {len(y_tr)} / 验证 {len(y_va)} / 测试 {len(y_te)}")

    return datasets


# ============================================================
# torchvision 在线数据集加载（★ 推荐方式 ★）
# ============================================================


def _dataset_to_arrays(dataset, max_samples=None, verbose=True, desc="加载"):
    """将 torchvision Dataset 转为 numpy 数组 (N, H, W) uint8 + (N,) int32。"""
    images, labels = [], []
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i, (img, label) in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        images.append(np.array(img, dtype=np.uint8))
        labels.append(label)
        if verbose and (i + 1) % 10000 == 0:
            print(f"    {desc}: {i + 1}/{total}")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


def _batch_preprocess_images(images, target_size=28, verbose=True, desc="预处理"):
    """批量预处理：高斯去噪 + OTSU二值化。"""
    N = len(images)
    processed = np.zeros((N, target_size, target_size), dtype=np.uint8)
    for i in range(N):
        _, processed[i] = preprocess_single(images[i], target_size=target_size)
        if verbose and (i + 1) % 10000 == 0:
            print(f"    {desc}: {i + 1}/{N}")
    return processed


def load_torchvision_data(dataset_name, split_val=True,
                           train_ratio=0.6, val_ratio=0.2,
                           random_seed=42, verbose=True,
                           max_train=None, max_test=None):
    """
    使用 torchvision 在线加载 MNIST / EMNIST-letters 数据集（★ 推荐方式 ★）。

    torchvision 自动管理下载和缓存（首次自动下载到 ./data/ 目录），
    无需手动运行 prepare_data.py。

    参数:
        dataset_name: 'mnist' 或 'letters'
        split_val:    是否划分验证集（从训练集中切出）
        train_ratio:  训练集比例（默认 0.6）
        val_ratio:    验证集比例（默认 0.2）
        random_seed:  随机种子
        verbose:      是否打印进度
        max_train:    最多加载训练样本数（None=全部，调试/演示时可设 500~2000）
        max_test:     最多加载测试样本数（None=全部）

    返回:
        dict: {
            'X_train': (N_tr, 28, 28) uint8,  'y_train': (N_tr,) int32,
            'X_val':   (N_va, 28, 28) uint8,  'y_val':   (N_va,) int32,
            'X_test':  (N_te, 28, 28) uint8,  'y_test':  (N_te,) int32,
            'classes': list[str]
        }

    示例:
        from image_loader import load_torchvision_data

        digits  = load_torchvision_data('mnist')                # 全量
        digits  = load_torchvision_data('mnist', max_train=500) # 演示用
    """
    from torchvision import datasets,transforms

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    emnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, [2])),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'mnist':
        if verbose:
            print("[image_loader] 通过 torchvision 加载 MNIST 手写数字数据集...")
        train_data = datasets.MNIST(
            root='./data', train=True, download=True, transform=mnist_transform
        )
        test_data = datasets.MNIST(
            root='./data', train=False, download=True, transform=mnist_transform
        )
        classes = [str(i) for i in range(10)]

    elif dataset_name == 'letters':
        if verbose:
            print("[image_loader] 通过 torchvision 加载 EMNIST-letters 手写字母数据集...")
        train_data = datasets.EMNIST(
            root='./data', split='letters', train=True, download=True, transform=emnist_transform
        )
        test_data = datasets.EMNIST(
            root='./data', split='letters', train=False, download=True, transform=emnist_transform
        )
        classes = [chr(ord('A') + i) for i in range(26)]

    else:
        raise ValueError(
            f"不支持的数据集: '{dataset_name}'，可选 'mnist' 或 'letters'"
        )


    result = {
        'train_data': train_data,
        'test_data': test_data,
        'classes': classes
    }

    return result


def load_prepared_data(dataset_name, base_dir=None, split_val=True,
                       train_ratio=0.6, val_ratio=0.2, random_seed=42):
    """
    加载 prepare_data.py 预处理的 .npz 数据集，并划分训练/验证/测试集。

    参数:
        dataset_name: 'mnist' 或 'letters'
        base_dir:     项目根目录，默认自动检测
        split_val:    是否划分验证集（否则仅 train/test 按 8:2）
        train_ratio, val_ratio, test_ratio: 划分比例
        random_seed:  随机种子

    返回:
        dict: {
            'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classes'
        }

    示例:
        digits = load_prepared_data('mnist')
        letters = load_prepared_data('letters')
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_name == 'mnist':
        path = os.path.join(base_dir, 'data', 'mnist', 'mnist_processed.npz')
        classes = [str(i) for i in range(10)]
    elif dataset_name == 'letters':
        path = os.path.join(base_dir, 'data', 'emnist_letters', 'emnist_letters_processed.npz')
        classes = [chr(ord('A') + i) for i in range(26)]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}, 可选 'mnist' 或 'letters'")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"预处理数据文件不存在: {path}\n"
            f"请先运行: python prepare_data.py"
        )

    data = np.load(path)
    X_train_all = data['X_train']
    y_train_all = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    if split_val:
        # 从训练集中切出验证集: train_ratio/(train+val) 和 val_ratio/(train+val) 构成100%
        _train_r = train_ratio / (train_ratio + val_ratio)
        _val_r = val_ratio / (train_ratio + val_ratio)
        (X_train, y_train), (X_val, y_val), _ = split_dataset(
            X_train_all, y_train_all,
            train_ratio=_train_r, val_ratio=_val_r,
            test_ratio=0.0, random_seed=random_seed
        )
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'classes': classes
        }
    else:
        return {
            'X_train': X_train_all, 'y_train': y_train_all,
            'X_val': None, 'y_val': None,
            'X_test': X_test, 'y_test': y_test,
            'classes': classes
        }
