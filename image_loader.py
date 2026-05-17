"""
图像数据加载与预处理模块 [成员A]

提供统一接口供成员B调用：
  - load_images()        加载原始图像并预处理
  - split_dataset()      按比例划分训练/验证/测试集
  - preprocess_single()  单张图像预处理（GUI画板实时识别用）
  - load_mnist()         加载MNIST数据集作为补充

预处理流水线：灰度化 → 高斯去噪 → OTSU二值化 → 尺寸归一化(28×28)
"""

import os
import struct
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# 预处理核心函数
# ============================================================


def _to_grayscale(img):
    """将图像转为灰度图，返回 (H, W) uint8 数组。"""
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        return img.astype(np.uint8)
    return np.array(img.convert('L'), dtype=np.uint8)


def _gaussian_blur(img, ksize=3):
    """高斯去噪（OpenCV-free 实现，3×3 高斯核）。"""
    if ksize == 3:
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0
    elif ksize == 5:
        kernel = np.array([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], dtype=np.float32) / 256.0
    else:
        raise ValueError(f"Unsupported ksize: {ksize}")

    h, w = img.shape
    pad = ksize // 2
    padded = np.pad(img, pad, mode='reflect')
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + ksize, j:j + ksize].astype(np.float32)
            result[i, j] = np.sum(patch * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)


def _otsu_threshold(img):
    """OTSU 大津算法二值化，返回 (二值图, 阈值)。"""
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    sum_all = np.dot(np.arange(256), hist)

    weight_bg = 0
    sum_bg = 0
    max_var = 0
    threshold = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    binary = (img >= threshold).astype(np.uint8) * 255
    return binary, threshold


def _resize(img, target_size=(28, 28)):
    """尺寸归一化（双线性插值，PIL 实现）。"""
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(target_size, Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def preprocess_single(img, target_size=28, denoise_ksize=3):
    """
    单张图像预处理流水线。

    参数:
        img: PIL.Image 或 numpy 数组 (H,W) / (H,W,3)
        target_size: 目标尺寸，默认 28
        denoise_ksize: 高斯核大小，默认 3

    返回:
        preprocessed: 预处理后图像 (target_size, target_size) uint8
        binary:      OTSU二值图 (target_size, target_size) uint8
    """
    gray = _to_grayscale(img)
    denoised = _gaussian_blur(gray, ksize=denoise_ksize)
    binary, _ = _otsu_threshold(denoised)
    resized = _resize(binary, target_size=(target_size, target_size))
    return resized, binary[:target_size, :target_size] if binary.shape != (target_size, target_size) else binary


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
                    processed, _ = preprocess_single(img, target_size=target_size)
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
        _, train_processed[i] = preprocess_single(train_images[i], target_size=target_size)
    for i in range(len(test_images)):
        _, test_processed[i] = preprocess_single(test_images[i], target_size=target_size)

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
    from torchvision import datasets

    if dataset_name == 'mnist':
        if verbose:
            print("[image_loader] 通过 torchvision 加载 MNIST 手写数字数据集...")
        train_data = datasets.MNIST(
            root='./data', train=True, download=True
        )
        test_data = datasets.MNIST(
            root='./data', train=False, download=True
        )
        classes = [str(i) for i in range(10)]

    elif dataset_name == 'letters':
        if verbose:
            print("[image_loader] 通过 torchvision 加载 EMNIST-letters 手写字母数据集...")
        train_data = datasets.EMNIST(
            root='./data', split='letters', train=True, download=True
        )
        test_data = datasets.EMNIST(
            root='./data', split='letters', train=False, download=True
        )
        classes = [chr(ord('A') + i) for i in range(26)]

    else:
        raise ValueError(
            f"不支持的数据集: '{dataset_name}'，可选 'mnist' 或 'letters'"
        )

    # 1. 转为 numpy 数组
    n_train = max_train if max_train is not None else len(train_data)
    n_test = max_test if max_test is not None else len(test_data)
    if verbose:
        print(f"  加载训练集 ({n_train} / {len(train_data)} 张)...")
    X_train_all, y_train_all = _dataset_to_arrays(
        train_data, max_samples=max_train, verbose=verbose, desc="训练集"
    )

    if verbose:
        print(f"  加载测试集 ({n_test} / {len(test_data)} 张)...")
    X_test, y_test = _dataset_to_arrays(
        test_data, max_samples=max_test, verbose=verbose, desc="测试集"
    )

    # 2. EMNIST 图像需要转置（纠正原始存储的旋转+镜像方向）
    if dataset_name == 'letters':
        if verbose:
            print("  校正 EMNIST 图像方向...")
        X_train_all = np.transpose(X_train_all, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

    # 3. 预处理流水线：灰度化 → 高斯去噪 → OTSU二值化 → 28×28归一化
    if verbose:
        print(f"  预处理训练集 ({len(X_train_all)} 张)...")
    X_train_all = _batch_preprocess_images(
        X_train_all, target_size=28, verbose=verbose, desc="训练集"
    )

    if verbose:
        print(f"  预处理测试集 ({len(X_test)} 张)...")
    X_test = _batch_preprocess_images(
        X_test, target_size=28, verbose=verbose, desc="测试集"
    )

    # 4. 划分训练/验证/测试集
    if split_val:
        # 归一化比例（test_ratio=0 时需要调整 train/val 和为 1）
        _sum = train_ratio + val_ratio
        _train_r = train_ratio / _sum
        _val_r = val_ratio / _sum
        (X_train, y_train), (X_val, y_val), _ = split_dataset(
            X_train_all, y_train_all,
            train_ratio=_train_r, val_ratio=_val_r,
            test_ratio=0.0, random_seed=random_seed
        )
    else:
        X_train, y_train = X_train_all, y_train_all
        X_val, y_val = None, None

    result = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'classes': classes
    }

    if verbose:
        n_val = len(y_val) if y_val is not None else 0
        print(f"  完成: 训练 {len(y_train)} / 验证 {n_val} / 测试 {len(y_test)}")

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
