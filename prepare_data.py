"""
一次性数据准备脚本：下载 MNIST + EMNIST-letters 并预处理。

运行: python prepare_data.py

数据集:
  - MNIST:          手写数字 0-9, 训练60000张 + 测试10000张
  - EMNIST-letters: 手写字母 A-Z, 训练124800张 + 测试20800张

预处理流水线: 灰度化 → OTSU二值化 → 28×28归一化
依赖: numpy, scikit-learn, requests (无需PyTorch)
"""

import os
import sys
import gzip
import struct
import numpy as np
from io import BytesIO

# 导入成员A的预处理函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_loader import preprocess_single

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _download(url, dest_path, retries=3):
    """下载文件，使用 curl 作为首选（绕过 SSL 证书吊销检查问题）。"""
    if os.path.exists(dest_path):
        print(f"  已存在: {os.path.basename(dest_path)}，跳过下载")
        return

    print(f"  下载: {os.path.basename(dest_path)}")
    import subprocess
    for attempt in range(retries):
        result = subprocess.run([
            'curl', '-L', '--ssl-no-revoke',
            '-A', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            '--retry', '3',
            '-o', dest_path, url,
            '--connect-timeout', '15', '--max-time', '600',
            '--progress-bar'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  完成 → {os.path.basename(dest_path)}")
            return
        print(f"  重试 {attempt + 1}/{retries}...")
        import time
        time.sleep(3)
    raise RuntimeError(f"下载失败: {url}")


def _read_idx(filename):
    """读取 IDX 格式文件（支持 .gz 压缩）。"""
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            data = f.read()
        buf = BytesIO(data)
    else:
        buf = open(filename, 'rb')

    with buf:
        magic, num = struct.unpack('>II', buf.read(8))
        if magic == 2051:  # 图像
            rows, cols = struct.unpack('>II', buf.read(8))
            arr = np.frombuffer(buf.read(), dtype=np.uint8)
            return arr.reshape(num, rows, cols)
        elif magic == 2049:  # 标签
            arr = np.frombuffer(buf.read(), dtype=np.uint8)
            return arr.astype(np.int32)
        else:
            raise ValueError(f"Unknown magic: {magic}")


# ============================================================
# MNIST
# ============================================================

MNIST_URLS = {
    'train-images-idx3-ubyte.gz':
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz':
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz':
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz':
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_and_preprocess_mnist():
    print("=" * 60)
    print("下载 & 预处理 MNIST 手写数字数据集...")
    print("=" * 60)

    mnist_dir = os.path.join(DATA_DIR, "data", "mnist")
    os.makedirs(mnist_dir, exist_ok=True)

    save_path = os.path.join(mnist_dir, "mnist_processed.npz")
    if os.path.exists(save_path):
        print(f"  MNIST 已预处理，跳过: {save_path}")
        data = np.load(save_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    # 下载
    for fname, url in MNIST_URLS.items():
        _download(url, os.path.join(mnist_dir, fname))

    # 读取
    X_train = _read_idx(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'))
    y_train = _read_idx(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'))
    X_test = _read_idx(os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = _read_idx(os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz'))

    # 预处理 (OTSU二值化)
    print(f"  预处理 {len(X_train)} 张训练图像...")
    X_train_proc = np.zeros_like(X_train)
    for i in range(len(X_train)):
        _, X_train_proc[i] = preprocess_single(X_train[i], target_size=28)
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{len(X_train)}")

    print(f"  预处理 {len(X_test)} 张测试图像...")
    X_test_proc = np.zeros_like(X_test)
    for i in range(len(X_test)):
        _, X_test_proc[i] = preprocess_single(X_test[i], target_size=28)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(X_test)}")

    # 保存
    save_path = os.path.join(mnist_dir, "mnist_processed.npz")
    np.savez_compressed(save_path,
                        X_train=X_train_proc, y_train=y_train,
                        X_test=X_test_proc, y_test=y_test)
    print(f"  MNIST 已保存: {save_path}")
    print(f"    训练: {X_train_proc.shape}, 测试: {X_test_proc.shape}")


# ============================================================
# EMNIST-letters
# ============================================================

EMNIST_URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'


def download_and_preprocess_emnist_letters():
    print()
    print("=" * 60)
    print("下载 & 预处理 EMNIST-letters 手写字母数据集...")
    print("=" * 60)

    emnist_dir = os.path.join(DATA_DIR, "data", "emnist_letters")
    os.makedirs(emnist_dir, exist_ok=True)

    save_path = os.path.join(emnist_dir, "emnist_letters_processed.npz")
    if os.path.exists(save_path):
        print(f"  EMNIST-letters 已预处理，跳过: {save_path}")
        data = np.load(save_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    zip_path = os.path.join(emnist_dir, "gzip.zip")

    if not os.path.exists(zip_path):
        print(f"  下载 EMNIST (约560MB，请耐心等待)...")
        # 尝试多个镜像源
        urls = [
            'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip',
            'https://s3.amazonaws.com/nist-srd/SD19/by_class.zip',  # 备选
        ]
        ok = False
        for url in urls:
            try:
                _download(url, zip_path)
                ok = True
                break
            except Exception as e:
                print(f"  来源失败: {url} — {e}")
        if not ok:
            raise RuntimeError("EMNIST 所有镜像下载失败，请手动下载 gzip.zip 放到 data/emnist_letters/")
        print(f"  下载完成")

    # 解压所需文件
    needed = [
        'emnist-letters-train-images-idx3-ubyte.gz',
        'emnist-letters-train-labels-idx1-ubyte.gz',
        'emnist-letters-test-images-idx3-ubyte.gz',
        'emnist-letters-test-labels-idx1-ubyte.gz',
    ]
    # EMNIST zip 内部文件有 gzip/ 前缀
    zip_prefix = 'gzip/'

    # 检查是否已解压
    all_extracted = all(
        os.path.exists(os.path.join(emnist_dir, f)) for f in needed
    )

    if not all_extracted:
        print("  解压 EMNIST 文件...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for fname in needed:
                if not os.path.exists(os.path.join(emnist_dir, fname)):
                    zf.extract(zip_prefix + fname, emnist_dir)
                    # 重命名：去掉 gzip/ 前缀
                    extracted = os.path.join(emnist_dir, zip_prefix + fname)
                    target = os.path.join(emnist_dir, fname)
                    os.rename(extracted, target)
                    # 清理空目录
                    gzip_dir = os.path.join(emnist_dir, 'gzip')
                    if os.path.isdir(gzip_dir) and not os.listdir(gzip_dir):
                        os.rmdir(gzip_dir)
                    print(f"    解压: {fname}")
        print("  解压完成")

    # 读取
    X_train_all = _read_idx(os.path.join(emnist_dir, needed[0]))
    y_train_all = _read_idx(os.path.join(emnist_dir, needed[1]))
    X_test = _read_idx(os.path.join(emnist_dir, needed[2]))
    y_test = _read_idx(os.path.join(emnist_dir, needed[3]))

    # EMNIST-letters 标签: 1-26 → A-Z，转换为 0-25
    y_train_all = y_train_all - 1
    y_test = y_test - 1

    # EMNIST 图像需要转置（镜像+旋转，纠正原始存储格式）
    print("  转置图像方向...")
    X_train_all = np.transpose(X_train_all, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # 预处理
    print(f"  预处理 {len(X_train_all)} 张训练图像...")
    X_train_proc = np.zeros_like(X_train_all)
    for i in range(len(X_train_all)):
        _, X_train_proc[i] = preprocess_single(X_train_all[i], target_size=28)
        if (i + 1) % 20000 == 0:
            print(f"    {i+1}/{len(X_train_all)}")

    print(f"  预处理 {len(X_test)} 张测试图像...")
    X_test_proc = np.zeros_like(X_test)
    for i in range(len(X_test)):
        _, X_test_proc[i] = preprocess_single(X_test[i], target_size=28)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(X_test)}")

    # 保存
    save_path = os.path.join(emnist_dir, "emnist_letters_processed.npz")
    LETTER_CLASSES = np.array([chr(ord('A') + i) for i in range(26)])
    np.savez_compressed(save_path,
                        X_train=X_train_proc, y_train=y_train_all,
                        X_test=X_test_proc, y_test=y_test,
                        classes=LETTER_CLASSES)
    print(f"  EMNIST-letters 已保存: {save_path}")
    print(f"    训练: {X_train_proc.shape}, 测试: {X_test_proc.shape}")


if __name__ == '__main__':
    download_and_preprocess_mnist()
    download_and_preprocess_emnist_letters()
    print()
    print("=" * 60)
    print("全部数据集准备完成!")
    print("=" * 60)
    print("""
使用方式:
    from image_loader import load_prepared_data

    digits  = load_prepared_data('mnist')
    letters = load_prepared_data('letters')
""")
