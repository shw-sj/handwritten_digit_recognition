"""
图像数据增强模块 [成员A]

对训练样本进行在线/离线增强，提升模型泛化能力：
  - 随机旋转    ±10°
  - 随机平移    ±3px
  - 随机缩放    ±10%
  - 弹性形变（模拟手写笔迹的自然抖动）

统一接口:
  - augment_dataset(images, labels, factor=3) → 扩充后的 (images, labels)
  - augment_single(img) → 随机增强单张图像（用于GUI实时识别测试）
"""

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


# ============================================================
# 单张图像增强变换
# ============================================================

def _rotate(img, angle_deg):
    """绕中心旋转指定角度，空白区域用背景色(0)填充。"""
    pil = Image.fromarray(img)
    rotated = pil.rotate(angle_deg, resample=Image.BILINEAR,
                         fillcolor=0, expand=False)
    return np.array(rotated, dtype=np.uint8)


def _translate(img, dx, dy):
    """平移 dx, dy 像素，空白区域用背景色(0)填充。"""
    pil = Image.fromarray(img)
    translated = pil.transform(
        pil.size, Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.BILINEAR, fillcolor=0
    )
    return np.array(translated, dtype=np.uint8)


def _scale(img, factor):
    """
    缩放图像，factor > 1 放大（裁剪中心），factor < 1 缩小（填充边距）。
    保持输出尺寸与输入一致。
    """
    H, W = img.shape
    pil = Image.fromarray(img)
    new_w = int(W * factor)
    new_h = int(H * factor)
    scaled = pil.resize((new_w, new_h), Image.BILINEAR)

    if factor >= 1.0:
        # 放大 → 中心裁剪
        left = (new_w - W) // 2
        top = (new_h - H) // 2
        result = scaled.crop((left, top, left + W, top + H))
    else:
        # 缩小 → 放置在中心，周围填 0
        result = Image.new('L', (W, H), 0)
        left = (W - new_w) // 2
        top = (H - new_h) // 2
        result.paste(scaled, (left, top))

    return np.array(result, dtype=np.uint8)


def _elastic_deform(img, alpha=4.0, sigma=2.0, random_state=None):
    """
    弹性形变：对图像施加随机位移场，模拟手写笔画自然抖动。

    方法: Simard et al., "Best Practices for Convolutional Neural Networks
          Applied to Visual Document Analysis", ICDAR 2003.

    参数:
        alpha: 形变强度  (越大扭曲越强)
        sigma: 平滑程度  (越大扭曲越平滑)
    """
    rng = np.random.RandomState(random_state)
    H, W = img.shape

    # 生成随机位移场
    dx = rng.randn(H, W) * alpha
    dy = rng.randn(H, W) * alpha

    # 高斯平滑位移场
    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(dx, sigma)
    dy = gaussian_filter(dy, sigma)

    # 生成采样坐标
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    # 映射
    deformed = map_coordinates(img.astype(np.float32), indices,
                               order=1, mode='constant', cval=0)
    deformed = np.clip(deformed, 0, 255).astype(np.uint8)
    return deformed.reshape(H, W)


# ============================================================
# 随机增强组合
# ============================================================

def augment_single(img, rng=None):
    """
    对单张图像施加一次随机增强。

    增强流水线（每种变换以一定概率独立应用）：
      1. 旋转     ±10°,  概率 0.7
      2. 平移     ±3px,  概率 0.7
      3. 缩放     ±10%,  概率 0.5
      4. 弹性形变         概率 0.4

    参数:
        img: (H, W) uint8 二值图像
        rng: np.random.RandomState 或 None

    返回:
        augmented: (H, W) uint8 增强后的图像
    """
    if rng is None:
        rng = np.random

    H, W = img.shape
    result = img.copy()

    # 1. 旋转 ±10°
    if rng.random() < 0.7:
        angle = rng.uniform(-10, 10)
        result = _rotate(result, angle)

    # 2. 平移 ±3px
    if rng.random() < 0.7:
        dx = rng.uniform(-3, 3)
        dy = rng.uniform(-3, 3)
        result = _translate(result, dx, dy)

    # 3. 缩放 ±10%
    if rng.random() < 0.5:
        factor = rng.uniform(0.9, 1.1)
        result = _scale(result, factor)

    # 4. 弹性形变
    if rng.random() < 0.4:
        result = _elastic_deform(result, alpha=rng.uniform(2, 6),
                                 sigma=rng.uniform(1.5, 3.0),
                                 random_state=rng.randint(0, 2**31 - 1))

    return result


# ============================================================
# 批量数据增强 —— 对外统一接口
# ============================================================

def augment_dataset(images, labels, factor=3, random_seed=42):
    """
    对训练数据集进行增强，扩充 factor 倍。

    策略：
      - 保留原始数据（不变）
      - 对每张原图生成 (factor - 1) 个随机增强版本
      - 增强版本使用不同的随机种子，确保多样性

    参数:
        images:      np.ndarray (N, H, W), uint8
        labels:      np.ndarray (N,), int
        factor:      扩充倍数，默认 3（最终数据量 = N * factor）
        random_seed: 随机种子

    返回:
        aug_images: np.ndarray (N * factor, H, W), uint8
        aug_labels: np.ndarray (N * factor,), int
    """
    N = images.shape[0]
    rng = np.random.RandomState(random_seed)

    aug_images = [images]
    aug_labels = [labels]

    for aug_round in range(factor - 1):
        batch_imgs = np.zeros_like(images)
        for i in range(N):
            # 每张图每轮使用不同种子
            seed_i = random_seed + aug_round * N + i
            batch_imgs[i] = augment_single(images[i],
                                           rng=np.random.RandomState(seed_i))
        aug_images.append(batch_imgs)
        aug_labels.append(labels)

    return np.concatenate(aug_images, axis=0), np.concatenate(aug_labels, axis=0)
