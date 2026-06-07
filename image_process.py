"""
图像数据加载与预处理模块 [成员A]

提供统一接口供成员B、C调用：

  - preprocess_single()  单张图像预处理（GUI画板实时识别用）
  - preprocess_letter()  单张字母图像预处理（GUI画板实时识别用）
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

import cv2
import numpy as np
from PIL import Image

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
