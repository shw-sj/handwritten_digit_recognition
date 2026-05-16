"""
图像特征提取模块 [成员A]

提供3种图像特征提取方法 + 统一接口供成员B调用：

  1. 像素特征（pixel）   — 28×28 / 14×14 网格展平
  2. 投影特征（projection）— 水平投影 + 垂直投影直方图
  3. 结构特征（structure） — Hu矩 + 笔画穿越次数 + 孔洞数 + 宽高比

统一接口:
  - extract_features(images, method='pixel', **kwargs) → np.ndarray (N, D)
  - get_feature_dim(method, **kwargs) → int
"""

import numpy as np


# ============================================================
# 1. 像素特征
# ============================================================

def _extract_pixel(images, grid_size=28, normalize=True):
    """
    像素特征：将图像缩放至 grid_size × grid_size 后展平为一维向量。

    参数:
        images:    (N, H, W) 二值图像数组, uint8
        grid_size: 缩放尺寸, 默认 28 (28×28=784维) 或 14 (14×14=196维)
        normalize: 是否归一化到 [0, 1]

    返回:
        features: (N, grid_size * grid_size) float32
    """
    N = images.shape[0]
    features = np.zeros((N, grid_size * grid_size), dtype=np.float32)

    for i in range(N):
        img = images[i]
        h, w = img.shape
        # 简单下采样：对每个网格块取均值
        block_h = h / grid_size
        block_w = w / grid_size
        for r in range(grid_size):
            for c in range(grid_size):
                r_start = int(r * block_h)
                r_end = int((r + 1) * block_h)
                c_start = int(c * block_w)
                c_end = int((c + 1) * block_w)
                block = img[r_start:r_end, c_start:c_end]
                features[i, r * grid_size + c] = np.mean(block)

    if normalize:
        # 像素值 0-255 → 0-1
        features /= 255.0

    return features


# ============================================================
# 2. 投影特征
# ============================================================

def _extract_projection(images, normalize=True):
    """
    投影特征：水平投影直方图 + 垂直投影直方图。

    对每张 H×W 的二值图像，计算：
      - 水平投影：(H,)  每行白色像素数 / 总列数
      - 垂直投影：(W,)  每列白色像素数 / 总行数
    拼接得到 (H+W,) 维特征向量。

    参数:
        images:    (N, H, W) 二值图像, uint8
        normalize: 是否归一化

    返回:
        features: (N, H+W) float32
    """
    N, H, W = images.shape
    features = np.zeros((N, H + W), dtype=np.float32)

    for i in range(N):
        img = images[i].astype(np.float32)
        # 白色像素 = 前景 (值为255)
        h_proj = np.sum(img, axis=1)  # (H,) 每行白色像素总和
        v_proj = np.sum(img, axis=0)  # (W,) 每列白色像素总和

        if normalize:
            h_proj = h_proj / (W * 255.0)
            v_proj = v_proj / (H * 255.0)

        features[i] = np.concatenate([h_proj, v_proj])

    return features


# ============================================================
# 3. 结构/轮廓特征
# ============================================================

def _moments(img):
    """计算图像的空间矩和中心矩（0-3阶）。"""
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    mass = img.astype(np.float32) / 255.0

    m00 = np.sum(mass)
    if m00 < 1e-9:
        return None  # 空白图像

    m10 = np.sum(x * mass)
    m01 = np.sum(y * mass)
    cx = m10 / m00
    cy = m01 / m00

    dx = x - cx
    dy = y - cy

    mu20 = np.sum(dx**2 * mass)
    mu02 = np.sum(dy**2 * mass)
    mu11 = np.sum(dx * dy * mass)
    mu30 = np.sum(dx**3 * mass)
    mu03 = np.sum(dy**3 * mass)
    mu21 = np.sum(dx**2 * dy * mass)
    mu12 = np.sum(dx * dy**2 * mass)

    return mu20, mu02, mu11, mu30, mu03, mu21, mu12, m00


def _hu_moments(img):
    """
    计算 7 个 Hu 不变矩（平移、旋转、缩放不变性）。

    参数:
        img: (H, W) 二值图像

    返回:
        hu: (7,) float32, 对数变换后的 Hu 矩
    """
    m = _moments(img)
    if m is None:
        return np.zeros(7, dtype=np.float32)

    mu20, mu02, mu11, mu30, mu03, mu21, mu12, m00 = m

    # 归一化中心矩
    eta = lambda mu_pq, gamma: mu_pq / (m00 ** (1 + gamma / 2.0))

    nu20 = eta(mu20, 2)
    nu02 = eta(mu02, 2)
    nu11 = eta(mu11, 2)
    nu30 = eta(mu30, 3)
    nu03 = eta(mu03, 3)
    nu21 = eta(mu21, 3)
    nu12 = eta(mu12, 3)

    # 7 个 Hu 矩
    hu = np.zeros(7, dtype=np.float64)
    hu[0] = nu20 + nu02
    hu[1] = (nu20 - nu02) ** 2 + 4 * nu11 ** 2
    hu[2] = (nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2
    hu[3] = (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2
    hu[4] = ((nu30 - 3 * nu12) * (nu30 + nu12) *
             ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) +
             (3 * nu21 - nu03) * (nu21 + nu03) *
             (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2))
    hu[5] = ((nu20 - nu02) * ((nu30 + nu12) ** 2 - (nu21 + nu03) ** 2) +
             4 * nu11 * (nu30 + nu12) * (nu21 + nu03))
    hu[6] = ((3 * nu21 - nu03) * (nu30 + nu12) *
             ((nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2) -
             (nu30 - 3 * nu12) * (nu21 + nu03) *
             (3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2))

    # 对数变换
    hu_log = np.sign(hu) * np.log1p(np.abs(hu))
    return hu_log.astype(np.float32)


def _count_stroke_crossings(img, num_lines=4):
    """
    统计笔画穿越次数：在图像中放置若干条扫描线，统计黑白跃变次数。

    对水平和垂直方向各放置 num_lines 条等距扫描线，
    每条线统计从背景→前景的跃变次数（笔画边缘），
    返回每条线的穿越次数，共 2 * num_lines 个特征。
    """
    H, W = img.shape
    binary = (img > 127).astype(np.int32)
    crossings = []

    # 水平扫描线
    for i in range(1, num_lines + 1):
        y = int(i * H / (num_lines + 1))
        line = binary[y, :]
        changes = np.sum(np.abs(np.diff(line)))
        crossings.append(changes)

    # 垂直扫描线
    for i in range(1, num_lines + 1):
        x = int(i * W / (num_lines + 1))
        line = binary[:, x]
        changes = np.sum(np.abs(np.diff(line)))
        crossings.append(changes)

    return np.array(crossings, dtype=np.float32)


def _count_holes(img):
    """
    孔洞数：使用连通域分析统计封闭的背景区域数。

    0, 6, 9 有 1 个洞；8 有 2 个洞；其余有 0 个。
    返回归一化后的孔洞数（除以最大可能孔洞数 3）。
    """
    binary = (img > 127).astype(np.uint8)
    H, W = binary.shape

    # 对背景区域 (0) 做 flood-fill 连通域计数
    # 排除与边界相连的外部背景
    visited = np.zeros((H, W), dtype=bool)
    hole_count = 0

    for i in range(H):
        for j in range(W):
            if binary[i, j] == 0 and not visited[i, j]:
                # BFS 该连通域
                stack = [(i, j)]
                visited[i, j] = True
                touches_border = False
                pixels = []

                while stack:
                    y, x = stack.pop()
                    pixels.append((y, x))
                    if y == 0 or y == H - 1 or x == 0 or x == W - 1:
                        touches_border = True
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if binary[ny, nx] == 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                if not touches_border:
                    hole_count += 1

    return np.float32(min(hole_count, 3) / 3.0)


def _aspect_ratio(img):
    """计算前景包围盒的宽高比（归一化）。"""
    binary = img > 127
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not np.any(rows):
        return np.float32(0.0)

    h = np.sum(rows)
    w = np.sum(cols)
    if h == 0:
        return np.float32(0.0)

    return np.float32(w / h)


def _extract_structure(images):
    """
    结构/轮廓特征：Hu矩(7) + 笔画穿越次数(8) + 孔洞数(1) + 宽高比(1) = 17维。

    参数:
        images: (N, H, W) 二值图像, uint8

    返回:
        features: (N, 17) float32
    """
    N = images.shape[0]
    features = np.zeros((N, 17), dtype=np.float32)

    for i in range(N):
        img = images[i]
        hu = _hu_moments(img)               # 7
        cross = _count_stroke_crossings(img)  # 8 (默认4条水平+4条垂直)
        holes = _count_holes(img)           # 1
        ar = _aspect_ratio(img)            # 1

        features[i] = np.concatenate([hu, cross, [holes], [ar]])

    return features


# ============================================================
# 对外统一接口（供成员B调用）
# ============================================================

# 方法名 → 提取函数映射
_METHOD_MAP = {
    'pixel':      _extract_pixel,
    'projection': _extract_projection,
    'structure':  _extract_structure,
}


def extract_features(images, method='pixel', **kwargs):
    """
    统一的特征提取接口。

    参数:
        images: np.ndarray (N, H, W), 二值图像, uint8
                —— 即 image_loader.load_images() 或 get_dataset() 返回的图像
        method: str, 特征提取方法:
                'pixel'      — 像素特征（默认）
                'projection' — 投影特征
                'structure'  — 结构/轮廓特征
        **kwargs: 传递给具体提取方法的参数, 如:
                grid_size=14 (pixel方法, 控制输出维度)

    返回:
        features: np.ndarray (N, D), float32, 特征矩阵

    示例:
        from image_feature import extract_features, get_feature_dim

        X = datasets['digits']['X_train']   # (N, 28, 28) uint8
        F = extract_features(X, method='pixel', grid_size=28)
        # F.shape → (N, 784)

        F = extract_features(X, method='projection')
        # F.shape → (N, 56)

        F = extract_features(X, method='structure')
        # F.shape → (N, 17)

        dim = get_feature_dim('pixel', grid_size=28)  # → 784
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"不支持的特征提取方法: '{method}'。"
            f"可选: {list(_METHOD_MAP.keys())}"
        )
    return _METHOD_MAP[method](images, **kwargs)


def get_feature_dim(method='pixel', image_size=28, **kwargs):
    """
    获取指定方法对应的特征向量维度（用于配置BP网络输入层大小）。

    参数:
        method:     str, 特征提取方法
        image_size: int, 输入图像尺寸（默认28）
        **kwargs:   pixel: grid_size (默认28)
                    projection: 无额外参数, 维度 = 2 * image_size
                    structure: 无额外参数, 固定 17 维

    返回:
        dim: int, 特征向量维度

    示例:
        get_feature_dim('pixel', grid_size=14)      # → 196
        get_feature_dim('projection', image_size=28) # → 56
        get_feature_dim('structure')                 # → 17
    """
    if method == 'pixel':
        gs = kwargs.get('grid_size', image_size)
        return gs * gs
    elif method == 'projection':
        return 2 * image_size
    elif method == 'structure':
        return 17
    else:
        raise ValueError(f"不支持的特征提取方法: '{method}'")


def get_all_methods():
    """返回所有支持的特征提取方法名称列表。"""
    return list(_METHOD_MAP.keys())
