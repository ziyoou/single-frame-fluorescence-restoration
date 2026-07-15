
# -*- coding: utf-8 -*-
"""
Poisson-Gaussian noise parameter estimation for single or multiple microscopic noisy images.

This program estimates parameters of the signal-dependent noise model from one or more
microscopic noisy images. The adopted noise model is:

    Var(Y | X) = alpha * X + sigma^2,

where X denotes the signal intensity after subtracting the sCMOS camera offset,
alpha is the signal-dependent Poisson noise coefficient, and sigma stands for the
standard deviation of signal-independent Gaussian readout noise.

Workflow overview:
    1. Load raw grayscale images;
    2. Subtract the sCMOS camera offset;
    3. Split images into local patches;
    4. Discard patches with excessive negative pixels or saturated pixels;
    5. Select low-texture patches based on gradient energy;
    6. Compute local mean and local noise variance for each low-texture patch;
    7. Robustly fit Var = alpha * mean + sigma^2 via RANSAC.

References
----------
[1] LIU X, TANAKA M, OKUTOMI M. Practical signal-dependent noise parameter
    estimation from a single noisy image[J]. IEEE Transactions on Image
    Processing, 2014, 23(10): 4361-4371.

[2] FOI A, TRIMECHE M, KATKOVNIK V, EGIAZARIAN K. Practical Poissonian-Gaussian
    noise modeling and fitting for single-image raw-data[J]. IEEE Transactions
    on Image Processing, 2008, 17(10): 1737-1754.
"""


import glob
import os
import random

import numpy as np
import tifffile as tiff
from PIL import Image

# ============================================================
# 0. Parameter settings
# ============================================================

IMAGE_DIR = r"./demo_data\denoise\only_noisy_img_Myosin_IIA\noisy_Img_from_BioSR"
AUTO_BIAS = True         # True: auto-estimate bias; False: use manual BIAS

BIAS = 0              # Manual bias, only used when AUTO_BIAS=False

BIAS_PERCENTILE = 0.5     # Low percentile for auto estimation, e.g., 0.1, 0.5, 1.0
BIAS_SAMPLE_MAX = 20      # Max number of images for bias estimation
MAX_GRAY = 200.0     # 8-bit grayscale max, avoid misclassifying highlights as saturated

PATCH_SIZE = 32
STRIDE = 4
NEG_RATIO_MAX = 0.08      # Skip patches with too many negative pixels
SAT_RATIO_MAX = 0.0001    # Skip patches with too many saturated pixels
WEAK_TEXTURE_KEEP_RATIO = 0.060  # Keep ratio of weakest-texture patches per intensity bin
NUM_BINS = 25             # Number of intensity bins
RANSAC_ITER = 30000
INLIER_THRESHOLD_FACTOR = 1.1

def read_gray_image_raw(path):
    """Load grayscale image as float32 without normalization."""
    ext = os.path.splitext(path)[-1].lower()

    if ext in [".tif", ".tiff"]:
        img = tiff.imread(path)
    else:
        img = Image.open(path).convert("F")

    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., 0]

    return img


def get_image_paths(folder):
    exts = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths = sorted(
        path
        for ext in exts
        for path in glob.glob(os.path.join(folder, ext))
    )
    if not paths:
        raise RuntimeError(f"No images found in {folder}")

    return paths


def estimate_bias_from_images(
    image_paths,
    percentile=0.5,
    sample_max=20,
    verbose=True
):
    """
    Auto-estimate effective bias from fluorescence images.

    Note: This is not a strict camera dark bias. It estimates an effective
    background offset from low-intensity percentiles. For accurate camera bias,
    use dark frame measurements.

    Args:
        image_paths: list of image paths.
        percentile: low-intensity percentile, e.g., 0.1, 0.5, 1.0.
        sample_max: max number of images used for estimation.
        verbose: whether to print diagnostics.

    Returns:
        bias_est: estimated bias.
    """

    if len(image_paths) == 0:
        raise RuntimeError("image_paths is empty, cannot estimate bias.")

    # Sample a subset if there are many images
    paths = image_paths[:min(len(image_paths), sample_max)]

    bias_candidates = []

    if verbose:
        print("")
        print("==========================================")
        print("Auto bias estimation")
        print(f"Use percentile: {percentile}%")
        print(f"Use images: {len(paths)} / {len(image_paths)}")
        print("==========================================")

    for i, path in enumerate(paths):
        img_raw = read_gray_image_raw(path)

        b = np.percentile(img_raw, percentile)
        bias_candidates.append(b)

        if verbose:
            print(f"[{i + 1}/{len(paths)}] {os.path.basename(path)}")
            print(f"    min  = {float(np.min(img_raw)):.4f}")
            print(f"    p{percentile} = {float(b):.4f}")
            print(f"    mean = {float(np.mean(img_raw)):.4f}")
            print(f"    max  = {float(np.max(img_raw)):.4f}")

    bias_candidates = np.array(bias_candidates, dtype=np.float32)

    # Median is more robust to outlier images than mean
    bias_est = float(np.median(bias_candidates))

    if verbose:
        print("------------------------------------------")
        print(f"Estimated effective BIAS = {bias_est:.4f}")
        print("==========================================")
        print("")

    return bias_est

# ============================================================
# 2. patch 提取
# ============================================================

def extract_patches(img, patch_size=32, stride=16):
    """从图像中滑动裁剪 patch。"""
    h, w = img.shape
    return [
        img[y:y + patch_size, x:x + patch_size]
        for y in range(0, h - patch_size + 1, stride)
        for x in range(0, w - patch_size + 1, stride)
    ]


# ============================================================
# 3. 3x3 局部均值
# ============================================================

def local_mean_3x3(patch):
    """
    不依赖 scipy 的 3x3 均值滤波。
    """

    p = np.pad(patch, ((1, 1), (1, 1)), mode="reflect")

    mean = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0

    return mean


# ============================================================
# 4. patch 纹理强度
# ============================================================

def gradient_energy(patch):
    """
    用梯度能量衡量 patch 的纹理强度。
    纹理越弱，值越小。
    """

    gx = patch[:, 1:] - patch[:, :-1]
    gy = patch[1:, :] - patch[:-1, :]

    e = np.mean(gx ** 2) + np.mean(gy ** 2)

    return e


# ============================================================
# 5. patch 噪声方差估计
# ============================================================

def estimate_patch_noise_variance(patch):
    """
    对一个弱纹理 patch 估计噪声方差。

    用 3x3 局部均值去掉低频结构：
        residual = patch - local_mean_3x3(patch)

    对白噪声 n，有：
        Var(n - mean_3x3(n)) ≈ 8/9 Var(n)

    所以用 8/9 做修正。
    """

    mean_img = local_mean_3x3(patch)
    residual = patch - mean_img

    correction = 8.0 / 9.0

    var_est = np.mean(residual ** 2) / correction

    return var_est


# ============================================================
# 6. 选择弱纹理 patch
# ============================================================

def collect_weak_texture_statistics(
    img_raw,
    bias=100.0,
    max_gray=2047.0,
    patch_size=32,
    stride=16,
    num_bins=20,
    weak_keep_ratio=0.25
):
    """
    从单张噪声图中选择弱纹理 patch，并收集：
        patch_mean
        patch_variance

    注意：
        这里在原始灰度单位下工作。
        先做:
            img = img_raw - bias

    最后拟合:
        variance_raw = alpha_raw * mean_raw + sigma_raw^2
    """

    img = img_raw.astype(np.float32) - bias
    records = []
    h, w = img.shape

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            raw_patch = img_raw[y:y + patch_size, x:x + patch_size]
            patch = img[y:y + patch_size, x:x + patch_size]

            # 负值比例
            neg_ratio = np.mean(patch < 0)

            if neg_ratio > NEG_RATIO_MAX:
                continue

            # 饱和比例，针对原始图像判断
            sat_ratio = np.mean(raw_patch >= max_gray)

            if sat_ratio > SAT_RATIO_MAX:
                continue

            patch_mean = np.mean(patch)

            # 均值太低的 patch 不参与 alpha 拟合，但背景可以帮助估计 sigma
            # 这里先保留，后面拟合时再处理
            patch_var = estimate_patch_noise_variance(patch)
            texture = gradient_energy(patch)

            if not (np.isfinite(patch_mean) and np.isfinite(patch_var)):
                continue
            if patch_var <= 0:
                continue

            records.append([patch_mean, patch_var, texture, neg_ratio, sat_ratio])

    if len(records) < 10:
        raise RuntimeError("有效 patch 太少，请检查 bias、图像亮度、patch_size 或 NEG_RATIO_MAX。")

    records = np.array(records, dtype=np.float32)

    means = records[:, 0]
    vars_ = records[:, 1]
    textures = records[:, 2]

    # 按亮度分 bin，每个 bin 内选择纹理最弱的一部分
    mean_min = np.percentile(means, 1)
    mean_max = np.percentile(means, 99)

    bins = np.linspace(mean_min, mean_max, num_bins + 1)

    selected_means = []
    selected_vars = []

    for i in range(num_bins):
        lo = bins[i]
        hi = bins[i + 1]

        idx = np.where((means >= lo) & (means < hi))[0]

        if len(idx) < 5:
            continue

        tex_bin = textures[idx]

        k = max(3, int(len(idx) * weak_keep_ratio))

        order = np.argsort(tex_bin)
        selected_idx = idx[order[:k]]

        selected_means.extend(means[selected_idx].tolist())
        selected_vars.extend(vars_[selected_idx].tolist())

    selected_means = np.array(selected_means, dtype=np.float32)
    selected_vars = np.array(selected_vars, dtype=np.float32)

    if len(selected_means) < 10:
        raise RuntimeError("弱纹理 patch 太少，请调大 WEAK_TEXTURE_KEEP_RATIO 或减小 NUM_BINS。")

    return selected_means, selected_vars, records

# ============================================================
# 7. 鲁棒拟合 var = alpha * mean + sigma^2
# ============================================================

def fit_poisson_gaussian_ransac(means, variances):
    """
    用 RANSAC 拟合：

        variance = alpha * mean + sigma^2

    返回:
        alpha
        sigma
        sigma^2
        inlier_mask
    """

    x = means.astype(np.float64)
    y = variances.astype(np.float64)

    # 只保留有限值
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)

    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        raise RuntimeError("用于拟合的点太少。")

    best_inliers = None
    best_score = -1

    # 用全局 MAD 估计大致残差尺度
    y_median = np.median(y)
    mad = np.median(np.abs(y - y_median)) + 1e-12
    threshold = INLIER_THRESHOLD_FACTOR * 1.4826 * mad

    if threshold <= 0:
        threshold = np.std(y) * 0.1 + 1e-12

    n = len(x)

    for _ in range(RANSAC_ITER):
        ids = random.sample(range(n), 2)

        x2 = x[ids]
        y2 = y[ids]

        if abs(x2[1] - x2[0]) < 1e-12:
            continue

        alpha = (y2[1] - y2[0]) / (x2[1] - x2[0])
        beta = y2[0] - alpha * x2[0]

        # 只限制 alpha >= 0；允许 beta，也就是 sigma^2，为负数。
        if alpha < 0:
            continue

        pred = alpha * x + beta
        residual = np.abs(y - pred)

        inliers = residual < threshold
        score = np.sum(inliers)

        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or np.sum(best_inliers) < 5:
        # RANSAC 失败，退化为普通最小二乘
        A = np.stack([x, np.ones_like(x)], axis=1)
        alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
        best_inliers = np.ones_like(x, dtype=bool)
    else:
        # 用内点重新做最小二乘
        xi = x[best_inliers]
        yi = y[best_inliers]

        A = np.stack([xi, np.ones_like(xi)], axis=1)
        alpha, beta = np.linalg.lstsq(A, yi, rcond=None)[0]

    alpha = max(float(alpha), 0.0)
    beta = float(beta)

    sigma2 = beta
    sigma = np.sqrt(sigma2) if sigma2 >= 0 else np.nan

    return alpha, sigma, sigma2, best_inliers

# ============================================================
# 8. 主函数
# ============================================================

def main():
    image_paths = get_image_paths(IMAGE_DIR)

    print(f"Found {len(image_paths)} images.")

    # 自动或手动确定 bias
    # ========================================================
    if AUTO_BIAS:
        used_bias = estimate_bias_from_images(
            image_paths,
            percentile=BIAS_PERCENTILE,
            sample_max=BIAS_SAMPLE_MAX,
            verbose=True
        )
    else:
        used_bias = BIAS
        print("")
        print("==========================================")
        print("Manual bias mode")
        print(f"Use manual BIAS = {used_bias:.4f}")
        print("==========================================")
        print("")

    all_means = []
    all_variances = []

    for i, path in enumerate(image_paths):
        print("")
        print(f"Processing [{i + 1}/{len(image_paths)}]: {path}")

        img_raw = read_gray_image_raw(path)


        print("raw min:", float(np.min(img_raw)))
        print("raw max:", float(np.max(img_raw)))
        print("raw mean:", float(np.mean(img_raw)))
        print("used bias:", float(used_bias))
        print("negative ratio after bias:", float(np.mean((img_raw - used_bias) < 0)))   #  减去bias 后  有多少比例的像素变为负值

        means, variances, _ = collect_weak_texture_statistics(
            img_raw,
            bias=used_bias,
            max_gray=MAX_GRAY,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            num_bins=NUM_BINS,
            weak_keep_ratio=WEAK_TEXTURE_KEEP_RATIO
        )

        print(f"selected weak-texture points from this image: {len(means)}")

        all_means.append(means)
        all_variances.append(variances)

    all_means = np.concatenate(all_means, axis=0)
    all_variances = np.concatenate(all_variances, axis=0)

    print("")
    print("Total selected weak-texture points:", len(all_means))

    alpha, sigma, sigma2, _ = fit_poisson_gaussian_ransac(
        all_means,
        all_variances
    )

    print("==========================================")
    print(f"used_bias  = {used_bias:.6f}")
    print(f"alpha_raw  = {alpha:.6f}")
    print(f"sigma_raw  = {sigma:.6f}")
    print(f"sigma_raw^2 = {sigma2:.6f}")
    print("==========================================")

if __name__ == "__main__":
    main()
