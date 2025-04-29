import numpy as np
import scipy as sp
import imageio.v2 as imageio
import os
from PIL import Image


def box(img, r):
    """
    O(1) box filter using cumulative sum.
    
    Args:
        img (np.ndarray): Input image, at least 2D.
        r (int): Radius of the box filter.

    Returns:
        np.ndarray: Box-filtered image.
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, axis=0)
    imDst[0:r + 1, :, ...] = imCum[r:2 * r + 1, :, ...]
    imDst[r + 1:rows - r, :, ...] = imCum[2 * r + 1:rows, :, ...] - imCum[0:rows - 2 * r - 1, :, ...]
    imDst[rows - r:rows, :, ...] = np.tile(imCum[rows - 1:rows, :, ...], tile) - imCum[rows - 2 * r - 1:rows - r - 1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, axis=1)
    imDst[:, 0:r + 1, ...] = imCum[:, r:2 * r + 1, ...]
    imDst[:, r + 1:cols - r, ...] = imCum[:, 2 * r + 1:cols, ...] - imCum[:, 0:cols - 2 * r - 1, ...]
    imDst[:, cols - r:cols, ...] = np.tile(imCum[:, cols - 1:cols, ...], tile) - imCum[:, cols - 2 * r - 1:cols - r - 1, ...]

    return imDst


def _gf_color(I, p, r, eps, s=None):
    """
    Color guided filter (supporting optional subsampling for speedup).

    Args:
        I (np.ndarray): Guide image (3 channels).
        p (np.ndarray): Filtering input (1 channel).
        r (int): Window radius.
        eps (float): Regularization parameter.
        s (int, optional): Subsampling factor for fast guided filter.

    Returns:
        np.ndarray: Filtered output.
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1 / s, 1 / s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1 / s, 1 / s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    # Mean values
    mI_r = box(I[:, :, 0], r) / N
    mI_g = box(I[:, :, 1], r) / N
    mI_b = box(I[:, :, 2], r) / N
    mP = box(p, r) / N

    # Mean of I*p
    mIp_r = box(I[:, :, 0] * p, r) / N
    mIp_g = box(I[:, :, 1] * p, r) / N
    mIp_b = box(I[:, :, 2] * p, r) / N

    # Covariance between I and p
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # Variance matrix of I
    var_I_rr = box(I[:, :, 0] * I[:, :, 0], r) / N - mI_r * mI_r
    var_I_rg = box(I[:, :, 0] * I[:, :, 1], r) / N - mI_r * mI_g
    var_I_rb = box(I[:, :, 0] * I[:, :, 2], r) / N - mI_r * mI_b
    var_I_gg = box(I[:, :, 1] * I[:, :, 1], r) / N - mI_g * mI_g
    var_I_gb = box(I[:, :, 1] * I[:, :, 2], r) / N - mI_g * mI_b
    var_I_bb = box(I[:, :, 2] * I[:, :, 2], r) / N - mI_b * mI_b

    # Solve linear systems
    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]]
            ])
            covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
            a[i, j, :] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:, :, 0] * mI_r - a[:, :, 1] * mI_g - a[:, :, 2] * mI_b

    meanA = box(a, r) / N[..., np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """
    Grayscale guided filter.

    Args:
        I (np.ndarray): Guide image (1 channel).
        p (np.ndarray): Filtering input.
        r (int): Window radius.
        eps (float): Regularization parameter.
        s (int, optional): Subsampling factor.

    Returns:
        np.ndarray: Filtered output.
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1 / s, order=1)
        Psub = sp.ndimage.zoom(p, 1 / s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p

    (rows, cols) = Isub.shape
    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N

    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """
    Automatically choose grayscale or color guided filter.

    Args:
        I (np.ndarray): Guide image.
        p (np.ndarray): Filtering input.
        r (int): Window radius.
        eps (float): Regularization parameter.
        s (int, optional): Subsampling factor.

    Returns:
        np.ndarray: Filtered output.
    """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        raise ValueError(f"Invalid guide dimensions: {I.shape}")


def guided_filter(I, p, r, eps, s=None):
    """
    Apply guided filtering channel-wise.

    Args:
        I (np.ndarray): Guide image (1 or 3 channels).
        p (np.ndarray): Filtering input (1 or more channels).
        r (int): Window radius.
        eps (float): Regularization parameter.
        s (int, optional): Subsampling factor.

    Returns:
        np.ndarray: Filtered output.
    """
    if p.ndim == 2:
        p = p[:, :, np.newaxis]

    out = np.zeros_like(p)
    for ch in range(p.shape[2]):
        out[:, :, ch] = _gf_colorgray(I, p[:, :, ch], r, eps, s)

    return np.squeeze(out) if out.shape[2] == 1 else out


# Dataset preprocessing
rgb_dir = os.listdir(os.path.join('/xxxx/', 'rgb_aug'))
rgb_path = '/xxxx/xxx/rgb_aug/'
tir_path = '/xxxx/xxx/nir_aug/'
rgb_lf_save_path = '/xxxx/xxx/rgb_aug_lf/'
rgb_hf_save_path = '/xxxx/xxx/rgb_aug_hf/'
tir_lf_save_path = '/xxxx/xxx/tir_aug_lf/'
tir_hf_save_path = '/xxxx/xxx/tir_aug_hf/'

# Create save directories
for path in [rgb_lf_save_path, rgb_hf_save_path, tir_lf_save_path, tir_hf_save_path]:
    os.makedirs(path, exist_ok=True)

# Process each image
for idx, filename in enumerate(rgb_dir):
    print(f"Processing {idx}/{len(rgb_dir)}: {filename}")

    rgb = np.array(imageio.imread(rgb_path + filename)).astype(np.float32) / 255.0
    tir = np.array(Image.open(tir_path + filename).convert('L')).astype(np.float32) / 255.0
    tir = np.repeat(tir[:, :, np.newaxis], repeats=3, axis=2)

    # Apply guided filtering
    rgb_lf = guided_filter(rgb, rgb, 8, 0.05, s=4)
    tir_lf = guided_filter(tir, tir, 8, 0.05, s=4)

    # Compute high-frequency components
    rgb_hf = rgb - rgb_lf
    tir_hf = tir - tir_lf

    # Save results
    imageio.imwrite(rgb_lf_save_path + filename, rgb_lf)
    imageio.imwrite(rgb_hf_save_path + filename, rgb_hf)
    imageio.imwrite(tir_lf_save_path + filename, tir_lf)
    imageio.imwrite(tir_hf_save_path + filename, tir_hf)
