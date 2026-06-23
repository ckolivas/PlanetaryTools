"""Adaptive deconvolution — numpy/scipy port of the GIMP plug-in."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import convolve, uniform_filter

from planetary_tools.core.color import clamp01, linear_luminance, oklab_to_rgb, rgb_to_oklab

# Moffat PSF (gamma=1, beta=2, 5x5) — matches GIMP plug-in
_GAMMA = 1.0
_BETA = 2.0
_PSF_SIZE = 5


def _generate_moffat_kernel(gamma: float = 1.0, beta: float = 2.0, size: int = 5) -> np.ndarray:
    half = size // 2
    psf = np.zeros((size, size), dtype=np.float64)
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            r = math.sqrt(dx * dx + dy * dy)
            psf[dy + half, dx + half] = (1.0 + (r / gamma) ** 2) ** (-beta)
    psf /= psf.sum()
    return psf.astype(np.float32)


_PSF = _generate_moffat_kernel(_GAMMA, _BETA, _PSF_SIZE)
_PSF_MIRROR = _PSF[::-1, ::-1]


def _std_windowed(lum: np.ndarray, win_size: tuple[int, int] = (7, 7)) -> np.ndarray:
    """Local standard deviation via box-filtered mean and mean-of-squares."""
    size = win_size[0]
    mean = uniform_filter(lum.astype(np.float64), size=size, mode="reflect")
    mean_sq = uniform_filter(lum.astype(np.float64) ** 2, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def _convolve2d(flat: np.ndarray, kernel: np.ndarray, width: int, height: int) -> np.ndarray:
    img = flat.reshape(height, width)
    out = convolve(img, kernel, mode="reflect")
    return out.ravel().astype(np.float32)


def _oklab_sharpen_rgb(
    rgb: np.ndarray,
    ratio: np.ndarray,
) -> np.ndarray:
    """Scale OKLab L by per-pixel ratio and convert back to linear RGB."""
    lab = rgb_to_oklab(rgb)
    lab[..., 0] = lab[..., 0] * ratio.reshape(rgb.shape[0], rgb.shape[1])
    return oklab_to_rgb(lab)


def adaptive_deconvolution(
    data: np.ndarray,
    is_grayscale: bool,
    amount: float = 10.0,
    adaptive: bool = True,
    oklab: bool = True,
) -> np.ndarray:
    strength = amount / math.pi
    src = clamp01(data)

    if is_grayscale:
        ch = src if src.ndim == 2 else src[..., 0]
        flat = ch.ravel().astype(np.float32)
        channels = 1
        is_gray = True
        oklab = False
    else:
        flat = src.reshape(-1, 3).astype(np.float32).ravel()
        channels = 3
        is_gray = False

    height, width = (src.shape[0], src.shape[1]) if src.ndim >= 2 else src.shape
    num_pixels = width * height

    lum = linear_luminance(src).ravel() if not is_gray else flat.copy()
    max_lum = float(lum.max()) if lum.size else 0.0

    contrast = _std_windowed(lum.reshape(height, width)).ravel()
    c_min = float(contrast.min())
    c_max = float(contrast.max())
    denom = c_max - c_min + 1e-10
    contrast_norm = (contrast - c_min) / denom
    sqrt_contrast = np.sqrt(contrast_norm) if adaptive else None

    if not oklab and not is_gray:
        red = flat[0::3].copy()
        green = flat[1::3].copy()
        blue = flat[2::3].copy()
        channel_data = [red, green, blue]
        corr_minus = []
        max_vals = []
        for ch_data in channel_data:
            max_vals.append(min(2.0 * float(ch_data.max()), 1.0))
            conv = _convolve2d(ch_data, _PSF, width, height)
            relative = ch_data / (conv + 1e-12)
            correction = _convolve2d(relative, _PSF_MIRROR, width, height)
            corr_minus.append(correction - 1.0)

        sharpened = []
        for idx, ch_data in enumerate(channel_data):
            if adaptive:
                damped = 1.0 + strength * sqrt_contrast * corr_minus[idx]
            else:
                damped = 1.0 + strength * corr_minus[idx]
            sharp = ch_data * damped * damped * damped
            sharp = np.minimum(sharp, max_vals[idx])
            sharpened.append(sharp)

        out = np.zeros(num_pixels * 3, dtype=np.float32)
        out[0::3] = sharpened[0]
        out[1::3] = sharpened[1]
        out[2::3] = sharpened[2]
        result = out.reshape(height, width, 3)
    else:
        max_val = min(2.0 * max_lum, 1.0)
        conv = _convolve2d(lum, _PSF, width, height)
        relative = lum / (conv + 1e-12)
        correction = _convolve2d(relative, _PSF_MIRROR, width, height)
        corr_minus_one = correction - 1.0

        if adaptive:
            damped = 1.0 + strength * sqrt_contrast * corr_minus_one
        else:
            damped = 1.0 + strength * corr_minus_one

        if is_gray:
            sharpened = lum * damped * damped * damped
            sharpened = np.minimum(sharpened, max_val)
            result = sharpened.reshape(height, width)
        else:
            result = _oklab_sharpen_rgb(src.reshape(height, width, 3), damped)

    return result.astype(np.float32)