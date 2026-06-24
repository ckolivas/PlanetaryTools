"""Linear RGB and OKLab colour conversions."""

from __future__ import annotations

import numpy as np

# sRGB transfer functions
_SRGB_BREAK = 0.0031308


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded values in [0, 1] to linear light."""
    v = np.asarray(value, dtype=np.float64)
    linear = np.where(
        v <= 0.04045,
        v / 12.92,
        np.power((v + 0.055) / 1.055, 2.4),
    )
    return linear.astype(np.float32)


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    """Convert linear light in [0, 1] to sRGB for display or 8-bit export."""
    v = np.asarray(value, dtype=np.float64)
    v = np.clip(v, 0.0, 1.0)
    encoded = np.where(
        v <= _SRGB_BREAK,
        v * 12.92,
        1.055 * np.power(v, 1.0 / 2.4) - 0.055,
    )
    return encoded.astype(np.float32)


# OKLab matrices (Björn Ottosson, https://bottosson.github.io/posts/oklab/)
_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)

_LMS_TO_OKLAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)

_OKLAB_TO_LMS = np.array(
    [
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480],
    ],
    dtype=np.float64,
)

_LMS_TO_RGB = np.array(
    [
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010],
    ],
    dtype=np.float64,
)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB (H, W, 3) -> OKLab (H, W, 3)."""
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    lms = flat @ _RGB_TO_LMS.T
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))
    lab = lms_cbrt @ _LMS_TO_OKLAB.T
    return lab.reshape(rgb.shape).astype(np.float32)


def oklab_to_rgb(lab: np.ndarray, *, clamp: bool = True) -> np.ndarray:
    """OKLab (H, W, 3) -> linear RGB (H, W, 3); optionally clamped to [0, 1]."""
    flat = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    lms_ = flat @ _OKLAB_TO_LMS.T
    lms = np.power(np.maximum(lms_, 0.0), 3.0)
    rgb = lms @ _LMS_TO_RGB.T
    if clamp:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.reshape(lab.shape).astype(np.float32)


def rgb_to_oklab_L(rgb: np.ndarray) -> np.ndarray:
    """Return OKLab L channel only."""
    return rgb_to_oklab(rgb)[..., 0]


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.709 luminance from linear RGB."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.212671 * r + 0.715160 * g + 0.072169 * b).astype(np.float32)


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)