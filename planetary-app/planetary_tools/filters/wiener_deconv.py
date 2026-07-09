"""Wiener filter using the same Moffat PSF as adaptive deconvolution.

Tooltip framing: PSF deconvolution denoising.

Uses the noise-regularized Wiener gain

    W(f) = |H(f)|² / (|H(f)|² + K)

where H is the Moffat PSF. Gain is always ≤ 1, so the filter damps
frequencies the PSF does not support rather than acting as an inverse
filter (which barely moves a 5×5 Moffat and easily amplifies noise).

Higher *amount* raises K. Adaptive mode applies the opposite spatial
weighting of adaptive deconvolution: less change in high-contrast areas,
using the same √(normalized local contrast) curve.
"""

from __future__ import annotations

import numpy as np

from planetary_tools.core.colour import clamp01, linear_luminance, oklab_to_rgb, rgb_to_oklab
from planetary_tools.filters.adaptive_deconv import _PSF, _std_windowed


def _pad_psf(psf: np.ndarray, height: int, width: int) -> np.ndarray:
    """Center a small PSF kernel in an FFT-sized array (ifftshift-ready origin)."""
    kh, kw = psf.shape
    out = np.zeros((height, width), dtype=np.float64)
    y0 = (height - kh) // 2
    x0 = (width - kw) // 2
    out[y0 : y0 + kh, x0 : x0 + kw] = np.asarray(psf, dtype=np.float64)
    return np.fft.ifftshift(out)


def _wiener_channel(channel: np.ndarray, nsr: float) -> np.ndarray:
    """Frequency-domain Wiener denoise of a single 2D channel (gain ≤ 1)."""
    img = np.asarray(channel, dtype=np.float64)
    height, width = img.shape
    h_freq = np.fft.rfft2(_pad_psf(_PSF, height, width))
    g_freq = np.fft.rfft2(img)
    h_abs2 = (h_freq.real * h_freq.real) + (h_freq.imag * h_freq.imag)
    # Denoise form: attenuate frequencies poorly supported by the PSF.
    # (Full inverse Wiener conj(H)/(|H|²+K) with this tiny Moffat is ~identity
    # after DC renorm and does not track amount usefully.)
    w_freq = h_abs2 / (h_abs2 + nsr)
    restored = np.fft.irfft2(g_freq * w_freq, s=(height, width))
    return restored.astype(np.float32)


def _adaptive_apply_weight(lum_2d: np.ndarray) -> np.ndarray:
    """Weight for applying the filter: high in flats, low on strong structure.

    Adaptive deconvolution multiplies its correction by √contrast_norm (more
    effect on edges). This is the complement: (1 − √contrast_norm).
    """
    contrast = _std_windowed(lum_2d).astype(np.float64)
    c_min = float(contrast.min())
    c_max = float(contrast.max())
    denom = c_max - c_min + 1e-10
    contrast_norm = (contrast - c_min) / denom
    return (1.0 - np.sqrt(contrast_norm)).astype(np.float32)


def _blend(original: np.ndarray, filtered: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """weight=1 → fully filtered; weight=0 → original."""
    w = weight.astype(np.float64)
    return (original.astype(np.float64) * (1.0 - w) + filtered.astype(np.float64) * w).astype(
        np.float32
    )


def _nsr_from_amount(amount: float) -> float:
    """Map UI amount to Wiener noise power K.

    Calibrated on planetary stacks so amount≈10 is a mild denoise and higher
    values continue to reduce fine residual energy (see sample smooth/noisy).
    """
    s = max(float(amount), 0.0) / 10.0
    # amount 10 → K=0.1; 20 → 0.4; 50 → 2.5; 100 → 10
    return max(0.1 * s * s, 1e-12)


def wiener_deconvolution(
    data: np.ndarray,
    is_grayscale: bool,
    amount: float = 10.0,
    adaptive: bool = True,
    oklab: bool = True,
) -> np.ndarray:
    """Wiener PSF denoise with optional contrast-adaptive blend."""
    if amount <= 0.0:
        return np.asarray(data, dtype=np.float32)

    src = clamp01(data)
    nsr = _nsr_from_amount(amount)

    if is_grayscale:
        ch = src if src.ndim == 2 else src[..., 0]
        ch = np.asarray(ch, dtype=np.float32)
        filtered = _wiener_channel(ch, nsr)
        if adaptive:
            weight = _adaptive_apply_weight(ch)
            filtered = _blend(ch, filtered, weight)
        return filtered.astype(np.float32)

    lum = linear_luminance(src)
    apply_weight = _adaptive_apply_weight(lum) if adaptive else None

    if oklab:
        lab = rgb_to_oklab(src)
        l_in = lab[..., 0].astype(np.float32)
        l_out = _wiener_channel(l_in, nsr)
        if apply_weight is not None:
            l_out = _blend(l_in, l_out, apply_weight)
        lab[..., 0] = l_out
        return oklab_to_rgb(lab).astype(np.float32)

    out = np.empty_like(src, dtype=np.float32)
    for c in range(3):
        ch = src[..., c]
        filtered = _wiener_channel(ch, nsr)
        if apply_weight is not None:
            filtered = _blend(ch, filtered, apply_weight)
        out[..., c] = filtered
    return out
