"""OKLab luminance, decompose, and compose filters.

DISABLED — implementations commented out; re-enable when needed.
"""

from __future__ import annotations

# import numpy as np
#
# from planetary_tools.core.color import clamp01, oklab_to_rgb, rgb_to_oklab
#
#
# def oklab_luminance(data: np.ndarray) -> np.ndarray:
#     """Desaturate to OKLab L (greyscale linear RGB with R=G=B=L)."""
#     rgb = clamp01(data)
#     if rgb.ndim == 2:
#         rgb = np.stack([rgb, rgb, rgb], axis=-1)
#     L = rgb_to_oklab(rgb)[..., 0]
#     out = np.stack([L, L, L], axis=-1)
#     return clamp01(out)
#
#
# def oklab_decompose(data: np.ndarray) -> dict[str, np.ndarray]:
#     """Split linear RGB into OKLab L, a (+0.5), b (+0.5) channel arrays."""
#     rgb = clamp01(data)
#     if rgb.ndim == 2:
#         rgb = np.stack([rgb, rgb, rgb], axis=-1)
#     lab = rgb_to_oklab(rgb)
#     return {
#         "L": lab[..., 0].astype(np.float32),
#         "a": (lab[..., 1] + 0.5).astype(np.float32),
#         "b": (lab[..., 2] + 0.5).astype(np.float32),
#     }
#
#
# def oklab_compose(
#     channel_l: np.ndarray,
#     channel_a: np.ndarray,
#     channel_b: np.ndarray,
# ) -> np.ndarray:
#     """Compose linear RGB from OKLab channel planes."""
#     L = channel_l.astype(np.float32)
#     a = channel_a.astype(np.float32) - 0.5
#     b = channel_b.astype(np.float32) - 0.5
#     lab = np.stack([L, a, b], axis=-1)
#     return oklab_to_rgb(lab)