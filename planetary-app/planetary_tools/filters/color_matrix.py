"""3×3 colour correction matrix for linear RGB."""

from __future__ import annotations

import numpy as np

IDENTITY_MATRIX: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


def matrix_from_params(params: dict) -> np.ndarray:
    """Return a 3×3 matrix from filter params."""
    raw = params.get("matrix", IDENTITY_MATRIX)
    mat = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    return mat


def apply_color_matrix(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply [R′, G′, B′] = M × [R, G, B] to linear RGB."""
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    flat = rgb.reshape(-1, 3)
    out = flat @ np.asarray(matrix, dtype=np.float64).T
    return out.reshape(rgb.shape).astype(np.float32)