"""3×3 colour correction matrix for linear RGB."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

IDENTITY_MATRIX: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

# Shipped sensor colour-correction matrices (linear RGB).
SENSOR_MATRICES: dict[str, list[list[float]]] = {
    # Recalculated from the supplied 400–680 nm graph; see docs/sensor-colour-matrices.md.
    "IMX183": [
        [1.080, -0.037, -0.043],
        [-0.037, 1.059, -0.022],
        [-0.042, -0.029, 1.071],
    ],
    # Recalculated from the supplied 400–680 nm graph; see docs/sensor-colour-matrices.md.
    "IMX224": [
        [1.081, -0.036, -0.045],
        [-0.049, 1.071, -0.022],
        [-0.066, -0.042, 1.108],
    ],
    # Recalculated from the supplied 400–680 nm graph; see docs/sensor-colour-matrices.md.
    "IMX290": [
        [1.072, -0.033, -0.039],
        [-0.044, 1.064, -0.020],
        [-0.052, -0.034, 1.086],
    ],
    # Recalculated from the supplied 400–680 nm graph; see docs/sensor-colour-matrices.md.
    "IMX462": [
        [1.076, -0.029, -0.047],
        [-0.023, 1.040, -0.017],
        [-0.021, -0.022, 1.043],
    ],
    "IMX485": [
        [1.180, -0.123, -0.018],
        [-0.124, 1.169, -0.087],
        [-0.344, -0.078, 1.185],
    ],
    "IMX571": [
        [1.208, -0.147, -0.061],
        [-0.005, 1.024, -0.018],
        [-0.041, -0.092, 1.133],
    ],
    "IMX585": [
        [1.025, -0.024, 0.000],
        [-0.130, 1.026, -0.021],
        [-0.019, -0.076, 1.024],
    ],
    "IMX662": [
        [1.179, -0.088, -0.004],
        [-0.155, 1.175, -0.031],
        [-0.331, -0.095, 1.177],
    ],
    "IMX664": [
        [1.107, -0.102, -0.005],
        [-0.139, 1.186, -0.047],
        [-0.331, -0.062, 1.393],
    ],
    "IMX676": [
        [1.178, -0.096, -0.001],
        [-0.156, 1.164, -0.078],
        [-0.398, -0.066, 1.187],
    ],
    "IMX678": [
        [1.176, -0.187, -0.080],
        [-0.005, 1.149, -0.035],
        [-0.370, -0.103, 1.176],
    ],
    "IMX715": [
        [1.174, -0.133, -0.009],
        [-0.169, 1.164, -0.046],
        [-0.439, -0.041, 1.178],
    ],
}

COLOUR_MATRIX_SENSOR_NAMES = frozenset(SENSOR_MATRICES)


def colour_matrix_sensor_presets(
    default_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return built-in sensor presets sharing non-matrix defaults with Default."""
    base = {k: v for k, v in deepcopy(default_params).items() if k != "matrix"}
    return {
        name: {**deepcopy(base), "matrix": [row[:] for row in matrix]}
        for name, matrix in SENSOR_MATRICES.items()
    }


def matrix_from_params(params: dict) -> np.ndarray:
    """Return a 3×3 matrix from filter params."""
    raw = params.get("matrix", IDENTITY_MATRIX)
    mat = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    return mat


def apply_colour_matrix(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply [R′, G′, B′] = M × [R, G, B] to linear RGB."""
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    flat = rgb.reshape(-1, 3)
    out = flat @ np.asarray(matrix, dtype=np.float64).T
    return out.reshape(rgb.shape).astype(np.float32)