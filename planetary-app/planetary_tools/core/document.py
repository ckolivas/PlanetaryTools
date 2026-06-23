"""In-memory image document stored as 32-bit float linear colour."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from planetary_tools.core.color import linear_to_srgb


@dataclass
class ImageDocument:
    """Single-layer image in 32-bit float linear colour space."""

    data: np.ndarray
    path: Path | None = None
    is_grayscale: bool = False
    modified: bool = False
    # oklab_channels: dict[str, np.ndarray] = field(default_factory=dict)  # OKLab decompose (disabled)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def clone_data(self) -> np.ndarray:
        return self.data.copy()

    def set_data(self, data: np.ndarray, *, grayscale: bool | None = None) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        if grayscale is not None:
            self.is_grayscale = grayscale
        self.modified = True

    def to_display_rgb(self) -> np.ndarray:
        """8-bit sRGB array (H, W, 3) for on-screen display."""
        if self.is_grayscale:
            g = linear_to_srgb(self.data)
            if g.ndim == 2:
                rgb = np.stack([g, g, g], axis=-1)
            else:
                rgb = np.repeat(g[..., None], 3, axis=-1)
        else:
            rgb = linear_to_srgb(self.data)
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def title(self) -> str:
        name = self.path.name if self.path else "Untitled"
        return f"{name}{'*' if self.modified else ''}"