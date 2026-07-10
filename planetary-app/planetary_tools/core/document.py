"""In-memory image document stored as 32-bit float linear colour."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from planetary_tools.core.colour import linear_to_srgb


@dataclass
class ImageDocument:
    """Single-layer image in 32-bit float linear colour space."""

    data: np.ndarray
    path: Path | None = None
    is_grayscale: bool = False
    modified: bool = False
    # Native channel precision of the source file (8, 16, or 32).
    storage_bits: int = 32
    # oklab_channels: dict[str, np.ndarray] = field(default_factory=dict)  # OKLab decompose (disabled)
    # Noise residual probe context pinned at load / first pin. Survives filter
    # applies so enhance dialogs keep the same absolute noise scale (sharpening
    # must not re-estimate a finer "texture scale" and drop the score).
    noise_texture_scale: float | None = field(default=None, repr=False)
    noise_chromatic: bool | None = field(default=None, repr=False)

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

    def pin_noise_context(self) -> None:
        """Capture noise probe context from the current pixels.

        Call when a new source image is established (load, compose). Do not call
        after enhance filters — the pinned scale must stay that of the stack as
        loaded so absolute noise remains comparable across dialogs.
        """
        from planetary_tools.core.noise import estimate_texture_scale, is_chromatic

        self.noise_texture_scale = estimate_texture_scale(
            self.data, self.is_grayscale
        )
        self.noise_chromatic = is_chromatic(self.data, self.is_grayscale)

    def noise_context(self) -> tuple[float, bool]:
        """Return (texture_scale, chromatic), pinning from current data if needed."""
        if self.noise_texture_scale is None or self.noise_chromatic is None:
            self.pin_noise_context()
        assert self.noise_texture_scale is not None
        assert self.noise_chromatic is not None
        return self.noise_texture_scale, self.noise_chromatic

    def set_data(self, data: np.ndarray, *, grayscale: bool | None = None) -> None:
        """Replace pixel data. Does **not** clear the pinned noise context."""
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