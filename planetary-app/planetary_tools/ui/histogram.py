"""RGB histogram widget for colour filter dialogs."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.core.colour import linear_to_srgb

HISTOGRAM_BINS = 256
_MAX_SAMPLES = 250_000
_PLOT_HEIGHT = 120
_HEADER_HEIGHT = 24

_CHANNEL_COLOURS = (
    QColor(235, 70, 70, 200),
    QColor(70, 210, 80, 200),
    QColor(80, 130, 235, 200),
)


def _subsample_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return (N, 3) samples for histogram computation."""
    flat = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    if flat.shape[0] <= _MAX_SAMPLES:
        return flat
    idx = np.linspace(0, flat.shape[0] - 1, _MAX_SAMPLES, dtype=np.int64)
    return flat[idx]


def _count_level_bins(channel: np.ndarray) -> np.ndarray:
    """GIMP-style 8-bit level bins for channel samples in [0, 1]."""
    levels = (np.clip(channel, 0.0, 1.0) * 255.0).astype(np.intp)
    return np.bincount(levels, minlength=HISTOGRAM_BINS)[:HISTOGRAM_BINS].astype(
        np.float32
    )


def compute_rgb_histograms(
    data: np.ndarray,
    *,
    perceptual: bool,
) -> np.ndarray:
    """Log-scaled RGB histograms shaped (3, 256) in [0, 1]."""
    rgb = np.asarray(data, dtype=np.float32)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    else:
        rgb = rgb[..., :3]

    if perceptual:
        rgb = linear_to_srgb(rgb)

    samples = _subsample_rgb(np.clip(rgb, 0.0, 1.0))
    out = np.zeros((3, HISTOGRAM_BINS), dtype=np.float32)
    for ch in range(3):
        out[ch] = _count_level_bins(samples[:, ch])

    # Log Y scale: a large bin-0 spike otherwise squashes the visible tail.
    out = np.log1p(out)
    peak = float(out.max())
    if peak > 0.0:
        out /= peak
    return out


class RgbHistogramWidget(QWidget):
    """Overlaid RGB histogram with linear / perceptual toggle."""

    space_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: np.ndarray | None = None
        self._is_grayscale = False
        self._perceptual = True
        self._histograms = np.zeros((3, HISTOGRAM_BINS), dtype=np.float32)

        self.setMinimumHeight(_HEADER_HEIGHT + _PLOT_HEIGHT + 8)
        self.setFixedHeight(_HEADER_HEIGHT + _PLOT_HEIGHT + 8)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QLabel("RGB histogram"))
        header.addStretch()
        self._space_combo = QComboBox()
        self._space_combo.addItem("Perceptual (sRGB)", True)
        self._space_combo.addItem("Linear", False)
        self._space_combo.setToolTip(
            "Perceptual matches GIMP default histogram encoding; "
            "linear shows radiometric channel values. "
            "Counts use a logarithmic vertical scale."
        )
        self._space_combo.currentIndexChanged.connect(self._on_space_changed)
        header.addWidget(self._space_combo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)

        self._plot = _HistogramPlot(self)
        self._plot.setFixedHeight(_PLOT_HEIGHT)
        layout.addWidget(self._plot)

    def set_data(self, data: np.ndarray, is_grayscale: bool) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self._is_grayscale = is_grayscale
        self._recompute()

    def _on_space_changed(self) -> None:
        self._perceptual = bool(self._space_combo.currentData())
        self._recompute()
        self.space_changed.emit()

    def _recompute(self) -> None:
        if self._data is None:
            self._histograms.fill(0.0)
        else:
            self._histograms = compute_rgb_histograms(
                self._data,
                perceptual=self._perceptual,
            )
        self._plot.set_histograms(self._histograms)
        self._plot.repaint()


class _HistogramPlot(QWidget):
    """Paint area for the overlaid channel curves."""

    _MARGIN_LEFT = 2
    _MARGIN_RIGHT = 2
    _MARGIN_TOP = 4
    _MARGIN_BOTTOM = 18

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._histograms = np.zeros((3, HISTOGRAM_BINS), dtype=np.float32)
        self.setMinimumHeight(_PLOT_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setAutoFillBackground(True)

    def set_histograms(self, histograms: np.ndarray) -> None:
        self._histograms = np.asarray(histograms, dtype=np.float32)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        left = self._MARGIN_LEFT
        right = w - self._MARGIN_RIGHT
        top = self._MARGIN_TOP
        bottom = h - self._MARGIN_BOTTOM
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)

        painter.fillRect(0, 0, w, h, QColor(36, 36, 40))
        painter.fillRect(left, top, plot_w, plot_h, QColor(22, 22, 24))
        painter.setPen(QPen(QColor(90, 90, 96), 1))
        painter.drawRect(left, top, plot_w - 1, plot_h - 1)

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        for ch in (1, 0, 2):
            hist = self._histograms[ch]
            if float(hist.max()) <= 0.0:
                continue
            path = QPainterPath()
            path.moveTo(left, bottom)
            for i, value in enumerate(hist):
                x = left + (i / max(HISTOGRAM_BINS - 1, 1)) * plot_w
                y = bottom - float(value) * plot_h
                path.lineTo(x, y)
            path.lineTo(right, bottom)
            path.closeSubpath()
            painter.fillPath(path, _CHANNEL_COLOURS[ch])

        painter.setPen(QPen(QColor(120, 120, 128), 1, Qt.PenStyle.DashLine))
        mid_x = left + plot_w * 0.5
        painter.drawLine(int(mid_x), top, int(mid_x), bottom)

        label_font = QFont(self.font())
        label_font.setPointSize(max(8, label_font.pointSize() - 1))
        painter.setFont(label_font)
        painter.setPen(QColor(190, 190, 198))
        painter.drawText(left + 2, h - 4, "0%")
        painter.drawText(int(mid_x) - 14, h - 4, "50%")
        painter.drawText(right - 30, h - 4, "100%")