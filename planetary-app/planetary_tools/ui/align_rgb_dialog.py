"""RGB registration controls hosted in the shared preview dock."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel

from planetary_tools.core.align import align_channel
from planetary_tools.ui.dialogs import _FilterDialog


class AlignRgbDialog(_FilterDialog):
    def __init__(self, parent=None) -> None:
        super().__init__("Align RGB", parent)

    def _build_filter_params(self) -> None:
        hint = QLabel("Align red and blue to green. Green and the canvas size stay unchanged.")
        hint.setWordWrap(True)
        self._form.addRow(hint)

        self._mask_enabled = QCheckBox("Mask dim pixels")
        self._mask_enabled.setChecked(True)
        self._mask_enabled.setToolTip(
            "Ignore dim pixels while estimating alignment.\n"
            "Masking does not remove pixels from the resulting image."
        )
        self._form.addRow(self._mask_enabled)
        self._mask_percent = QDoubleSpinBox()
        self._mask_percent.setRange(0, 100)
        self._mask_percent.setDecimals(1)
        self._mask_percent.setValue(25)
        self._mask_percent.setSuffix(" %")
        self._mask_percent.setKeyboardTracking(False)
        self._mask_percent.setToolTip(
            "Exclude this bottom percentage of each channel's\n"
            "own minimum-to-maximum perceptual brightness range."
        )
        self._form.addRow("Brightness mask", self._mask_percent)

        self._derotate = QCheckBox("Derotate")
        self._derotate.setChecked(True)
        self._derotate.setToolTip(
            "Rotate red and blue to match green as well as shifting them.\n"
            "Uncheck for faster, shift-only alignment."
        )
        self._form.addRow(self._derotate)
        self._max_angle = QDoubleSpinBox()
        self._max_angle.setRange(.5, 180)
        self._max_angle.setDecimals(2)
        self._max_angle.setValue(45)
        self._max_angle.setSuffix(" °")
        self._max_angle.setKeyboardTracking(False)
        self._max_angle.setToolTip("Search this many degrees either side of zero.")
        self._form.addRow("Max search angle", self._max_angle)

        self._mask_enabled.toggled.connect(self._mask_percent.setEnabled)
        self._derotate.toggled.connect(self._max_angle.setEnabled)
        self._mask_enabled.toggled.connect(self.params_changed.emit)
        self._mask_percent.valueChanged.connect(self.params_changed.emit)
        self._derotate.toggled.connect(self.params_changed.emit)
        self._max_angle.valueChanged.connect(self.params_changed.emit)

    def build_filter_func(self):
        # Snapshot controls before the preview worker starts. Each render uses
        # the original RGB source, never a previously aligned preview.
        fraction = self._mask_percent.value() / 100 if self._mask_enabled.isChecked() else None
        rotate = self._derotate.isChecked()
        angle = self._max_angle.value()

        def apply(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
            if is_grayscale or data.ndim != 3 or data.shape[2] != 3:
                raise ValueError("Align RGB requires an RGB image.")
            green = data[..., 1]
            channels = [
                green if c == 1 else align_channel(
                    green, data[..., c], mask_fraction=fraction, rotate=rotate, max_angle=angle,
                )
                for c in range(3)
            ]
            return np.stack(channels, axis=-1).astype(np.float32)

        return apply
