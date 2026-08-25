"""Rotate image dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class RotateImageDialog(QDialog):
    """Set rotation angle and optional crop-to-original."""

    def __init__(self, width: int, height: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rotate Image")
        self._orig_w = max(1, int(width))
        self._orig_h = max(1, int(height))

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"Original size: {self._orig_w} × {self._orig_h} px")
        )
        layout.addWidget(
            QLabel(
                "Positive angles are counter-clockwise. The canvas expands by "
                "default to fit the rotated image. Resampled with bicubic "
                "interpolation (highest quality Pillow allows for float data)."
            )
        )

        form = QFormLayout()
        self._angle = QDoubleSpinBox()
        self._angle.setRange(-3600.0, 3600.0)
        self._angle.setDecimals(2)
        self._angle.setSingleStep(0.01)
        self._angle.setSuffix(" °")
        self._angle.setValue(0.0)
        self._angle.setToolTip(
            "Rotation angle in degrees. Positive = counter-clockwise (CCW). "
            "Uses high-quality bicubic resampling (best Pillow allows for float)."
        )
        form.addRow("Angle", self._angle)

        presets = QHBoxLayout()
        btn_ccw = QPushButton("90° CCW")
        btn_ccw.setToolTip("Rotate 90° counter-clockwise.")
        btn_ccw.clicked.connect(lambda: self._angle.setValue(90.0))
        presets.addWidget(btn_ccw)

        btn_cw = QPushButton("90° CW")
        btn_cw.setToolTip("Rotate 90° clockwise (−90°).")
        btn_cw.clicked.connect(lambda: self._angle.setValue(-90.0))
        presets.addWidget(btn_cw)

        btn_180 = QPushButton("180°")
        btn_180.setToolTip("Rotate 180°.")
        btn_180.clicked.connect(lambda: self._angle.setValue(180.0))
        presets.addWidget(btn_180)
        form.addRow("Presets", presets)

        self._crop = QCheckBox("Crop to original size")
        self._crop.setChecked(False)
        self._crop.setToolTip(
            "After expanding to fit the rotated image, centre-crop back to "
            "the original width and height."
        )
        form.addRow(self._crop)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def angle_deg(self) -> float:
        return float(self._angle.value())

    def crop_to_original(self) -> bool:
        return self._crop.isChecked()
