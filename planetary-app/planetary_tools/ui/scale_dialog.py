"""Scale image dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class ScaleImageDialog(QDialog):
    """Set output size by percentage, width, or height."""

    def __init__(
        self,
        width: int,
        height: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale Image")
        self._orig_w = max(1, int(width))
        self._orig_h = max(1, int(height))
        self._syncing = False
        max_dim = max(65535, self._orig_w * 10, self._orig_h * 10)

        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(f"Original size: {self._orig_w} × {self._orig_h} px")
        )

        form = QFormLayout()
        self._percent = QDoubleSpinBox()
        self._percent.setRange(0.01, 10_000.0)
        self._percent.setDecimals(2)
        self._percent.setSingleStep(1.0)
        self._percent.setSuffix(" %")
        self._percent.setValue(100.0)
        self._percent.valueChanged.connect(self._on_percent_changed)
        form.addRow("Scale", self._percent)

        self._width = QSpinBox()
        self._width.setRange(1, max_dim)
        self._width.setValue(self._orig_w)
        self._width.valueChanged.connect(self._on_width_changed)
        form.addRow("Width", self._width)

        self._height = QSpinBox()
        self._height.setRange(1, max_dim)
        self._height.setValue(self._orig_h)
        self._height.valueChanged.connect(self._on_height_changed)
        form.addRow("Height", self._height)

        self._aspect = QCheckBox("Maintain aspect ratio")
        self._aspect.setChecked(True)
        self._aspect.toggled.connect(self._on_aspect_toggled)
        form.addRow(self._aspect)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def output_size(self) -> tuple[int, int]:
        return self._width.value(), self._height.value()

    def _on_aspect_toggled(self, _enabled: bool) -> None:
        if self._aspect.isChecked():
            self._sync_from_width()

    def _on_percent_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            scale = float(value) / 100.0
            self._width.setValue(max(1, round(self._orig_w * scale)))
            if self._aspect.isChecked():
                self._height.setValue(max(1, round(self._orig_h * scale)))
        finally:
            self._syncing = False

    def _on_width_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if self._aspect.isChecked():
                ratio = self._orig_h / self._orig_w
                self._height.setValue(max(1, round(value * ratio)))
                self._percent.setValue(value / self._orig_w * 100.0)
            else:
                self._percent.setValue(value / self._orig_w * 100.0)
        finally:
            self._syncing = False

    def _on_height_changed(self, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            if self._aspect.isChecked():
                ratio = self._orig_w / self._orig_h
                self._width.setValue(max(1, round(value * ratio)))
                self._percent.setValue(value / self._orig_h * 100.0)
            else:
                self._percent.setValue(value / self._orig_h * 100.0)
        finally:
            self._syncing = False

    def _sync_from_width(self) -> None:
        self._on_width_changed(self._width.value())