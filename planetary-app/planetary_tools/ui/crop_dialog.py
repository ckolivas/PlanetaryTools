"""Crop/expand image panel: size, centre offset, autocrop, live canvas outline."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.core.crop import (
    DEFAULT_BORDER_PX,
    DEFAULT_MIN_BRIGHTNESS_PCT,
    CropRect,
    autocrop_rect,
    offset_from_rect,
    rect_from_size_offset,
)


class CropImageDialog(QWidget):
    """Hosted in the main-window dock so the canvas overlay stays visible."""

    rect_changed = pyqtSignal(int, int, int, int)
    accepted = pyqtSignal()
    rejected = pyqtSignal()
    filter_id = ""

    def __init__(
        self,
        width: int,
        height: int,
        data: np.ndarray,
        is_grayscale: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._img_w = max(1, int(width))
        self._img_h = max(1, int(height))
        self._data = data
        self._is_grayscale = is_grayscale
        self._syncing = False
        max_dim = max(65535, self._img_w * 10, self._img_h * 10)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"Original size: {self._img_w} × {self._img_h} px"))
        hint = QLabel(
            "The dashed outline is the proposed canvas. Sizes larger than the "
            "image expand it (new pixels are black). Offsets are from the "
            "image centre (positive = right / down)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._width = QSpinBox()
        self._width.setRange(1, max_dim)
        self._width.setValue(self._img_w)
        self._width.setSuffix(" px")
        self._width.setToolTip(
            "Output width in pixels. Larger than the image expands the canvas."
        )
        self._width.valueChanged.connect(self._on_geometry_changed)
        form.addRow("Width", self._width)

        self._height = QSpinBox()
        self._height.setRange(1, max_dim)
        self._height.setValue(self._img_h)
        self._height.setSuffix(" px")
        self._height.setToolTip(
            "Output height in pixels. Larger than the image expands the canvas."
        )
        self._height.valueChanged.connect(self._on_geometry_changed)
        form.addRow("Height", self._height)

        self._offset_x = QSpinBox()
        self._offset_x.setRange(-max_dim, max_dim)
        self._offset_x.setValue(0)
        self._offset_x.setSuffix(" px")
        self._offset_x.setToolTip(
            "Horizontal shift of the output centre from the image middle. "
            "Positive moves the output to the right."
        )
        self._offset_x.valueChanged.connect(self._on_geometry_changed)
        form.addRow("Offset X", self._offset_x)

        self._offset_y = QSpinBox()
        self._offset_y.setRange(-max_dim, max_dim)
        self._offset_y.setValue(0)
        self._offset_y.setSuffix(" px")
        self._offset_y.setToolTip(
            "Vertical shift of the output centre from the image middle. "
            "Positive moves the output down."
        )
        self._offset_y.valueChanged.connect(self._on_geometry_changed)
        form.addRow("Offset Y", self._offset_y)

        self._region_label = QLabel()
        self._region_label.setWordWrap(True)
        form.addRow(self._region_label)
        layout.addLayout(form)

        auto_box = QGroupBox("Autocrop")
        auto_form = QFormLayout(auto_box)
        self._border = QSpinBox()
        self._border.setRange(0, max_dim)
        self._border.setValue(DEFAULT_BORDER_PX)
        self._border.setSuffix(" px")
        self._border.setToolTip(
            "Pixels kept around the detected object on every side. "
            "If that reaches past the frame, the canvas expands."
        )
        auto_form.addRow("Border", self._border)

        self._min_bright = QSpinBox()
        self._min_bright.setRange(0, 100)
        self._min_bright.setValue(int(DEFAULT_MIN_BRIGHTNESS_PCT))
        self._min_bright.setSuffix(" %")
        self._min_bright.setToolTip(
            "Pixels at least this bright relative to the image peak are "
            "treated as the central object. Default 10%."
        )
        auto_form.addRow("Min brightness", self._min_bright)

        auto_btn = QPushButton("Autocrop")
        auto_btn.setToolTip(
            "Detect the central bright object and set the region to its "
            "bounding box plus the border."
        )
        auto_btn.clicked.connect(self._run_autocrop)
        auto_form.addRow(auto_btn)
        layout.addWidget(auto_box)

        reset_btn = QPushButton("Full image")
        reset_btn.setToolTip("Reset to the entire image (no crop or expand).")
        reset_btn.clicked.connect(self._reset_full)
        layout.addWidget(reset_btn)

        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self._reject)
        layout.addWidget(buttons)

        self._refresh_region_label()

    def crop_rect(self) -> CropRect:
        return rect_from_size_offset(
            self._img_w,
            self._img_h,
            self._width.value(),
            self._height.value(),
            self._offset_x.value(),
            self._offset_y.value(),
        )

    def emit_current_rect(self) -> None:
        rect = self.crop_rect()
        self.rect_changed.emit(rect.x, rect.y, rect.width, rect.height)

    def _accept(self) -> None:
        self.accepted.emit()

    def _reject(self) -> None:
        self.rejected.emit()

    def _on_geometry_changed(self, _value: int = 0) -> None:
        if self._syncing:
            return
        self._apply_rect(self.crop_rect(), emit=True)

    def _apply_rect(self, rect: CropRect, *, emit: bool) -> None:
        ox, oy = offset_from_rect(self._img_w, self._img_h, rect)
        self._syncing = True
        try:
            self._width.setValue(rect.width)
            self._height.setValue(rect.height)
            self._offset_x.setValue(ox)
            self._offset_y.setValue(oy)
        finally:
            self._syncing = False
        self._refresh_region_label()
        if emit:
            self.rect_changed.emit(rect.x, rect.y, rect.width, rect.height)

    def _refresh_region_label(self) -> None:
        rect = self.crop_rect()
        if rect.extends_outside(self._img_w, self._img_h):
            mode = "expand"
        elif rect.width < self._img_w or rect.height < self._img_h:
            mode = "crop"
        else:
            mode = "full"
        self._region_label.setText(
            f"Region: {rect.x}, {rect.y}  {rect.width} × {rect.height} px "
            f"({mode})"
        )

    def _run_autocrop(self) -> None:
        rect, found = autocrop_rect(
            self._data,
            self._is_grayscale,
            border_px=self._border.value(),
            min_brightness_pct=float(self._min_bright.value()),
        )
        self._apply_rect(rect, emit=True)
        if not found:
            QMessageBox.warning(
                self,
                "Autocrop",
                "No central object found at this minimum brightness. "
                "Try a lower value.",
            )

    def _reset_full(self) -> None:
        self._apply_rect(CropRect(0, 0, self._img_w, self._img_h), emit=True)
