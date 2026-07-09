"""Filter parameter dialogs with presets, brightness readout, and live preview."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.core.brightness import BrightnessInfo, measure_brightness
from planetary_tools.core.presets import (
    ensure_builtin_presets,
    reserved_preset_names,
    save_presets,
)
from planetary_tools.filters.colour_matrix import IDENTITY_MATRIX
from planetary_tools.filters.levels import (
    LEVEL_CHANNELS,
    auto_balance_levels,
    default_levels_params,
    identity_levels,
    normalize_levels_params,
)
from planetary_tools.filters.registry import (
    CLAMP_FILTER_IDS,
    CLAMP_LOW_PARAM,
    CLAMP_PARAM,
    ENHANCE_FILTER_IDS,
    FILTERS,
    apply_filter,
)
from planetary_tools.ui.histogram import RgbHistogramWidget

FilterFunc = Callable[[np.ndarray, bool], np.ndarray]

COLOUR_FILTER_IDS = frozenset({
    "stretch_contrast",
    "colour_matrix",
    "saturation_vibrance",
    "levels",
})

_MATRIX_INPUT_LABELS = ("R in", "G in", "B in")
_MATRIX_OUTPUT_LABELS = ("R out", "G out", "B out")

_LEVEL_CHANNEL_LABELS = {
    "L": "Luminance",
    "R": "Red",
    "G": "Green",
    "B": "Blue",
}

_LEVEL_PCT_SPIN_WIDTH = 102
FILTER_PANEL_WIDTH = 330


def _make_level_pct_spin(
    default_pct: float,
    *,
    on_change: Callable[[], None] | None = None,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setDecimals(1)
    spin.setSingleStep(0.5)
    spin.setSuffix(" %")
    spin.setValue(default_pct)
    spin.setMaximumWidth(_LEVEL_PCT_SPIN_WIDTH)
    if on_change is not None:
        spin.valueChanged.connect(lambda _: on_change())
    return spin


def _make_level_pct_pair_row(
    min_default: float,
    max_default: float,
    *,
    on_change: Callable[[], None] | None = None,
) -> tuple[QWidget, QDoubleSpinBox, QDoubleSpinBox]:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    min_spin = _make_level_pct_spin(min_default, on_change=on_change)
    max_spin = _make_level_pct_spin(max_default, on_change=on_change)
    layout.addWidget(min_spin)
    dash = QLabel("–")
    dash.setFixedWidth(10)
    dash.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(dash)
    layout.addWidget(max_spin)
    layout.addStretch()
    return row, min_spin, max_spin


def _matrix_from_widgets(
    widgets: list[list[QDoubleSpinBox]],
) -> list[list[float]]:
    return [[spin.value() for spin in row] for row in widgets]


def _set_matrix_widgets(
    widgets: list[list[QDoubleSpinBox]],
    matrix: list[list[float]],
) -> None:
    for row, values in zip(widgets, matrix):
        for spin, value in zip(row, values):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)


def _make_matrix_grid(
    matrix: list[list[float]],
    *,
    on_change: Callable[[], None] | None = None,
) -> tuple[QWidget, list[list[QDoubleSpinBox]]]:
    """Build a labelled 3×3 matrix editor."""
    panel = QWidget()
    grid = QGridLayout(panel)
    grid.setContentsMargins(0, 0, 0, 0)

    for col, label in enumerate(_MATRIX_INPUT_LABELS, start=1):
        header = QLabel(label)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(header, 0, col)

    widgets: list[list[QDoubleSpinBox]] = []
    for row_idx, row_label in enumerate(_MATRIX_OUTPUT_LABELS):
        grid.addWidget(QLabel(row_label), row_idx + 1, 0)
        row_widgets: list[QDoubleSpinBox] = []
        for col_idx in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-10.0, 10.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.01)
            spin.setMaximumWidth(72)
            spin.setValue(float(matrix[row_idx][col_idx]))
            if on_change is not None:
                spin.valueChanged.connect(lambda _: on_change())
            grid.addWidget(spin, row_idx + 1, col_idx + 1)
            row_widgets.append(spin)
        widgets.append(row_widgets)

    return panel, widgets


class _FilterDialog(QWidget):
    """Filter parameter panel hosted in the main-window dock (not a separate window)."""

    params_changed = pyqtSignal()
    preview_now = pyqtSignal()
    preview_toggled = pyqtSignal(bool)
    accepted = pyqtSignal()
    rejected = pyqtSignal()

    filter_id: str = ""
    supports_presets: bool = False
    supports_clamp: bool = False

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._input_info: BrightnessInfo | None = None

        root = QVBoxLayout(self)

        self._histogram_source: np.ndarray | None = None
        self._histogram_is_grayscale = False
        self._histogram: RgbHistogramWidget | None = None
        if self.filter_id in COLOUR_FILTER_IDS:
            self._histogram = RgbHistogramWidget()
            root.addWidget(self._histogram)

        self._input_label = QLabel("Input —")
        self._input_label.setStyleSheet("font-weight: bold;")
        self._output_label = QLabel("Output —")
        self._output_label.setStyleSheet("font-weight: bold;")
        self._increase_label: QLabel | None = None
        self._grain_label: QLabel | None = None
        if self.filter_id in ENHANCE_FILTER_IDS:
            self._increase_label = QLabel("Brightness increase —")
            self._increase_label.setStyleSheet("font-weight: bold;")
            self._increase_label.setToolTip(
                "Peak channel increase from the filter result before "
                "any clipping is applied."
            )
            self._grain_label = QLabel("Grain —")
            self._grain_label.setStyleSheet("font-weight: bold;")
            self._grain_label.setToolTip(
                "Fine-scale residual energy in low-contrast regions of the "
                "subject (black sky/background excluded), divided by peak "
                "luminance so contrast stretch does not inflate the score. "
                "Measured on the filter result before any clipping. "
                "Higher values mean more grain/noise. Scale is arbitrary "
                "and can be tuned (GRAIN_DISPLAY_SCALE)."
            )
        root.addWidget(self._input_label)
        root.addWidget(self._output_label)
        if self._increase_label is not None:
            root.addWidget(self._increase_label)
        if self._grain_label is not None:
            root.addWidget(self._grain_label)

        if self.supports_presets:
            root.addWidget(self._make_preset_row())

        self._form = QFormLayout()
        root.addLayout(self._form)

        self._build_filter_params()

        self.clamp_channels: QCheckBox | None = None
        self.clamp_low: QCheckBox | None = None
        if self.supports_clamp:
            self.clamp_channels = QCheckBox("Clamp to 100% when clipping")
            self.clamp_channels.setChecked(True)
            self.clamp_channels.setToolTip(
                "Scale all channel levels proportionally so the brightest "
                "channel value becomes 100% when any value exceeds 100%."
            )
            self.clamp_channels.toggled.connect(self._on_clamp_high_toggled)
            self.clamp_channels.toggled.connect(lambda _: self.params_changed.emit())
            self._form.addRow(self.clamp_channels)

            self.clamp_low = QCheckBox("Clamp to 0% when clipping")
            self.clamp_low.setChecked(False)
            self.clamp_low.setEnabled(False)
            self.clamp_low.setToolTip(
                "When clamping highlights to 100%, scale the darkest channel "
                "value to 0% instead of preserving the black point. While "
                "enabled, automatic flooring of negative values is deferred "
                "until highlight clamping runs."
            )
            self.clamp_low.toggled.connect(lambda _: self.params_changed.emit())
            self._form.addRow(self.clamp_low)

        self.preview = QCheckBox("Preview on canvas")
        self.preview.setChecked(True)
        self.preview.toggled.connect(self.preview_toggled.emit)
        self.preview.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self._reject)
        root.addWidget(buttons)

        if self.filter_id and self.supports_presets:
            fdef = FILTERS[self.filter_id]
            self._presets = ensure_builtin_presets(self.filter_id, fdef.default_params)
            self._populate_preset_combo()
            self._set_combo_to("Last")
            self.set_params(self._presets["Last"])

        self.setFixedWidth(FILTER_PANEL_WIDTH)

    def _make_preset_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.currentTextChanged.connect(self._on_preset_selected)
        layout.addWidget(self._preset_combo, stretch=1)
        save_btn = QPushButton("Save…")
        save_btn.clicked.connect(self._save_preset)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_preset)
        layout.addWidget(save_btn)
        layout.addWidget(del_btn)
        return row

    def _populate_preset_combo(self) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        for name in sorted(self._presets.keys()):
            self._preset_combo.addItem(name)
        self._preset_combo.blockSignals(False)

    def _set_combo_to(self, name: str) -> None:
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _on_preset_selected(self, name: str) -> None:
        if name and name in self._presets:
            self.set_params(self._presets[name])
            self.params_changed.emit()

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset As", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in reserved_preset_names(self.filter_id):
            QMessageBox.warning(self, "Save Preset", f'"{name}" is a reserved preset name.')
            return
        is_new = name not in self._presets
        self._presets[name] = self.get_params()
        save_presets(self.filter_id, self._presets)
        if is_new:
            self._preset_combo.addItem(name)
        self._set_combo_to(name)

    def _delete_preset(self) -> None:
        name = self._preset_combo.currentText()
        if name in reserved_preset_names(self.filter_id):
            QMessageBox.warning(self, "Delete Preset", f'Cannot delete the "{name}" preset.')
            return
        if name not in self._presets:
            return
        del self._presets[name]
        save_presets(self.filter_id, self._presets)
        self._preset_combo.removeItem(self._preset_combo.currentIndex())
        self._set_combo_to("Last")

    def save_last_preset(self) -> None:
        if not self.filter_id or not self.supports_presets:
            return
        self._presets["Last"] = self.get_params()
        save_presets(self.filter_id, self._presets)

    def set_input_brightness(self, data: np.ndarray, is_grayscale: bool) -> None:
        self._input_info = measure_brightness(data, is_grayscale)
        self._input_data = np.asarray(data, dtype=np.float32)
        self._input_is_grayscale = is_grayscale
        self._input_label.setText(self._input_info.format_line("Input — "))
        self._histogram_source = self._input_data
        self._histogram_is_grayscale = is_grayscale
        self.update_histogram_display(data)

    def update_histogram_display(self, data: np.ndarray | None) -> None:
        if self._histogram is None:
            return
        if data is None:
            if self._histogram_source is not None:
                self._histogram.set_data(
                    self._histogram_source,
                    self._histogram_is_grayscale,
                )
            return
        self._histogram.set_data(data, self._histogram_is_grayscale)

    def update_output_brightness(
        self,
        info: BrightnessInfo | None,
        increase_pct: float | None = None,
        grain_level: float | None = None,
    ) -> None:
        if info is None:
            self._output_label.setText("Output — (preview off)")
            if self._increase_label is not None:
                self._increase_label.setText("Brightness increase —")
            if self._grain_label is not None:
                self._grain_label.setText("Grain —")
            return
        self._output_label.setText(info.format_line("Output — "))
        if self._increase_label is not None:
            if increase_pct is None:
                self._increase_label.setText("Brightness increase —")
            else:
                sign = "+" if increase_pct >= 0 else ""
                self._increase_label.setText(
                    f"Brightness increase — {sign}{increase_pct:.1f}%  (before clipping)"
                )
        if self._grain_label is not None:
            if grain_level is None:
                self._grain_label.setText("Grain —")
            else:
                self._grain_label.setText(
                    f"Grain — {grain_level:.2f}  (before clipping)"
                )

    def _add_double(
        self,
        label: str,
        default: float,
        minimum: float,
        maximum: float,
        step: float = 0.1,
        decimals: int = 1,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.valueChanged.connect(lambda _: self.params_changed.emit())
        self._form.addRow(label, spin)
        return spin

    def _on_clamp_high_toggled(self, enabled: bool) -> None:
        if self.clamp_low is not None:
            self.clamp_low.setEnabled(enabled)
            if not enabled:
                self.clamp_low.setChecked(False)

    def get_params(self) -> dict[str, Any]:
        p: dict[str, Any] = {}
        if self.clamp_channels is not None:
            p[CLAMP_PARAM] = self.clamp_channels.isChecked()
        if self.clamp_low is not None:
            p[CLAMP_LOW_PARAM] = (
                self.clamp_low.isChecked() and self.clamp_channels is not None
                and self.clamp_channels.isChecked()
            )
        return p

    def _build_filter_params(self) -> None:
        """Subclasses add filter-specific controls here."""

    def set_params(self, params: dict[str, Any]) -> None:
        clamp_val = params.get(CLAMP_PARAM, params.get("rescale", False))
        if self.clamp_channels is not None:
            self.clamp_channels.blockSignals(True)
            self.clamp_channels.setChecked(clamp_val)
            self.clamp_channels.blockSignals(False)
        if self.clamp_low is not None:
            low_val = params.get(CLAMP_LOW_PARAM, False)
            self.clamp_low.blockSignals(True)
            self.clamp_low.setEnabled(bool(clamp_val))
            self.clamp_low.setChecked(low_val and clamp_val)
            self.clamp_low.blockSignals(False)

    def build_filter_func(self) -> FilterFunc:
        if not self.filter_id:
            raise NotImplementedError
        params = self.get_params()

        def func(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
            return apply_filter(self.filter_id, data, is_grayscale, params)

        return func

    def _accept(self) -> None:
        self.accepted.emit()

    def _reject(self) -> None:
        self.rejected.emit()


class WaveletSharpenDialog(_FilterDialog):
    filter_id = "wavelet_sharpen"
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent: QWidget | None = None) -> None:
        self._auto_running = False
        super().__init__("Wavelet Sharpen", parent)

    def _build_filter_params(self) -> None:
        from planetary_tools.filters.wavelet_auto import auto_wavelet_sharpen_params

        self._auto_wavelet_sharpen_params = auto_wavelet_sharpen_params

        fdef = FILTERS[self.filter_id]
        self.fine = self._add_double("Fine", fdef.default_params["fine"], 0.0, 300.0)
        self.medium = self._add_double("Medium", fdef.default_params["medium"], 0.0, 300.0)
        self.coarse = self._add_double("Coarse", fdef.default_params["coarse"], 0.0, 300.0)

        auto_row = QWidget()
        auto_layout = QHBoxLayout(auto_row)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        self.auto = QCheckBox("Auto")
        self.auto.setToolTip(
            "Enable target grain/contrast controls. Click Calculate to search "
            "fine/medium/coarse without exceeding either target."
        )
        self.auto.setChecked(False)
        self.auto.toggled.connect(self._on_auto_toggled)
        auto_layout.addWidget(self.auto)
        self.auto_apply = QPushButton("Calculate")
        self.auto_apply.setToolTip(
            "Run the auto search with the current grain and contrast targets."
        )
        self.auto_apply.clicked.connect(self._run_auto_search)
        auto_layout.addWidget(self.auto_apply)
        auto_layout.addStretch(1)
        self._form.addRow(auto_row)

        self.target_grain = QDoubleSpinBox()
        self.target_grain.setRange(0.0, 10.0)
        self.target_grain.setDecimals(1)
        self.target_grain.setSingleStep(0.1)
        self.target_grain.setValue(float(fdef.default_params.get("target_grain", 3.0)))
        self.target_grain.setToolTip("Maximum peak-normalized grain score to allow.")
        self._form.addRow("Target grain", self.target_grain)

        self.target_contrast = QDoubleSpinBox()
        self.target_contrast.setRange(0.0, 100.0)
        self.target_contrast.setDecimals(0)
        self.target_contrast.setSingleStep(1.0)
        self.target_contrast.setValue(
            float(fdef.default_params.get("target_contrast", 15.0))
        )
        self.target_contrast.setToolTip(
            "Target brightness increase (percent), matched without going over."
        )
        self._form.addRow("Target contrast", self.target_contrast)

        self._sync_auto_enabled_state()

    def _sync_auto_enabled_state(self) -> None:
        auto_on = self.auto.isChecked()
        running = self._auto_running
        self.fine.setEnabled(not auto_on and not running)
        self.medium.setEnabled(not auto_on and not running)
        self.coarse.setEnabled(not auto_on and not running)
        self.target_grain.setEnabled(auto_on and not running)
        self.target_contrast.setEnabled(auto_on and not running)
        self.auto.setEnabled(not running)
        self.auto_apply.setEnabled(auto_on and not running)

    def _on_auto_toggled(self, checked: bool) -> None:
        self._sync_auto_enabled_state()
        if checked:
            self.fine.blockSignals(True)
            self.medium.blockSignals(True)
            self.coarse.blockSignals(True)
            self.fine.setValue(0.0)
            self.medium.setValue(0.0)
            self.coarse.setValue(0.0)
            self.fine.blockSignals(False)
            self.medium.blockSignals(False)
            self.coarse.blockSignals(False)
            self.params_changed.emit()
        else:
            self.params_changed.emit()

    def _run_auto_search(self) -> None:
        if self._auto_running or not self.auto.isChecked():
            return
        data = getattr(self, "_input_data", None)
        if data is None:
            return
        is_grayscale = bool(getattr(self, "_input_is_grayscale", False))

        self._auto_running = True
        self._sync_auto_enabled_state()
        try:
            from PyQt6.QtWidgets import QApplication

            def progress(
                fine: float,
                medium: float,
                coarse: float,
                _grain: float,
                _contrast: float,
            ) -> None:
                self.fine.blockSignals(True)
                self.medium.blockSignals(True)
                self.coarse.blockSignals(True)
                self.fine.setValue(fine)
                self.medium.setValue(medium)
                self.coarse.setValue(coarse)
                self.fine.blockSignals(False)
                self.medium.blockSignals(False)
                self.coarse.blockSignals(False)
                QApplication.processEvents()

            result = self._auto_wavelet_sharpen_params(
                data,
                is_grayscale,
                target_grain=self.target_grain.value(),
                target_contrast=self.target_contrast.value(),
                progress=progress,
            )
            self.fine.blockSignals(True)
            self.medium.blockSignals(True)
            self.coarse.blockSignals(True)
            self.fine.setValue(result.fine)
            self.medium.setValue(result.medium)
            self.coarse.setValue(result.coarse)
            self.fine.blockSignals(False)
            self.medium.blockSignals(False)
            self.coarse.blockSignals(False)
        finally:
            self._auto_running = False
            self._sync_auto_enabled_state()
        self.params_changed.emit()

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "fine": self.fine.value(),
            "medium": self.medium.value(),
            "coarse": self.coarse.value(),
            "auto": self.auto.isChecked(),
            "target_grain": self.target_grain.value(),
            "target_contrast": self.target_contrast.value(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.fine.blockSignals(True)
        self.medium.blockSignals(True)
        self.coarse.blockSignals(True)
        self.auto.blockSignals(True)
        self.target_grain.blockSignals(True)
        self.target_contrast.blockSignals(True)
        self.fine.setValue(params.get("fine", self.fine.value()))
        self.medium.setValue(params.get("medium", self.medium.value()))
        self.coarse.setValue(params.get("coarse", self.coarse.value()))
        # Auto is a session control: always open off so targets can be set
        # before Apply (do not restore from Last/presets).
        self.auto.setChecked(False)
        self.target_grain.setValue(
            float(params.get("target_grain", self.target_grain.value()))
        )
        self.target_contrast.setValue(
            float(params.get("target_contrast", self.target_contrast.value()))
        )
        self.fine.blockSignals(False)
        self.medium.blockSignals(False)
        self.coarse.blockSignals(False)
        self.auto.blockSignals(False)
        self.target_grain.blockSignals(False)
        self.target_contrast.blockSignals(False)
        self._sync_auto_enabled_state()


class WaveletDenoiseDialog(_FilterDialog):
    filter_id = "wavelet_denoise"
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Wavelet Denoise", parent)

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.fine = self._add_double("Fine", fdef.default_params["fine"], 0.0, 20.0)
        self.medium = self._add_double("Medium", fdef.default_params["medium"], 0.0, 20.0)
        self.coarse = self._add_double("Coarse", fdef.default_params["coarse"], 0.0, 20.0)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "fine": self.fine.value(),
            "medium": self.medium.value(),
            "coarse": self.coarse.value(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.fine.blockSignals(True)
        self.medium.blockSignals(True)
        self.coarse.blockSignals(True)
        self.fine.setValue(params.get("fine", self.fine.value()))
        self.medium.setValue(params.get("medium", self.medium.value()))
        self.coarse.setValue(params.get("coarse", self.coarse.value()))
        self.fine.blockSignals(False)
        self.medium.blockSignals(False)
        self.coarse.blockSignals(False)


class AdaptiveDeconvDialog(_FilterDialog):
    filter_id = "adaptive_deconv"
    supports_presets = True
    supports_clamp = True

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None) -> None:
        self._is_grayscale = is_grayscale
        super().__init__("Adaptive Deconvolution", parent)

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.amount = self._add_double(
            "Amount", fdef.default_params["amount"], 0.0, 200.0, step=0.1, decimals=1
        )
        self.adaptive = QCheckBox("Contrast Adaptive")
        self.adaptive.setToolTip(
            "Increases sharpening in areas of higher contrast."
        )
        self.adaptive.setChecked(fdef.default_params["adaptive"])
        self.adaptive.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.adaptive)
        self.oklab = QCheckBox("OKLab luminance")
        self.oklab.setToolTip(
            "Sharpens on luminance layer decreasing colour noise but lowers saturation."
        )
        self.oklab.setChecked(fdef.default_params["oklab"] and not self._is_grayscale)
        self.oklab.setEnabled(not self._is_grayscale)
        self.oklab.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.oklab)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "amount": self.amount.value(),
            "adaptive": self.adaptive.isChecked(),
            "oklab": self.oklab.isChecked(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.amount.blockSignals(True)
        self.adaptive.blockSignals(True)
        self.oklab.blockSignals(True)
        self.amount.setValue(params.get("amount", self.amount.value()))
        self.adaptive.setChecked(params.get("adaptive", self.adaptive.isChecked()))
        self.oklab.setChecked(params.get("oklab", self.oklab.isChecked()) and not self._is_grayscale)
        self.amount.blockSignals(False)
        self.adaptive.blockSignals(False)
        self.oklab.blockSignals(False)


class WienerDeconvDialog(_FilterDialog):
    filter_id = "wiener_deconv"
    supports_presets = True
    supports_clamp = True

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None) -> None:
        self._is_grayscale = is_grayscale
        super().__init__("Wiener Deconvolution", parent)

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.amount = self._add_double(
            "Amount", fdef.default_params["amount"], 0.0, 200.0, step=0.1, decimals=1
        )
        self.adaptive = QCheckBox("Contrast Adaptive")
        self.adaptive.setToolTip(
            "Reduces the filter effect in areas of higher contrast "
            "(opposite of adaptive deconvolution)."
        )
        self.adaptive.setChecked(fdef.default_params["adaptive"])
        self.adaptive.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.adaptive)
        self.oklab = QCheckBox("OKLab luminance")
        self.oklab.setToolTip(
            "Filters on the luminance layer, reducing colour noise but "
            "leaving chroma detail unchanged."
        )
        self.oklab.setChecked(fdef.default_params["oklab"] and not self._is_grayscale)
        self.oklab.setEnabled(not self._is_grayscale)
        self.oklab.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.oklab)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "amount": self.amount.value(),
            "adaptive": self.adaptive.isChecked(),
            "oklab": self.oklab.isChecked(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.amount.blockSignals(True)
        self.adaptive.blockSignals(True)
        self.oklab.blockSignals(True)
        self.amount.setValue(params.get("amount", self.amount.value()))
        self.adaptive.setChecked(params.get("adaptive", self.adaptive.isChecked()))
        self.oklab.setChecked(
            params.get("oklab", self.oklab.isChecked()) and not self._is_grayscale
        )
        self.amount.blockSignals(False)
        self.adaptive.blockSignals(False)
        self.oklab.blockSignals(False)


class StretchContrastDialog(_FilterDialog):
    """Stretch Contrast OKLab — preview only, no presets or clamp."""

    filter_id = "stretch_contrast"
    supports_presets = False
    supports_clamp = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Stretch Contrast OKLab", parent)

    def _build_filter_params(self) -> None:
        self._form.addRow(QLabel(
            "Stretches OKLab luminance to the full range using proportional RGB scaling."
        ))


class SaturationVibranceDialog(_FilterDialog):
    """OKLab saturation and vibrance with preset save/load."""

    filter_id = "saturation_vibrance"
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Saturation & Vibrance", parent)

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.saturation = self._add_double(
            "Saturation",
            fdef.default_params["saturation"],
            0.0,
            10.0,
            step=0.05,
            decimals=2,
        )
        self.saturation.setToolTip("1.00 = 100% saturation (no change).")
        self.vibrance = self._add_double(
            "Vibrance",
            fdef.default_params["vibrance"],
            0.0,
            10.0,
            step=0.05,
            decimals=2,
        )
        self.vibrance.setToolTip(
            "1.00 = 100% vibrance (no change). Boosts low-chroma areas more than "
            "already-saturated colours."
        )
        self._form.addRow(QLabel(
            "Adjusts chroma in OKLab. 1.00 on each control leaves the image unchanged."
        ))

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset saturation and vibrance to 1.00 (no change).")
        reset_btn.clicked.connect(self._reset_defaults)
        self._form.addRow(reset_btn)

    def _reset_defaults(self) -> None:
        self.set_params(FILTERS[self.filter_id].default_params)
        self.params_changed.emit()
        self.preview_now.emit()

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "saturation": self.saturation.value(),
            "vibrance": self.vibrance.value(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.saturation.blockSignals(True)
        self.vibrance.blockSignals(True)
        self.saturation.setValue(params.get("saturation", self.saturation.value()))
        self.vibrance.setValue(params.get("vibrance", self.vibrance.value()))
        self.saturation.blockSignals(False)
        self.vibrance.blockSignals(False)


class LevelsDialog(_FilterDialog):
    """Per-channel input/output levels with preset save/load."""

    filter_id = "levels"
    supports_presets = True
    supports_clamp = False

    def __init__(self, parent: QWidget | None = None) -> None:
        self._channel_params = default_levels_params()
        self._loading_channel = False
        self._editing_channel = "L"
        self._input_data: np.ndarray | None = None
        self._is_grayscale = False
        super().__init__("Levels", parent)
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint,
        )

    def _build_filter_params(self) -> None:
        self._channel_combo = QComboBox()
        for ch in LEVEL_CHANNELS:
            self._channel_combo.addItem(_LEVEL_CHANNEL_LABELS[ch], ch)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._form.addRow("Channel", self._channel_combo)

        in_row, self._in_min, self._in_max = _make_level_pct_pair_row(
            0.0, 100.0, on_change=self._on_level_spin_changed,
        )
        self._in_min.setToolTip("Input minimum")
        self._in_max.setToolTip("Input maximum")
        self._form.addRow("Input", in_row)

        self._gamma = QDoubleSpinBox()
        self._gamma.setRange(0.10, 10.0)
        self._gamma.setDecimals(2)
        self._gamma.setSingleStep(0.05)
        self._gamma.setValue(1.0)
        self._gamma.setMaximumWidth(_LEVEL_PCT_SPIN_WIDTH)
        self._gamma.valueChanged.connect(lambda _: self._on_level_spin_changed())
        self._form.addRow("Gamma", self._gamma)
        self._gamma.setToolTip("1.00 = no midtone change. Values above 1 brighten midtones.")

        out_row, self._out_min, self._out_max = _make_level_pct_pair_row(
            0.0, 100.0, on_change=self._on_level_spin_changed,
        )
        self._out_min.setToolTip("Output minimum")
        self._out_max.setToolTip("Output maximum")
        self._form.addRow("Output", out_row)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        auto_balance = QPushButton("Auto Balance")
        auto_balance.setToolTip(
            "Set RGB input levels from 2% / 98% histogram percentiles "
            "(GIMP Auto Input Levels), then set RGB output maximum to the "
            "lowest input maximum. Luminance is not changed."
        )
        auto_balance.clicked.connect(self._auto_balance)
        reset_ch = QPushButton("Reset")
        reset_ch.setToolTip("Reset the selected channel")
        reset_ch.clicked.connect(self._reset_channel)
        reset_all = QPushButton("Reset all")
        reset_all.clicked.connect(self._reset_all)
        layout.addWidget(auto_balance)
        layout.addWidget(reset_ch)
        layout.addWidget(reset_all)
        self._form.addRow(row)

        note = QLabel(
            "Luminance uses OKLab. RGB matches GIMP perceptual gamma "
            "(sRGB-encoded); output is applied before input."
        )
        note.setWordWrap(True)
        self._form.addRow(note)
        self._load_channel_into_spins(self._current_channel())

    def _current_channel(self) -> str:
        data = self._channel_combo.currentData()
        return str(data) if data is not None else "L"

    def _on_channel_changed(self) -> None:
        if self._loading_channel:
            return
        new_ch = self._current_channel()
        if new_ch != self._editing_channel:
            self._store_spins_in_channel(self._editing_channel)
            self._editing_channel = new_ch
        self._load_channel_into_spins(new_ch)
        self.params_changed.emit()

    def _on_level_spin_changed(self) -> None:
        if self._loading_channel:
            return
        self._store_spins_in_channel(self._editing_channel)
        self.params_changed.emit()

    def _store_spins_in_channel(self, channel: str) -> None:
        self._channel_params[channel] = {
            "in_min": self._in_min.value() / 100.0,
            "in_max": self._in_max.value() / 100.0,
            "gamma": self._gamma.value(),
            "out_min": self._out_min.value() / 100.0,
            "out_max": self._out_max.value() / 100.0,
        }

    def _level_spins(self) -> tuple[QDoubleSpinBox, ...]:
        return (self._in_min, self._in_max, self._gamma, self._out_min, self._out_max)

    def _load_channel_into_spins(self, channel: str) -> None:
        levels = self._channel_params[channel]
        self._loading_channel = True
        for spin in self._level_spins():
            spin.blockSignals(True)
        self._in_min.setValue(levels["in_min"] * 100.0)
        self._in_max.setValue(levels["in_max"] * 100.0)
        self._gamma.setValue(levels["gamma"])
        self._out_min.setValue(levels["out_min"] * 100.0)
        self._out_max.setValue(levels["out_max"] * 100.0)
        for spin in self._level_spins():
            spin.blockSignals(False)
        self._loading_channel = False

    def _notify_params_changed(self, *, immediate: bool = False) -> None:
        self._store_spins_in_channel(self._editing_channel)
        self.params_changed.emit()
        if immediate:
            self.preview_now.emit()

    def _reset_channel(self) -> None:
        ch = self._editing_channel
        self._channel_params[ch] = identity_levels()
        self._load_channel_into_spins(ch)
        self._notify_params_changed(immediate=True)

    def _reset_all(self) -> None:
        self._channel_params = default_levels_params()
        self._load_channel_into_spins(self._editing_channel)
        self._notify_params_changed(immediate=True)

    def set_input_brightness(self, data: np.ndarray, is_grayscale: bool) -> None:
        super().set_input_brightness(data, is_grayscale)
        self._input_data = np.asarray(data, dtype=np.float32)
        self._is_grayscale = is_grayscale

    def _auto_balance(self) -> None:
        if self._input_data is None:
            return
        self._store_spins_in_channel(self._editing_channel)
        self._channel_params = auto_balance_levels(
            self._input_data,
            is_grayscale=self._is_grayscale,
        )
        self._load_channel_into_spins(self._editing_channel)
        self._notify_params_changed(immediate=True)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        self._store_spins_in_channel(self._editing_channel)
        p["channels"] = {
            ch: dict(self._channel_params[ch]) for ch in LEVEL_CHANNELS
        }
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self._channel_params = normalize_levels_params(params)
        self._editing_channel = self._current_channel()
        self._load_channel_into_spins(self._editing_channel)


class MergeWaveletDetailDialog(_FilterDialog):
    """Merge fine wavelet detail from a secondary (NIR) image into the main image."""

    filter_id = "merge_wavelet_detail"
    supports_presets = False
    supports_clamp = True

    def __init__(
        self,
        secondary_path: str,
        secondary_data: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        self._secondary_path = secondary_path
        self._secondary_data = secondary_data
        super().__init__("Merge Wavelet Detail", parent)

    def _build_filter_params(self) -> None:
        name = os.path.basename(self._secondary_path) if self._secondary_path else "(none)"
        lbl = QLabel(name)
        lbl.setWordWrap(True)
        self._form.addRow("Secondary:", lbl)

        self._scales_spin = QSpinBox()
        self._scales_spin.setRange(1, 3)
        self._scales_spin.setValue(3)
        self._scales_spin.setToolTip(
            "Number of finest wavelet scales to take from the secondary image.\n"
            "1 = fine scale only, 2 = fine + medium, 3 = all three scales."
        )
        self._scales_spin.valueChanged.connect(lambda _: self.params_changed.emit())
        self._form.addRow("Scales:", self._scales_spin)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "n_secondary_scales": self._scales_spin.value(),
            "secondary_data": self._secondary_data,
        })
        return p


class ColourMatrixDialog(_FilterDialog):
    """3×3 colour correction matrix with preset save/load."""

    filter_id = "colour_matrix"
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Colour Correction Matrix", parent)

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        matrix = fdef.default_params["matrix"]
        panel, self._matrix_widgets = _make_matrix_grid(
            matrix,
            on_change=lambda: self.params_changed.emit(),
        )
        self._form.addRow(QLabel("Matrix (linear RGB):"), panel)

        reset_btn = QPushButton("Reset to identity")
        reset_btn.clicked.connect(self._reset_identity)
        self._form.addRow(reset_btn)

    def _reset_identity(self) -> None:
        _set_matrix_widgets(self._matrix_widgets, IDENTITY_MATRIX)
        self.params_changed.emit()

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p["matrix"] = _matrix_from_widgets(self._matrix_widgets)
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        matrix = params.get("matrix", IDENTITY_MATRIX)
        _set_matrix_widgets(self._matrix_widgets, matrix)


# class InstantFilterDialog(_FilterDialog):
#     """Dialog for parameterless Colour filters (e.g. OKLab Luminance)."""
#
#     supports_presets = False
#     supports_clamp = False
#
#     def __init__(self, title: str, filter_id: str, parent: QWidget | None = None) -> None:
#         self.filter_id = filter_id
#         super().__init__(title, parent)
#
#     def _build_filter_params(self) -> None:
#         self._form.addRow(QLabel("This filter has no adjustable parameters."))


def edit_filter_params(
    filter_id: str,
    params: dict[str, Any],
    is_grayscale: bool,
    parent: QWidget | None = None,
) -> dict[str, Any] | None:
    """Small modal editor for batch pipeline step parameters."""
    fdef = FILTERS[filter_id]
    dlg = QDialog(parent)
    dlg.setWindowTitle(fdef.label)
    dlg.setWindowFlags(
        Qt.WindowType.Tool
        | Qt.WindowType.WindowTitleHint
        | Qt.WindowType.WindowCloseButtonHint
    )
    dlg.setModal(False)
    dlg.setFixedWidth(FILTER_PANEL_WIDTH)
    layout = QVBoxLayout(dlg)
    form = QFormLayout()
    layout.addLayout(form)
    widgets: dict[str, Any] = {}

    if filter_id == "wavelet_sharpen":
        for key in ("fine", "medium", "coarse"):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 300.0)
            spin.setValue(params.get(key, fdef.default_params[key]))
            form.addRow(key.capitalize(), spin)
            widgets[key] = spin
        auto = QCheckBox("Auto")
        auto.setChecked(bool(params.get("auto", fdef.default_params.get("auto", False))))
        form.addRow(auto)
        widgets["auto"] = auto
        target_grain = QDoubleSpinBox()
        target_grain.setRange(0.0, 10.0)
        target_grain.setDecimals(1)
        target_grain.setSingleStep(0.1)
        target_grain.setValue(
            float(params.get("target_grain", fdef.default_params.get("target_grain", 3.0)))
        )
        form.addRow("Target grain", target_grain)
        widgets["target_grain"] = target_grain
        target_contrast = QDoubleSpinBox()
        target_contrast.setRange(0.0, 100.0)
        target_contrast.setDecimals(0)
        target_contrast.setSingleStep(1.0)
        target_contrast.setValue(
            float(
                params.get(
                    "target_contrast",
                    fdef.default_params.get("target_contrast", 15.0),
                )
            )
        )
        form.addRow("Target contrast", target_contrast)
        widgets["target_contrast"] = target_contrast
    elif filter_id == "wavelet_denoise":
        for key in ("fine", "medium", "coarse"):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 20.0)
            spin.setValue(params.get(key, fdef.default_params[key]))
            form.addRow(key.capitalize(), spin)
            widgets[key] = spin
    elif filter_id == "adaptive_deconv":
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 200.0)
        amount.setDecimals(1)
        amount.setSingleStep(0.1)
        amount.setValue(params.get("amount", fdef.default_params["amount"]))
        form.addRow("Amount", amount)
        widgets["amount"] = amount
        adaptive = QCheckBox("Contrast Adaptive")
        adaptive.setToolTip(
            "Increases sharpening in areas of higher contrast."
        )
        adaptive.setChecked(params.get("adaptive", fdef.default_params["adaptive"]))
        form.addRow(adaptive)
        widgets["adaptive"] = adaptive
        oklab = QCheckBox("OKLab luminance")
        oklab.setToolTip(
            "Sharpens on luminance layer decreasing colour noise but lowers saturation."
        )
        oklab.setChecked(params.get("oklab", fdef.default_params["oklab"]) and not is_grayscale)
        oklab.setEnabled(not is_grayscale)
        form.addRow(oklab)
        widgets["oklab"] = oklab
    elif filter_id == "wiener_deconv":
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 200.0)
        amount.setDecimals(1)
        amount.setSingleStep(0.1)
        amount.setValue(params.get("amount", fdef.default_params["amount"]))
        form.addRow("Amount", amount)
        widgets["amount"] = amount
        adaptive = QCheckBox("Contrast Adaptive")
        adaptive.setToolTip(
            "Reduces the filter effect in areas of higher contrast "
            "(opposite of adaptive deconvolution)."
        )
        adaptive.setChecked(params.get("adaptive", fdef.default_params["adaptive"]))
        form.addRow(adaptive)
        widgets["adaptive"] = adaptive
        oklab = QCheckBox("OKLab luminance")
        oklab.setToolTip(
            "Filters on the luminance layer, reducing colour noise but "
            "leaving chroma detail unchanged."
        )
        oklab.setChecked(params.get("oklab", fdef.default_params["oklab"]) and not is_grayscale)
        oklab.setEnabled(not is_grayscale)
        form.addRow(oklab)
        widgets["oklab"] = oklab
    elif filter_id == "stretch_contrast":
        layout.addWidget(QLabel("No parameters — stretch is applied automatically."))
    elif filter_id == "colour_matrix":
        matrix = params.get("matrix", fdef.default_params["matrix"])
        panel, matrix_widgets = _make_matrix_grid(matrix)
        form.addRow(QLabel("Matrix (linear RGB):"), panel)
        widgets["matrix"] = matrix_widgets
    elif filter_id == "saturation_vibrance":
        for key, label in (("saturation", "Saturation"), ("vibrance", "Vibrance")):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(params.get(key, fdef.default_params[key]))
            form.addRow(label, spin)
            widgets[key] = spin
    elif filter_id == "levels":
        channel_data = normalize_levels_params(params)
        channel_combo = QComboBox()
        for ch in LEVEL_CHANNELS:
            channel_combo.addItem(_LEVEL_CHANNEL_LABELS[ch], ch)
        form.addRow("Channel", channel_combo)

        level_spins: dict[str, QDoubleSpinBox] = {}

        in_row, level_spins["in_min"], level_spins["in_max"] = _make_level_pct_pair_row(
            0.0, 100.0,
        )
        form.addRow("Input", in_row)

        gamma_spin = QDoubleSpinBox()
        gamma_spin.setRange(0.10, 10.0)
        gamma_spin.setDecimals(2)
        form.addRow("Gamma", gamma_spin)
        level_spins["gamma"] = gamma_spin

        out_row, level_spins["out_min"], level_spins["out_max"] = _make_level_pct_pair_row(
            0.0, 100.0,
        )
        form.addRow("Output", out_row)

        def _load_batch_channel(ch: str) -> None:
            lv = channel_data[ch]
            level_spins["in_min"].setValue(lv["in_min"] * 100.0)
            level_spins["in_max"].setValue(lv["in_max"] * 100.0)
            level_spins["gamma"].setValue(lv["gamma"])
            level_spins["out_min"].setValue(lv["out_min"] * 100.0)
            level_spins["out_max"].setValue(lv["out_max"] * 100.0)

        def _store_batch_channel(ch: str) -> None:
            channel_data[ch] = {
                "in_min": level_spins["in_min"].value() / 100.0,
                "in_max": level_spins["in_max"].value() / 100.0,
                "gamma": level_spins["gamma"].value(),
                "out_min": level_spins["out_min"].value() / 100.0,
                "out_max": level_spins["out_max"].value() / 100.0,
            }

        def _on_batch_channel_changed() -> None:
            prev = channel_combo.property("_prev_channel")
            if prev:
                _store_batch_channel(str(prev))
            ch = str(channel_combo.currentData())
            channel_combo.setProperty("_prev_channel", ch)
            _load_batch_channel(ch)

        channel_combo.currentIndexChanged.connect(_on_batch_channel_changed)
        channel_combo.setProperty("_prev_channel", "L")
        _load_batch_channel("L")
        widgets["_levels_channels"] = channel_data
        widgets["_levels_channel_combo"] = channel_combo
        widgets["_levels_store"] = _store_batch_channel
    else:
        layout.addWidget(QLabel("No numeric parameters for this filter."))

    if filter_id in CLAMP_FILTER_IDS:
        clamp_box = QCheckBox("Clamp to 100% when clipping")
        clamp_val = params.get(
            CLAMP_PARAM,
            params.get("rescale", fdef.default_params.get(CLAMP_PARAM, False)),
        )
        clamp_box.setChecked(clamp_val)
        form.addRow(clamp_box)
        widgets[CLAMP_PARAM] = clamp_box

        clamp_low_box = QCheckBox("Clamp to 0% when clipping")
        clamp_low_box.setChecked(params.get(
            CLAMP_LOW_PARAM,
            fdef.default_params.get(CLAMP_LOW_PARAM, False),
        ) and clamp_val)
        clamp_low_box.setEnabled(clamp_val)
        form.addRow(clamp_low_box)
        widgets[CLAMP_LOW_PARAM] = clamp_low_box

        def _sync_clamp_low(checked: bool) -> None:
            clamp_low_box.setEnabled(checked)
            if not checked:
                clamp_low_box.setChecked(False)

        clamp_box.toggled.connect(_sync_clamp_low)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != dlg.DialogCode.Accepted:
        return None

    out = dict(params)
    if "_levels_channels" in widgets:
        store = widgets["_levels_store"]
        combo = widgets["_levels_channel_combo"]
        store(str(combo.currentData()))
        out["channels"] = {
            ch: dict(widgets["_levels_channels"][ch]) for ch in LEVEL_CHANNELS
        }
    for key, widget in widgets.items():
        if key.startswith("_levels"):
            continue
        if key == "matrix":
            out[key] = _matrix_from_widgets(widget)
        elif isinstance(widget, QDoubleSpinBox):
            out[key] = widget.value()
        elif isinstance(widget, QCheckBox):
            out[key] = widget.isChecked()
    if CLAMP_LOW_PARAM in out and not out.get(CLAMP_PARAM, False):
        out[CLAMP_LOW_PARAM] = False
    return out