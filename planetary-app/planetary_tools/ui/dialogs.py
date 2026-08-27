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
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QStyle,
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
    channel_input_peak,
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

FILTER_PANEL_WIDTH = 330


def _fit_spin_width(spin: QDoubleSpinBox | QSpinBox) -> None:
    """Fix a spin box to its natural width for the current font and DPI.

    Hard-coded pixel widths clip the displayed value on high-DPI or
    large-font systems (e.g. some Windows laptops, where only the first
    digit is visible). The style's own size hint sizes the box to fit the
    digits, suffix, frame, and spin buttons at the active display scale.
    Call this after the range, decimals, and suffix are set so the hint
    reflects the widest value.
    """
    spin.setFixedWidth(spin.sizeHint().width())

# Preset combo entry shown once any parameter is edited by hand; selecting a
# real preset (including the previous one) re-applies it.
_PRESET_NONE = "(None)"


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
    _fit_spin_width(spin)
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
    dash.setFixedWidth(dash.fontMetrics().horizontalAdvance("–") + 6)
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
            spin.setValue(float(matrix[row_idx][col_idx]))
            _fit_spin_width(spin)
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
        # True while params_changed is emitted by something that is not a
        # hand edit (applying a preset, toggling preview), so the preset
        # combo is not switched to (None).
        self._suppress_preset_dirty = False

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
        self._noise_label: QLabel | None = None
        if self.filter_id in ENHANCE_FILTER_IDS:
            self._increase_label = QLabel("Brightness increase —")
            self._increase_label.setStyleSheet("font-weight: bold;")
            self._increase_label.setToolTip(
                "Peak channel increase from the filter result before "
                "any clipping is applied."
            )
            self._noise_label = QLabel("Noise —")
            self._noise_label.setStyleSheet("font-weight: bold;")
            self._noise_label.setToolTip(
                "Hybrid noise score on subject flats (sky excluded), peak-"
                "normalized. Residual filter sizes track an estimated texture/"
                "PSF scale of the *source* image (softer stacks look for "
                "coarser structure). Combines fine MAD, heavy-tail (p99) "
                "excess, and band-pass MAD. Before clipping. Higher = more "
                "noise/speckle.\n\n"
                "Shows source → result when the filter changes the score. "
                "Source noise is the same in wavelet sharpen, denoise, and "
                "adaptive deconvolution for a given image."
            )
        self._help_icon = QLabel()
        self._help_icon.hide()
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_label)
        input_row.addStretch(1)
        input_row.addWidget(self._help_icon)
        root.addLayout(input_row)
        root.addWidget(self._output_label)
        if self._increase_label is not None:
            root.addWidget(self._increase_label)
        if self._noise_label is not None:
            root.addWidget(self._noise_label)

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
        # Preview visibility is not a preset parameter — don't let its
        # params_changed emission flip the preset combo to (None).
        self.preview.toggled.connect(lambda _: self._emit_params_changed_quietly())
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
            self.params_changed.connect(self._mark_preset_dirty)

        # Minimum, not fixed: with large fonts the content can need more
        # room, and the dock is sized to the dialog's hint when shown.
        self.setMinimumWidth(FILTER_PANEL_WIDTH)

    def set_help_text(self, text: str) -> None:
        """Show a question-mark icon whose hover tooltip holds the help text."""
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion)
        size = self.fontMetrics().height()
        self._help_icon.setPixmap(icon.pixmap(size, size))
        # Rich text makes the tooltip word-wrap instead of one long line.
        self._help_icon.setToolTip(f"<qt>{text}</qt>")
        self._help_icon.show()

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
        self._preset_combo.addItem(_PRESET_NONE)
        for name in sorted(self._presets.keys()):
            self._preset_combo.addItem(name)
        self._preset_combo.blockSignals(False)

    def _emit_params_changed_quietly(self) -> None:
        self._suppress_preset_dirty = True
        try:
            self.params_changed.emit()
        finally:
            self._suppress_preset_dirty = False

    def _mark_preset_dirty(self) -> None:
        """Switch the combo to (None) when params are edited by hand."""
        if self._suppress_preset_dirty:
            return
        if self._preset_combo.currentText() != _PRESET_NONE:
            self._preset_combo.blockSignals(True)
            self._set_combo_to(_PRESET_NONE)
            self._preset_combo.blockSignals(False)

    def _set_combo_to(self, name: str) -> None:
        idx = self._preset_combo.findText(name)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _on_preset_selected(self, name: str) -> None:
        if name and name in self._presets:
            self._suppress_preset_dirty = True
            try:
                self.set_params(self._presets[name])
                self.params_changed.emit()
            finally:
                self._suppress_preset_dirty = False

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset As", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name == _PRESET_NONE or name in reserved_preset_names(self.filter_id):
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

    def set_input_brightness(
        self,
        data: np.ndarray,
        is_grayscale: bool,
        *,
        noise_texture_scale: float | None = None,
        noise_chromatic: bool | None = None,
    ) -> None:
        self._input_info = measure_brightness(data, is_grayscale)
        self._input_data = np.asarray(data, dtype=np.float32)
        self._input_is_grayscale = is_grayscale
        # Session-pinned noise context (from document at load). Survives filter
        # applies so absolute noise stays comparable across enhance dialogs.
        self._noise_texture_scale = noise_texture_scale
        self._noise_chromatic = noise_chromatic
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
        noise_level: float | None = None,
        source_noise_level: float | None = None,
    ) -> None:
        if info is None:
            self._output_label.setText("Output — (preview off)")
            if self._increase_label is not None:
                self._increase_label.setText("Brightness increase —")
            if self._noise_label is not None:
                self._noise_label.setText("Noise —")
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
        if self._noise_label is not None:
            if noise_level is None and source_noise_level is None:
                self._noise_label.setText("Noise —")
            elif (
                source_noise_level is not None
                and noise_level is not None
                and abs(source_noise_level - noise_level) >= 0.005
            ):
                # Source is identical across enhance dialogs; result tracks params.
                self._noise_label.setText(
                    f"Noise — {source_noise_level:.2f} → {noise_level:.2f}"
                    f"  (before clipping)"
                )
            else:
                shown = noise_level if noise_level is not None else source_noise_level
                self._noise_label.setText(
                    f"Noise — {shown:.2f}  (before clipping)"
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

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None) -> None:
        self._is_grayscale = is_grayscale
        self._auto_running = False
        super().__init__("Wavelet Sharpen", parent)

    def _build_filter_params(self) -> None:
        from planetary_tools.filters.wavelet_auto import auto_wavelet_sharpen_params

        self._auto_wavelet_sharpen_params = auto_wavelet_sharpen_params

        fdef = FILTERS[self.filter_id]
        self.fine = self._add_double("Fine", fdef.default_params["fine"], 0.0, 300.0)
        self.medium = self._add_double("Medium", fdef.default_params["medium"], 0.0, 300.0)
        self.coarse = self._add_double("Coarse", fdef.default_params["coarse"], 0.0, 300.0)
        self.chunky = self._add_double("Chunky", fdef.default_params["chunky"], 0.0, 300.0)

        self.luminance = QCheckBox("Luminance (BT.709)")
        self.luminance.setToolTip(
            "Sharpens BT.709 luminance and adds the change back to RGB. "
            "Decreases colour noise but lowers saturation."
        )
        self.luminance.setChecked(
            fdef.default_params.get("luminance", False) and not self._is_grayscale
        )
        self.luminance.setEnabled(not self._is_grayscale)
        self.luminance.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.luminance)

        auto_row = QWidget()
        auto_layout = QHBoxLayout(auto_row)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_apply = QPushButton("Auto")
        self.auto_apply.setToolTip(
            "Search fine/medium/coarse to meet the target noise and contrast "
            "without exceeding either."
        )
        self.auto_apply.clicked.connect(self._run_auto_search)
        auto_layout.addWidget(self.auto_apply)
        auto_layout.addStretch(1)
        self._form.addRow(auto_row)

        self.target_noise = QDoubleSpinBox()
        self.target_noise.setRange(0.0, 20.0)
        self.target_noise.setDecimals(1)
        self.target_noise.setSingleStep(0.1)
        self.target_noise.setValue(float(fdef.default_params.get("target_noise", 3.0)))
        self.target_noise.setToolTip("Maximum peak-normalized noise score to allow.")
        self._form.addRow("Target noise", self.target_noise)

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

    def _set_amount_spins(
        self, fine: float, medium: float, coarse: float, chunky: float
    ) -> None:
        spins = (
            (self.fine, fine),
            (self.medium, medium),
            (self.coarse, coarse),
            (self.chunky, chunky),
        )
        for spin, value in spins:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _run_auto_search(self) -> None:
        if self._auto_running:
            return
        data = getattr(self, "_input_data", None)
        if data is None:
            return
        is_grayscale = bool(getattr(self, "_input_is_grayscale", False))

        prior = (
            self.fine.value(),
            self.medium.value(),
            self.coarse.value(),
            self.chunky.value(),
        )

        progress_dlg = QProgressDialog("Calculating…", "Cancel", 0, 0, self)
        progress_dlg.setWindowTitle("Auto Wavelet Sharpen")
        progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.show()

        class _Cancelled(Exception):
            pass

        from PyQt6.QtWidgets import QApplication

        def progress(
            fine: float,
            medium: float,
            coarse: float,
            chunky: float,
            _noise: float,
            _contrast: float,
        ) -> None:
            self._set_amount_spins(fine, medium, coarse, chunky)
            QApplication.processEvents()
            if progress_dlg.wasCanceled():
                raise _Cancelled

        self._auto_running = True
        self.auto_apply.setEnabled(False)
        try:
            result = self._auto_wavelet_sharpen_params(
                data,
                is_grayscale,
                target_noise=self.target_noise.value(),
                target_contrast=self.target_contrast.value(),
                progress=progress,
                texture_scale=getattr(self, "_noise_texture_scale", None),
                chromatic=getattr(self, "_noise_chromatic", None),
                luminance=self.luminance.isChecked(),
            )
            self._set_amount_spins(
                result.fine, result.medium, result.coarse, result.chunky
            )
        except _Cancelled:
            self._set_amount_spins(*prior)
        finally:
            self._auto_running = False
            self.auto_apply.setEnabled(True)
            progress_dlg.close()
        self.params_changed.emit()

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "fine": self.fine.value(),
            "medium": self.medium.value(),
            "coarse": self.coarse.value(),
            "chunky": self.chunky.value(),
            "luminance": self.luminance.isChecked(),
            "target_noise": self.target_noise.value(),
            "target_contrast": self.target_contrast.value(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        spins = (
            self.fine, self.medium, self.coarse, self.chunky,
            self.target_noise, self.target_contrast,
        )
        for spin in spins:
            spin.blockSignals(True)
        self.luminance.blockSignals(True)
        self.fine.setValue(params.get("fine", self.fine.value()))
        self.medium.setValue(params.get("medium", self.medium.value()))
        self.coarse.setValue(params.get("coarse", self.coarse.value()))
        # Older presets predate the chunky scale: treat missing as 0.
        self.chunky.setValue(params.get("chunky", 0.0))
        self.luminance.setChecked(
            bool(params.get("luminance", self.luminance.isChecked()))
            and not self._is_grayscale
        )
        self.target_noise.setValue(
            float(params.get("target_noise", self.target_noise.value()))
        )
        self.target_contrast.setValue(
            float(params.get("target_contrast", self.target_contrast.value()))
        )
        for spin in spins:
            spin.blockSignals(False)
        self.luminance.blockSignals(False)


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


class MoonEnhanceDialog(_FilterDialog):
    filter_id = "moon_enhance"
    supports_presets = True
    supports_clamp = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Moon Enhance", parent)
        self.set_help_text(
            "Detects faint moons outside the planet's disk and ring envelope "
            "and brightens each toward the target brightness. The planet and "
            "empty sky are left untouched. Detection re-runs when Sensitivity, "
            "Planet margin, or Max moons change; Moon brightness and Radius "
            "scale update instantly."
        )

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.brightness = self._add_double(
            "Moon brightness", fdef.default_params["brightness"], 0.0, 100.0,
            step=1.0, decimals=0,
        )
        self.brightness.setSuffix(" %")
        self.brightness.setToolTip(
            "Brightness each detected moon's peak is driven toward, as a "
            "percentage of full scale."
        )
        self.sensitivity = self._add_double(
            "Sensitivity", fdef.default_params["sensitivity"], 1.0, 50.0,
            step=0.5, decimals=1,
        )
        self.sensitivity.setToolTip(
            "Detection threshold in local signal-to-noise. Lower finds "
            "fainter moons but risks false positives from sky noise."
        )
        self.radius_scale = self._add_double(
            "Radius scale", fdef.default_params["radius_scale"], 0.2, 3.0,
            step=0.1, decimals=1,
        )
        self.radius_scale.setToolTip(
            "Multiplier on each moon's measured size for the brightening "
            "region."
        )
        self.planet_margin = QDoubleSpinBox()
        self.planet_margin.setRange(1.0, 2000.0)
        self.planet_margin.setDecimals(0)
        self.planet_margin.setSingleStep(10.0)
        self.planet_margin.setValue(50.0)
        self.planet_margin.setToolTip(
            "Exclusion distance around the planet in pixels."
        )
        self.planet_margin.valueChanged.connect(lambda _: self.params_changed.emit())
        self.margin_auto = QCheckBox("Auto")
        self.margin_auto.setChecked(True)
        self.margin_auto.setToolTip(
            "Half the planet's equivalent radius, enough for limb/ring glow "
            "without swallowing nearby moons."
        )
        self.margin_auto.toggled.connect(self._on_margin_auto_toggled)
        self.margin_auto.toggled.connect(lambda _: self.params_changed.emit())
        self.planet_margin.setEnabled(False)
        margin_row = QWidget()
        margin_layout = QHBoxLayout(margin_row)
        margin_layout.setContentsMargins(0, 0, 0, 0)
        margin_layout.addWidget(self.planet_margin, stretch=1)
        margin_layout.addWidget(self.margin_auto)
        self._form.addRow("Planet margin", margin_row)
        self.max_moons = self._add_double(
            "Max moons", fdef.default_params["max_moons"], 1.0, 100.0,
            step=1.0, decimals=0,
        )
        self.max_moons.setToolTip(
            "Keep at most this many detections, strongest first."
        )

    def _on_margin_auto_toggled(self, checked: bool) -> None:
        self.planet_margin.setEnabled(not checked)
        if checked:
            self.planet_margin.blockSignals(True)
            self._refresh_auto_margin_display()
            self.planet_margin.blockSignals(False)

    def _refresh_auto_margin_display(self) -> None:
        data = getattr(self, "_input_data", None)
        if data is None:
            return
        from planetary_tools.filters.moon_enhance import auto_planet_margin

        gray = bool(getattr(self, "_input_is_grayscale", False))
        self.planet_margin.setValue(auto_planet_margin(data, gray))

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "brightness": self.brightness.value(),
            "sensitivity": self.sensitivity.value(),
            "radius_scale": self.radius_scale.value(),
            "planet_margin": (
                0.0 if self.margin_auto.isChecked() else self.planet_margin.value()
            ),
            "max_moons": int(self.max_moons.value()),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        spins = (
            (self.brightness, "brightness"),
            (self.sensitivity, "sensitivity"),
            (self.radius_scale, "radius_scale"),
            (self.max_moons, "max_moons"),
        )
        for spin, key in spins:
            spin.blockSignals(True)
            spin.setValue(float(params.get(key, spin.value())))
            spin.blockSignals(False)
        margin = float(params.get("planet_margin", 0.0))
        auto = margin <= 0.0
        self.margin_auto.blockSignals(True)
        self.margin_auto.setChecked(auto)
        self.margin_auto.blockSignals(False)
        self.planet_margin.blockSignals(True)
        if auto:
            self._refresh_auto_margin_display()
        else:
            self.planet_margin.setValue(margin)
        self.planet_margin.setEnabled(not auto)
        self.planet_margin.blockSignals(False)

    def set_input_brightness(
        self,
        data: np.ndarray,
        is_grayscale: bool,
        *,
        noise_texture_scale: float | None = None,
        noise_chromatic: bool | None = None,
    ) -> None:
        super().set_input_brightness(
            data,
            is_grayscale,
            noise_texture_scale=noise_texture_scale,
            noise_chromatic=noise_chromatic,
        )
        if self.margin_auto.isChecked():
            self.planet_margin.blockSignals(True)
            self._refresh_auto_margin_display()
            self.planet_margin.blockSignals(False)


class AdaptiveDeconvDialog(_FilterDialog):
    filter_id = "adaptive_deconv"
    supports_presets = True
    supports_clamp = True

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None) -> None:
        self._is_grayscale = is_grayscale
        self._auto_running = False
        super().__init__("Adaptive Deconvolution", parent)

    def _build_filter_params(self) -> None:
        from planetary_tools.filters.adaptive_deconv_auto import (
            auto_adaptive_deconv_params,
        )

        self._auto_adaptive_deconv_params = auto_adaptive_deconv_params

        fdef = FILTERS[self.filter_id]
        self.amount = self._add_double(
            "Amount", fdef.default_params["amount"], 0.0, 100.0, step=0.1, decimals=1
        )
        self.adaptive = QCheckBox("Contrast Adaptive")
        self.adaptive.setToolTip(
            "Increases sharpening in areas of higher contrast."
        )
        self.adaptive.setChecked(fdef.default_params["adaptive"])
        self.adaptive.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.adaptive)
        self.luminance = QCheckBox("Luminance (BT.709)")
        self.luminance.setToolTip(
            "Sharpens BT.709 luminance and adds the change back to RGB. "
            "Decreases colour noise but lowers saturation."
        )
        self.luminance.setChecked(
            fdef.default_params["luminance"] and not self._is_grayscale
        )
        self.luminance.setEnabled(not self._is_grayscale)
        self.luminance.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.luminance)

        auto_row = QWidget()
        auto_layout = QHBoxLayout(auto_row)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_apply = QPushButton("Auto")
        self.auto_apply.setToolTip(
            "Binary-search amount to meet the target noise and contrast "
            "without exceeding either."
        )
        self.auto_apply.clicked.connect(self._run_auto_search)
        auto_layout.addWidget(self.auto_apply)
        auto_layout.addStretch(1)
        self._form.addRow(auto_row)

        self.target_noise = QDoubleSpinBox()
        self.target_noise.setRange(0.0, 20.0)
        self.target_noise.setDecimals(1)
        self.target_noise.setSingleStep(0.1)
        self.target_noise.setValue(float(fdef.default_params.get("target_noise", 3.5)))
        self.target_noise.setToolTip("Maximum peak-normalized noise score to allow.")
        self._form.addRow("Target noise", self.target_noise)

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

    def _set_amount_spin(self, amount: float) -> None:
        self.amount.blockSignals(True)
        self.amount.setValue(amount)
        self.amount.blockSignals(False)

    def _run_auto_search(self) -> None:
        if self._auto_running:
            return
        data = getattr(self, "_input_data", None)
        if data is None:
            return
        is_grayscale = bool(getattr(self, "_input_is_grayscale", False))
        prior = self.amount.value()

        progress_dlg = QProgressDialog("Calculating…", "Cancel", 0, 0, self)
        progress_dlg.setWindowTitle("Auto Adaptive Deconvolution")
        progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.show()

        class _Cancelled(Exception):
            pass

        from PyQt6.QtWidgets import QApplication

        def progress(amount: float, _noise: float, _contrast: float) -> None:
            self._set_amount_spin(amount)
            QApplication.processEvents()
            if progress_dlg.wasCanceled():
                raise _Cancelled

        self._auto_running = True
        self.auto_apply.setEnabled(False)
        try:
            result = self._auto_adaptive_deconv_params(
                data,
                is_grayscale,
                target_noise=self.target_noise.value(),
                target_contrast=self.target_contrast.value(),
                adaptive=self.adaptive.isChecked(),
                luminance=self.luminance.isChecked(),
                progress=progress,
                texture_scale=getattr(self, "_noise_texture_scale", None),
                chromatic=getattr(self, "_noise_chromatic", None),
            )
            self._set_amount_spin(result.amount)
        except _Cancelled:
            self._set_amount_spin(prior)
        finally:
            self._auto_running = False
            self.auto_apply.setEnabled(True)
            progress_dlg.close()
        self.params_changed.emit()

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p.update({
            "amount": self.amount.value(),
            "adaptive": self.adaptive.isChecked(),
            "luminance": self.luminance.isChecked(),
            "target_noise": self.target_noise.value(),
            "target_contrast": self.target_contrast.value(),
        })
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.amount.blockSignals(True)
        self.adaptive.blockSignals(True)
        self.luminance.blockSignals(True)
        self.target_noise.blockSignals(True)
        self.target_contrast.blockSignals(True)
        self.amount.setValue(params.get("amount", self.amount.value()))
        self.adaptive.setChecked(params.get("adaptive", self.adaptive.isChecked()))
        self.luminance.setChecked(
            params.get("luminance", self.luminance.isChecked())
            and not self._is_grayscale
        )
        self.target_noise.setValue(
            float(params.get("target_noise", self.target_noise.value()))
        )
        self.target_contrast.setValue(
            float(params.get("target_contrast", self.target_contrast.value()))
        )
        self.amount.blockSignals(False)
        self.adaptive.blockSignals(False)
        self.luminance.blockSignals(False)
        self.target_noise.blockSignals(False)
        self.target_contrast.blockSignals(False)


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
        fdef = FILTERS[self.filter_id]
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.0, 100.0)
        self.amount.setDecimals(0)
        self.amount.setSingleStep(1.0)
        self.amount.setSuffix(" %")
        self.amount.setValue(float(fdef.default_params.get("amount", 100.0)))
        self.amount.setToolTip(
            "Peak level to stretch to. 100% stretches luminance to the full "
            "range; lower values stretch or contract the image to that peak; "
            "0% is black."
        )
        self.amount.valueChanged.connect(lambda _: self.params_changed.emit())
        self._form.addRow("Stretch to", self.amount)

        self.set_help_text(
            "Stretches OKLab luminance using proportional RGB scaling so the "
            "peak lands at the chosen level: 100% is the full range; lower "
            "values contract the image to that peak."
        )

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p["amount"] = self.amount.value()
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        self.amount.blockSignals(True)
        self.amount.setValue(float(params.get("amount", 100.0)))
        self.amount.blockSignals(False)


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
        self.set_help_text(
            "Adjusts chroma in OKLab. 1.00 on each control leaves the image unchanged."
        )

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
        ch_row = QWidget()
        ch_layout = QHBoxLayout(ch_row)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        ch_layout.addWidget(self._channel_combo, stretch=1)
        stretch_btn = QPushButton("Stretch")
        stretch_btn.setToolTip(
            "Set this channel's input maximum to its peak value."
        )
        stretch_btn.clicked.connect(self._stretch_channel)
        ch_layout.addWidget(stretch_btn)
        self._form.addRow("Channel", ch_row)

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
        _fit_spin_width(self._gamma)
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

        self.set_help_text(
            "Luminance uses OKLab L as the brightness measure, applied by "
            "scaling RGB like Stretch Contrast OKLab. RGB matches GIMP "
            "perceptual gamma (sRGB-encoded); output is applied before input."
        )
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

    def set_input_brightness(
        self,
        data: np.ndarray,
        is_grayscale: bool,
        *,
        noise_texture_scale: float | None = None,
        noise_chromatic: bool | None = None,
    ) -> None:
        super().set_input_brightness(
            data,
            is_grayscale,
            noise_texture_scale=noise_texture_scale,
            noise_chromatic=noise_chromatic,
        )
        self._input_data = np.asarray(data, dtype=np.float32)
        self._is_grayscale = is_grayscale

    def _stretch_channel(self) -> None:
        if self._input_data is None:
            return
        self._store_spins_in_channel(self._editing_channel)
        ch = self._editing_channel
        peak = channel_input_peak(self._input_data, ch)
        if peak < 1e-6:
            peak = 1.0
        self._channel_params[ch]["in_max"] = peak
        self._load_channel_into_spins(ch)
        self._notify_params_changed(immediate=True)

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
        self._scales_spin.setRange(0, 4)
        self._scales_spin.setValue(3)
        self._scales_spin.setToolTip(
            "Number of finest wavelet scales to take from the secondary image.\n"
            "0 = none (main image unchanged, for comparison),\n"
            "1 = fine only, 2 = fine + medium, 3 = fine + medium + coarse (default),\n"
            "4 = all four scales including the coarsest detail band."
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


class ExtractComponentDialog(_FilterDialog):
    """Extract one colour component as a greyscale image."""

    filter_id = ""
    supports_presets = False
    supports_clamp = False
    # Result is always greyscale (2-D plane after apply).
    result_is_grayscale = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Extract Component", parent)

    def _build_filter_params(self) -> None:
        from planetary_tools.filters.extract_component import (
            COMPONENT_LABELS,
            COMPONENT_ORDER,
        )

        self.component = QComboBox()
        for key in COMPONENT_ORDER:
            self.component.addItem(COMPONENT_LABELS[key], key)
        self.component.setToolTip(
            "Component to extract as greyscale. CMY use the mean of the two "
            "primaries that form that secondary (Cyan = (G+B)/2, "
            "Magenta = (R+B)/2, Yellow = (R+G)/2)."
        )
        self.component.currentIndexChanged.connect(lambda _: self.params_changed.emit())
        self._form.addRow("Component", self.component)

    def get_params(self) -> dict[str, Any]:
        p = super().get_params()
        p["component"] = str(self.component.currentData() or "luminance")
        return p

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        key = str(params.get("component", "luminance"))
        idx = self.component.findData(key)
        if idx < 0:
            idx = 0
        self.component.blockSignals(True)
        self.component.setCurrentIndex(idx)
        self.component.blockSignals(False)

    def build_filter_func(self) -> FilterFunc:
        from planetary_tools.filters.extract_component import extract_component

        component = str(self.component.currentData() or "luminance")

        def func(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
            # RGB triple for preview/canvas; apply path collapses to 2-D greyscale.
            return extract_component(data, is_grayscale, component, as_rgb=True)

        return func


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
        self._form.addRow(QLabel("Matrix (linear RGB):"))
        self._form.addRow(panel)

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


def _apply_params_to_batch_widgets(
    filter_id: str,
    widgets: dict[str, Any],
    params: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    """Push a params dict into widgets built by ``edit_filter_params``."""
    del filter_id  # reserved for filter-specific handling
    merged = {**defaults, **(params or {})}

    if "_levels_channels" in widgets:
        channel_data = widgets["_levels_channels"]
        normalized = normalize_levels_params(merged)
        for ch in LEVEL_CHANNELS:
            channel_data[ch] = dict(normalized[ch])
        if "_levels_reload" in widgets:
            widgets["_levels_reload"]()

    for key, widget in widgets.items():
        if key.startswith("_") or key.startswith("_levels"):
            continue
        if key not in merged:
            continue
        value = merged[key]
        if key == "matrix":
            _set_matrix_widgets(widget, value)
        elif isinstance(widget, QDoubleSpinBox):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        elif isinstance(widget, QCheckBox):
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)

    # Refresh auto/manual enable state (or scale aspect linking) after a preset.
    sync_auto = widgets.get("_sync_auto")
    if callable(sync_auto):
        if "auto" in widgets:
            sync_auto(bool(widgets["auto"].isChecked()))
        else:
            sync_auto()


def edit_filter_params(
    filter_id: str,
    params: dict[str, Any],
    is_grayscale: bool,
    parent: QWidget | None = None,
    *,
    preset_name: str | None = None,
) -> tuple[dict[str, Any], str | None] | None:
    """Small modal editor for batch pipeline step parameters.

    Returns ``(params, preset_name)`` on accept, or ``None`` on cancel.
    ``preset_name`` is the selected saved preset, or ``None`` if parameters
    were hand-edited away from a named preset.
    """
    fdef = FILTERS[filter_id]
    dlg = QDialog(parent)
    dlg.setWindowTitle(fdef.label)
    dlg.setWindowFlags(
        Qt.WindowType.Tool
        | Qt.WindowType.WindowTitleHint
        | Qt.WindowType.WindowCloseButtonHint
    )
    dlg.setModal(False)
    dlg.setMinimumWidth(FILTER_PANEL_WIDTH)
    layout = QVBoxLayout(dlg)
    form = QFormLayout()
    layout.addLayout(form)
    widgets: dict[str, Any] = {}

    presets = ensure_builtin_presets(filter_id, fdef.default_params)
    selected_preset: list[str | None] = [
        preset_name if preset_name in presets else None
    ]
    suppress_dirty = [False]

    preset_row = QWidget()
    preset_layout = QHBoxLayout(preset_row)
    preset_layout.setContentsMargins(0, 0, 0, 0)
    preset_layout.addWidget(QLabel("Preset:"))
    preset_combo = QComboBox()
    preset_combo.addItem(_PRESET_NONE)
    for name in sorted(presets.keys()):
        preset_combo.addItem(name)
    if selected_preset[0] is not None:
        idx = preset_combo.findText(selected_preset[0])
        if idx >= 0:
            preset_combo.setCurrentIndex(idx)
    else:
        preset_combo.setCurrentIndex(0)
    preset_layout.addWidget(preset_combo, stretch=1)
    layout.insertWidget(0, preset_row)

    if filter_id == "wavelet_sharpen":
        for key in ("fine", "medium", "coarse", "chunky"):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 300.0)
            spin.setValue(params.get(key, fdef.default_params.get(key, 0.0)))
            form.addRow(key.capitalize(), spin)
            widgets[key] = spin
        luminance = QCheckBox("Luminance (BT.709)")
        luminance.setToolTip(
            "Sharpens BT.709 luminance and adds the change back to RGB. "
            "Decreases colour noise but lowers saturation."
        )
        luminance.setChecked(
            bool(params.get("luminance", fdef.default_params.get("luminance", False)))
            and not is_grayscale
        )
        luminance.setEnabled(not is_grayscale)
        form.addRow(luminance)
        widgets["luminance"] = luminance
        auto_box = QCheckBox("Auto")
        auto_box.setToolTip(
            "Per image, search fine/medium/coarse/chunky to meet the noise and "
            "contrast targets without exceeding either. Manual amounts are ignored."
        )
        auto_box.setChecked(bool(params.get("auto", fdef.default_params.get("auto", False))))
        form.addRow(auto_box)
        widgets["auto"] = auto_box
        target_noise = QDoubleSpinBox()
        target_noise.setRange(0.0, 20.0)
        target_noise.setDecimals(1)
        target_noise.setSingleStep(0.1)
        target_noise.setValue(
            float(params.get("target_noise", fdef.default_params.get("target_noise", 3.0)))
        )
        form.addRow("Target noise", target_noise)
        widgets["target_noise"] = target_noise
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

        def _sync_wavelet_auto(checked: bool) -> None:
            for key in ("fine", "medium", "coarse", "chunky"):
                widgets[key].setEnabled(not checked)
            widgets["target_noise"].setEnabled(checked)
            widgets["target_contrast"].setEnabled(checked)
            preset_combo.setEnabled(not checked)

        auto_box.toggled.connect(_sync_wavelet_auto)
        _sync_wavelet_auto(auto_box.isChecked())
        widgets["_sync_auto"] = _sync_wavelet_auto
    elif filter_id == "wavelet_denoise":
        for key in ("fine", "medium", "coarse"):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 20.0)
            spin.setValue(params.get(key, fdef.default_params[key]))
            form.addRow(key.capitalize(), spin)
            widgets[key] = spin
    elif filter_id == "adaptive_deconv":
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 100.0)
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
        luminance = QCheckBox("Luminance (BT.709)")
        luminance.setToolTip(
            "Sharpens BT.709 luminance and adds the change back to RGB. "
            "Decreases colour noise but lowers saturation."
        )
        luminance.setChecked(
            params.get("luminance", fdef.default_params["luminance"]) and not is_grayscale
        )
        luminance.setEnabled(not is_grayscale)
        form.addRow(luminance)
        widgets["luminance"] = luminance
        auto_box = QCheckBox("Auto")
        auto_box.setToolTip(
            "Per image, binary-search amount to meet the noise and contrast "
            "targets without exceeding either. Manual amount is ignored."
        )
        auto_box.setChecked(bool(params.get("auto", fdef.default_params.get("auto", False))))
        form.addRow(auto_box)
        widgets["auto"] = auto_box
        target_noise = QDoubleSpinBox()
        target_noise.setRange(0.0, 20.0)
        target_noise.setDecimals(1)
        target_noise.setSingleStep(0.1)
        target_noise.setValue(
            float(params.get("target_noise", fdef.default_params.get("target_noise", 3.5)))
        )
        form.addRow("Target noise", target_noise)
        widgets["target_noise"] = target_noise
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

        def _sync_deconv_auto(checked: bool) -> None:
            widgets["amount"].setEnabled(not checked)
            widgets["target_noise"].setEnabled(checked)
            widgets["target_contrast"].setEnabled(checked)
            preset_combo.setEnabled(not checked)

        auto_box.toggled.connect(_sync_deconv_auto)
        _sync_deconv_auto(auto_box.isChecked())
        widgets["_sync_auto"] = _sync_deconv_auto
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
        amount = QDoubleSpinBox()
        amount.setRange(0.0, 100.0)
        amount.setDecimals(0)
        amount.setSingleStep(1.0)
        amount.setSuffix(" %")
        amount.setValue(
            float(params.get("amount", fdef.default_params.get("amount", 100.0)))
        )
        form.addRow("Stretch to", amount)
        widgets["amount"] = amount
    elif filter_id == "colour_matrix":
        matrix = params.get("matrix", fdef.default_params["matrix"])
        panel, matrix_widgets = _make_matrix_grid(matrix)
        form.addRow(QLabel("Matrix (linear RGB):"))
        form.addRow(panel)
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
            for spin in level_spins.values():
                spin.blockSignals(True)
            level_spins["in_min"].setValue(lv["in_min"] * 100.0)
            level_spins["in_max"].setValue(lv["in_max"] * 100.0)
            level_spins["gamma"].setValue(lv["gamma"])
            level_spins["out_min"].setValue(lv["out_min"] * 100.0)
            level_spins["out_max"].setValue(lv["out_max"] * 100.0)
            for spin in level_spins.values():
                spin.blockSignals(False)

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

        def _reload_levels_current() -> None:
            ch = str(channel_combo.currentData() or "L")
            _load_batch_channel(ch)

        channel_combo.currentIndexChanged.connect(_on_batch_channel_changed)
        channel_combo.setProperty("_prev_channel", "L")
        _load_batch_channel("L")
        widgets["_levels_channels"] = channel_data
        widgets["_levels_channel_combo"] = channel_combo
        widgets["_levels_store"] = _store_batch_channel
        widgets["_levels_reload"] = _reload_levels_current

        for spin in level_spins.values():
            spin.valueChanged.connect(lambda _: _mark_preset_dirty())
    elif filter_id == "scale_image":
        percent = QDoubleSpinBox()
        percent.setRange(0.01, 10_000.0)
        percent.setDecimals(2)
        percent.setSingleStep(1.0)
        percent.setSuffix(" %")
        percent.setValue(float(params.get("percent", fdef.default_params["percent"])))
        percent.setToolTip(
            "Scale factor applied to each image. Width and height follow this "
            "when aspect ratio is maintained."
        )
        form.addRow("Scale", percent)
        widgets["percent"] = percent

        width_percent = QDoubleSpinBox()
        width_percent.setRange(0.01, 10_000.0)
        width_percent.setDecimals(2)
        width_percent.setSingleStep(1.0)
        width_percent.setSuffix(" %")
        width_percent.setValue(
            float(params.get("width_percent", fdef.default_params["width_percent"]))
        )
        width_percent.setToolTip("Horizontal scale relative to each image's original width.")
        form.addRow("Width", width_percent)
        widgets["width_percent"] = width_percent

        height_percent = QDoubleSpinBox()
        height_percent.setRange(0.01, 10_000.0)
        height_percent.setDecimals(2)
        height_percent.setSingleStep(1.0)
        height_percent.setSuffix(" %")
        height_percent.setValue(
            float(params.get("height_percent", fdef.default_params["height_percent"]))
        )
        height_percent.setToolTip("Vertical scale relative to each image's original height.")
        form.addRow("Height", height_percent)
        widgets["height_percent"] = height_percent

        aspect = QCheckBox("Maintain aspect ratio")
        aspect.setChecked(
            bool(params.get("maintain_aspect", fdef.default_params["maintain_aspect"]))
        )
        aspect.setToolTip("Keep width and height scale factors equal.")
        form.addRow(aspect)
        widgets["maintain_aspect"] = aspect

        syncing = [False]

        def _sync_scale_aspect(_checked: bool | None = None) -> None:
            linked = aspect.isChecked()
            if linked and not syncing[0]:
                syncing[0] = True
                try:
                    height_percent.setValue(width_percent.value())
                    percent.setValue(width_percent.value())
                finally:
                    syncing[0] = False

        def _on_scale_percent(value: float) -> None:
            if syncing[0]:
                return
            syncing[0] = True
            try:
                width_percent.setValue(value)
                if aspect.isChecked():
                    height_percent.setValue(value)
            finally:
                syncing[0] = False

        def _on_scale_width(value: float) -> None:
            if syncing[0]:
                return
            syncing[0] = True
            try:
                percent.setValue(value)
                if aspect.isChecked():
                    height_percent.setValue(value)
            finally:
                syncing[0] = False

        def _on_scale_height(value: float) -> None:
            if syncing[0]:
                return
            syncing[0] = True
            try:
                if aspect.isChecked():
                    width_percent.setValue(value)
                    percent.setValue(value)
            finally:
                syncing[0] = False

        percent.valueChanged.connect(_on_scale_percent)
        width_percent.valueChanged.connect(_on_scale_width)
        height_percent.valueChanged.connect(_on_scale_height)
        aspect.toggled.connect(_sync_scale_aspect)
        widgets["_sync_auto"] = _sync_scale_aspect
    elif filter_id == "rotate_image":
        angle = QDoubleSpinBox()
        angle.setRange(-3600.0, 3600.0)
        angle.setDecimals(2)
        angle.setSingleStep(0.01)
        angle.setSuffix(" °")
        angle.setValue(float(params.get("angle", fdef.default_params["angle"])))
        angle.setToolTip(
            "Rotation angle in degrees. Positive = counter-clockwise (CCW). "
            "The canvas expands to fit unless cropped."
        )
        form.addRow("Angle", angle)
        widgets["angle"] = angle

        presets_row = QHBoxLayout()
        btn_ccw = QPushButton("90° CCW")
        btn_ccw.setToolTip("Rotate 90° counter-clockwise.")
        btn_ccw.clicked.connect(lambda: angle.setValue(90.0))
        presets_row.addWidget(btn_ccw)
        btn_cw = QPushButton("90° CW")
        btn_cw.setToolTip("Rotate 90° clockwise (−90°).")
        btn_cw.clicked.connect(lambda: angle.setValue(-90.0))
        presets_row.addWidget(btn_cw)
        btn_180 = QPushButton("180°")
        btn_180.setToolTip("Rotate 180°.")
        btn_180.clicked.connect(lambda: angle.setValue(180.0))
        presets_row.addWidget(btn_180)
        form.addRow("Presets", presets_row)

        crop = QCheckBox("Crop to original size")
        crop.setChecked(
            bool(params.get("crop_to_original", fdef.default_params["crop_to_original"]))
        )
        crop.setToolTip(
            "After expanding to fit the rotated image, centre-crop back to "
            "the original width and height."
        )
        form.addRow(crop)
        widgets["crop_to_original"] = crop
    elif filter_id == "crop_image":
        autocrop = QCheckBox("Autocrop")
        autocrop.setChecked(
            bool(params.get("autocrop", fdef.default_params["autocrop"]))
        )
        autocrop.setToolTip(
            "Detect the central bright object on each image and set the "
            "canvas to its bounding box plus the border. If the border "
            "reaches past the frame, the canvas expands (new pixels are "
            "black). Recommended for mixed image sizes."
        )
        form.addRow(autocrop)
        widgets["autocrop"] = autocrop

        border = QDoubleSpinBox()
        border.setRange(0.0, 65535.0)
        border.setDecimals(0)
        border.setSuffix(" px")
        border.setValue(float(params.get("border_px", fdef.default_params["border_px"])))
        border.setToolTip(
            "Pixels kept around the detected object on every side. "
            "If that reaches past the frame, the canvas expands."
        )
        form.addRow("Border", border)
        widgets["border_px"] = border

        min_bright = QDoubleSpinBox()
        min_bright.setRange(0.0, 100.0)
        min_bright.setDecimals(1)
        min_bright.setSuffix(" %")
        min_bright.setValue(
            float(
                params.get(
                    "min_brightness_pct",
                    fdef.default_params["min_brightness_pct"],
                )
            )
        )
        min_bright.setToolTip(
            "Pixels at least this bright relative to each image's peak "
            "luminance are treated as the central object."
        )
        form.addRow("Min brightness", min_bright)
        widgets["min_brightness_pct"] = min_bright

        width = QDoubleSpinBox()
        width.setRange(0.0, 65535.0)
        width.setDecimals(0)
        width.setSuffix(" px")
        width.setSpecialValueText("Same as image")
        width.setValue(float(params.get("width", fdef.default_params["width"])))
        width.setToolTip(
            "Output width in pixels. Same as image uses each file's full "
            "width. Larger than an image expands the canvas (black padding)."
        )
        form.addRow("Width", width)
        widgets["width"] = width

        height = QDoubleSpinBox()
        height.setRange(0.0, 65535.0)
        height.setDecimals(0)
        height.setSuffix(" px")
        height.setSpecialValueText("Same as image")
        height.setValue(float(params.get("height", fdef.default_params["height"])))
        height.setToolTip(
            "Output height in pixels. Same as image uses each file's full "
            "height. Larger than an image expands the canvas (black padding)."
        )
        form.addRow("Height", height)
        widgets["height"] = height

        offset_x = QDoubleSpinBox()
        offset_x.setRange(-65535.0, 65535.0)
        offset_x.setDecimals(0)
        offset_x.setSuffix(" px")
        offset_x.setValue(
            float(params.get("offset_x", fdef.default_params["offset_x"]))
        )
        offset_x.setToolTip(
            "Horizontal shift of the crop centre from each image's middle. "
            "Positive moves the crop to the right."
        )
        form.addRow("Offset X", offset_x)
        widgets["offset_x"] = offset_x

        offset_y = QDoubleSpinBox()
        offset_y.setRange(-65535.0, 65535.0)
        offset_y.setDecimals(0)
        offset_y.setSuffix(" px")
        offset_y.setValue(
            float(params.get("offset_y", fdef.default_params["offset_y"]))
        )
        offset_y.setToolTip(
            "Vertical shift of the crop centre from each image's middle. "
            "Positive moves the crop down."
        )
        form.addRow("Offset Y", offset_y)
        widgets["offset_y"] = offset_y

        def _sync_crop_mode(_checked: bool | None = None) -> None:
            auto = autocrop.isChecked()
            border.setEnabled(auto)
            min_bright.setEnabled(auto)
            width.setEnabled(not auto)
            height.setEnabled(not auto)
            offset_x.setEnabled(not auto)
            offset_y.setEnabled(not auto)

        autocrop.toggled.connect(_sync_crop_mode)
        _sync_crop_mode()
        widgets["_sync_auto"] = _sync_crop_mode
    elif filter_id == "moon_enhance":
        specs = (
            ("brightness", "Moon brightness", 0.0, 100.0, 0, " %"),
            ("sensitivity", "Sensitivity", 1.0, 50.0, 1, ""),
            ("radius_scale", "Radius scale", 0.2, 3.0, 1, ""),
            ("planet_margin", "Planet margin", 0.0, 2000.0, 0, ""),
            ("max_moons", "Max moons", 1.0, 100.0, 0, ""),
        )
        for key, label, lo, hi, decimals, suffix in specs:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(decimals)
            if suffix:
                spin.setSuffix(suffix)
            if key == "planet_margin":
                spin.setSpecialValueText("Auto")
                spin.setToolTip(
                    "Set to 0 for Auto (half the planet's equivalent radius)."
                )
            spin.setValue(float(params.get(key, fdef.default_params[key])))
            form.addRow(label, spin)
            widgets[key] = spin
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

    def _mark_preset_dirty() -> None:
        if suppress_dirty[0]:
            return
        if preset_combo.currentText() != _PRESET_NONE:
            preset_combo.blockSignals(True)
            preset_combo.setCurrentIndex(0)
            preset_combo.blockSignals(False)
        selected_preset[0] = None

    def _on_preset_chosen(name: str) -> None:
        if name == _PRESET_NONE or name not in presets:
            selected_preset[0] = None
            return
        suppress_dirty[0] = True
        try:
            _apply_params_to_batch_widgets(
                filter_id, widgets, presets[name], fdef.default_params
            )
            selected_preset[0] = name
        finally:
            suppress_dirty[0] = False

    preset_combo.currentTextChanged.connect(_on_preset_chosen)

    for key, widget in widgets.items():
        if key.startswith("_levels"):
            continue
        if isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(lambda _: _mark_preset_dirty())
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(lambda _: _mark_preset_dirty())
        elif key == "matrix":
            for row in widget:
                for spin in row:
                    spin.valueChanged.connect(lambda _: _mark_preset_dirty())

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
    name = selected_preset[0]
    if name == _PRESET_NONE:
        name = None
    return out, name