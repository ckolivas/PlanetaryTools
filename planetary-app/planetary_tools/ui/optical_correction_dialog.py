"""Correct Optical Aberrations: pick a moon and use it as the image's PSF."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QCheckBox, QGridLayout, QLabel, QWidget

from planetary_tools.filters import optical_correction as oc
from planetary_tools.filters.registry import FILTERS
from planetary_tools.ui.dialogs import _FilterDialog

_THUMB_PX = 96


def _psf_pixmap(psf: np.ndarray, size: int = _THUMB_PX) -> QPixmap:
    """Square-root stretched PSF, enlarged with visible pixels."""
    arr = np.clip(np.asarray(psf, dtype=np.float64), 0.0, None)
    peak = float(arr.max())
    if peak > 0:
        arr = np.sqrt(arr/peak)
    img8 = np.ascontiguousarray((arr*255.0 + 0.5).astype(np.uint8))
    h, w = img8.shape
    image = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(image).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)


class OpticalCorrectionDialog(_FilterDialog):
    filter_id = "optical_correction"
    supports_presets = True
    supports_clamp = True
    supports_difference = True
    # MainWindow connects canvas clicks/boxes and the moon markers.
    picks_canvas_region = True

    # [(x, y, radius), ...] of the moons in use; empty hides the markers.
    markers_changed = pyqtSignal(object)

    _ABERRATIONS = (
        ("astigmatism", "Astigmatism", 2,
         "Elongated blur. Detail along the long axis is boosted until the "
         "moon is round (Target size, Max boost)."),
        ("coma", "Coma/collimation", 1,
         "One-sided blur, a comet-like tail. Light is moved back from the "
         "tail; nothing is amplified. Off-axis coma varies across the field: "
         "turn it off if the moons disagree or the planet shows none."),
        ("trefoil", "Trefoil", 3,
         "Three-lobed, triangular blur, for example from pinched optics. "
         "On an SCT, miscollimation also gives an unbalanced trefoil with "
         "brighter wings towards the coma, so tick it with Coma/collimation. "
         "Light is moved back from the lobes; nothing is amplified."),
    )

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None) -> None:
        self._is_grayscale_doc = is_grayscale
        self._moon_x = -1.0
        self._moon_y = -1.0
        self._measurement: oc.PsfMeasurement | None = None
        self._maps = None  # protect-background maps for the measurement
        super().__init__("Correct Optical Aberrations", parent)
        self.set_help_text(
            "Uses the moons in the image as its point spread function and "
            "corrects the aberrations they show: astigmatism (elongation), "
            "coma (a one-sided tail) and trefoil (three lobes). Each is "
            "measured on the moons; untick any you do not want corrected. "
            "Auto combines every suitable moon (bright, compact, isolated, "
            "unclipped and off the planet's glow; a resolved disc such as "
            "Titan is avoided) and warns when they disagree, as off-axis "
            "coma does. Or click a moon on the image or drag a box around "
            "it; Shift-drag pans. Use it on the unsharpened stack, then "
            "sharpen as usual (Wavelet Auto picks suitable amounts); always "
            "before Moon Enhance, denoise or clamping, which change the "
            "moons' shape. In a batch, Auto measures every image again and a "
            "chosen moon is followed within 30 px."
        )

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        self.auto = QCheckBox("Auto-select moons")
        self.auto.setChecked(bool(fdef.default_params["auto"]))
        self.auto.setToolTip(
            "Combine every suitable moon: bright, compact, away from other "
            "moons, not clipped and off the planet's glow. In a batch it "
            "chooses again for each image."
        )
        self.auto.toggled.connect(self._on_auto_toggled)
        self._form.addRow(self.auto)

        self.pick_hint = QLabel("Or click a moon on the image, or drag a box around it.")
        self.pick_hint.setWordWrap(True)
        self._form.addRow(self.pick_hint)

        self.moon_info = QLabel("Moon —")
        self.moon_info.setWordWrap(True)
        self.moon_info.setToolTip(
            "Full width at half maximum along the PSF's long and short axes, "
            "their ratio and the long axis angle (anticlockwise from "
            "horizontal), and the moon's detection signal-to-noise."
        )
        self._form.addRow(self.moon_info)

        thumbs = QWidget()
        grid = QGridLayout(thumbs)
        grid.setContentsMargins(0, 0, 0, 0)
        self.measured_view = QLabel()
        self.corrected_view = QLabel()
        for col, (view, caption) in enumerate(((self.measured_view, "Measured moon"),
                                               (self.corrected_view, "Corrected moon"))):
            view.setFixedSize(_THUMB_PX, _THUMB_PX)
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setStyleSheet("background: black;")
            label = QLabel(caption)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(view, 0, col, Qt.AlignmentFlag.AlignHCenter)
            grid.addWidget(label, 1, col)
        thumbs.setToolTip("The best moon cut from the image before and after the "
                          "correction the image gets, square-root stretched to "
                          "show its faint glow and any noise.")
        self._form.addRow(thumbs)

        self._aberration_boxes: dict[str, QCheckBox] = {}
        for key, label, _order, tip in self._ABERRATIONS:
            box = QCheckBox(label)
            box.setChecked(bool(fdef.default_params[key]))
            box.setToolTip(tip + " The figure is its measured strength relative "
                                 "to the round part of the moon.")
            box.toggled.connect(lambda _: self._update_labels())
            box.toggled.connect(lambda _: self._refresh_corrected_view())
            box.toggled.connect(lambda _: self.params_changed.emit())
            self._form.addRow(box)
            self._aberration_boxes[key] = box
        self._amounts: dict[str, float | None] = {}
        self.agreement_warning = QLabel()
        self.agreement_warning.setWordWrap(True)
        self.agreement_warning.setStyleSheet("color: #d08000;")
        self.agreement_warning.hide()
        self._form.addRow(self.agreement_warning)

        self.target_size = self._add_double(
            "Target size", fdef.default_params["target_size"], 25.0, 100.0, step=5.0, decimals=0)
        self.target_size.setSuffix(" %")
        self.target_size.setToolTip(
            "Astigmatism: size of the round PSF the moon is corrected to, as a "
            "percentage of its narrowest width. 100% removes only the "
            "elongation; lower values also sharpen and amplify noise.")
        self.max_boost = self._add_double(
            "Max boost", fdef.default_params["max_boost"], 1.0, 20.0, step=0.5, decimals=1)
        self.max_boost.setSuffix(" ×")
        self.max_boost.setToolTip(
            "Astigmatism: largest amplification of detail along the moon's "
            "long axis. 1× leaves it uncorrected; the default 3× corrects most of a "
            "moderate elongation and 8× usually makes the moon round, with "
            "more noise and edge overshoot. Nothing is ever smoothed.")
        for spin in (self.target_size, self.max_boost):
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda _: self._refresh_corrected_view())

        self.per_channel = QCheckBox("Measure each colour channel")
        self.per_channel.setChecked(bool(fdef.default_params["per_channel"]))
        self.per_channel.setToolTip(
            "Measure the moons separately in red, green and blue, so each "
            "channel's own blur is corrected. A channel too faint to measure "
            "uses the luminance PSF.")
        self.per_channel.setEnabled(not self._is_grayscale_doc)
        self.per_channel.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.per_channel)

        self.protect_background = QCheckBox("Protect background")
        self.protect_background.setChecked(bool(fdef.default_params["protect_background"]))
        self.protect_background.setToolTip(
            "Boost detail for astigmatism only where the signal is strong "
            "enough that its noise stays hidden; leave sky and faint glow, "
            "including the moons' outer glow, unchanged so their speckle is "
            "not boosted.")
        self.protect_background.toggled.connect(lambda _: self._refresh_corrected_view())
        self.protect_background.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.protect_background)

    # --- moon selection -------------------------------------------------

    def _data(self) -> np.ndarray | None:
        return getattr(self, "_input_data", None)

    def _on_auto_toggled(self, checked: bool) -> None:
        if not checked and self._moon_x < 0 and self._measurement is not None:
            # Keep the moon Auto rated best as the starting manual choice.
            self._moon_x, self._moon_y = self._measurement.x, self._measurement.y
        self._update_measurement()
        self.params_changed.emit()

    def set_picked_region(self, x: int, y: int, width: int, height: int) -> None:
        """A canvas click (zero size) or dragged box chooses the moon."""
        data = self._data()
        if data is None:
            return
        cx, cy = x + width/2.0, y + height/2.0
        search = max(width, height)/2.0 + 4.0 if max(width, height) > 0 else 30.0
        choice = oc.locate_moon(data, self._input_is_grayscale, cx, cy, search)
        if choice is None:
            return
        self._moon_x, self._moon_y = round(choice.x, 1), round(choice.y, 1)
        self.auto.blockSignals(True)
        self.auto.setChecked(False)
        self.auto.blockSignals(False)
        self._update_measurement()
        self.params_changed.emit()

    def _selection_params(self) -> dict[str, Any]:
        return {"auto": self.auto.isChecked(), "moon_x": self._moon_x, "moon_y": self._moon_y}

    def _aberrations(self) -> dict[str, bool]:
        return {key: box.isChecked() for key, box in self._aberration_boxes.items()}

    def _update_measurement(self) -> None:
        data = self._data()
        if data is None:
            return
        moons = oc.choose_moons(data, self._input_is_grayscale, self._selection_params())
        m = oc.measure_moons(data, self._input_is_grayscale, moons) if moons else None
        self._measurement = m
        amounts = {"astigmatism": None, "coma": None, "trefoil": None}
        if m is None:
            if not moons and self.auto.isChecked():
                text = "No suitable moon found. Click a moon or drag a box around it."
            elif not moons:
                text = "Click a moon on the image or drag a box around it."
            else:
                choice = moons[0][0]
                text = f"No measurable moon at ({choice.x:.0f}, {choice.y:.0f})."
            self.moon_info.setText(text)
            self.measured_view.clear()
            self.corrected_view.clear()
            self.agreement_warning.hide()
        else:
            count = len(m.moons)
            if self.auto.isChecked():
                source = f"Auto: {count} moons combined" if count > 1 else "Auto: moon"
            else:
                source = "Moon"
            where = "" if count > 1 else f" at ({m.x:.0f}, {m.y:.0f})"
            text = (f"{source}{where} — FWHM {m.fwhm_major:.1f} × {m.fwhm_minor:.1f} px, "
                    f"{m.elongation:.2f}:1 at {m.angle_deg:.0f}°, SNR {m.snr:.0f}")
            self.moon_info.setText(text)
            self._maps = oc.signal_maps(self._luminance(), m)
            amounts = {"astigmatism": m.astigmatism_amount, "coma": m.coma_amount,
                       "trefoil": m.trefoil_amount}
            differing = oc.disagreements(m)
            notes = []
            if differing:
                notes.append(
                    f"{' and '.join(differing)} differs between the moons; it "
                    "probably varies across the field, so it may not suit the "
                    "planet. Consider unticking it or picking the nearest moon.")
            if oc.elongation_differs(m):
                notes.append(
                    "The moons are elongated by different amounts, as drift "
                    "during a long stack does; only the elongation they share "
                    f"({100*m.shared_elongation:.0f}%) is corrected as astigmatism.")
            if notes:
                self.agreement_warning.setText("\n".join(notes))
                self.agreement_warning.show()
            else:
                self.agreement_warning.hide()
        self._amounts = amounts
        self._update_labels()
        self._refresh_corrected_view()
        self.refresh_markers()

    def _update_labels(self) -> None:
        """Each aberration's measured strength and, for astigmatism whose
        elongation differs between the moons, the shared part corrected."""
        m = self._measurement
        for key, label, _order, _tip in self._ABERRATIONS:
            value = self._amounts.get(key)
            if value is None or m is None:
                self._aberration_boxes[key].setText(label)
                continue
            text = f"{label} ({100*value:.0f}%"
            if (key == "astigmatism" and oc.elongation_differs(m)
                    and self._aberration_boxes[key].isChecked()):
                # Only the share of the elongation the moons have in common.
                text += f", {100*value*m.shared_elongation:.0f}% corrected"
            self._aberration_boxes[key].setText(text + ")")

    def _luminance(self) -> np.ndarray:
        return oc._plane(self._data(), self._input_is_grayscale)

    def _refresh_corrected_view(self) -> None:
        """Both thumbnails: the best moon cut from the image, before and after
        the same correction the image gets."""
        m = self._measurement
        if m is None or self._data() is None:
            return
        maps = self._maps if self.protect_background.isChecked() else None
        before, after = oc.moon_preview(
            self._luminance(), m, maps, target_size=self.target_size.value(),
            max_boost=self.max_boost.value(), **self._aberrations())
        self.measured_view.setPixmap(_psf_pixmap(before))
        self.corrected_view.setPixmap(_psf_pixmap(after))

    def refresh_markers(self) -> None:
        m = self._measurement
        self.markers_changed.emit(list(m.moons) if m is not None else [])

    # --- params ---------------------------------------------------------

    def set_input_brightness(self, data, is_grayscale, **kwargs) -> None:
        super().set_input_brightness(data, is_grayscale, **kwargs)
        self._update_measurement()

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "auto": self.auto.isChecked(),
            "moon_x": float(self._moon_x),
            "moon_y": float(self._moon_y),
            "target_size": self.target_size.value(),
            "max_boost": self.max_boost.value(),
            # Kept as set for colour images; greyscale ignores it.
            "per_channel": self.per_channel.isChecked(),
            "protect_background": self.protect_background.isChecked(),
            **self._aberrations(),
        })
        return params

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        defaults = FILTERS[self.filter_id].default_params
        for spin, key in ((self.target_size, "target_size"), (self.max_boost, "max_boost")):
            spin.blockSignals(True)
            spin.setValue(float(params.get(key, defaults[key])))
            spin.blockSignals(False)
        boxes = [(self.auto, "auto"), (self.per_channel, "per_channel"),
                 (self.protect_background, "protect_background"),
                 *((box, key) for key, box in self._aberration_boxes.items())]
        for box, key in boxes:
            box.blockSignals(True)
            box.setChecked(bool(params.get(key, defaults[key])))
            box.blockSignals(False)
        self._moon_x = float(params.get("moon_x", -1.0))
        self._moon_y = float(params.get("moon_y", -1.0))
        self._update_measurement()
