"""Correct Optical Aberrations: pick a moon and use it as the image's PSF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QWidget,
)

from planetary_tools.filters import optical_correction as oc
from planetary_tools.filters.registry import FILTERS
from planetary_tools.ui.dialogs import _FilterDialog
from planetary_tools.ui.file_dialog import QFileDialog
from planetary_tools.ui.recent_files import last_open_directory, last_save_directory

PSF_FILE_FILTER = f"Planetary Tools PSF (*{oc.PSF_FILE_SUFFIX})"

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
    # No presets: Save PSF and Load PSF carry a measurement between captures.
    supports_presets = False
    supports_clamp = True
    supports_difference = True
    # MainWindow connects canvas clicks/boxes and the moon markers.
    picks_canvas_region = True

    # [(x, y, radius), ...] of the moons in use; empty hides the markers.
    markers_changed = pyqtSignal(object)
    # Marker radius for a selected moon that could not be measured, and the
    # distance within which a click removes a selected moon.
    _MARKER_RADIUS = 12.0

    # Astigmatism is not offered: its correction left a worse result after
    # sharpening (see optical_correction.ASTIGMATISM_ENABLED).
    _ABERRATIONS = (
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

    def __init__(self, is_grayscale: bool, parent: QWidget | None = None,
                 source_path: str | Path | None = None) -> None:
        self._is_grayscale_doc = is_grayscale
        self._source_path = Path(source_path) if source_path else None
        # A PSF loaded from a file (and its saved form), used instead of the
        # image's moons.
        self._saved: oc.ImagePsf | None = None
        self._saved_dict: dict | None = None
        # Hand-picked moons, (x, y) in image pixels.
        self._picks: list[tuple[float, float]] = []
        self._measurement: oc.PsfMeasurement | None = None
        self._maps = None  # protect-background maps for the measurement
        super().__init__("Correct Optical Aberrations", parent)
        self.set_help_text(
            "Uses the moons in the image as its point spread function and "
            "corrects the aberrations they show: coma (a one-sided tail) and "
            "trefoil (three lobes). Each is "
            "measured on the moons; untick any you do not want corrected. "
            "It starts with every suitable moon combined (bright, compact, "
            "isolated, unclipped and off the planet's glow; a resolved disc "
            "such as Titan is avoided) and warns when they disagree, as "
            "off-axis coma does. Click moons on the image to enable or "
            "disable them; drag a box around a moon among others; Shift-drag "
            "pans. Use it on the unsharpened stack, then "
            "sharpen as usual (Wavelet Auto picks suitable amounts); always "
            "before Moon Enhance, denoise or clamping, which change the "
            "moons' shape. In a batch, the moons are chosen again in every "
            "image until you click one; clicked moons are each followed "
            "within 30 px. For a planet with no moon "
            "in the frame, Save PSF on a capture that has moons (or a star) "
            "and Load PSF here; both captures need the same telescope, "
            "camera, Barlow and camera angle, and neither may be resized."
        )

    def _build_filter_params(self) -> None:
        fdef = FILTERS[self.filter_id]
        # The suitable moons are chosen automatically until one is clicked.
        self._auto = bool(fdef.default_params["auto"])
        self.pick_hint = QLabel()
        self._set_pick_hint()
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

        files = QWidget()
        row = QHBoxLayout(files)
        row.setContentsMargins(0, 0, 0, 0)
        self.save_psf_button = QPushButton("Save PSF…")
        self.save_psf_button.setToolTip(
            "Save the PSF measured on these moons (and on each colour "
            "channel), to correct a capture with no moon in the frame: one "
            "taken with the same telescope, camera, Barlow and camera angle, "
            "and not resized.")
        self.save_psf_button.clicked.connect(self._save_psf)
        self.load_psf_button = QPushButton("Load PSF…")
        self.load_psf_button.setToolTip(
            "Correct this image with a PSF saved from another capture, such "
            "as a moon or star test, instead of its own moons. Click a moon "
            "to measure this image's moons again.")
        self.load_psf_button.clicked.connect(self._load_psf)
        row.addWidget(self.save_psf_button)
        row.addWidget(self.load_psf_button)
        self._form.addRow(files)

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
            "Leave sky that holds only noise unchanged, so the correction "
            "does not move its speckle about.")
        self.protect_background.toggled.connect(lambda _: self._refresh_corrected_view())
        self.protect_background.toggled.connect(lambda _: self.params_changed.emit())
        self._form.addRow(self.protect_background)

    # --- moon selection -------------------------------------------------

    def _data(self) -> np.ndarray | None:
        return getattr(self, "_input_data", None)

    def _set_pick_hint(self, text: str = "") -> None:
        self.pick_hint.setText(text or (
            "Click moons on the image to enable or disable them. Drag a box "
            "around a moon among others."))

    def set_picked_region(self, x: int, y: int, width: int, height: int) -> None:
        """A canvas click (zero size) or dragged box selects a moon, or
        removes a selected one."""
        data = self._data()
        if data is None:
            return
        cx, cy = x + width/2.0, y + height/2.0
        size = max(width, height)
        # The first click changes the moons chosen automatically; with a
        # saved PSF loaded it starts a new selection.
        if self._saved is not None:
            picks = []
        elif self._auto:
            measured = self._measurement.moons if self._measurement is not None else ()
            picks = [(round(mx, 1), round(my, 1)) for mx, my, _ in measured]
        else:
            picks = list(self._picks)
        hit = max(self._MARKER_RADIUS, size/2.0)
        near = [p for p in picks if np.hypot(p[0] - cx, p[1] - cy) <= hit]
        if near:
            picks.remove(min(near, key=lambda p: np.hypot(p[0] - cx, p[1] - cy)))
            self._set_pick_hint()
        else:
            search = size/2.0 + 4.0 if size > 0 else oc.SELECT_SEARCH_PX
            found = oc.select_moon(data, self._input_is_grayscale, cx, cy, search)
            if found is None:
                self._set_pick_hint("No moon found there. Click closer to it, or "
                                    "drag a box around it.")
                return
            self._set_pick_hint()
            if any(np.hypot(p[0] - found[0], p[1] - found[1]) <= 3.0 for p in picks):
                return
            picks.append((float(found[0]), float(found[1])))
        self._picks = picks
        self._clear_saved()
        self._auto = False
        self._update_measurement()
        self.params_changed.emit()

    def _selection_params(self) -> dict[str, Any]:
        return {"auto": self._auto, "moons": [list(p) for p in self._picks]}

    def _aberrations(self) -> dict[str, bool]:
        return {key: box.isChecked() for key, box in self._aberration_boxes.items()}

    # --- saved PSFs -----------------------------------------------------

    def _clear_saved(self) -> None:
        self._saved = None
        self._saved_dict = None

    def _set_saved(self, saved: dict | None) -> None:
        """Use a PSF in its saved form; ValueError if it is not one."""
        if saved is None:
            self._clear_saved()
            return
        self._saved = oc.psf_from_dict(saved)
        self._saved_dict = saved

    def _psf_path_suggestion(self) -> str:
        if self._source_path is not None:
            return str(self._source_path.with_suffix(oc.PSF_FILE_SUFFIX))
        return str(Path(last_save_directory())/f"psf{oc.PSF_FILE_SUFFIX}")

    def _save_psf(self) -> None:
        data = self._data()
        if self._saved_dict is not None:
            saved = self._saved_dict
        elif data is None:
            return
        else:
            try:
                found = oc.measure_image_psf(
                    data, self._input_is_grayscale, **self._selection_params(),
                    per_channel=True)
            except oc.NoMoonError as exc:
                QMessageBox.warning(self, "Save PSF", str(exc))
                return
            source = self._source_path.name if self._source_path else "an unsaved image"
            saved = oc.psf_to_dict(found, source)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PSF", self._psf_path_suggestion(), PSF_FILE_FILTER)
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != oc.PSF_FILE_SUFFIX:
            path = path.with_name(path.name + oc.PSF_FILE_SUFFIX)
        try:
            path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Save PSF", f"Could not save {path.name}: {exc}")

    def _load_psf(self) -> None:
        start = str(self._source_path.parent) if self._source_path else last_open_directory()
        path, _ = QFileDialog.getOpenFileName(self, "Load PSF", start, PSF_FILE_FILTER)
        if not path:
            return
        try:
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            self._set_saved(saved)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ValueError) and not isinstance(
                exc, json.JSONDecodeError) else "This is not a Planetary Tools PSF file."
            QMessageBox.warning(self, "Load PSF", f"Could not load {Path(path).name}: {message}")
            return
        self._auto = False
        self._update_measurement()
        self.params_changed.emit()

    def _update_measurement(self) -> None:
        data = self._data()
        if data is None:
            return
        if self._saved is not None:
            moons = []
            m = self._saved.luminance
        else:
            moons = oc.choose_moons(data, self._input_is_grayscale, self._selection_params())
            m = oc.measure_moons(data, self._input_is_grayscale, moons) if moons else None
        self._measurement = m
        self.save_psf_button.setEnabled(m is not None)
        amounts = {"coma": None, "trefoil": None}
        if m is None:
            if not moons and self._auto:
                text = ("No suitable moon found. Click a moon, drag a box "
                        "around it or load a saved PSF.")
            elif not moons:
                text = "Click moons on the image, drag a box around one or load a saved PSF."
            else:
                choice = moons[0][0]
                text = f"No measurable moon at ({choice.x:.0f}, {choice.y:.0f})."
            self.moon_info.setText(text)
            self.measured_view.clear()
            self.corrected_view.clear()
            self.agreement_warning.hide()
        else:
            count = len(m.moons)
            if self._saved is not None:
                text = oc.psf_summary(self._saved)
            elif self._auto:
                source = f"Auto: {count} moons combined" if count > 1 else "Auto: moon"
            else:
                source = f"{count} moons combined" if count > 1 else "Moon"
            if self._saved is None:
                where = "" if count > 1 else f" at ({m.x:.0f}, {m.y:.0f})"
                text = (f"{source}{where} — FWHM {m.fwhm_major:.1f} × {m.fwhm_minor:.1f} px, "
                        f"{m.elongation:.2f}:1 at {m.angle_deg:.0f}°, SNR {m.snr:.0f}")
            self.moon_info.setText(text)
            self._maps = oc.signal_maps(self._luminance(), m)
            amounts = {"coma": m.coma_amount, "trefoil": m.trefoil_amount}
            differing = oc.disagreements(m)
            notes = []
            if differing:
                notes.append(
                    f"{' and '.join(differing)} differs between the moons; it "
                    "probably varies across the field, so it may not suit the "
                    "planet. Consider unticking it or picking the nearest moon.")
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
        """Each aberration's measured strength."""
        m = self._measurement
        for key, label, _order, _tip in self._ABERRATIONS:
            value = self._amounts.get(key)
            if value is None or m is None:
                self._aberration_boxes[key].setText(label)
                continue
            self._aberration_boxes[key].setText(f"{label} ({100*value:.0f}%)")

    def _luminance(self) -> np.ndarray:
        return oc._plane(self._data(), self._input_is_grayscale)

    def _refresh_corrected_view(self) -> None:
        """Both thumbnails: the best moon cut from the image, before and after
        the same correction the image gets."""
        m = self._measurement
        if m is None or self._data() is None:
            return
        if self._saved is not None:
            before, after = oc.psf_preview(m, **self._aberrations())
        else:
            maps = self._maps if self.protect_background.isChecked() else None
            before, after = oc.moon_preview(
                self._luminance(), m, maps, **self._aberrations())
        self.measured_view.setPixmap(_psf_pixmap(before))
        self.corrected_view.setPixmap(_psf_pixmap(after))

    def refresh_markers(self) -> None:
        if self._saved is not None:
            self.markers_changed.emit([])
            return
        measured = list(self._measurement.moons) if self._measurement is not None else []
        if self._auto:
            self.markers_changed.emit(measured)
            return
        # Every selected moon, measured or not, so each can be removed again.
        markers = []
        for x, y in self._picks:
            sized = [r for mx, my, r in measured if np.hypot(mx - x, my - y) <= 3.0]
            markers.append((x, y, sized[0] if sized else self._MARKER_RADIUS))
        self.markers_changed.emit(markers)

    # --- params ---------------------------------------------------------

    def set_input_brightness(self, data, is_grayscale, **kwargs) -> None:
        super().set_input_brightness(data, is_grayscale, **kwargs)
        self._update_measurement()

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "auto": self._auto,
            "moons": [[float(x), float(y)] for x, y in self._picks],
            # The single-moon form older workflows saved; "moons" replaces it.
            "moon_x": -1.0,
            "moon_y": -1.0,
            # Kept as set for colour images; greyscale ignores it.
            "per_channel": self.per_channel.isChecked(),
            "protect_background": self.protect_background.isChecked(),
            **self._aberrations(),
            "psf": self._saved_dict,
        })
        return params

    def set_params(self, params: dict[str, Any]) -> None:
        super().set_params(params)
        defaults = FILTERS[self.filter_id].default_params
        self._auto = bool(params.get("auto", defaults["auto"]))
        boxes = [(self.per_channel, "per_channel"),
                 (self.protect_background, "protect_background"),
                 *((box, key) for key, box in self._aberration_boxes.items())]
        for box, key in boxes:
            box.blockSignals(True)
            box.setChecked(bool(params.get(key, defaults[key])))
            box.blockSignals(False)
        self._picks = oc.manual_moons(params)
        try:
            self._set_saved(params.get("psf"))
        except ValueError:
            self._clear_saved()
        self._update_measurement()
