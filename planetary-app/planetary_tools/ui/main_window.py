"""Planetary Tools by Con Kolivas <kernel@kolivas.org>"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from planetary_tools import __version__
from planetary_tools.core.align import align_channel
from planetary_tools.core.document import ImageDocument
from planetary_tools.core.scale import scale_image
from planetary_tools.core.undo import UndoManager
# from planetary_tools.filters import oklab_compose, oklab_decompose
from planetary_tools.filters.registry import output_filter_stats
from planetary_tools.io.loader import (
    load_image,
    save_channel,
    save_image,
    supported_extensions,
)
from planetary_tools.ui.batch_dialog import BatchDialog
from planetary_tools.ui.canvas import ZOOM_LEVELS, ImageCanvas
from planetary_tools.ui.compose_dialog import RGBComposeDialog, detect_channel
from planetary_tools.ui.dialogs import (
    FILTER_PANEL_WIDTH,
    AdaptiveDeconvDialog,
    ColourMatrixDialog,
    LevelsDialog,
    MergeWaveletDetailDialog,
    # InstantFilterDialog,
    SaturationVibranceDialog,
    StretchContrastDialog,
    WaveletDenoiseDialog,
    WaveletSharpenDialog,
    WienerDeconvDialog,
    _FilterDialog,
)
from planetary_tools.ui.preview import PreviewController, array_to_display_rgb
from planetary_tools.ui.scale_dialog import ScaleImageDialog
from planetary_tools.ui.recent_files import (
    add_recent,
    last_open_directory,
    last_save_directory,
    last_save_filter,
    list_recent,
    remember_open_path,
    remember_save_filter,
    remember_save_path,
    remove_recent,
)

# Rec. 601 luma weights, used to derive a missing RGB channel from the other two.
_LUMA_WEIGHTS = {"Red": 0.299, "Green": 0.587, "Blue": 0.114}

# Preferred reference channel order when aligning compose-from-files channels.
_ALIGN_PRIORITY = ("Green", "Red", "Blue")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Planetary Tools")
        self.resize(1200, 800)

        self._document: ImageDocument | None = None
        self._active_filter_dlg: _FilterDialog | None = None
        self._filter_dialog_open = False
        self._undo = UndoManager(on_change=self._update_undo_actions)
        self._preview = PreviewController(parent=self)
        self._preview.preview_updated.connect(self._refresh_canvas_display)
        self._preview.preview_failed.connect(self._on_preview_failed)
        self._preview.busy_changed.connect(self._on_preview_busy)

        self._canvas = ImageCanvas()
        self._canvas.zoom_changed.connect(self._sync_zoom_combo)
        self.setCentralWidget(self._canvas)

        self._filter_dock = QDockWidget(self)
        self._filter_dock.setObjectName("FilterDock")
        self._filter_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._filter_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._filter_host = QWidget()
        self._filter_host_layout = QVBoxLayout(self._filter_host)
        self._filter_host_layout.setContentsMargins(8, 8, 8, 8)
        self._filter_dock.setWidget(self._filter_host)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._filter_dock)
        self._filter_dock.setFixedWidth(FILTER_PANEL_WIDTH)
        self._filter_dock.hide()
        self._filter_loop: QEventLoop | None = None
        self._filter_accepted = False

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._build_toolbar()
        self._build_menus()
        self._update_actions()
        self._status.showMessage("Open an image to begin. All images are stored as 32-bit float linear colour.")

    def _build_toolbar(self) -> None:
        bar = QToolBar("View")
        bar.setMovable(False)
        self.addToolBar(bar)

        self._zoom_combo = QComboBox()
        for z in ZOOM_LEVELS:
            self._zoom_combo.addItem(f"{int(z * 100)}%", z)
        self._zoom_combo.setCurrentText("100%")
        self._zoom_combo.currentIndexChanged.connect(self._on_zoom_combo)
        bar.addWidget(QLabel(" Zoom: "))
        bar.addWidget(self._zoom_combo)

        fit_act = QAction("Fit", self)
        fit_act.setToolTip("Fit the image to the window.")
        fit_act.triggered.connect(self._canvas.zoom_to_fit)
        bar.addAction(fit_act)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._open_act = QAction("&Open…", self)
        self._open_act.setShortcut(QKeySequence.StandardKey.Open)
        self._open_act.setToolTip("Open an image file for editing.")
        self._open_act.triggered.connect(self._open_file)
        file_menu.addAction(self._open_act)

        self._recent_menu = QMenu("Open &Recent", self)
        self._recent_menu.setToolTipsVisible(True)
        self._recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addMenu(self._recent_menu)

        self._save_act = QAction("&Save", self)
        self._save_act.setShortcut(QKeySequence.StandardKey.Save)
        self._save_act.setToolTip("Save changes to the current file.")
        self._save_act.triggered.connect(self._save_file)
        file_menu.addAction(self._save_act)

        self._save_as_act = QAction("Save &As…", self)
        self._save_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_as_act.setToolTip(
            "Save the current image to a new file, choosing format and bit depth."
        )
        self._save_as_act.triggered.connect(self._save_file_as)
        file_menu.addAction(self._save_as_act)

        file_menu.addSeparator()
        self._batch_act = QAction("&Batch Processing…", self)
        self._batch_act.setToolTip(
            "Apply one or more filters to every image in a folder."
        )
        self._batch_act.triggered.connect(self._run_batch)
        file_menu.addAction(self._batch_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.setToolTip("Close Planetary Tools.")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)
        file_menu.setToolTipsVisible(True)

        edit_menu = self.menuBar().addMenu("&Edit")
        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_act.setToolTip("Undo the last change.")
        self._undo_act.triggered.connect(self._undo_action)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("&Redo", self)
        self._redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_act.setToolTip("Redo the last undone change.")
        self._redo_act.triggered.connect(self._redo_action)
        edit_menu.addAction(self._redo_act)

        edit_menu.addSeparator()
        self._scale_act = QAction("&Scale Image…", self)
        self._scale_act.setToolTip(
            "Resize the image by percentage, width, or height."
        )
        self._scale_act.triggered.connect(self._run_scale_image)
        edit_menu.addAction(self._scale_act)
        edit_menu.setToolTipsVisible(True)

        enhance_menu = self.menuBar().addMenu("&Enhance")
        self._sharpen_act = QAction("Wavelet &Sharpen…", self)
        self._sharpen_act.setToolTip(
            "Sharpen fine detail using multi-scale wavelet decomposition."
        )
        self._sharpen_act.triggered.connect(self._run_wavelet_sharpen)
        enhance_menu.addAction(self._sharpen_act)

        self._denoise_act = QAction("Wavelet &Denoise…", self)
        self._denoise_act.setToolTip(
            "Reduce noise using multi-scale wavelet decomposition."
        )
        self._denoise_act.triggered.connect(self._run_wavelet_denoise)
        enhance_menu.addAction(self._denoise_act)

        self._deconv_act = QAction("Adaptive &Deconvolution…", self)
        self._deconv_act.setToolTip(
            "Sharpen detail using adaptive deconvolution, useful for final sharpening."
        )
        self._deconv_act.triggered.connect(self._run_adaptive_deconv)
        enhance_menu.addAction(self._deconv_act)

        # Wiener deconvolution is implemented but not exposed: weaker denoise
        # than wavelet for typical planetary stacks. Re-enable when improved.
        self._wiener_act = QAction("&Wiener Deconvolution…", self)
        self._wiener_act.setToolTip(
            "PSF deconvolution denoising. (Disabled — not as effective as wavelet denoise.)"
        )
        self._wiener_act.triggered.connect(self._run_wiener_deconv)
        self._wiener_act.setEnabled(False)
        self._wiener_act.setVisible(False)
        enhance_menu.addAction(self._wiener_act)

        enhance_menu.addSeparator()
        self._merge_detail_act = QAction("&Merge Wavelet Detail…", self)
        self._merge_detail_act.setToolTip(
            "Load an aligned higher resolution (eg. NIR) image to merge its detail into a colour image."
        )
        self._merge_detail_act.triggered.connect(self._run_merge_wavelet_detail)
        enhance_menu.addAction(self._merge_detail_act)
        enhance_menu.setToolTipsVisible(True)

        colours_menu = self.menuBar().addMenu("&Colours")
        self._stretch_act = QAction("Stretch Contrast &OKLab", self)
        self._stretch_act.setToolTip(
            "Stretch contrast in OKLab lightness while preserving colour."
        )
        self._stretch_act.triggered.connect(self._run_stretch)
        colours_menu.addAction(self._stretch_act)

        self._colour_matrix_act = QAction("Colour Correction &Matrix…", self)
        self._colour_matrix_act.setToolTip(
            "Apply a 3×3 colour correction matrix, with predefined camera matrices available."
        )
        self._colour_matrix_act.triggered.connect(self._run_colour_matrix)
        colours_menu.addAction(self._colour_matrix_act)

        self._saturation_act = QAction("Saturation && &Vibrance…", self)
        self._saturation_act.setToolTip(
            "Adjust overall saturation and targeted vibrance."
        )
        self._saturation_act.triggered.connect(self._run_saturation_vibrance)
        colours_menu.addAction(self._saturation_act)

        self._levels_act = QAction("&Levels…", self)
        self._levels_act.setToolTip("Adjust per-channel black and white points.")
        self._levels_act.triggered.connect(self._run_levels)
        colours_menu.addAction(self._levels_act)

        colours_menu.addSeparator()
        self._rgb_decompose_act = QAction("RGB &Decompose to Files…", self)
        self._rgb_decompose_act.setToolTip(
            "Save the red, green, and blue channels of the current image as separate files.\n"
            "The chosen filename will have -red, -green, and -blue appended before the extension."
        )
        self._rgb_decompose_act.triggered.connect(self._run_rgb_decompose)
        colours_menu.addAction(self._rgb_decompose_act)

        self._rgb_compose_act = QAction("RGB &Compose from Files…", self)
        self._rgb_compose_act.setToolTip(
            "Combine 2 or 3 separate channel image files into a single colour image."
        )
        self._rgb_compose_act.triggered.connect(self._run_rgb_compose)
        colours_menu.addAction(self._rgb_compose_act)
        colours_menu.setToolTipsVisible(True)

        # self._lum_act = QAction("OKLab &Luminance", self)
        # self._lum_act.triggered.connect(self._run_luminance)
        # colours_menu.addAction(self._lum_act)
        #
        # colours_menu.addSeparator()
        # self._decompose_act = QAction("OKLab &Decompose…", self)
        # self._decompose_act.triggered.connect(self._run_decompose)
        # colours_menu.addAction(self._decompose_act)
        #
        # self._compose_act = QAction("OKLab &Compose…", self)
        # self._compose_act.triggered.connect(self._run_compose)
        # colours_menu.addAction(self._compose_act)
        # self._compose_act.setEnabled(True)

        view_menu = self.menuBar().addMenu("&View")
        zoom_in = QAction("Zoom &In", self)
        zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in.setToolTip("Increase the zoom level.")
        zoom_in.triggered.connect(lambda: self._canvas.set_zoom(self._canvas.zoom * 1.25))
        view_menu.addAction(zoom_in)

        zoom_out = QAction("Zoom &Out", self)
        zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out.setToolTip("Decrease the zoom level.")
        zoom_out.triggered.connect(lambda: self._canvas.set_zoom(self._canvas.zoom / 1.25))
        view_menu.addAction(zoom_out)

        actual = QAction("&Actual Size (100%)", self)
        actual.setToolTip("Reset the zoom level to 100%.")
        actual.triggered.connect(lambda: self._canvas.set_zoom(1.0))
        view_menu.addAction(actual)
        view_menu.setToolTipsVisible(True)

        help_menu = self.menuBar().addMenu("&Help")
        about_act = QAction("&About Planetary Tools", self)
        about_act.setToolTip("Show version and author information.")
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)
        help_menu.setToolTipsVisible(True)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Planetary Tools",
            f"Planetary Tools {__version__}\nby Con Kolivas <kernel@kolivas.org>",
        )

    def _update_actions(self) -> None:
        has_doc = self._document is not None
        for act in (
            self._save_act, self._save_as_act, self._scale_act,
            self._sharpen_act, self._denoise_act, self._deconv_act,
            self._merge_detail_act,
            self._stretch_act, self._colour_matrix_act, self._saturation_act,
            self._levels_act, self._rgb_decompose_act,
            # self._lum_act, self._decompose_act,
        ):
            act.setEnabled(has_doc)
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        self._undo_act.setEnabled(self._undo.stack.can_undo())
        self._redo_act.setEnabled(self._undo.stack.can_redo())
        if self._undo.stack.can_undo():
            self._undo_act.setText(f"Undo {self._undo.stack.undo_label()}")
        else:
            self._undo_act.setText("Undo")
        if self._undo.stack.can_redo():
            self._redo_act.setText(f"Redo {self._undo.stack.redo_label()}")
        else:
            self._redo_act.setText("Redo")

    def _on_preview_busy(self, busy: bool) -> None:
        if busy:
            self._status.showMessage("Updating preview…")

    def _on_preview_failed(self, message: str) -> None:
        self._status.showMessage(f"Preview failed: {message}")

    def _refresh_canvas_display(self) -> None:
        if self._document is None:
            return
        if self._preview.is_active:
            data = self._preview.display_data()
            if data is not None:
                rgb = array_to_display_rgb(data, self._document.is_grayscale)
                self._canvas.show_rgb_uint8(rgb)
                return
        self._canvas.refresh()

    def _sync_zoom_combo(self, zoom: float) -> None:
        pct = f"{int(round(zoom * 100))}%"
        idx = self._zoom_combo.findText(pct)
        if idx >= 0:
            self._zoom_combo.blockSignals(True)
            self._zoom_combo.setCurrentIndex(idx)
            self._zoom_combo.blockSignals(False)

    def _on_zoom_combo(self, index: int) -> None:
        if index < 0:
            return
        zoom = self._zoom_combo.itemData(index)
        if zoom is not None:
            self._canvas.set_zoom(float(zoom))

    def _set_document(self, doc: ImageDocument) -> None:
        self._document = doc
        self._undo.clear()
        self._canvas.set_document(doc)
        self._canvas.set_zoom(1.0)
        self._zoom_combo.setCurrentText("100%")
        self.setWindowTitle(f"Planetary Tools — {doc.title()}")
        self._update_actions()
        self._status.showMessage(
            f"{doc.width}×{doc.height}  {'Greyscale' if doc.is_grayscale else 'RGB'}  "
            f"32-bit float linear"
        )

    def _file_filter(self) -> str:
        exts = " ".join(f"*{e}" for e in supported_extensions())
        return f"Images ({exts});;All Files (*)"

    def _save_as_filters(self) -> str:
        return (
            "TIFF float32 (*.tif *.tiff);;"
            "TIFF 16-bit (*.tif *.tiff);;"
            "PNG 16-bit (*.png);;"
            "PNG 8-bit (*.png);;"
            "JPEG (*.jpg *.jpeg);;"
            "All Files (*)"
        )

    def _default_save_as_filter(self) -> str:
        remembered = last_save_filter()
        if remembered:
            return remembered
        if self._document is None:
            return "PNG 16-bit (*.png)"
        if self._document.storage_bits <= 8:
            return "PNG 8-bit (*.png)"
        return "PNG 16-bit (*.png)"

    def _bit_depth_for_save(self, path: str, selected_filter: str) -> int:
        sel = selected_filter.lower()
        if "float32" in sel:
            return 32
        if "png 8-bit" in sel or ("png" in sel and "8-bit" in sel):
            return 8
        if "png 16-bit" in sel or ("png" in sel and "16-bit" in sel):
            return 16
        if "tiff 16-bit" in sel or ("tiff" in sel and "16-bit" in sel):
            return 16
        if "jpeg" in sel:
            return 8
        suffix = Path(path).suffix.lower()
        if suffix in {".tif", ".tiff"}:
            return 16
        if suffix == ".png" and self._document is not None:
            bits = self._document.storage_bits
            return bits if bits in (8, 16) else 16
        return 8

    def _save_as_dialog_options(self) -> QFileDialog.Option:
        # Windows/macOS native dialogs merge filters that share the same
        # extension, so PNG 8-bit and PNG 16-bit collapse to a single entry.
        return QFileDialog.Option.DontUseNativeDialog

    def _write_document(self, path: str, bit_depth: int) -> None:
        if self._document is None:
            return
        save_image(self._document, path, bit_depth=bit_depth)
        self.setWindowTitle(f"Planetary Tools — {self._document.title()}")
        self._status.showMessage(f"Saved {path}")

    def _populate_recent_menu(self) -> None:
        self._recent_menu.clear()
        paths = list_recent()
        if not paths:
            empty = self._recent_menu.addAction("No recent files")
            empty.setEnabled(False)
            return
        for path in paths:
            label = Path(path).name
            if not Path(path).is_file():
                label = f"{label} (not found)"
            action = self._recent_menu.addAction(label)
            action.setToolTip(path)
            action.setEnabled(Path(path).is_file())
            action.triggered.connect(
                lambda _checked=False, p=path: self._open_recent(p),
            )

    def _open_path(self, path: str | Path, *, confirm_unsaved: bool = True) -> bool:
        if confirm_unsaved and not self._confirm_unsaved_changes(
            "before opening another file"
        ):
            return False
        try:
            doc = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return False
        self._set_document(doc)
        add_recent(path)
        return True

    def _open_recent(self, path: str) -> None:
        if not Path(path).is_file():
            remove_recent(path)
            self._populate_recent_menu()
            QMessageBox.warning(self, "Open failed", f"File not found:\n{path}")
            return
        self._open_path(path)

    def _open_file(self) -> None:
        if not self._confirm_unsaved_changes("before opening another file"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            last_open_directory(),
            self._file_filter(),
        )
        if not path:
            return
        self._open_path(path, confirm_unsaved=False)

    def _try_save_document_as(self) -> bool:
        if self._document is None:
            return True
        selected_filter = self._default_save_as_filter()
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            last_save_directory(),
            self._save_as_filters(),
            selected_filter,
            options=self._save_as_dialog_options(),
        )
        if not path:
            return False
        if selected:
            selected_filter = selected
        if not Path(path).suffix:
            if "png" in selected_filter.lower():
                path += ".png"
            elif "jpeg" in selected_filter.lower() or "jpg" in selected_filter.lower():
                path += ".jpg"
            elif "tiff" in selected_filter.lower() or "tif" in selected_filter.lower():
                path += ".tif"
        bit_depth = self._bit_depth_for_save(path, selected_filter)
        try:
            self._write_document(path, bit_depth)
            remember_save_path(path)
            remember_save_filter(selected_filter)
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False

    def _try_save_document(self) -> bool:
        if self._document is None:
            return True
        if self._document.path is None:
            return self._try_save_document_as()
        try:
            save_image(self._document, self._document.path)
            self.setWindowTitle(f"Planetary Tools — {self._document.title()}")
            self._status.showMessage(f"Saved {self._document.path}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False

    def _save_file(self) -> None:
        if self._document is None:
            return
        if self._document.path is None:
            self._try_save_document_as()
            return
        try:
            save_image(self._document, self._document.path)
            self.setWindowTitle(f"Planetary Tools — {self._document.title()}")
            self._status.showMessage(f"Saved {self._document.path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _save_file_as(self) -> None:
        self._try_save_document_as()

    def _confirm_unsaved_changes(self, action: str) -> bool:
        if self._document is None or not self._document.modified:
            return True

        name = self._document.path.name if self._document.path else "Untitled"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText(f'Save changes to "{name}" {action}?')
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        save_as_btn = box.addButton("Save As…", QMessageBox.ButtonRole.ActionRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked == cancel_btn:
            return False
        if clicked == discard_btn:
            return True
        if clicked == save_btn:
            return self._try_save_document()
        if clicked == save_as_btn:
            return self._try_save_document_as()
        return False

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._confirm_unsaved_changes("before closing"):
            event.accept()
        else:
            event.ignore()

    def _clear_filter_panel(self) -> None:
        while self._filter_host_layout.count():
            item = self._filter_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _on_filter_dock_visibility(self, visible: bool) -> None:
        if (
            not visible
            and self._filter_dialog_open
            and self._filter_loop is not None
            and self._filter_loop.isRunning()
        ):
            self._filter_accepted = False
            self._filter_loop.quit()

    def _run_filter_dialog(self, dlg: _FilterDialog, label: str) -> None:
        """Show filter controls in the dock with optional live canvas preview."""
        if self._document is None or self._filter_dialog_open:
            return

        self._filter_dialog_open = True
        self._filter_accepted = False
        try:
            self._clear_filter_panel()
            self._filter_host_layout.addWidget(dlg)
            self._filter_dock.setWindowTitle(label)
            self._filter_dock.show()

            dlg.set_input_brightness(self._document.data, self._document.is_grayscale)

            self._preview.start(self._document.data, self._document.is_grayscale)
            self._preview.set_filter_func(dlg.build_filter_func())
            self._preview.set_preview_enabled(dlg.preview.isChecked())

            dlg.params_changed.connect(self._on_dialog_params_changed)
            dlg.preview_now.connect(self._preview.update_now)
            dlg.preview_toggled.connect(self._preview.set_preview_enabled)
            dlg.preview_toggled.connect(self._refresh_canvas_display)
            dlg.preview_toggled.connect(self._update_dialog_brightness)
            self._preview.preview_updated.connect(self._update_dialog_brightness)
            self._filter_dock.visibilityChanged.connect(self._on_filter_dock_visibility)

            def on_accept() -> None:
                self._filter_accepted = True
                if self._filter_loop is not None:
                    self._filter_loop.quit()

            def on_reject() -> None:
                self._filter_accepted = False
                if self._filter_loop is not None:
                    self._filter_loop.quit()

            dlg.accepted.connect(on_accept)
            dlg.rejected.connect(on_reject)

            self._active_filter_dlg = dlg
            self._preview.update_now()
            self._update_dialog_brightness()

            self._filter_loop = QEventLoop(self)
            self._filter_loop.exec()
            self._filter_loop = None

            dlg.params_changed.disconnect(self._on_dialog_params_changed)
            dlg.preview_now.disconnect(self._preview.update_now)
            dlg.preview_toggled.disconnect(self._preview.set_preview_enabled)
            dlg.preview_toggled.disconnect(self._refresh_canvas_display)
            dlg.preview_toggled.disconnect(self._update_dialog_brightness)
            self._preview.preview_updated.disconnect(self._update_dialog_brightness)
            self._filter_dock.visibilityChanged.disconnect(self._on_filter_dock_visibility)
            dlg.accepted.disconnect(on_accept)
            dlg.rejected.disconnect(on_reject)
            self._active_filter_dlg = None

            if self._filter_accepted:
                if getattr(dlg, "supports_presets", True):
                    dlg.save_last_preset()
                result = self._preview.finish(apply=True)
                if result is not None:
                    self._commit_filter(label, result)
                else:
                    self._preview.finish(apply=False)
                    self._canvas.refresh()
            else:
                self._preview.finish(apply=False)
                self._canvas.refresh()
        finally:
            self._clear_filter_panel()
            self._filter_dock.hide()
            self._filter_dialog_open = False

    def _update_dialog_brightness(self) -> None:
        dlg = getattr(self, "_active_filter_dlg", None)
        if dlg is None or self._document is None or not dlg.filter_id:
            return
        preview_on = self._preview.is_active and dlg.preview.isChecked()
        if not preview_on:
            dlg.update_output_brightness(None)
            dlg.update_histogram_display(None)
            return
        original = self._preview.original_data()
        if original is None:
            return
        stats = output_filter_stats(
            dlg.filter_id,
            original,
            self._document.is_grayscale,
            dlg.get_params(),
        )
        dlg.update_output_brightness(
            stats.brightness,
            stats.brightness_increase_pct,
            stats.noise_level,
        )
        preview_data = self._preview.display_data()
        if preview_data is not None:
            dlg.update_histogram_display(preview_data)
        else:
            dlg.update_histogram_display(None)

    def _on_dialog_params_changed(self) -> None:
        sender = self.sender()
        if isinstance(sender, _FilterDialog):
            self._preview.set_filter_func(sender.build_filter_func())
        self._preview.schedule_update()

    def _commit_filter(self, label: str, result: np.ndarray) -> None:
        if self._document is None:
            return
        self._undo.record(
            self._document.data,
            self._document.is_grayscale,
            label,
        )
        try:
            self._document.set_data(result)
            self._canvas.refresh()
            self.setWindowTitle(f"Planetary Tools — {self._document.title()}")
            self._status.showMessage(f"Applied {label}")
            self._update_undo_actions()
        except Exception as exc:
            QMessageBox.critical(self, "Filter failed", str(exc))
            restored = self._undo.stack.undo(
                self._document.data,
                self._document.is_grayscale,
                "",
            )
            if restored:
                data, grayscale = restored
                self._document.set_data(data, grayscale=grayscale)
                self._canvas.refresh()
            self._update_undo_actions()

    def _run_wavelet_sharpen(self) -> None:
        self._run_filter_dialog(WaveletSharpenDialog(self), "Wavelet Sharpen")

    def _run_wavelet_denoise(self) -> None:
        self._run_filter_dialog(WaveletDenoiseDialog(self), "Wavelet Denoise")

    def _run_adaptive_deconv(self) -> None:
        if self._document is None:
            return
        self._run_filter_dialog(
            AdaptiveDeconvDialog(self._document.is_grayscale, self),
            "Adaptive Deconvolution",
        )

    def _run_wiener_deconv(self) -> None:
        if self._document is None:
            return
        self._run_filter_dialog(
            WienerDeconvDialog(self._document.is_grayscale, self),
            "Wiener Deconvolution",
        )

    def _run_merge_wavelet_detail(self) -> None:
        if self._document is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load secondary image for wavelet detail merge",
            last_open_directory(),
            self._file_filter(),
        )
        if not path:
            return
        try:
            sec_doc = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        remember_open_path(path)
        self._run_filter_dialog(
            MergeWaveletDetailDialog(path, sec_doc.data, self),
            "Merge Wavelet Detail",
        )

    def _run_stretch(self) -> None:
        if self._document is None or self._document.is_grayscale:
            QMessageBox.information(self, "Stretch Contrast OKLab", "This filter requires an RGB image.")
            return
        self._run_filter_dialog(StretchContrastDialog(self), "Stretch Contrast OKLab")

    def _run_colour_matrix(self) -> None:
        if self._document is None or self._document.is_grayscale:
            QMessageBox.information(
                self, "Colour Correction Matrix", "This filter requires an RGB image."
            )
            return
        self._run_filter_dialog(ColourMatrixDialog(self), "Colour Correction Matrix")

    def _run_saturation_vibrance(self) -> None:
        if self._document is None or self._document.is_grayscale:
            QMessageBox.information(
                self, "Saturation & Vibrance", "This filter requires an RGB image."
            )
            return
        self._run_filter_dialog(SaturationVibranceDialog(self), "Saturation & Vibrance")

    def _run_levels(self) -> None:
        if self._document is None or self._document.is_grayscale:
            QMessageBox.information(self, "Levels", "This filter requires an RGB image.")
            return
        self._run_filter_dialog(LevelsDialog(self), "Levels")

    def _run_rgb_decompose(self) -> None:
        if self._document is None:
            return
        selected_filter = self._default_save_as_filter()
        path, selected = QFileDialog.getSaveFileName(
            self,
            "RGB Decompose to Files",
            last_save_directory(),
            self._save_as_filters(),
            selected_filter,
            options=self._save_as_dialog_options(),
        )
        if not path:
            return
        if selected:
            selected_filter = selected
        if not Path(path).suffix:
            if "png" in selected_filter.lower():
                path += ".png"
            elif "jpeg" in selected_filter.lower() or "jpg" in selected_filter.lower():
                path += ".jpg"
            elif "tiff" in selected_filter.lower() or "tif" in selected_filter.lower():
                path += ".tif"
        bit_depth = self._bit_depth_for_save(path, selected_filter)
        base = Path(path)
        out_paths = {
            name: base.with_name(f"{base.stem}-{name}{base.suffix}")
            for name in ("red", "green", "blue")
        }
        existing = [str(p) for p in out_paths.values() if p.exists()]
        if existing:
            reply = QMessageBox.question(
                self,
                "RGB Decompose to Files",
                "The following file(s) already exist and will be overwritten:\n\n"
                + "\n".join(existing)
                + "\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            for name, idx in (("red", 0), ("green", 1), ("blue", 2)):
                save_channel(self._document.data[..., idx], out_paths[name], bit_depth=bit_depth)
            remember_save_path(base)
            remember_save_filter(selected_filter)
            self._status.showMessage(f"Saved RGB channels to {base.parent}")
        except Exception as exc:
            QMessageBox.critical(self, "RGB Decompose failed", str(exc))

    def _run_rgb_compose(self) -> None:
        if not self._confirm_unsaved_changes("before composing a new image"):
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 2 or 3 channel images (Red, Green, Blue)",
            last_open_directory(),
            self._file_filter(),
        )
        if not paths:
            return
        if len(paths) not in (2, 3):
            QMessageBox.warning(
                self,
                "RGB Compose from Files",
                "Select either two or three channel images.",
            )
            return
        remember_open_path(paths[0])
        paths = [Path(p) for p in paths]
        guesses = [detect_channel(p) for p in paths]
        dlg = RGBComposeDialog(paths, guesses, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        assignment = dlg.channel_assignment()
        try:
            channels: dict[str, np.ndarray] = {}
            for name, path in assignment.items():
                channels[name] = load_image(path).data[..., 0]
            if len({arr.shape for arr in channels.values()}) != 1:
                raise ValueError("All channel images must have identical dimensions.")
            if dlg.align_channels() and len(channels) > 1:
                reference_name = next(
                    name for name in _ALIGN_PRIORITY if name in channels
                )
                reference = channels[reference_name]
                for name in list(channels.keys()):
                    if name != reference_name:
                        channels[name] = align_channel(reference, channels[name])
            if len(channels) == 2:
                missing = ({"Red", "Green", "Blue"} - channels.keys()).pop()
                name_a, name_b = channels.keys()
                w_a, w_b = _LUMA_WEIGHTS[name_a], _LUMA_WEIGHTS[name_b]
                calculated = (
                    w_a * channels[name_a] + w_b * channels[name_b]
                ) / (w_a + w_b)
                channels[missing] = calculated.astype(np.float32)
            composed = np.stack(
                [channels["Red"], channels["Green"], channels["Blue"]], axis=-1
            ).astype(np.float32)
            doc = ImageDocument(data=composed, is_grayscale=False, modified=True)
            self._set_document(doc)
            self._status.showMessage("Composed RGB image from files")
        except Exception as exc:
            QMessageBox.critical(self, "RGB Compose failed", str(exc))

    # def _run_luminance(self) -> None:
    #     if self._document is None or self._document.is_grayscale:
    #         QMessageBox.information(self, "OKLab Luminance", "This filter requires an RGB image.")
    #         return
    #     self._run_filter_dialog(
    #         InstantFilterDialog("OKLab Luminance", "oklab_luminance", self),
    #         "OKLab Luminance",
    #     )

    def _run_batch(self) -> None:
        dlg = BatchDialog(self)
        dlg.exec()

    # def _run_decompose(self) -> None:
    #     if self._document is None or self._document.is_grayscale:
    #         QMessageBox.information(self, "OKLab Decompose", "This filter requires an RGB image.")
    #         return
    #     channels = oklab_decompose(self._document.data)
    #     self._document.oklab_channels = channels
    #     folder = QFileDialog.getExistingDirectory(self, "Save OKLab channels to folder")
    #     if folder:
    #         import tifffile
    #         base = self._document.path.stem if self._document.path else "image"
    #         for name, arr in channels.items():
    #             out = Path(folder) / f"{base}_oklab_{name}.tif"
    #             tifffile.imwrite(out, arr.astype(np.float32))
    #         self._status.showMessage(f"Saved OKLab channels to {folder}")
    #     else:
    #         self._status.showMessage("OKLab channels stored in memory — use Compose or save to folder")
    #
    # def _run_compose(self) -> None:
    #     paths, _ = QFileDialog.getOpenFileNames(
    #         self, "Select OKLab L, a, and b channel images", "",
    #         "TIFF (*.tif *.tiff);;All Files (*)",
    #     )
    #     if len(paths) != 3:
    #         QMessageBox.warning(self, "OKLab Compose", "Select exactly three channel images (L, a, b).")
    #         return
    #     try:
    #         import tifffile
    #         channels = [tifffile.imread(p).astype(np.float32) for p in paths]
    #         if not all(c.shape == channels[0].shape for c in channels):
    #             raise ValueError("All three channels must have identical dimensions.")
    #         composed = oklab_compose(channels[0], channels[1], channels[2])
    #         doc = ImageDocument(data=composed, is_grayscale=False, modified=True)
    #         self._set_document(doc)
    #         self._status.showMessage("Composed image from OKLab channels")
    #     except Exception as exc:
    #         QMessageBox.critical(self, "Compose failed", str(exc))

    def _undo_action(self) -> None:
        if self._document is None:
            return
        result = self._undo.stack.undo(
            self._document.data,
            self._document.is_grayscale,
            "",
        )
        if result is None:
            return
        data, grayscale = result
        self._document.set_data(data, grayscale=grayscale)
        self._canvas.refresh()
        self._update_undo_actions()
        self.setWindowTitle(f"Planetary Tools — {self._document.title()}")

    def _redo_action(self) -> None:
        if self._document is None:
            return
        result = self._undo.stack.redo(
            self._document.data,
            self._document.is_grayscale,
            "",
        )
        if result is None:
            return
        data, grayscale = result
        self._document.set_data(data, grayscale=grayscale)
        self._canvas.refresh()
        self._update_undo_actions()
        self.setWindowTitle(f"Planetary Tools — {self._document.title()}")

    def _run_scale_image(self) -> None:
        if self._document is None:
            return
        dlg = ScaleImageDialog(
            self._document.width,
            self._document.height,
            self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        new_w, new_h = dlg.output_size()
        if new_w == self._document.width and new_h == self._document.height:
            return
        try:
            result = scale_image(self._document.data, new_w, new_h)
        except Exception as exc:
            QMessageBox.critical(self, "Scale Image", str(exc))
            return
        self._undo.record(
            self._document.data,
            self._document.is_grayscale,
            "Scale Image",
        )
        self._document.set_data(result)
        self._canvas.refresh()
        self._update_undo_actions()
        self.setWindowTitle(f"Planetary Tools — {self._document.title()}")
        self._status.showMessage(
            f"{new_w}×{new_h}  "
            f"{'Greyscale' if self._document.is_grayscale else 'RGB'}  "
            f"32-bit float linear"
        )


def run_app(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    app = QApplication(argv)
    app.setApplicationName("Planetary Tools")
    app.setOrganizationName("PlanetaryTools")
    window = MainWindow()
    window.show()
    if len(argv) > 1 and not argv[1].startswith("-"):
        window._open_path(argv[1])
    return app.exec()
