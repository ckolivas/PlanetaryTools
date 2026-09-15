"""Animate — write GIF / APNG / WebP animations or MP4 video from stills."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.core.animate import (
    FORMAT_SUFFIX,
    AnimationResult,
    apply_format_suffix,
    duration_ms,
    natural_sort_key,
    write_animation,
)
from planetary_tools.core.animation_interpolation import (
    build_timeline, filename_timestamp, rounded_timestamp,
)
from planetary_tools.io.loader import supported_extensions
from planetary_tools.ui.file_filters import image_file_filters
from planetary_tools.ui.recent_files import (
    last_output_option,
    last_output_number,
    remember_output_option,
    last_open_directory,
    last_save_directory,
    remember_open_path,
    remember_save_path,
)

_COL_FILE = 0
_FORMAT_FILTERS = {
    "gif": "GIF (*.gif)",
    "apng": "Animated PNG (*.png)",
    "webp": "WebP (*.webp)",
    "mp4": "MP4 video (*.mp4)",
}


class _RunWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[Path],
        output: Path,
        fps: float,
        fmt: str,
        gif_quality: str,
        back_and_forth: bool,
        mp4_crf: int = 0,
        *,
        motion_interpolation: bool = False,
        frame_interval_minutes: float = 1.0,
    ) -> None:
        super().__init__()
        self._paths = paths
        self._output = output
        self._fps = fps
        self._fmt = fmt
        self._gif_quality = gif_quality
        self._back_and_forth = back_and_forth
        self._mp4_crf = mp4_crf
        self._motion_interpolation = motion_interpolation
        self._frame_interval_minutes = frame_interval_minutes

    def run(self) -> None:
        try:
            result = write_animation(
                self._paths,
                self._output,
                fps=self._fps,
                fmt=self._fmt,
                gif_quality=self._gif_quality,
                back_and_forth=self._back_and_forth,
                mp4_crf=self._mp4_crf,
                motion_interpolation=self._motion_interpolation,
                frame_interval_minutes=self._frame_interval_minutes,
                on_progress=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class AnimateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Animate")
        self.setMinimumWidth(560)
        self._paths: list[Path] = []
        self._worker: QThread | None = None
        self._auto_output = True

        root = QVBoxLayout(self)
        description = QLabel(
            "Build an animation or video from stills. Frames are sorted by "
            "filename, or by capture time when interpolating. Smaller frames "
            "are centred on a black canvas that fits the largest."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        pick = QHBoxLayout()
        btn_add = QPushButton("Add files…")
        btn_add.clicked.connect(self._add_files)
        btn_folder = QPushButton("Add folder…")
        btn_folder.clicked.connect(self._pick_folder)
        pick.addWidget(btn_add)
        pick.addWidget(btn_folder)
        pick.addStretch()
        fl.addLayout(pick)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File", "Captured (UTC)", "Rounded (UTC)"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            self._table.setColumnHidden(column, True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(140)
        fl.addWidget(self._table)

        order = QHBoxLayout()
        btn_up = QPushButton("Move up")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down = QPushButton("Move down")
        btn_down.clicked.connect(lambda: self._move(1))
        self._order_buttons = (btn_up, btn_down)
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        order.addWidget(btn_up)
        order.addWidget(btn_down)
        order.addWidget(btn_remove)
        order.addWidget(btn_clear)
        order.addStretch()
        fl.addLayout(order)
        root.addWidget(files)

        opts = QGroupBox("Animation")
        of = QFormLayout(opts)
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.1, 60.0)
        self._fps.setDecimals(1)
        self._fps.setSingleStep(0.1)
        self._fps.setValue(last_output_number("animationFps", 10.0, 0.1, 60.0))
        self._fps.setSuffix(" fps")
        self._fps.valueChanged.connect(self._update_delay_hint)
        self._fps.valueChanged.connect(lambda value: remember_output_option("animationFps", value))
        of.addRow("Frame rate", self._fps)

        self._motion_interpolation = QCheckBox("Generate frames with motion interpolation")
        self._motion_interpolation.setChecked(
            last_output_option("animationMotionInterpolation", "0", ("0", "1")) == "1"
        )
        self._motion_interpolation.setToolTip(
            "Read WinJUPOS or PVOL filename timestamps, sort chronologically, and generate "
            "missing frames using built-in motion interpolation. Use aligned images for best results."
        )
        of.addRow(self._motion_interpolation)
        self._frame_interval = QDoubleSpinBox()
        self._frame_interval.setRange(0.1, 1440.0)
        self._frame_interval.setDecimals(1)
        self._frame_interval.setSingleStep(1.0)
        self._frame_interval.setValue(last_output_number("animationFrameInterval", 1.0, 0.1, 1440.0))
        self._frame_interval.setSuffix(" min")
        self._frame_interval.setKeyboardTracking(False)
        self._frame_interval.setEnabled(False)
        self._frame_interval.setToolTip(
            "Observation time between generated frames; playback speed is set by Frame rate. "
            "Capture times round to the nearest interval (halfway rounds up). "
            "Files rounding to the same time need a smaller interval or one file removed."
        )
        of.addRow("Frame interval", self._frame_interval)
        self._timing_hint = QLabel("")
        self._timing_hint.setWordWrap(True)
        self._timing_hint.setMinimumHeight(self._timing_hint.fontMetrics().lineSpacing() * 3 + 8)
        self._timing_hint.hide()
        of.addRow(self._timing_hint)
        self._motion_interpolation.toggled.connect(self._on_interpolation_changed)
        self._motion_interpolation.toggled.connect(
            lambda enabled: remember_output_option("animationMotionInterpolation", int(enabled))
        )
        self._frame_interval.valueChanged.connect(lambda _: self._refresh_table())
        self._frame_interval.valueChanged.connect(
            lambda value: remember_output_option("animationFrameInterval", value)
        )

        self._back_and_forth = QCheckBox("Back and forth")
        self._back_and_forth.setChecked(True)
        self._back_and_forth.setToolTip(
            "After the last frame, play the sequence in reverse (without "
            "repeating the first or last frame) so the loop does not jump."
        )
        of.addRow(self._back_and_forth)

        self._format = QComboBox()
        self._format.addItem("GIF", "gif")
        self._format.addItem("Animated PNG", "apng")
        self._format.addItem("WebP", "webp")
        self._format.addItem("MP4 video (H.264 RGB)", "mp4")
        self._format.setItemData(self._format.findData("mp4"),
            "Playback requires a player supporting H.264 RGB (4:4:4).",
            Qt.ItemDataRole.ToolTipRole)
        fmt = last_output_option("animationFormat", "gif", tuple(_FORMAT_FILTERS))
        self._format.setCurrentIndex(self._format.findData(fmt))
        self._format.currentIndexChanged.connect(self._on_format_changed)
        of.addRow("Format", self._format)

        self._gif_quality = QComboBox()
        self._gif_quality.addItem("Best", "best")
        self._gif_quality.addItem("High", "high")
        self._gif_quality.addItem("Medium", "medium")
        self._gif_quality.addItem("Low", "low")
        self._gif_quality.setToolTip(
            "GIF only. Best uses 256 colours with Floyd–Steinberg dither "
            "and a palette per frame. Lower presets cut colours (and High "
            "drops dither) for a smaller file."
        )
        of.addRow("GIF quality", self._gif_quality)

        self._mp4_crf = QSpinBox()
        self._mp4_crf.setRange(0, 51)
        self._mp4_crf.setValue(0)
        self._mp4_crf.setSpecialValueText("0 (lossless)")
        self._mp4_crf.setKeyboardTracking(False)
        self._mp4_crf.setToolTip(
            "MP4 constant quality (CRF): 0 is lossless; higher values give "
            "lower quality and smaller files. Lossless preserves the 8-bit "
            "RGB animation frames, without colour subsampling."
        )
        of.addRow("MP4 constant quality", self._mp4_crf)

        self._delay_hint = QLabel("")
        self._delay_hint.setWordWrap(True)
        of.addRow(self._delay_hint)
        root.addWidget(opts)

        out = QGroupBox("Output")
        ouf = QFormLayout(out)
        out_row = QHBoxLayout()
        self._output = QLineEdit()
        self._output.textEdited.connect(self._on_output_edited)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._output)
        out_row.addWidget(browse)
        ouf.addRow("File", out_row)
        root.addWidget(out)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        root.addWidget(self._progress)
        root.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        buttons.addButton(self._run_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._on_format_changed()
        self._on_interpolation_changed(self._motion_interpolation.isChecked())

    def _fmt(self) -> str:
        return str(self._format.currentData())

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _add_files(self) -> None:
        if self._busy():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add animation frames", last_open_directory(), image_file_filters()
        )
        if not paths:
            return
        remember_open_path(paths[0])
        self._append_paths([Path(raw) for raw in paths])

    def _append_paths(self, paths: list[Path]) -> None:
        existing = {p.resolve() for p in self._paths}
        for path in paths:
            key = path.resolve()
            if key in existing:
                continue
            self._paths.append(path)
            existing.add(key)
        self._paths.sort(key=natural_sort_key)
        self._refresh_table()
        self._maybe_default_output()

    def _pick_folder(self) -> None:
        if self._busy():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of frames", last_open_directory()
        )
        if not folder:
            return
        remember_open_path(folder)
        exts = {e.lower() for e in supported_extensions()}
        paths = [
            p
            for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
        paths.sort(key=natural_sort_key)
        self._append_paths(paths)

    def _move(self, delta: int) -> None:
        if self._busy() or self._motion_interpolation.isChecked():
            return
        row = self._table.currentRow()
        dest = row + delta
        if row < 0 or dest < 0 or dest >= len(self._paths):
            return
        self._paths[row], self._paths[dest] = self._paths[dest], self._paths[row]
        self._refresh_table()
        self._table.setCurrentCell(dest, _COL_FILE)

    def _remove(self) -> None:
        if self._busy():
            return
        row = self._table.currentRow()
        if row < 0 or row >= len(self._paths):
            return
        del self._paths[row]
        self._refresh_table()
        if self._paths:
            self._table.setCurrentCell(min(row, len(self._paths) - 1), _COL_FILE)
        elif self._auto_output:
            self._output.clear()

    def _clear(self) -> None:
        if self._busy():
            return
        self._paths = []
        self._refresh_table()
        if self._auto_output:
            self._output.clear()

    def _refresh_table(self) -> None:
        motion = self._motion_interpolation.isChecked()
        interval = self._frame_interval.value()
        timing = "Add at least two timestamped images."
        if motion and len(self._paths) >= 2:
            try:
                timeline = build_timeline(self._paths, interval)
                self._paths = [frame.path for frame in timeline.sources]
                timing = (f"{timeline.frame_count} forward frames: {len(self._paths)} originals + "
                          f"{timeline.frame_count-len(self._paths)} generated, {interval:g} min apart. "
                          "Times round to the nearest interval; halfway rounds up.")
            except ValueError as exc:
                timing = str(exc)
        self._timing_hint.setText(timing)
        self._table.setRowCount(len(self._paths))
        for i, path in enumerate(self._paths):
            item = QTableWidgetItem(path.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(str(path))
            self._table.setItem(i, _COL_FILE, item)
            if motion:
                try:
                    captured = filename_timestamp(path)
                    times = [captured, rounded_timestamp(captured, interval)]
                    labels = [stamp.strftime('%Y-%m-%d %H:%M:%S') for stamp in times]
                except ValueError:
                    labels = ['Unrecognised timestamp', '—']
                for column, label in enumerate(labels, 1):
                    cell = QTableWidgetItem(label)
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(i, column, cell)
        self._status.setText(f"{len(self._paths)} frame(s)")

    def _on_interpolation_changed(self, enabled: bool) -> None:
        self._frame_interval.setEnabled(enabled)
        self._timing_hint.setVisible(enabled)
        for column in (1, 2):
            self._table.setColumnHidden(column, not enabled)
        for button in self._order_buttons:
            button.setEnabled(not enabled)
        self._refresh_table()

    def _maybe_default_output(self) -> None:
        if not self._auto_output or not self._paths:
            return
        directory = self._paths[0].parent
        stem = self._paths[0].stem
        self._output.setText(str(directory / f"{stem}{FORMAT_SUFFIX[self._fmt()]}"))

    def _on_output_edited(self, _text: str) -> None:
        self._auto_output = False

    def _on_format_changed(self) -> None:
        remember_output_option("animationFormat", self._fmt())
        current = self._output.text().strip()
        if current:
            self._output.setText(str(apply_format_suffix(current, self._fmt())))
        elif self._auto_output:
            self._maybe_default_output()
        gif = self._fmt() == "gif"
        self._gif_quality.setEnabled(gif)
        self._mp4_crf.setEnabled(self._fmt() == "mp4")
        self._update_delay_hint()

    def _update_delay_hint(self) -> None:
        fmt = self._fmt()
        fps = float(self._fps.value())
        if fmt == "mp4":
            self._delay_hint.setText(
                f"Video at {fps:g} fps. Looping is controlled by the player."
            )
            return
        try:
            delay = duration_ms(fmt, fps)
        except ValueError:
            self._delay_hint.setText("")
            return
        actual = 1000.0 / delay
        if fmt == "gif" and abs(actual - fps) > 0.05:
            self._delay_hint.setText(
                f"GIF frame delay {delay} ms (≈ {actual:.1f} fps)."
            )
        else:
            self._delay_hint.setText(f"Frame delay {delay} ms.")

    def _browse_output(self) -> None:
        if self._busy():
            return
        fmt = self._fmt()
        start = self._output.text().strip()
        if not start:
            start = last_save_directory()
        path, selected = QFileDialog.getSaveFileName(
            self, "Save animation", start,
            ";;".join([*_FORMAT_FILTERS.values(), "All Files (*)"]), _FORMAT_FILTERS[fmt],
        )
        if not path:
            return
        fmt = next((key for key, value in _FORMAT_FILTERS.items() if value == selected), fmt)
        if selected == "All Files (*)":
            fmt = next((key for key, ext in FORMAT_SUFFIX.items()
                        if Path(path).suffix.lower() == ext), fmt)
        self._format.setCurrentIndex(self._format.findData(fmt))
        out = apply_format_suffix(path, fmt)
        if not out.suffix:
            out = out.with_suffix(FORMAT_SUFFIX[fmt])
        self._auto_output = False
        self._output.setText(str(out))

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)
        self._motion_interpolation.setEnabled(not running)
        self._frame_interval.setEnabled(not running and self._motion_interpolation.isChecked())

    def _run(self) -> None:
        if self._busy():
            return
        if len(self._paths) < 2:
            QMessageBox.warning(self, "Animate", "Select at least two images.")
            return
        if self._motion_interpolation.isChecked():
            try:
                build_timeline(self._paths, self._frame_interval.value())
            except ValueError as exc:
                QMessageBox.warning(self, "Animate", str(exc))
                return
        out_text = self._output.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Animate", "Choose an output file.")
            return
        output = apply_format_suffix(out_text, self._fmt())
        self._output.setText(str(output))
        if output.exists():
            reply = QMessageBox.warning(
                self,
                "Overwrite existing file?",
                f"{output}\n\nalready exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._set_running(True)
        self._status.setText("Writing…")
        self._worker = _RunWorker(
            list(self._paths),
            output,
            float(self._fps.value()),
            self._fmt(),
            str(self._gif_quality.currentData()),
            self._back_and_forth.isChecked(),
            self._mp4_crf.value(),
            motion_interpolation=self._motion_interpolation.isChecked(),
            frame_interval_minutes=self._frame_interval.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ran)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_ran(self, result: object) -> None:
        self._set_running(False)
        r = result  # type: AnimationResult
        remember_save_path(r.path)
        timing = f"{r.fps_requested:g} fps" if r.fmt == "mp4" else f"{r.duration_ms} ms/frame"
        msg = (
            f"Wrote {r.frames} frames, {r.width}×{r.height}, "
            f"{timing} → {r.path}"
        )
        self._status.setText(msg)
        parent = self.parent()
        if isinstance(parent, QMainWindow):
            parent.statusBar().showMessage(
                f"Wrote {r.frames} frames at {r.fps_requested:g} fps → {r.path}"
            )
        QMessageBox.information(self, "Animate", msg)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(min(current, total))
        self._status.setText(f"{message} ({current}/{total})")

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._status.setText("Failed")
        QMessageBox.critical(self, "Animate", message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._busy():
            QMessageBox.warning(
                self, "Animate", "Wait for the run to finish before closing."
            )
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self) -> None:
        # The Close button and Escape bypass closeEvent on a QDialog.
        if self._busy():
            QMessageBox.warning(self, "Animate", "Wait for the run to finish before closing.")
            return
        super().reject()
