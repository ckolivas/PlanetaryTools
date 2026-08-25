"""Animate — write a looping GIF / APNG / WebP from a sequence of stills."""

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
from planetary_tools.io.loader import supported_extensions
from planetary_tools.ui.recent_files import (
    last_open_directory,
    last_save_directory,
    remember_open_path,
    remember_save_path,
)

_COL_FILE = 0


def _image_filter() -> str:
    exts = " ".join(f"*{e}" for e in supported_extensions())
    return f"Images ({exts});;All files (*)"


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
    ) -> None:
        super().__init__()
        self._paths = paths
        self._output = output
        self._fps = fps
        self._fmt = fmt
        self._gif_quality = gif_quality
        self._back_and_forth = back_and_forth

    def run(self) -> None:
        try:
            result = write_animation(
                self._paths,
                self._output,
                fps=self._fps,
                fmt=self._fmt,
                gif_quality=self._gif_quality,
                back_and_forth=self._back_and_forth,
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
        root.addWidget(
            QLabel(
                "Build a looping animation from stills. Frames are sorted by "
                "filename; use Move up / Move down to reorder. Smaller frames "
                "are centred on a black canvas that fits the largest."
            )
        )

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        pick = QHBoxLayout()
        btn_add = QPushButton("Add files…")
        btn_add.clicked.connect(self._add_files)
        btn_folder = QPushButton("Select folder…")
        btn_folder.clicked.connect(self._pick_folder)
        pick.addWidget(btn_add)
        pick.addWidget(btn_folder)
        pick.addStretch()
        fl.addLayout(pick)

        self._table = QTableWidget(0, 1)
        self._table.setHorizontalHeaderLabels(["File"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        fl.addWidget(self._table)

        order = QHBoxLayout()
        btn_up = QPushButton("Move up")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down = QPushButton("Move down")
        btn_down.clicked.connect(lambda: self._move(1))
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
        self._fps.setValue(10.0)
        self._fps.setSuffix(" fps")
        self._fps.valueChanged.connect(self._update_delay_hint)
        of.addRow("Frame rate", self._fps)

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

        self._update_delay_hint()

    def _fmt(self) -> str:
        return str(self._format.currentData())

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _add_files(self) -> None:
        if self._busy():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add animation frames", last_open_directory(), _image_filter()
        )
        if not paths:
            return
        remember_open_path(paths[0])
        existing = {p.resolve() for p in self._paths}
        for raw in paths:
            path = Path(raw)
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
            self, "Select folder of frames", last_open_directory()
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
        self._paths = paths
        self._auto_output = True
        self._refresh_table()
        self._maybe_default_output()

    def _move(self, delta: int) -> None:
        if self._busy():
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
        self._table.setRowCount(len(self._paths))
        for i, path in enumerate(self._paths):
            item = QTableWidgetItem(path.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(str(path))
            self._table.setItem(i, _COL_FILE, item)
        self._status.setText(f"{len(self._paths)} frame(s)")

    def _maybe_default_output(self) -> None:
        if not self._auto_output or not self._paths:
            return
        directory = self._paths[0].parent
        stem = self._paths[0].stem
        self._output.setText(str(directory / f"{stem}{FORMAT_SUFFIX[self._fmt()]}"))

    def _on_output_edited(self, _text: str) -> None:
        self._auto_output = False

    def _on_format_changed(self) -> None:
        current = self._output.text().strip()
        if current:
            self._output.setText(str(apply_format_suffix(current, self._fmt())))
        elif self._auto_output:
            self._maybe_default_output()
        gif = self._fmt() == "gif"
        self._gif_quality.setEnabled(gif)
        self._update_delay_hint()

    def _update_delay_hint(self) -> None:
        fmt = self._fmt()
        fps = float(self._fps.value())
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
        suffix = FORMAT_SUFFIX[fmt]
        filters = {
            "gif": "GIF (*.gif)",
            "apng": "Animated PNG (*.png)",
            "webp": "WebP (*.webp)",
        }
        start = self._output.text().strip()
        if not start:
            start = last_save_directory()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save animation", start, filters[fmt]
        )
        if not path:
            return
        out = apply_format_suffix(path, fmt)
        if not out.suffix:
            out = out.with_suffix(suffix)
        self._auto_output = False
        self._output.setText(str(out))

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def _run(self) -> None:
        if self._busy():
            return
        if len(self._paths) < 2:
            QMessageBox.warning(self, "Animate", "Select at least two images.")
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
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ran)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_ran(self, result: object) -> None:
        self._set_running(False)
        r = result  # type: AnimationResult
        remember_save_path(r.path)
        msg = (
            f"Wrote {r.frames} frames, {r.width}×{r.height}, "
            f"{r.duration_ms} ms/frame → {r.path}"
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
