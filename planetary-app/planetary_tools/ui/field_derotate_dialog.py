"""Field derotation — align/rotate a set to a chosen reference image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.core.field_derotate import (
    IDENTITY_MATCH,
    DerotateSetResult,
    RigidMatch,
    derotate_set,
    estimate_rigid,
)
from planetary_tools.io.loader import load_image, supported_extensions
from planetary_tools.ui.recent_files import last_open_directory, remember_open_path

_COL_FILE = 0
_COL_ANGLE = 1
_COL_DX = 2
_COL_DY = 3
_COL_SCORE = 4
_COL_STATUS = 5


def _image_filter() -> str:
    exts = " ".join(f"*{e}" for e in supported_extensions())
    return f"Images ({exts});;All files (*)"


@dataclass
class _Row:
    path: Path
    match: RigidMatch | None = None
    status: str = ""


class _EstimateWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[Path],
        ref_index: int,
        max_angle: float,
    ) -> None:
        super().__init__()
        self._paths = paths
        self._ref_index = ref_index
        self._max_angle = max_angle

    def run(self) -> None:
        try:
            matches: list[RigidMatch] = [IDENTITY_MATCH] * len(self._paths)
            ref_path = self._paths[self._ref_index]
            self.progress.emit(0, len(self._paths), f"Loading reference {ref_path.name}")
            ref = load_image(ref_path).data
            matches[self._ref_index] = IDENTITY_MATCH
            for i, path in enumerate(self._paths):
                if i == self._ref_index:
                    continue
                self.progress.emit(i, len(self._paths), f"Matching {path.name}")
                tgt = load_image(path).data
                matches[i] = estimate_rigid(ref, tgt, max_angle=self._max_angle)
            self.finished_ok.emit(matches)
        except Exception as exc:
            self.failed.emit(str(exc))


class _RunWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[Path],
        matches: list[RigidMatch],
        output_dir: Path,
        suffix: str,
        bit_depth: int,
    ) -> None:
        super().__init__()
        self._paths = paths
        self._matches = matches
        self._output_dir = output_dir
        self._suffix = suffix
        self._bit_depth = bit_depth

    def run(self) -> None:
        try:
            items = []
            total = len(self._paths)
            for i, (path, match) in enumerate(zip(self._paths, self._matches)):
                self.progress.emit(i, total, f"Loading {path.name}")
                doc = load_image(path)
                items.append((path, doc.data, match))
            result = derotate_set(
                items,
                self._output_dir,
                suffix=self._suffix,
                bit_depth=self._bit_depth,
                on_progress=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class FieldDerotateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Field Derotation")
        self.setMinimumWidth(740)
        self._rows: list[_Row] = []
        self._ref_index = 0
        self._worker: QThread | None = None

        root = QVBoxLayout(self)
        root.addWidget(
            QLabel(
                "Align and rotate stacked stills to a chosen reference image "
                "by best luminance match. This is not WinJUPOS CM / longitude "
                "derotation, and it does not use site or sky coordinates."
            )
        )

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        pick = QHBoxLayout()
        btn_files = QPushButton("Select files…")
        btn_files.clicked.connect(self._pick_files)
        btn_folder = QPushButton("Select folder…")
        btn_folder.clicked.connect(self._pick_folder)
        pick.addWidget(btn_files)
        pick.addWidget(btn_folder)
        pick.addStretch()
        fl.addLayout(pick)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["File", "Δ°", "dx", "dy", "Score", "Status"]
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_FILE, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        fl.addWidget(self._table)

        ref_row = QHBoxLayout()
        set_ref = QPushButton("Set as reference")
        set_ref.setToolTip(
            "Use the selected file as the alignment reference "
            "(WaveSharp “Set Reference”)."
        )
        set_ref.clicked.connect(self._set_reference)
        ref_row.addWidget(set_ref)
        ref_row.addStretch()
        fl.addLayout(ref_row)
        root.addWidget(files)

        opts = QGroupBox("Match")
        of = QFormLayout(opts)
        self._max_angle = QDoubleSpinBox()
        self._max_angle.setRange(0.5, 180.0)
        self._max_angle.setDecimals(2)
        self._max_angle.setSingleStep(1.0)
        self._max_angle.setValue(45.0)
        self._max_angle.setSuffix(" °")
        self._max_angle.setToolTip(
            "Search this many degrees either side of 0. Warns if the "
            "best match sits on the limit."
        )
        of.addRow("Max search angle", self._max_angle)
        root.addWidget(opts)

        out = QGroupBox("Output")
        ouf = QFormLayout(out)
        out_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_output)
        out_row.addWidget(self._output_dir)
        out_row.addWidget(browse)
        ouf.addRow("Folder", out_row)
        self._suffix = QLineEdit("_derot")
        ouf.addRow("Suffix", self._suffix)
        self._bit_depth = QComboBox()
        self._bit_depth.addItem("32-bit float", 32)
        self._bit_depth.addItem("16-bit", 16)
        self._bit_depth.addItem("8-bit", 8)
        ouf.addRow("Bit depth", self._bit_depth)
        root.addWidget(out)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        root.addWidget(self._progress)
        root.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self._est_btn = QPushButton("Estimate")
        self._est_btn.clicked.connect(self._estimate)
        self._run_btn = QPushButton("Run…")
        self._run_btn.clicked.connect(self._run)
        buttons.addButton(self._est_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self._run_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select stacked images", last_open_directory(), _image_filter()
        )
        if paths:
            remember_open_path(paths[0])
            self._set_paths([Path(p) for p in paths])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder of stacked images", last_open_directory()
        )
        if not folder:
            return
        remember_open_path(folder)
        exts = {e.lower() for e in supported_extensions()}
        paths = sorted(
            p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts
        )
        self._set_paths(paths)

    def _pick_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_dir.setText(folder)

    def _set_paths(self, paths: list[Path]) -> None:
        self._rows = [_Row(path=p, status="") for p in paths]
        self._ref_index = 0
        if paths and not self._output_dir.text().strip():
            self._output_dir.setText(str(paths[0].parent))
        self._refresh_table()

    def _set_reference(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._rows):
            QMessageBox.information(
                self, "Field Derotation", "Select a file in the table first."
            )
            return
        self._ref_index = row
        for r in self._rows:
            r.match = None
            r.status = ""
        self._refresh_table()
        self._status.setText(f"Reference: {self._rows[row].path.name}")

    def _refresh_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            name = row.path.name
            if i == self._ref_index:
                name = "★ " + name
            file_item = QTableWidgetItem(name)
            file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            file_item.setToolTip(str(row.path))
            self._table.setItem(i, _COL_FILE, file_item)

            m = row.match
            vals = (
                "" if m is None else f"{m.angle_deg:.2f}",
                "" if m is None else f"{m.dx:.1f}",
                "" if m is None else f"{m.dy:.1f}",
                "" if m is None else f"{m.score:.3f}",
            )
            status = row.status
            if i == self._ref_index and not status:
                status = "Reference"
            for col, text in zip(
                (_COL_ANGLE, _COL_DX, _COL_DY, _COL_SCORE), vals
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)
            st = QTableWidgetItem(status)
            st.setFlags(st.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, _COL_STATUS, st)

    def _set_running(self, running: bool) -> None:
        self._est_btn.setEnabled(not running)
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)

    def _estimate(self) -> None:
        if self._busy():
            return
        if len(self._rows) < 2:
            QMessageBox.warning(
                self, "Field Derotation", "Select at least two images."
            )
            return
        self._set_running(True)
        self._status.setText("Estimating…")
        self._worker = _EstimateWorker(
            [r.path for r in self._rows],
            self._ref_index,
            float(self._max_angle.value()),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_estimated)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_estimated(self, matches: object) -> None:
        self._set_running(False)
        typed = matches  # type: list[RigidMatch]
        for row, match in zip(self._rows, typed):
            row.match = match
            row.status = match.status
        self._refresh_table()
        self._status.setText("Estimate complete. Review Δ° / dx / dy, then Run.")

    def _run(self) -> None:
        if self._busy():
            return
        if any(r.match is None for r in self._rows):
            QMessageBox.warning(
                self,
                "Field Derotation",
                "Run Estimate first so each file has a match to the reference.",
            )
            return
        out_text = self._output_dir.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Field Derotation", "Select an output folder.")
            return
        output_dir = Path(out_text)
        suffix = self._suffix.text().strip() or "_derot"
        planned = [
            output_dir / f"{r.path.stem}{suffix}{r.path.suffix}" for r in self._rows
        ]
        existing = [p for p in planned if p.exists()]
        if existing:
            sample = "\n".join(str(p) for p in existing[:12])
            if len(existing) > 12:
                sample += f"\n… and {len(existing) - 12} more"
            reply = QMessageBox.warning(
                self,
                "Overwrite existing files?",
                (
                    f"{len(existing)} of {len(planned)} output file(s) already exist:\n\n"
                    f"{sample}\n\nOverwrite them?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._set_running(True)
        self._status.setText("Writing…")
        assert all(r.match is not None for r in self._rows)
        self._worker = _RunWorker(
            [r.path for r in self._rows],
            [r.match for r in self._rows],  # type: ignore[misc]
            output_dir,
            suffix,
            int(self._bit_depth.currentData()),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ran)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_ran(self, result: object) -> None:
        self._set_running(False)
        r = result  # type: DerotateSetResult
        size = ""
        if r.canvas_size:
            size = f" Canvas {r.canvas_size[0]}×{r.canvas_size[1]}."
        msg = f"Wrote {r.processed} image(s).{size}"
        if r.failed:
            msg += f" {len(r.failed)} failed."
            details = "\n".join(f"{p}: {e}" for p, e in r.failed[:10])
            QMessageBox.warning(self, "Field Derotation", msg + "\n\n" + details)
        else:
            QMessageBox.information(self, "Field Derotation", msg)
        self._status.setText(msg)

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(min(current, total))
        self._status.setText(f"{message} ({current}/{total})")

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self._status.setText("Failed")
        QMessageBox.critical(self, "Field Derotation", message)

    def closeEvent(self, event) -> None:
        if self._busy():
            QMessageBox.warning(
                self, "Field Derotation", "Wait for the run to finish before closing."
            )
            event.ignore()
            return
        super().closeEvent(event)
