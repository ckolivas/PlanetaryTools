"""Batch processing dialog."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.batch.pipeline import (
    WORKFLOW_EXTENSION,
    BatchResult,
    BatchWorkflow,
    PipelineStep,
    collect_paths,
    existing_output_paths,
    load_workflow,
    planned_output_paths,
    run_batch,
    save_workflow,
)
from planetary_tools.core.presets import ensure_builtin_presets
from planetary_tools.filters.registry import FILTERS, batch_filters
from planetary_tools.ui.dialogs import edit_filter_params
from planetary_tools.ui.recent_files import last_open_directory, remember_open_path


class _BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        input_paths: list[Path],
        output_dir: Path,
        steps: list[PipelineStep],
        suffix: str,
        bit_depth: int,
        preserve_tree: bool,
        input_root: Path | None,
    ) -> None:
        super().__init__()
        self._input_paths = input_paths
        self._output_dir = output_dir
        self._steps = steps
        self._suffix = suffix
        self._bit_depth = bit_depth
        self._preserve_tree = preserve_tree
        self._input_root = input_root

    def run(self) -> None:
        try:
            result = run_batch(
                self._input_paths,
                self._output_dir,
                self._steps,
                suffix=self._suffix,
                bit_depth=self._bit_depth,
                preserve_tree=self._preserve_tree,
                input_root=self._input_root,
                on_progress=lambda c, t, m: self.progress.emit(c, t, m),
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class BatchDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumWidth(520)
        self._steps: list[PipelineStep] = []
        self._input_files: list[Path] = []
        self._input_folder: Path | None = None
        self._worker: _BatchWorker | None = None

        root = QVBoxLayout(self)

        # Input
        in_group = QGroupBox("Input")
        in_layout = QFormLayout(in_group)
        self._input_label = QLabel("No input selected")
        self._input_label.setWordWrap(True)
        pick_files = QPushButton("Select files…")
        pick_files.clicked.connect(self._pick_files)
        pick_folder = QPushButton("Select folder…")
        pick_folder.clicked.connect(self._pick_folder)
        in_btns = QHBoxLayout()
        in_btns.addWidget(pick_files)
        in_btns.addWidget(pick_folder)
        in_layout.addRow(self._input_label)
        in_layout.addRow(in_btns)
        self._recursive = QCheckBox("Include subfolders")
        in_layout.addRow(self._recursive)
        root.addWidget(in_group)

        # Pipeline
        pipe_group = QGroupBox("Filter pipeline")
        pipe_layout = QVBoxLayout(pipe_group)
        self._step_list = QListWidget()
        pipe_layout.addWidget(self._step_list)

        filter_row = QHBoxLayout()
        self._filter_combo = QComboBox()
        for fdef in batch_filters():
            self._filter_combo.addItem(fdef.label, fdef.id)
        self._filter_combo.currentIndexChanged.connect(self._refresh_preset_combo)
        filter_row.addWidget(QLabel("Filter:"))
        filter_row.addWidget(self._filter_combo, stretch=1)
        pipe_layout.addLayout(filter_row)

        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.setToolTip(
            "Saved preset to apply when adding this filter to the pipeline "
            "(Default, Last, and any user presets for that filter)."
        )
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self._preset_combo, stretch=1)
        pipe_layout.addLayout(preset_row)

        step_btns = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_step)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit_step)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_step)
        up_btn = QPushButton("Up")
        up_btn.clicked.connect(lambda: self._move_step(-1))
        down_btn = QPushButton("Down")
        down_btn.clicked.connect(lambda: self._move_step(1))
        step_btns.addWidget(add_btn)
        step_btns.addWidget(edit_btn)
        step_btns.addWidget(remove_btn)
        step_btns.addWidget(up_btn)
        step_btns.addWidget(down_btn)
        pipe_layout.addLayout(step_btns)

        wf_row = QHBoxLayout()
        save_wf_btn = QPushButton("Save workflow…")
        save_wf_btn.setToolTip(
            "Save the filter pipeline and output options to a reusable workflow file."
        )
        save_wf_btn.clicked.connect(self._save_workflow)
        load_wf_btn = QPushButton("Load workflow…")
        load_wf_btn.setToolTip(
            "Load a previously saved batch workflow (pipeline and output options)."
        )
        load_wf_btn.clicked.connect(self._load_workflow)
        wf_row.addWidget(save_wf_btn)
        wf_row.addWidget(load_wf_btn)
        wf_row.addStretch(1)
        pipe_layout.addLayout(wf_row)
        root.addWidget(pipe_group)

        self._refresh_preset_combo()

        # Output
        out_group = QGroupBox("Output")
        out_layout = QFormLayout(out_group)
        self._output_dir = QLineEdit()
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._pick_output)
        out_row = QHBoxLayout()
        out_row.addWidget(self._output_dir)
        out_row.addWidget(browse_out)
        out_layout.addRow("Folder", out_row)
        self._suffix = QLineEdit("_processed")
        out_layout.addRow("Filename suffix", self._suffix)
        self._bit_depth = QComboBox()
        self._bit_depth.addItem("32-bit float TIFF", 32)
        self._bit_depth.addItem("16-bit TIFF / PNG", 16)
        self._bit_depth.addItem("8-bit PNG / JPEG", 8)
        out_layout.addRow("Output depth", self._bit_depth)
        self._preserve_tree = QCheckBox("Preserve subfolder structure")
        out_layout.addRow(self._preserve_tree)
        root.addWidget(out_group)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        root.addWidget(self._progress)
        root.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close
        )
        self._run_btn = QPushButton("Run batch")
        self._run_btn.clicked.connect(self._run_batch)
        self._buttons.addButton(self._run_btn, QDialogButtonBox.ButtonRole.ActionRole)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._refresh_step_list()

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select input images", last_open_directory()
        )
        if paths:
            remember_open_path(paths[0])
            self._input_files = [Path(p) for p in paths]
            self._input_folder = None
            self._input_label.setText(f"{len(paths)} file(s) selected")

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self._input_folder = Path(folder)
            self._input_files = []
            self._input_label.setText(str(self._input_folder))

    def _pick_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._output_dir.setText(folder)

    def _refresh_step_list(self) -> None:
        self._step_list.clear()
        for i, step in enumerate(self._steps):
            item = QListWidgetItem(f"{i + 1}. {step.label}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._step_list.addItem(item)

    def _refresh_preset_combo(self) -> None:
        fid = self._filter_combo.currentData()
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        if not fid:
            self._preset_combo.blockSignals(False)
            return
        fdef = FILTERS[fid]
        presets = ensure_builtin_presets(fid, fdef.default_params)
        # Prefer Last when present (most recent interactive settings), else Default.
        preferred = "Last" if "Last" in presets else "Default"
        for name in sorted(presets.keys()):
            self._preset_combo.addItem(name)
        idx = self._preset_combo.findText(preferred)
        self._preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._preset_combo.blockSignals(False)

    def _params_from_selected_preset(self) -> tuple[dict, str]:
        fid = self._filter_combo.currentData()
        fdef = FILTERS[fid]
        presets = ensure_builtin_presets(fid, fdef.default_params)
        name = self._preset_combo.currentText()
        if name not in presets:
            name = "Default" if "Default" in presets else next(iter(presets), "")
        if name and name in presets:
            params = deepcopy(presets[name])
        else:
            params = deepcopy(fdef.default_params)
            name = "Default"
        # Merge defaults so older presets missing new keys still run.
        merged = {**fdef.default_params, **params}
        return merged, name

    def _add_step(self) -> None:
        fid = self._filter_combo.currentData()
        if not fid:
            return
        params, preset_name = self._params_from_selected_preset()
        self._steps.append(
            PipelineStep(filter_id=fid, params=params, preset_name=preset_name)
        )
        self._refresh_step_list()
        self._step_list.setCurrentRow(len(self._steps) - 1)

    def _selected_index(self) -> int | None:
        row = self._step_list.currentRow()
        return row if row >= 0 else None

    def _edit_step(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        step = self._steps[idx]
        result = edit_filter_params(
            step.filter_id,
            step.params,
            is_grayscale=False,
            parent=self,
            preset_name=step.preset_name,
        )
        if result is not None:
            step.params, step.preset_name = result
            self._refresh_step_list()
            self._step_list.setCurrentRow(idx)

    def _remove_step(self) -> None:
        idx = self._selected_index()
        if idx is not None:
            self._steps.pop(idx)
            self._refresh_step_list()

    def _move_step(self, delta: int) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        new_idx = idx + delta
        if 0 <= new_idx < len(self._steps):
            self._steps[idx], self._steps[new_idx] = self._steps[new_idx], self._steps[idx]
            self._refresh_step_list()
            self._step_list.setCurrentRow(new_idx)

    def _current_workflow(self) -> BatchWorkflow:
        bit = self._bit_depth.currentData()
        return BatchWorkflow(
            steps=[
                PipelineStep(
                    filter_id=s.filter_id,
                    params=deepcopy(s.params),
                    preset_name=s.preset_name,
                )
                for s in self._steps
            ],
            suffix=self._suffix.text().strip() or "_processed",
            bit_depth=int(bit) if bit is not None else 32,
            preserve_tree=self._preserve_tree.isChecked(),
            recursive=self._recursive.isChecked(),
            output_dir=self._output_dir.text().strip(),
        )

    def _apply_workflow(self, workflow: BatchWorkflow) -> None:
        self._steps = [
            PipelineStep(
                filter_id=s.filter_id,
                params=deepcopy(s.params),
                preset_name=s.preset_name,
            )
            for s in workflow.steps
        ]
        self._suffix.setText(workflow.suffix or "_processed")
        idx = self._bit_depth.findData(int(workflow.bit_depth))
        if idx >= 0:
            self._bit_depth.setCurrentIndex(idx)
        self._preserve_tree.setChecked(bool(workflow.preserve_tree))
        self._recursive.setChecked(bool(workflow.recursive))
        if workflow.output_dir:
            self._output_dir.setText(workflow.output_dir)
        self._refresh_step_list()

    def _workflow_file_filter(self) -> str:
        ext = WORKFLOW_EXTENSION.lstrip(".")
        return f"Batch workflows (*.{ext});;All Files (*)"

    def _save_workflow(self) -> None:
        if not self._steps:
            QMessageBox.warning(
                self,
                "Save workflow",
                "Add at least one filter step before saving a workflow.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save batch workflow",
            last_open_directory(),
            self._workflow_file_filter(),
        )
        if not path:
            return
        try:
            save_workflow(path, self._current_workflow())
            out = Path(path)
            if out.suffix.lower() != WORKFLOW_EXTENSION:
                out = out.with_suffix(WORKFLOW_EXTENSION)
            remember_open_path(str(out))
            self._status.setText(f"Saved workflow to {out.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save workflow failed", str(exc))

    def _load_workflow(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load batch workflow",
            last_open_directory(),
            self._workflow_file_filter(),
        )
        if not path:
            return
        try:
            workflow, warnings = load_workflow(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load workflow failed", str(exc))
            return
        if not workflow.steps:
            QMessageBox.warning(
                self,
                "Load workflow",
                "The workflow file contains no usable filter steps.",
            )
            return
        if self._steps:
            reply = QMessageBox.question(
                self,
                "Load workflow",
                "Replace the current pipeline with the loaded workflow?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._apply_workflow(workflow)
        remember_open_path(path)
        msg = f"Loaded workflow ({len(workflow.steps)} step(s))"
        self._status.setText(msg)
        if warnings:
            QMessageBox.warning(
                self,
                "Load workflow",
                msg + " with warnings:\n\n" + "\n".join(warnings),
            )

    def _run_batch(self) -> None:
        if not self._steps:
            QMessageBox.warning(self, "Batch", "Add at least one filter to the pipeline.")
            return

        paths = collect_paths(
            files=self._input_files or None,
            folder=self._input_folder,
            recursive=self._recursive.isChecked(),
        )
        if not paths:
            QMessageBox.warning(self, "Batch", "Select input files or a folder containing images.")
            return

        out_text = self._output_dir.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Batch", "Select an output folder.")
            return
        output_dir = Path(out_text)

        suffix = self._suffix.text().strip() or "_processed"
        preserve_tree = self._preserve_tree.isChecked()
        input_root = self._input_folder if preserve_tree else None

        planned = planned_output_paths(
            paths,
            output_dir,
            suffix=suffix,
            preserve_tree=preserve_tree,
            input_root=input_root,
        )
        existing = existing_output_paths(planned)
        if existing:
            sample = "\n".join(str(p) for p in existing[:12])
            if len(existing) > 12:
                sample += f"\n… and {len(existing) - 12} more"
            reply = QMessageBox.warning(
                self,
                "Overwrite existing files?",
                (
                    f"{len(existing)} of {len(planned)} output file(s) already exist:\n\n"
                    f"{sample}\n\n"
                    "Overwrite them?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(paths))
        self._progress.setValue(0)
        self._status.setText("Starting…")

        self._worker = _BatchWorker(
            paths,
            output_dir,
            list(self._steps),
            suffix,
            self._bit_depth.currentData(),
            preserve_tree,
            input_root,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(min(current, total))
        self._status.setText(f"Processing {message} ({current}/{total})")

    def _on_finished(self, result: object) -> None:
        self._run_btn.setEnabled(True)
        r = result  # type: BatchResult
        msg = f"Processed {r.processed} image(s)."
        if r.failed:
            msg += f" {len(r.failed)} failed."
        self._status.setText(msg)
        if r.failed:
            details = "\n".join(f"{p}: {e}" for p, e in r.failed[:10])
            if len(r.failed) > 10:
                details += f"\n… and {len(r.failed) - 10} more"
            QMessageBox.warning(self, "Batch complete with errors", msg + "\n\n" + details)
        else:
            QMessageBox.information(self, "Batch complete", msg)

    def _on_failed(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText("Batch failed")
        QMessageBox.critical(self, "Batch failed", message)

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Batch", "Wait for the batch to finish before closing.")
            event.ignore()
            return
        super().closeEvent(event)