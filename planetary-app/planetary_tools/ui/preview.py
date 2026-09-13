"""Live filter preview on the canvas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.filters.registry import FilterOutputStats


@dataclass
class PreviewResult:
    data: np.ndarray
    stats: FilterOutputStats | None = None


FilterFunc = Callable[[np.ndarray, bool], np.ndarray | PreviewResult]


def _evaluate(func: FilterFunc, data: np.ndarray, grayscale: bool) -> PreviewResult:
    result = func(data, grayscale)
    return result if isinstance(result, PreviewResult) else PreviewResult(result)


class _PreviewWorker(QThread):
    result_ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self._job: tuple[FilterFunc, np.ndarray, bool, int] | None = None
        self.completed: tuple[int, PreviewResult] | None = None

    def configure(self, func: FilterFunc, data: np.ndarray,
                  is_grayscale: bool, generation: int) -> None:
        # Only configure an idle worker. A running job must keep its snapshot.
        self._job = (func, data, is_grayscale, generation)
        self.completed = None

    def run(self) -> None:
        if self._job is None:
            return
        func, data, grayscale, generation = self._job
        try:
            result = _evaluate(func, data, grayscale)
            self.completed = (generation, result)
            self.result_ready.emit(generation, result)
        except Exception as exc:
            self.failed.emit(generation, str(exc))
        finally:
            self._job = None


class PreviewController(QObject):
    """Debounced worker with one cached result for the current input/settings."""

    preview_updated = pyqtSignal()
    preview_failed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, debounce_ms: int = 500, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._debounce_ms = debounce_ms
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._run_preview)
        self._original: np.ndarray | None = None
        self._evaluation: PreviewResult | None = None
        self._result_generation = -1
        self._is_grayscale = False
        self._active = False
        self._preview_enabled = True
        self._filter_func: FilterFunc | None = None
        self._generation = 0
        self._needs_rerun = False
        self._worker = _PreviewWorker()
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_worker_finished)

    @property
    def is_active(self) -> bool:
        return self._active

    def display_data(self) -> np.ndarray | None:
        if not self._active:
            return None
        if self._preview_enabled and self._evaluation is not None:
            return self._evaluation.data
        return self._original

    def output_stats(self) -> FilterOutputStats | None:
        if not self._active or not self._preview_enabled or self._evaluation is None:
            return None
        return self._evaluation.stats

    def start(self, data: np.ndarray, is_grayscale: bool) -> None:
        self._original = data.copy()
        self._evaluation = None
        self._is_grayscale = is_grayscale
        self._active = True
        self._needs_rerun = False
        self._generation += 1

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        if enabled:
            self.schedule_update()
        else:
            self._timer.stop()
            self._needs_rerun = False
            self.busy_changed.emit(False)
        self.preview_updated.emit()

    def set_filter_func(self, func: FilterFunc) -> None:
        if func is not self._filter_func:
            self._filter_func = func
            # Invalidate immediately, including while waiting for debounce.
            self._generation += 1

    def schedule_update(self) -> None:
        if self._active and self._preview_enabled and self._filter_func is not None:
            if self._evaluation is not None and self._result_generation == self._generation:
                return
            self._timer.start(self._debounce_ms)

    def update_now(self) -> None:
        if not self._active or not self._preview_enabled or self._filter_func is None:
            return
        self._timer.stop()
        self._run_preview()

    def _run_preview(self) -> None:
        if not self._active or not self._preview_enabled:
            return
        if self._evaluation is not None and self._result_generation == self._generation:
            self.preview_updated.emit()
            return
        self.busy_changed.emit(True)
        self._start_worker()

    def _start_worker(self) -> None:
        if self._original is None or self._filter_func is None:
            return
        if self._worker.isRunning():
            self._needs_rerun = True
            return
        self._needs_rerun = False
        self._worker.configure(self._filter_func, self._original,
                               self._is_grayscale, self._generation)
        self._worker.start()

    def _on_worker_result(self, generation: int, result: PreviewResult) -> None:
        if not self._active or generation != self._generation:
            return
        self._evaluation = result
        self._result_generation = generation
        self.busy_changed.emit(False)
        self.preview_updated.emit()

    def _on_worker_failed(self, generation: int, message: str) -> None:
        if self._active and generation == self._generation:
            self.busy_changed.emit(False)
            self.preview_failed.emit(message)

    def _on_worker_finished(self) -> None:
        if self._active and self._preview_enabled and self._needs_rerun:
            self._needs_rerun = False
            self._run_preview()

    def finish(self, apply: bool) -> np.ndarray | None:
        self._timer.stop()
        self._active = False
        self._needs_rerun = False
        # Wait for an immutable in-flight snapshot before reading its result.
        self._worker.wait()
        result = None
        try:
            if apply and self._original is not None and self._filter_func is not None:
                evaluation = self._evaluation if self._result_generation == self._generation else None
                if evaluation is None and self._worker.completed is not None:
                    generation, completed = self._worker.completed
                    if generation == self._generation:
                        evaluation = completed
                if evaluation is None:
                    evaluation = _evaluate(self._filter_func, self._original, self._is_grayscale)
                result = evaluation.data
            return result
        finally:
            self._original = None
            self._evaluation = None
            self._worker.completed = None
            self._filter_func = None
            self._generation += 1
            self.busy_changed.emit(False)

    def original_data(self) -> np.ndarray | None:
        return self._original


def array_to_display_rgb(data: np.ndarray, is_grayscale: bool) -> np.ndarray:
    """Convert linear float array to 8-bit sRGB for canvas display."""
    if is_grayscale:
        g = linear_to_srgb(data)
        if g.ndim == 2:
            rgb = np.stack([g, g, g], axis=-1)
        else:
            rgb = np.repeat(g[..., None], 3, axis=-1)
    else:
        rgb = linear_to_srgb(data)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
