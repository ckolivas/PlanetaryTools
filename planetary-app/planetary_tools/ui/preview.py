"""Live filter preview on the canvas."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from planetary_tools.core.colour import linear_to_srgb

FilterFunc = Callable[[np.ndarray, bool], np.ndarray]


class _PreviewWorker(QThread):
    result_ready = pyqtSignal(int, object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._func: FilterFunc | None = None
        self._data: np.ndarray | None = None
        self._is_grayscale = False
        self._generation = 0

    def configure(
        self,
        func: FilterFunc,
        data: np.ndarray,
        is_grayscale: bool,
        generation: int,
    ) -> None:
        self._func = func
        self._data = data
        self._is_grayscale = is_grayscale
        self._generation = generation

    def run(self) -> None:
        if self._func is None or self._data is None:
            return
        gen = self._generation
        try:
            result = self._func(self._data, self._is_grayscale)
            self.result_ready.emit(gen, result)
        except Exception as exc:
            self.failed.emit(str(exc))


class PreviewController(QObject):
    """Debounced background preview; restores original on cancel."""

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
        self._preview_result: np.ndarray | None = None
        self._is_grayscale = False
        self._active = False
        self._preview_enabled = True
        self._filter_func: FilterFunc | None = None
        self._generation = 0
        self._needs_rerun = False
        self._worker = _PreviewWorker()
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.failed.connect(self._on_worker_failed)

    @property
    def is_active(self) -> bool:
        return self._active

    def display_data(self) -> np.ndarray | None:
        if not self._active:
            return None
        if self._preview_enabled and self._preview_result is not None:
            return self._preview_result
        return self._original

    def start(self, data: np.ndarray, is_grayscale: bool) -> None:
        self._original = data.copy()
        self._preview_result = None
        self._is_grayscale = is_grayscale
        self._active = True
        self._generation = 0

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        if enabled:
            self.schedule_update()
        else:
            self._preview_result = None
            self.preview_updated.emit()

    def set_filter_func(self, func: FilterFunc) -> None:
        self._filter_func = func

    def schedule_update(self) -> None:
        if not self._active or not self._preview_enabled or self._filter_func is None:
            return
        self._timer.start(self._debounce_ms)

    def update_now(self) -> None:
        if not self._active or not self._preview_enabled or self._filter_func is None:
            return
        self._timer.stop()
        self._run_preview()

    def _run_preview(self) -> None:
        if self._original is None or self._filter_func is None:
            return
        self._generation += 1
        self._preview_result = None
        self.preview_updated.emit()
        self.busy_changed.emit(True)
        self._start_worker()

    def _start_worker(self) -> None:
        if self._original is None or self._filter_func is None:
            return
        self._worker.configure(
            self._filter_func,
            self._original,
            self._is_grayscale,
            self._generation,
        )
        if self._worker.isRunning():
            self._needs_rerun = True
        else:
            self._needs_rerun = False
            self._worker.start()

    def _on_worker_result(self, generation: int, result: object) -> None:
        if generation != self._generation:
            if not self._worker.isRunning():
                self._start_worker()
            return
        self._preview_result = result  # type: ignore[assignment]
        self.busy_changed.emit(False)
        self.preview_updated.emit()
        if self._needs_rerun and not self._worker.isRunning():
            self._needs_rerun = False
            self._start_worker()

    def _on_worker_failed(self, message: str) -> None:
        self.busy_changed.emit(False)
        self.preview_failed.emit(message)

    def finish(self, apply: bool) -> np.ndarray | None:
        self._timer.stop()
        if self._worker.isRunning():
            self._worker.wait(60000)

        result: np.ndarray | None = None
        if apply and self._original is not None and self._filter_func is not None:
            result = self._filter_func(self._original, self._is_grayscale)

        self._active = False
        self._original = None
        self._preview_result = None
        self._filter_func = None
        self.busy_changed.emit(False)
        return result

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