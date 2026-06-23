"""Undo/redo stack for image documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class _Snapshot:
    data: np.ndarray
    is_grayscale: bool
    label: str


class UndoStack:
    """Stores image snapshots for undo/redo."""

    def __init__(self, max_depth: int = 30) -> None:
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._max_depth = max_depth

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, data: np.ndarray, is_grayscale: bool, label: str = "") -> None:
        snap = _Snapshot(data.copy(), is_grayscale, label)
        self._undo.append(snap)
        if len(self._undo) > self._max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo_label(self) -> str:
        return self._undo[-1].label if self._undo else ""

    def redo_label(self) -> str:
        return self._redo[-1].label if self._redo else ""

    def undo(
        self,
        current_data: np.ndarray,
        current_grayscale: bool,
        current_label: str,
    ) -> tuple[np.ndarray, bool] | None:
        if not self._undo:
            return None
        self._redo.append(
            _Snapshot(current_data.copy(), current_grayscale, current_label)
        )
        snap = self._undo.pop()
        return snap.data, snap.is_grayscale

    def redo(
        self,
        current_data: np.ndarray,
        current_grayscale: bool,
        current_label: str,
    ) -> tuple[np.ndarray, bool] | None:
        if not self._redo:
            return None
        self._undo.append(
            _Snapshot(current_data.copy(), current_grayscale, current_label)
        )
        snap = self._redo.pop()
        return snap.data, snap.is_grayscale


class UndoManager:
    """Higher-level undo helper used by the main window."""

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.stack = UndoStack()
        self._on_change = on_change

    def _notify(self) -> None:
        if self._on_change:
            self._on_change()

    def record(self, data: np.ndarray, is_grayscale: bool, label: str) -> None:
        self.stack.push(data, is_grayscale, label)
        self._notify()

    def clear(self) -> None:
        self.stack.clear()
        self._notify()