"""Undo/redo stack for image documents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AppliedOperation:
    """One edit applied to the image, as a batch step where one exists."""

    label: str
    # None: the edit has no batch equivalent (e.g. a caption).
    filter_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Snapshot:
    data: np.ndarray
    is_grayscale: bool
    label: str


class UndoStack:
    """Stores snapshots labelled by the operation each history step reverses.

    Alongside the depth-limited snapshots it keeps every applied operation
    since the image was loaded, following undo/redo, so the edits currently
    shown can be rebuilt as a batch workflow.
    """

    def __init__(self, max_depth: int = 30) -> None:
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._max_depth = max_depth
        self._applied: list[AppliedOperation] = []
        self._reverted: list[AppliedOperation] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._applied.clear()
        self._reverted.clear()

    def push(
        self,
        data: np.ndarray,
        is_grayscale: bool,
        label: str = "",
        operation: AppliedOperation | None = None,
    ) -> None:
        snap = _Snapshot(data.copy(), is_grayscale, label)
        self._undo.append(snap)
        if len(self._undo) > self._max_depth:
            self._undo.pop(0)
        self._redo.clear()
        self._applied.append(operation or AppliedOperation(label))
        self._reverted.clear()

    def applied_operations(self) -> list[AppliedOperation]:
        """Edits behind the current image, oldest first."""
        return [deepcopy(op) for op in self._applied]

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
    ) -> tuple[np.ndarray, bool] | None:
        if not self._undo:
            return None
        self._redo.append(
            _Snapshot(current_data.copy(), current_grayscale, self._undo[-1].label)
        )
        snap = self._undo.pop()
        if self._applied:
            self._reverted.append(self._applied.pop())
        return snap.data, snap.is_grayscale

    def redo(
        self,
        current_data: np.ndarray,
        current_grayscale: bool,
    ) -> tuple[np.ndarray, bool] | None:
        if not self._redo:
            return None
        self._undo.append(
            _Snapshot(current_data.copy(), current_grayscale, self._redo[-1].label)
        )
        snap = self._redo.pop()
        if self._reverted:
            self._applied.append(self._reverted.pop())
        return snap.data, snap.is_grayscale


class UndoManager:
    """Higher-level undo helper used by the main window."""

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.stack = UndoStack()
        self._on_change = on_change

    def _notify(self) -> None:
        if self._on_change:
            self._on_change()

    def record(
        self,
        data: np.ndarray,
        is_grayscale: bool,
        label: str,
        operation: AppliedOperation | None = None,
    ) -> None:
        self.stack.push(data, is_grayscale, label, operation)
        self._notify()

    def clear(self) -> None:
        self.stack.clear()
        self._notify()