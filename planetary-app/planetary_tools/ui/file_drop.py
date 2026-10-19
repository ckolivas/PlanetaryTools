"""Local image drops and model-backed row reordering for file lists."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, QItemSelectionModel, QObject, Qt, pyqtSignal
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QTableWidget

from planetary_tools.io.loader import supported_extensions


def image_paths(mime) -> list[Path]:
    """Accept existing local image files, preserving order and removing aliases."""
    extensions = set(supported_extensions())
    paths, seen = [], set()
    for url in mime.urls() if mime.hasUrls() else ():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def row_order(count: int, rows: list[int], destination: int) -> list[int]:
    """Move selected rows together to a gap in the original row sequence."""
    selected = sorted({row for row in rows if 0 <= row < count})
    destination = max(0, min(destination, count))
    remaining = [row for row in range(count) if row not in selected]
    insert = destination - sum(row < destination for row in selected)
    return remaining[:insert] + selected + remaining[insert:]


class ImageDropTarget(QObject):
    """Receive file drops on a window or viewport using its existing add path."""
    def __init__(self, widget, callback, *, enabled=lambda: True, single=False):
        super().__init__(widget)
        self.callback = callback
        self.enabled = enabled
        self.single = single
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return False
        paths = image_paths(event.mimeData()) if self.enabled() else []
        if not paths or (self.single and len(paths) != 1):
            event.ignore()
            return True
        # Opening/adding files must never ask the source file manager to move
        # or delete them, regardless of keyboard modifiers during the drop.
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        if event.type() == QEvent.Type.Drop:
            self.callback(paths)
        return True


class _FileRows:
    """Mixin: controllers own order; Qt never moves/deletes model cells itself."""
    def _init_file_rows(self):
        self.can_drop = lambda: True
        self.can_reorder = lambda: True
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setToolTip("Drop image files here to add them. Drag rows to reorder.")

    def startDrag(self, supported_actions):
        if not self.can_drop() or not self.can_reorder():
            return
        indexes = self.selectedIndexes()
        if not indexes:
            return
        drag = QDrag(self)
        drag.setMimeData(self.model().mimeData(indexes))
        # Unlike the default item-view implementation, do not remove source
        # rows after a Move: the controller applies the full permutation once.
        drag.exec(Qt.DropAction.MoveAction)

    def _accept_drag(self, event, *, moving=False):
        if not self.can_drop():
            event.ignore()
        elif event.source() is self and self.can_reorder():
            if moving:
                super().dragMoveEvent(event)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        elif event.source() is not self and image_paths(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event):
        self._accept_drag(event)

    def dragMoveEvent(self, event):
        self._accept_drag(event, moving=True)

    def dropEvent(self, event):
        self._accept_drag(event)
        if not event.isAccepted():
            return
        if event.source() is self:
            index = self.indexAt(event.position().toPoint())
            destination = self.model().rowCount()
            if index.isValid():
                destination = index.row()
                if event.position().y() > self.visualRect(index).center().y():
                    destination += 1
            rows = sorted({index.row() for index in self.selectedIndexes()})
            order = row_order(self.model().rowCount(), rows, destination)
            if order != list(range(len(order))):
                self.order_changed.emit(order)
                self.clearSelection()
                selection = self.selectionModel()
                for row in rows:
                    selection.select(
                        self.model().index(order.index(row), 0),
                        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                    )
                if rows:
                    selection.setCurrentIndex(
                        self.model().index(order.index(rows[0]), 0),
                        QItemSelectionModel.SelectionFlag.NoUpdate,
                    )
        else:
            self.files_dropped.emit(image_paths(event.mimeData()))
        self.viewport().update()


class FileDropList(_FileRows, QListWidget):
    files_dropped = pyqtSignal(list)
    order_changed = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_file_rows()


class FileDropTable(_FileRows, QTableWidget):
    files_dropped = pyqtSignal(list)
    order_changed = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_file_rows()
