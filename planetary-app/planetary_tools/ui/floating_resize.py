"""A wider, invisible resize border for floating tool docks."""

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt
from PyQt6.QtGui import QRegion
from PyQt6.QtWidgets import QAbstractButton, QWidget


class FloatingResizeBorder(QWidget):
    MARGIN = 10  # Logical pixels, so the grab area also scales on HiDPI screens.

    def __init__(self, dock):
        super().__init__(dock)
        self._dock = dock
        self._drag = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        dock.installEventFilter(self)
        dock.topLevelChanged.connect(self._update_border)
        self.hide()

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                            QEvent.Type.LayoutRequest, QEvent.Type.WindowStateChange):
            self._update_border()
        return False

    def _update_border(self, *_):
        dock = self._dock
        if not dock.isFloating() or dock.isMaximized() or dock.isFullScreen():
            self.hide()
            return
        self.setGeometry(dock.rect())
        margin = self.MARGIN
        border = QRegion(self.rect()) - QRegion(
            self.rect().adjusted(margin, margin, -margin, -margin))
        # Preserve title-bar buttons even where they overlap the wider border.
        for name in ('qt_dockwidget_floatbutton', 'qt_dockwidget_closebutton'):
            button = dock.findChild(QAbstractButton, name)
            if button is not None and button.isVisible():
                border -= QRegion(QRect(button.mapTo(dock, QPoint()), button.size()))
        self.setMask(border)
        self.show()
        self.raise_()

    def _edges_at(self, point):
        edges = Qt.Edge(0)
        if point.x() < self.MARGIN:
            edges |= Qt.Edge.LeftEdge
        elif point.x() >= self.width() - self.MARGIN:
            edges |= Qt.Edge.RightEdge
        if point.y() < self.MARGIN:
            edges |= Qt.Edge.TopEdge
        elif point.y() >= self.height() - self.MARGIN:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _set_resize_cursor(self, edges):
        left, right = Qt.Edge.LeftEdge, Qt.Edge.RightEdge
        top, bottom = Qt.Edge.TopEdge, Qt.Edge.BottomEdge
        if edges in (left | top, right | bottom):
            cursor = Qt.CursorShape.SizeFDiagCursor
        elif edges in (right | top, left | bottom):
            cursor = Qt.CursorShape.SizeBDiagCursor
        elif edges & (left | right):
            cursor = Qt.CursorShape.SizeHorCursor
        else:
            cursor = Qt.CursorShape.SizeVerCursor
        self.setCursor(cursor)

    def enterEvent(self, event):
        self._set_resize_cursor(self._edges_at(event.position().toPoint()))
        super().enterEvent(event)

    def mousePressEvent(self, event):
        edges = self._edges_at(event.position().toPoint())
        if event.button() != Qt.MouseButton.LeftButton or not edges:
            event.ignore()
            return
        self._set_resize_cursor(edges)
        self._drag = None
        window = self._dock.windowHandle()
        # Use the window manager when available (required for Wayland).
        if window is None or not window.startSystemResize(edges):
            self._drag = (edges, event.globalPosition().toPoint(), self._dock.geometry())
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            self._set_resize_cursor(self._edges_at(event.position().toPoint()))
            return
        edges, origin, initial = self._drag
        delta = event.globalPosition().toPoint() - origin
        dock = self._dock
        minimum = dock.minimumSize().expandedTo(dock.minimumSizeHint())
        rect = QRect(initial)
        if edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge):
            left = bool(edges & Qt.Edge.LeftEdge)
            width = max(minimum.width(), min(dock.maximumWidth(),
                        initial.width() + (-delta.x() if left else delta.x())))
            rect.setWidth(width)
            if left:
                rect.moveRight(initial.right())
        if edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge):
            top = bool(edges & Qt.Edge.TopEdge)
            height = max(minimum.height(), min(dock.maximumHeight(),
                         initial.height() + (-delta.y() if top else delta.y())))
            rect.setHeight(height)
            if top:
                rect.moveBottom(initial.bottom())
        dock.setGeometry(rect)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None
        event.accept()

    def hideEvent(self, event):
        self._drag = None
        super().hideEvent(event)
