"""Zoomable image canvas."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from planetary_tools.core.document import ImageDocument

ZOOM_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 8.00]


class ImageCanvas(QGraphicsView):
    """Scrollable view with zoom support, defaulting to 100%."""

    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

        self._zoom = 1.0
        self._document: ImageDocument | None = None

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_document(self, doc: ImageDocument | None) -> None:
        self._document = doc
        if doc is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(0, 0, 0, 0)
            return
        self.refresh()

    def refresh(self) -> None:
        if self._document is None:
            return
        self.show_rgb_uint8(self._document.to_display_rgb())

    def show_rgb_uint8(self, rgb) -> None:
        """Display an 8-bit sRGB RGB888 array without changing the document."""
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(image.copy()))
        self._scene.setSceneRect(0, 0, w, h)
        self._apply_zoom()

    def set_zoom(self, factor: float) -> None:
        self._zoom = max(0.05, min(factor, 32.0))
        self._apply_zoom()
        self.zoom_changed.emit(self._zoom)

    def zoom_to_fit(self) -> None:
        if self._document is None:
            return
        view_rect = self.viewport().rect()
        if view_rect.width() <= 0 or view_rect.height() <= 0:
            return
        img_w = self._document.width
        img_h = self._document.height
        if img_w == 0 or img_h == 0:
            return
        scale_x = view_rect.width() / img_w
        scale_y = view_rect.height() / img_h
        self.set_zoom(min(scale_x, scale_y) * 0.95)

    def _apply_zoom(self) -> None:
        self.resetTransform()
        self.scale(self._zoom, self._zoom)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.set_zoom(self._zoom * 1.15)
            elif delta < 0:
                self.set_zoom(self._zoom / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def nearest_zoom_index(self) -> int:
        best = 0
        best_diff = abs(ZOOM_LEVELS[0] - self._zoom)
        for i, z in enumerate(ZOOM_LEVELS):
            diff = abs(z - self._zoom)
            if diff < best_diff:
                best = i
                best_diff = diff
        return best