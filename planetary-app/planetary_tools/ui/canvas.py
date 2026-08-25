"""Zoomable image canvas."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

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

        self._crop_dim = QGraphicsPathItem()
        self._crop_dim.setBrush(QColor(0, 0, 0, 140))
        self._crop_dim.setPen(QPen(Qt.PenStyle.NoPen))
        self._crop_dim.setZValue(1)
        self._crop_dim.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._crop_dim.hide()
        self._scene.addItem(self._crop_dim)

        self._crop_pad = QGraphicsPathItem()
        self._crop_pad.setBrush(QColor(45, 50, 62, 200))
        self._crop_pad.setPen(QPen(Qt.PenStyle.NoPen))
        self._crop_pad.setZValue(1)
        self._crop_pad.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._crop_pad.hide()
        self._scene.addItem(self._crop_pad)

        self._crop_border = QGraphicsRectItem()
        border_pen = QPen(QColor(255, 220, 40))
        border_pen.setWidth(2)
        border_pen.setCosmetic(True)
        border_pen.setStyle(Qt.PenStyle.DashLine)
        self._crop_border.setPen(border_pen)
        self._crop_border.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._crop_border.setZValue(2)
        self._crop_border.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._crop_border.hide()
        self._scene.addItem(self._crop_border)

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
        self.clear_crop_overlay()
        if doc is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(0, 0, 0, 0)
            return
        self.refresh()

    def set_crop_overlay(self, x: int, y: int, width: int, height: int) -> None:
        """Draw a crop/expand rectangle in image pixels.

        Area kept from the image is undimmed. Source pixels outside the
        rectangle are dimmed. New canvas (expand) is a distinct fill.
        """
        pix = self._pixmap_item.pixmap()
        if pix.isNull() or width < 1 or height < 1:
            self.clear_crop_overlay()
            return
        img_w, img_h = pix.width(), pix.height()
        x = int(x)
        y = int(y)
        width = max(1, int(width))
        height = max(1, int(height))

        img_path = QPainterPath()
        img_path.addRect(QRectF(0, 0, img_w, img_h))
        crop_path = QPainterPath()
        crop_path.addRect(QRectF(x, y, width, height))

        dim_path = img_path.subtracted(crop_path)
        if dim_path.isEmpty():
            self._crop_dim.hide()
        else:
            self._crop_dim.setPath(dim_path)
            self._crop_dim.show()

        pad_path = crop_path.subtracted(img_path)
        if pad_path.isEmpty():
            self._crop_pad.hide()
        else:
            self._crop_pad.setPath(pad_path)
            self._crop_pad.show()

        # Inset by 0.5 px so the dashed stroke sits on pixel centres.
        self._crop_border.setRect(QRectF(x + 0.5, y + 0.5, width - 1.0, height - 1.0))
        self._crop_border.show()

        left = min(0, x)
        top = min(0, y)
        right = max(img_w, x + width)
        bottom = max(img_h, y + height)
        self._scene.setSceneRect(QRectF(left, top, right - left, bottom - top))

    def clear_crop_overlay(self) -> None:
        self._crop_dim.hide()
        self._crop_pad.hide()
        self._crop_border.hide()
        pix = self._pixmap_item.pixmap()
        if pix.isNull():
            self._scene.setSceneRect(0, 0, 0, 0)
        else:
            self._scene.setSceneRect(0, 0, pix.width(), pix.height())

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
        if self._crop_border.isVisible():
            scene = self._scene.sceneRect()
            img_w = scene.width()
            img_h = scene.height()
        else:
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