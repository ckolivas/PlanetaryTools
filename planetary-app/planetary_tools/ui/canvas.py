"""Zoomable image canvas."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
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
    crop_selected = pyqtSignal(int, int, int, int)

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

        self._crop_handles = []
        for _ in range(4):
            handle = QGraphicsRectItem(-4, -4, 8, 8)
            handle.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIgnoresTransformations)
            handle.setBrush(QColor(255, 220, 40))
            handle.setPen(QPen(Qt.GlobalColor.black))
            handle.setZValue(3)
            handle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            handle.hide()
            self._scene.addItem(handle)
            self._crop_handles.append(handle)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setMouseTracking(True)

        self._zoom = 1.0
        self._document: ImageDocument | None = None
        self._crop_selection_enabled = False
        self._crop_drag_start: tuple[int, int] | None = None
        self._crop_rect: tuple[int, int, int, int] | None = None
        self._crop_drag_rect: tuple[int, int, int, int] | None = None
        self._crop_drag_mode = "draw"
        self._crop_corner = 0

    def set_crop_selection_enabled(self, enabled: bool) -> None:
        self._crop_selection_enabled = enabled
        self._crop_drag_start = None
        self._crop_drag_rect = None
        for handle in self._crop_handles:
            handle.setVisible(enabled and self._crop_border.isVisible())
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def _crop_corners(self):
        if self._crop_rect is None:
            return ()
        x, y, w, h = self._crop_rect
        return ((x, y), (x+w, y), (x+w, y+h), (x, y+h))

    def _crop_hit(self, position) -> tuple[str, int]:
        # Hit areas stay the same size on screen at every zoom level.
        candidates = []
        for index, (x, y) in enumerate(self._crop_corners()):
            corner = self.mapFromScene(float(x), float(y))
            dx, dy = corner.x()-position.x(), corner.y()-position.y()
            if abs(dx) <= 7 and abs(dy) <= 7:
                candidates.append((dx*dx + dy*dy, index))
        if candidates:
            return "resize", min(candidates)[1]
        point = self.mapToScene(position)
        if self._crop_rect is not None and QRectF(*self._crop_rect).contains(point):
            return "move", 0
        return "draw", 0

    def _update_crop_cursor(self, event: QMouseEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            cursor = Qt.CursorShape.OpenHandCursor
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            cursor = Qt.CursorShape.CrossCursor
        else:
            mode, corner = self._crop_hit(event.position().toPoint())
            cursor = (Qt.CursorShape.SizeFDiagCursor if corner % 2 == 0 else
                      Qt.CursorShape.SizeBDiagCursor) if mode == "resize" else (
                          Qt.CursorShape.OpenHandCursor if mode == "move" else
                          Qt.CursorShape.CrossCursor)
        self.viewport().setCursor(cursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (self._crop_selection_enabled and self._document is not None
                and event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            point = self.mapToScene(event.position().toPoint())
            mode, corner = self._crop_hit(event.position().toPoint())
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                mode = "draw"
            if mode != "draw" or (0 <= point.x() <= self._document.width
                                   and 0 <= point.y() <= self._document.height):
                self._crop_drag_start = (round(point.x()), round(point.y()))
                self._crop_drag_rect = self._crop_rect
                self._crop_drag_mode = mode
                self._crop_corner = corner
                if mode == "move":
                    self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def _emit_drag_crop(self, event: QMouseEvent, *, finish: bool = False) -> None:
        if self._crop_drag_start is None or self._document is None:
            return
        point = self.mapToScene(event.position().toPoint())
        x, y = round(point.x()), round(point.y())
        start_x, start_y = self._crop_drag_start
        rect = self._crop_drag_rect
        if self._crop_drag_mode == "move" and rect is not None:
            result = (rect[0]+x-start_x, rect[1]+y-start_y, rect[2], rect[3])
        elif self._crop_drag_mode == "resize" and rect is not None:
            left, top, w, h = rect
            corners = ((left, top), (left+w, top), (left+w, top+h), (left, top+h))
            anchor_x, anchor_y = corners[(self._crop_corner+2) % 4]
            # Preserve the grab offset when pressing near, rather than exactly
            # on, a corner; crossing the fixed opposite corner is supported.
            corner_x, corner_y = corners[self._crop_corner]
            x, y = corner_x+x-start_x, corner_y+y-start_y
            if x == anchor_x:
                x += 1 if corner_x > anchor_x else -1
            if y == anchor_y:
                y += 1 if corner_y > anchor_y else -1
            result = (min(x, anchor_x), min(y, anchor_y), abs(x-anchor_x), abs(y-anchor_y))
        else:
            x = max(0, min(self._document.width, x))
            y = max(0, min(self._document.height, y))
            result = (min(x, start_x), min(y, start_y), abs(x-start_x), abs(y-start_y))
        if finish:
            self._crop_drag_start = None
            self._crop_drag_rect = None
        if result[2] == 0 or result[3] == 0:
            return
        self.crop_selected.emit(*result)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._crop_drag_start is not None:
            self._emit_drag_crop(event)
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self._crop_selection_enabled and not event.buttons():
            self._update_crop_cursor(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._crop_drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._emit_drag_crop(event, finish=True)
            self._crop_drag_start = None
            self._update_crop_cursor(event)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._crop_selection_enabled:
            self._update_crop_cursor(event)

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_document(self, doc: ImageDocument | None) -> None:
        self.set_crop_selection_enabled(False)
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
        self._crop_rect = (x, y, width, height)
        for handle, (cx, cy) in zip(self._crop_handles, self._crop_corners()):
            handle.setPos(cx, cy)
            handle.setVisible(self._crop_selection_enabled)

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
        # Keep the image stationary while replacing an expanded rectangle by
        # dragging. Updating the scrollable bounds mid-drag can shift the view.
        if self._crop_drag_start is None:
            self._scene.setSceneRect(QRectF(left, top, right - left, bottom - top))

    def clear_crop_overlay(self) -> None:
        self._crop_rect = None
        self._crop_drag_start = None
        self._crop_drag_rect = None
        for handle in self._crop_handles:
            handle.hide()
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
