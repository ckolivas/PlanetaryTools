"""Mouse crop selection uses image coordinates and the existing crop lifecycle."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.crop import CropRect
from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.canvas import ImageCanvas
from planetary_tools.ui.crop_dialog import CropImageDialog
from planetary_tools.ui.main_window import MainWindow


class CropSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def drag(self, canvas, start, end, modifiers=Qt.KeyboardModifier.NoModifier):
        first = canvas.mapFromScene(QPointF(*start))
        last = canvas.mapFromScene(QPointF(*end))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, modifiers, first)
        QTest.mouseMove(canvas.viewport(), last)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, modifiers, last)
        self.app.processEvents()

    def draw(self, canvas, start, end):
        self.drag(canvas, start, end, Qt.KeyboardModifier.ControlModifier)

    def editing_canvas(self):
        data = np.zeros((100, 160, 3), dtype=np.float32)
        canvas = ImageCanvas()
        canvas.resize(700, 500)
        canvas.set_document(ImageDocument(data))
        panel = CropImageDialog(160, 100, data, False)
        canvas.crop_selected.connect(panel.set_selected_rect)
        panel.rect_changed.connect(canvas.set_crop_overlay)
        canvas.set_crop_selection_enabled(True)
        panel.emit_current_rect()
        canvas.show()
        self.addCleanup(canvas.close)
        self.addCleanup(panel.close)
        self.app.processEvents()
        return canvas, panel

    def test_all_corners_resize_with_fixed_opposite_corner_at_each_zoom(self):
        canvas, panel = self.editing_canvas()
        cases = [((20, 10), (10, 4), CropRect(10, 4, 90, 66)),
                 ((100, 10), (120, 4), CropRect(20, 4, 100, 66)),
                 ((100, 70), (120, 80), CropRect(20, 10, 100, 70)),
                 ((20, 70), (10, 80), CropRect(10, 10, 90, 70))]
        for zoom in (.5, 1, 2):
            canvas.set_zoom(zoom)
            for start, end, expected in cases:
                with self.subTest(zoom=zoom, start=start):
                    panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
                    self.app.processEvents()
                    self.drag(canvas, start, end)
                    self.assertEqual(panel.crop_rect(), expected)
                    self.assertEqual(canvas._crop_rect, expected.as_tuple())
                    for handle in canvas._crop_handles:
                        self.assertTrue(handle.isVisible())
                        rect = handle.deviceTransform(canvas.viewportTransform()).mapRect(handle.rect())
                        self.assertAlmostEqual(rect.width(), 8)

    def test_move_keeps_size_and_allows_expansion(self):
        canvas, panel = self.editing_canvas()
        for zoom in (.5, 1, 2):
            canvas.set_zoom(zoom)
            panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
            self.app.processEvents()
            self.drag(canvas, (50, 40), (80, 60))
            self.assertEqual(panel.crop_rect(), CropRect(50, 30, 80, 60))
            self.assertEqual((panel._width.value(), panel._height.value()), (80, 60))
        self.drag(canvas, (80, 60), (10, 20))
        self.assertEqual(panel.crop_rect(), CropRect(-20, -10, 80, 60))
        canvas.set_crop_selection_enabled(False)
        self.assertTrue(all(not h.isVisible() for h in canvas._crop_handles))
        canvas.clear_crop_overlay()
        self.assertIsNone(canvas._crop_rect)

    def test_edges_resize_one_axis_at_each_zoom(self):
        canvas, panel = self.editing_canvas()
        # Grab away from the midpoint handles and move diagonally: the
        # perpendicular dimension must stay fixed, even outside the image.
        cases = [((44, 10), (54, -10), CropRect(20, -10, 80, 80)),
                 ((100, 34), (120, 44), CropRect(20, 10, 100, 60)),
                 ((76, 70), (66, 90), CropRect(20, 10, 80, 80)),
                 ((20, 46), (-10, 36), CropRect(-10, 10, 110, 60))]
        for zoom in (.5, 1, 2):
            canvas.set_zoom(zoom)
            for start, end, expected in cases:
                with self.subTest(zoom=zoom, start=start):
                    panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
                    self.app.processEvents()
                    self.drag(canvas, start, end)
                    self.assertEqual(panel.crop_rect(), expected)
                    self.assertEqual(canvas._crop_rect, expected.as_tuple())
                    self.assertEqual(len(canvas._crop_handles), 8)

    def test_edges_cross_opposite_edge_and_preserve_grab_offset(self):
        canvas, panel = self.editing_canvas()
        cases = [((60, 10), (60, 80), CropRect(20, 70, 80, 10)),
                 ((100, 40), (10, 40), CropRect(10, 10, 10, 60)),
                 ((60, 70), (60, 0), CropRect(20, 0, 80, 10)),
                 ((20, 40), (110, 40), CropRect(100, 10, 10, 60)),
                 ((60, 10), (60, 70), CropRect(20, 69, 80, 1)),
                 ((100, 40), (20, 40), CropRect(20, 10, 1, 60)),
                 ((60, 70), (60, 10), CropRect(20, 10, 80, 1)),
                 ((20, 40), (100, 40), CropRect(99, 10, 1, 60)),
                 ((60, 13), (70, 7), CropRect(20, 4, 80, 66)),
                 ((97, 40), (107, 50), CropRect(20, 10, 90, 60))]
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
                self.drag(canvas, start, end)
                self.assertEqual(panel.crop_rect(), expected)

    def test_cross_corner_minimum_size_and_grab_offset(self):
        canvas, panel = self.editing_canvas()
        panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
        self.drag(canvas, (20, 10), (110, 80))
        self.assertEqual(panel.crop_rect(), CropRect(100, 70, 10, 10))
        panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
        self.drag(canvas, (20, 10), (100, 70))
        self.assertEqual(panel.crop_rect(), CropRect(99, 69, 1, 1))
        panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
        self.drag(canvas, (23, 13), (13, 7))
        self.assertEqual(panel.crop_rect(), CropRect(10, 4, 90, 66))

    def test_expanding_drag_keeps_view_stationary_until_release(self):
        canvas, panel = self.editing_canvas()
        panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
        first = canvas.mapFromScene(QPointF(20, 10))
        last = canvas.mapFromScene(QPointF(-30, -20))
        bounds = canvas.scene().sceneRect()
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseMove(canvas.viewport(), last)
        self.assertEqual(canvas.scene().sceneRect(), bounds)
        self.assertEqual(panel.crop_rect(), CropRect(-30, -20, 130, 90))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=last)
        self.assertEqual(panel.crop_rect(), CropRect(-30, -20, 130, 90))
        self.assertLess(canvas.scene().sceneRect().left(), 0)

    def test_hover_cursors_and_drawing_outside_the_box(self):
        canvas, panel = self.editing_canvas()
        panel._apply_rect(CropRect(20, 10, 80, 60), emit=True)
        for point, cursor in (((20, 10), Qt.CursorShape.SizeFDiagCursor),
                              ((100, 10), Qt.CursorShape.SizeBDiagCursor),
                              ((60, 10), Qt.CursorShape.SizeVerCursor),
                              ((100, 40), Qt.CursorShape.SizeHorCursor),
                              ((60, 70), Qt.CursorShape.SizeVerCursor),
                              ((20, 40), Qt.CursorShape.SizeHorCursor),
                              ((60, 40), Qt.CursorShape.OpenHandCursor),
                              ((5, 5), Qt.CursorShape.CrossCursor)):
            QTest.mouseMove(canvas.viewport(), canvas.mapFromScene(QPointF(*point)))
            self.assertEqual(canvas.viewport().cursor().shape(), cursor)
        self.drag(canvas, (5, 5), (15, 25))
        self.assertEqual(panel.crop_rect(), CropRect(5, 5, 10, 20))

    def test_zoom_reverse_drag_bounds_and_numeric_sync(self):
        data = np.zeros((100, 160, 3), dtype=np.float32)
        canvas = ImageCanvas()
        canvas.resize(700, 500)
        canvas.set_document(ImageDocument(data))
        panel = CropImageDialog(160, 100, data, False)
        canvas.crop_selected.connect(panel.set_selected_rect)
        panel.rect_changed.connect(canvas.set_crop_overlay)
        canvas.set_crop_selection_enabled(True)
        canvas.show()
        self.addCleanup(canvas.close)
        self.addCleanup(panel.close)
        self.app.processEvents()
        for zoom in (.5, 1, 2):
            canvas.set_zoom(zoom)
            self.app.processEvents()
            for start, end in (((20, 10), (100, 70)), ((100, 70), (20, 10))):
                self.draw(canvas, start, end)
                self.assertEqual(panel.crop_rect(), CropRect(20, 10, 80, 60))
                self.assertEqual((panel._width.value(), panel._height.value()), (80, 60))
        self.draw(canvas, (20, 10), (180, 120))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 140, 90))
        self.draw(canvas, (40, 40), (40, 40))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 140, 90))
        panel._apply_rect(CropRect(-20, -20, 200, 140), emit=True)
        self.draw(canvas, (20, 10), (100, 70))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 80, 60))
        canvas.resize(300, 240)
        canvas.set_zoom(4)
        canvas.centerOn(80, 50)
        self.app.processEvents()
        self.draw(canvas, (60, 40), (100, 60))
        self.assertEqual(panel.crop_rect(), CropRect(60, 40, 40, 20))
        panel._width.setValue(100)
        self.assertEqual(canvas._crop_border.rect().width(), 99)
        before = panel.crop_rect()
        self.drag(canvas, (20, 20), (40, 50), Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(panel.crop_rect(), before)
        canvas.set_crop_selection_enabled(False)
        self.drag(canvas, (20, 20), (40, 50))
        self.assertEqual(panel.crop_rect(), before)

    def test_apply_cancel_reopen_and_undo(self):
        window = MainWindow()
        data = np.arange(100*160*3, dtype=np.float32).reshape(100, 160, 3) / 50000
        window._set_document(ImageDocument(data.copy()))
        window.show()
        self.app.processEvents()
        errors = []
        def interact(accept):
            try:
                panel = window._active_filter_dlg
                self.assertTrue(window._canvas._crop_selection_enabled)
                self.draw(window._canvas, (10, 10), (70, 50))
                self.drag(window._canvas, (40, 30), (50, 30))
                self.drag(window._canvas, (80, 50), (100, 70))
                self.drag(window._canvas, (100, 40), (110, 40))
                self.drag(window._canvas, (110, 40), (100, 40))
                self.assertEqual(panel.crop_rect(), CropRect(20, 10, 80, 60))
                (panel._accept if accept else panel._reject)()
            except BaseException as exc:
                errors.append(exc)
                window._active_filter_dlg._reject()
        try:
            for accept in (False, True):
                QTimer.singleShot(30, lambda accept=accept: interact(accept))
                window._run_crop_image()
                self.assertFalse(errors, errors)
                self.assertFalse(window._canvas._crop_selection_enabled)
                self.assertFalse(window._canvas._crop_border.isVisible())
                expected = data[10:70, 20:100] if accept else data
                np.testing.assert_array_equal(window._document.data, expected)
            self.assertIn('80 × 60 px', window.windowTitle())
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, data)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, data[10:70, 20:100])
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
