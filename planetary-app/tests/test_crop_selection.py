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
                self.drag(canvas, start, end)
                self.assertEqual(panel.crop_rect(), CropRect(20, 10, 80, 60))
                self.assertEqual((panel._width.value(), panel._height.value()), (80, 60))
        self.drag(canvas, (20, 10), (180, 120))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 140, 90))
        self.drag(canvas, (40, 40), (40, 40))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 140, 90))
        panel._apply_rect(CropRect(-20, -20, 200, 140), emit=True)
        self.drag(canvas, (20, 10), (100, 70))
        self.assertEqual(panel.crop_rect(), CropRect(20, 10, 80, 60))
        canvas.resize(300, 240)
        canvas.set_zoom(4)
        canvas.centerOn(80, 50)
        self.app.processEvents()
        self.drag(canvas, (60, 40), (100, 60))
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
                self.drag(window._canvas, (20, 10), (100, 70))
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
