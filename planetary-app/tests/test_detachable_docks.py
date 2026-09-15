"""Floating/redocking must preserve edits; closing a dock must cancel them."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication, QDockWidget, QWidget

from planetary_tools.core.crop import CropRect
from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.dialogs import _FilterDialog
from planetary_tools.ui.main_window import MainWindow


class _TestFilter(_FilterDialog):
    def _build_filter_params(self):
        pass

    def build_filter_func(self):
        return lambda data, grayscale: data + 0.1


class DetachableDockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_edit_survives_detach_redock_and_close_still_cancels(self):
        for tool in ('filter', 'crop'):
            for action in ('cancel', 'apply', 'close_window'):
                with self.subTest(tool=tool, action=action):
                    self.exercise_dock(tool, action)

    def test_background_is_painted_after_repeated_redocking(self):
        window = MainWindow()
        dock = window._filter_dock
        host = window._filter_host
        window.show()
        dock.show()
        try:
            self.assertFalse(window.isAnimated())
            for _ in range(5):
                for area in (Qt.DockWidgetArea.LeftDockWidgetArea,
                             Qt.DockWidgetArea.RightDockWidgetArea):
                    dock.setFloating(True)
                    self.app.processEvents()
                    window.addDockWidget(area, dock)
                    dock.setFloating(False)
                    self.app.processEvents()
                    self.assertFalse(dock._redock_refresh.isActive())
                    # Render without forcing a window background. The host
                    # must paint its own opaque surface after reparenting.
                    image = QImage(host.size(), QImage.Format.Format_ARGB32)
                    image.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(image)
                    try:
                        host.render(painter, flags=QWidget.RenderFlag.DrawChildren)
                    finally:
                        painter.end()
                    self.assertEqual(image.pixelColor(1, 1).alpha(), 255)
                    self.assertEqual(image.pixelColor(1, 1).rgb(),
                                     host.palette().color(host.backgroundRole()).rgb())
            # A queued refresh must not reopen a closed dock or affect a
            # second detach performed before the queued update is delivered.
            dock.setFloating(True)
            dock.setFloating(False)
            dock.hide()
            self.app.processEvents()
            self.assertFalse(dock.isVisible())
            dock.show()
            dock.setFloating(True)
            dock.setFloating(False)
            dock.setFloating(True)
            self.app.processEvents()
            self.assertTrue(dock.isFloating())
            self.assertFalse(dock._redock_refresh.isActive())
        finally:
            window.close()

    def exercise_dock(self, tool, action):
        accept = action == 'apply'
        window = MainWindow()
        data = np.full((40, 60, 3), 0.25, dtype=np.float32)
        window._set_document(ImageDocument(data.copy()))
        window.show()
        errors = []
        dock = window._filter_dock

        def interact():
            try:
                panel = window._active_filter_dlg
                required = (QDockWidget.DockWidgetFeature.DockWidgetMovable
                            | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
                for child in window.findChildren(QDockWidget):
                    self.assertEqual(child.features() & required, required)
                if tool == 'crop':
                    panel.set_selected_rect(5, 6, 20, 15)
                for area in (Qt.DockWidgetArea.LeftDockWidgetArea,
                             Qt.DockWidgetArea.RightDockWidgetArea):
                    dock.setFloating(True)
                    self.app.processEvents()
                    self.assertTrue(dock.isFloating())
                    self.assertTrue(window._filter_loop.isRunning())
                    # Window managers can temporarily hide a floating dock.
                    dock.hide()
                    self.app.processEvents()
                    dock.show()
                    window.addDockWidget(area, dock)
                    dock.setFloating(False)
                    self.app.processEvents()
                    self.assertFalse(dock.isFloating())
                    self.assertEqual(window.dockWidgetArea(dock), area)
                    self.assertTrue(window._filter_loop.isRunning())
                    self.assertIs(window._active_filter_dlg, panel)
                    if tool == 'crop':
                        self.assertEqual(panel.crop_rect(), CropRect(5, 6, 20, 15))
                        self.assertTrue(window._canvas._crop_selection_enabled)
                    else:
                        self.assertTrue(window._preview.is_active)
                dock.setFloating(True)
                if accept:
                    panel._accept()
                elif action == 'close_window':
                    window.close()
                else:
                    dock.close()
            except BaseException as exc:
                errors.append(exc)
                window._active_filter_dlg._reject()

        try:
            QTimer.singleShot(30, interact)
            if tool == 'crop':
                window._run_crop_image()
            else:
                window._run_filter_dialog(_TestFilter('Test'), 'Test')
            self.assertFalse(errors, errors)
            self.assertFalse(window._filter_dialog_open)
            self.assertFalse(window._preview.is_active)
            self.assertFalse(dock.isVisible())
            expected = data if not accept else (data[6:21, 5:25] if tool == 'crop' else data + 0.1)
            np.testing.assert_array_equal(window._document.data, expected)
            if accept:
                window._undo_action()
                np.testing.assert_array_equal(window._document.data, data)
            else:
                self.assertFalse(window._undo.stack.can_undo())
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
