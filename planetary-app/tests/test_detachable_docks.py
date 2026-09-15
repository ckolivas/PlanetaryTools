"""Floating/redocking must preserve edits; closing a dock must cancel them."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QDockWidget

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
