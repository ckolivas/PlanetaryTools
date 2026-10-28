"""File > Close unloads the image, honouring unsaved changes and open filters."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.main_window import MainWindow


class CloseDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.addCleanup(self.close_window)

    def close_window(self):
        # Never block teardown on the unsaved-changes prompt.
        if self.window._document is not None:
            self.window._document.modified = False
        self.window.close()

    def load(self, modified=False):
        doc = ImageDocument(np.full((8, 12, 3), .5, dtype=np.float32), modified=modified)
        self.window._set_document(doc)
        self.window._undo.record(doc.data.copy(), False, 'Edit')
        return doc

    def file_actions(self):
        menu = next(a.menu() for a in self.window.menuBar().actions() if a.text() == '&File')
        return [a.text().replace('&', '') for a in menu.actions() if not a.isSeparator()]

    def test_menu_entry_shortcut_and_enabled_state(self):
        actions = self.file_actions()
        self.assertEqual(actions[-2:], ['Close', 'Quit'])
        self.assertEqual(self.window._close_act.shortcut(), QKeySequence(QKeySequence.StandardKey.Close))
        self.assertFalse(self.window._close_act.isEnabled())
        self.load()
        self.assertTrue(self.window._close_act.isEnabled())

    def test_close_unloads_image_and_resets_state(self):
        self.load()
        self.assertTrue(self.window._undo.stack.can_undo())
        self.window._close_act.trigger()
        self.assertIsNone(self.window._document)
        self.assertFalse(self.window._undo.stack.can_undo())
        self.assertEqual(self.window.windowTitle(), 'Planetary Tools')
        self.assertFalse(self.window._close_act.isEnabled())
        self.assertFalse(self.window._save_act.isEnabled())
        self.assertTrue(self.window._canvas._pixmap_item.pixmap().isNull())

    def test_unsaved_changes_prompt_can_cancel(self):
        doc = self.load(modified=True)
        with patch.object(self.window, '_confirm_unsaved_changes', return_value=False) as confirm:
            self.window._close_document()
        confirm.assert_called_once_with('before closing')
        self.assertIs(self.window._document, doc)
        self.assertTrue(self.window._close_act.isEnabled())
        with patch.object(self.window, '_confirm_unsaved_changes', return_value=True):
            self.window._close_document()
        self.assertIsNone(self.window._document)

    def test_open_filter_is_resolved_first(self):
        doc = self.load()
        with patch.object(self.window, '_guard_filter_dialog', return_value=True) as guard:
            self.window._close_document()
        guard.assert_called_once()
        self.assertEqual(guard.call_args.args[0], 'Close')
        self.assertIs(self.window._document, doc)


if __name__ == '__main__':
    unittest.main()
