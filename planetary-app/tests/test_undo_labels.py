"""Undo/redo operation names survive traversal and branching history."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.main_window import MainWindow


class UndoLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_menu_labels_through_multiple_undo_redo_and_new_edit(self):
        window = MainWindow()
        source = np.full((16, 16), .1, np.float32)
        window._set_document(ImageDocument(source.copy(), is_grayscale=True))
        def check(undo, redo):
            self.assertEqual(window._undo_act.text(), 'Undo' + (' '+undo if undo else ''))
            self.assertEqual(window._redo_act.text(), 'Redo' + (' '+redo if redo else ''))
            self.assertEqual(window._undo_act.isEnabled(), bool(undo))
            self.assertEqual(window._redo_act.isEnabled(), bool(redo))
        try:
            check('', '')
            window._commit_filter('Levels', source*2)
            check('Levels', '')
            window._commit_filter('Curves', source*3)
            check('Curves', '')
            window._undo_action()
            check('Levels', 'Curves')
            np.testing.assert_array_equal(window._document.data, source*2)
            window._undo_action()
            check('', 'Levels')
            np.testing.assert_array_equal(window._document.data, source)
            window._redo_action()
            check('Levels', 'Curves')
            window._redo_action()
            check('Curves', '')
            np.testing.assert_array_equal(window._document.data, source*3)
            window._undo_action()
            window._commit_filter('Saturation & Vibrance', source*4)
            check('Saturation && Vibrance', '')
            self.assertEqual(window._undo_act.toolTip(), 'Undo Saturation & Vibrance')
            window._undo_action()
            check('Levels', 'Saturation && Vibrance')
            window._set_document(ImageDocument(source.copy(), is_grayscale=True))
            check('', '')
            # Legacy/unnamed history still permits undo, without a trailing space.
            window._undo.record(source, True, '')
            self.assertTrue(window._undo_act.isEnabled())
            self.assertEqual(window._undo_act.text(), 'Undo')
            window._undo_action()
            self.assertTrue(window._redo_act.isEnabled())
            self.assertEqual(window._redo_act.text(), 'Redo')
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
