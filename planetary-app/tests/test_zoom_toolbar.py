"""Toolbar zoom buttons: Fit, then 100% back to actual size."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication, QToolBar

from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.main_window import MainWindow


class ZoomToolbarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_actual_size_button_follows_fit_and_resets_zoom(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.resize(800, 600)
        window.show()
        window._set_document(ImageDocument(np.full((2000, 3000, 3), .5, dtype=np.float32)))
        bar = window.findChild(QToolBar)
        actions = {a.text(): a for a in bar.actions()}
        texts = [a.text() for a in bar.actions() if a.text()]
        self.assertEqual(texts[texts.index('Fit') + 1], '100%')
        actions['Fit'].trigger()
        self.app.processEvents()
        self.assertLess(window._canvas._zoom, 1.0)
        self.assertNotEqual(window._zoom_combo.currentText(), '100%')
        actions['100%'].trigger()
        self.assertEqual(window._canvas._zoom, 1.0)
        self.assertEqual(window._zoom_combo.currentText(), '100%')


if __name__ == '__main__':
    unittest.main()
