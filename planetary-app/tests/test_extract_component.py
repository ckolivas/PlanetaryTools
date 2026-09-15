"""Component inversion preserves source data and the preview/apply lifecycle."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QCheckBox

from planetary_tools.core.document import ImageDocument
from planetary_tools.filters.extract_component import (
    COMPONENT_ORDER, extract_component, extract_component_plane,
)
from planetary_tools.ui.dialogs import ExtractComponentDialog
from planetary_tools.ui.main_window import MainWindow


class ExtractComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_invert_image_without_mutating_or_clipping_source(self):
        rgb = np.array([[[0, .25, 1], [1.25, -.25, .5]]], dtype=np.float32)
        for source, grayscale in ((rgb, False), (rgb[..., 0], True)):
            original = source.copy()
            for component in COMPONENT_ORDER:
                with self.subTest(component=component, grayscale=grayscale):
                    plane = extract_component_plane(source, grayscale, component)
                    result = extract_component(source, grayscale, component)
                    if component == 'invert':
                        expected = 1-source
                        if grayscale:
                            expected = np.repeat(expected[..., None], 3, axis=-1)
                        np.testing.assert_array_equal(result, expected)
                    else:
                        np.testing.assert_array_equal(result, np.repeat(plane[..., None], 3, axis=-1))
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(
            extract_component_plane(rgb, False, 'red'), [[0, 1.25]])

    def test_component_selection_snapshot_preview_apply_cancel_and_undo(self):
        source = np.array([[[0, .25, 1], [.75, 1, .5]]], dtype=np.float32)
        panel = ExtractComponentDialog()
        self.addCleanup(panel.close)
        self.assertNotIn('invert', panel.get_params())
        self.assertFalse(any(box.text() == 'Invert' for box in panel.findChildren(QCheckBox)))
        panel.set_params({'component': 'invert'})
        self.assertEqual(panel.component.currentText(), 'Invert')
        self.assertIn('Invert the whole image', panel.component.toolTip())
        snapshot = panel.build_filter_func()
        changes = []
        panel.params_changed.connect(lambda: changes.append(True))
        panel.component.setCurrentIndex(panel.component.findData('red'))
        self.assertEqual(changes, [True])
        expected = 1-source
        np.testing.assert_array_equal(snapshot(source, False), expected)
        np.testing.assert_array_equal(panel.build_filter_func()(source, False),
                                      np.repeat(source[..., :1], 3, axis=-1))
        panel.set_params({'component': 'green'})
        self.assertEqual(panel.get_params()['component'], 'green')

        window = MainWindow()
        window._set_document(ImageDocument(source.copy()))
        errors = []

        def edit(accept):
            active = window._active_filter_dlg
            try:
                active.component.setCurrentIndex(active.component.findData('invert'))
                np.testing.assert_array_equal(active.build_filter_func()(source, False), expected)
                active.preview.setChecked(False)
                active.preview.setChecked(True)
                (active._accept if accept else active._reject)()
            except BaseException as exc:
                errors.append(exc)
                active._reject()

        try:
            for accept in (False, True):
                QTimer.singleShot(0, lambda accept=accept: edit(accept))
                window._run_extract_component()
                self.assertFalse(errors, errors)
                np.testing.assert_array_equal(window._document.data, expected if accept else source)
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, source)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, expected)
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
