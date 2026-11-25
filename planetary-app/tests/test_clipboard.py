"""Edit > Copy Image and Paste Image move images through the system clipboard."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QMimeData, QUrl
from PyQt6.QtGui import QImage, QKeySequence

from planetary_tools.core.document import ImageDocument
from planetary_tools.io.loader import load_image, save_image
from planetary_tools.ui.main_window import MainWindow
from test_sticky_auto import Isolated


def image():
    return np.random.default_rng(3).uniform(0, 1, (9, 14, 3)).astype(np.float32)


class ClipboardTests(Isolated):
    def window(self, data=None):
        window = MainWindow()
        self.addCleanup(window.close)
        if data is not None:
            window._set_document(ImageDocument(data, path=Path('source.png')))
        return window

    def close_unsaved(self, window):
        if window._document is not None:
            window._document.modified = False

    def test_menu_actions_use_the_standard_shortcuts(self):
        window = self.window()
        self.assertEqual(window._copy_act.shortcut(), QKeySequence(QKeySequence.StandardKey.Copy))
        self.assertEqual(window._paste_act.shortcut(), QKeySequence(QKeySequence.StandardKey.Paste))
        self.assertFalse(window._copy_act.isEnabled())
        self.assertTrue(window._paste_act.isEnabled())
        window._set_document(ImageDocument(image()))
        self.assertTrue(window._copy_act.isEnabled())

    def test_copy_and_paste_match_a_16_bit_png(self):
        data = image()
        source = self.window(data)
        source._copy_image()
        self.assertEqual(self.app.clipboard().image().format(), QImage.Format.Format_RGBX64)

        window = self.window()
        window._paste_image()
        pasted = window._document
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'saved.png'
            save_image(ImageDocument(data), path, bit_depth=16)
            np.testing.assert_array_equal(pasted.data, load_image(path).data)
        self.assertIsNone(pasted.path)
        self.assertTrue(pasted.modified)
        self.assertEqual(pasted.storage_bits, 16)
        self.assertIn('Untitled', window.windowTitle())
        self.close_unsaved(window)

    def test_paste_an_8_bit_image_with_transparency(self):
        picture = QImage(5, 3, QImage.Format.Format_ARGB32)
        picture.fill(0x80336699)  # Half transparent: the colour is kept.
        self.app.clipboard().setImage(picture)
        window = self.window()
        window._paste_image()
        doc = window._document
        self.assertEqual((doc.data.shape, doc.storage_bits), ((3, 5, 3), 8))
        np.testing.assert_allclose(doc.data[0, 0], [0.0331, 0.1329, 0.3185], atol=1e-4)
        self.close_unsaved(window)

    def test_paste_without_an_image_leaves_the_image_open(self):
        data = image()
        window = self.window(data)
        self.app.clipboard().setText('not an image')
        with patch.object(window, '_confirm_unsaved_changes') as confirm:
            window._paste_image()
        confirm.assert_not_called()
        self.assertEqual(window._document.path, Path('source.png'))
        self.assertIn('no image', window._status.currentMessage())

    def test_declining_to_discard_changes_keeps_the_image(self):
        window = self.window(image())
        window._document.modified = True
        self.app.clipboard().setImage(QImage(4, 4, QImage.Format.Format_RGB32))
        with patch.object(window, '_confirm_unsaved_changes', return_value=False):
            window._paste_image()
        self.assertEqual(window._document.path, Path('source.png'))
        self.close_unsaved(window)

    def test_paste_a_copied_image_file_opens_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'copied.png'
            save_image(ImageDocument(image()), path, bit_depth=16)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(Path(folder) / 'notes.txt')),
                          QUrl.fromLocalFile(str(path))])
            (Path(folder) / 'notes.txt').write_text('x')
            self.app.clipboard().setMimeData(mime)
            window = self.window()
            window._paste_image()
            self.assertEqual(window._document.path, path)
            self.assertFalse(window._document.modified)
