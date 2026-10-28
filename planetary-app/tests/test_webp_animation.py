"""WebP quality: lossless at 100, lossy below, validation and UI forwarding."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image, ImageSequence
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from planetary_tools.core import animate
from planetary_tools.ui.animate_dialog import AnimateDialog, _RunWorker


def decode(path):
    with Image.open(path) as im:
        return np.stack([np.asarray(f.convert('RGB')) for f in ImageSequence.Iterator(im)])


class WebpTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        rng = np.random.default_rng(271)
        base = rng.integers(0, 256, (40, 56, 3), dtype=np.uint8)
        self.frames = [np.clip(base.astype(int) + rng.integers(-20, 21, base.shape), 0, 255).astype(np.uint8)
                       for _ in range(4)]

    def test_100_is_lossless_and_lower_is_smaller_lossy(self):
        sizes = {}
        for q in (100, 95, 50):
            path = self.root/f'q{q}.webp'
            animate.encode_frames(self.frames, path, fps=10, fmt='webp', webp_quality=q, back_and_forth=False)
            actual = decode(path)
            self.assertEqual(actual.shape, (4, 40, 56, 3))
            if q == 100:
                np.testing.assert_array_equal(actual, self.frames)
            else:
                self.assertFalse(np.array_equal(actual, self.frames))
            sizes[q] = path.stat().st_size
        self.assertLess(sizes[95], sizes[100])
        self.assertLess(sizes[50], sizes[95])

    def test_default_is_lossy_99(self):
        default, explicit = self.root/'default.webp', self.root/'q99.webp'
        animate.encode_frames(self.frames, default, fps=10, fmt='webp', back_and_forth=False)
        animate.encode_frames(self.frames, explicit, fps=10, fmt='webp', webp_quality=99, back_and_forth=False)
        np.testing.assert_array_equal(decode(default), decode(explicit))

    def test_validation(self):
        path = self.root/'bad.webp'
        for q in (0, 101, 95.0, '95', True):
            with self.subTest(q=q), self.assertRaisesRegex(ValueError, 'WebP quality'):
                animate.encode_frames(self.frames, path, fps=10, fmt='webp', webp_quality=q)
        self.assertFalse(path.exists())
        animate.encode_frames(self.frames, self.root/'a.gif', fps=10, fmt='gif', webp_quality=0)

    def test_write_animation_and_worker_forward_quality(self):
        paths = []
        for i, frame in enumerate(self.frames[:2]):
            paths.append(self.root/f'frame{i}.png')
            Image.fromarray(frame).save(paths[-1])
        result = animate.write_animation(paths, self.root/'out.webp', fps=10, fmt='webp',
                                         webp_quality=100, back_and_forth=False)
        np.testing.assert_array_equal(decode(result.path), self.frames[:2])
        worker = _RunWorker(paths, result.path, 10, 'webp', 'best', False, webp_quality=80)
        with patch('planetary_tools.ui.animate_dialog.write_animation', return_value=result) as write:
            worker.run()
        self.assertEqual(write.call_args.kwargs['webp_quality'], 80)


class WebpDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_control_enabled_for_webp_and_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            def settings():
                return QSettings(str(Path(folder)/'prefs.ini'), QSettings.Format.IniFormat)
            with patch('planetary_tools.ui.recent_files._settings', settings):
                dialog = AnimateDialog()
                try:
                    self.assertEqual(dialog._webp_quality.value(), 99)
                    dialog._format.setCurrentIndex(dialog._format.findData('gif'))
                    self.assertFalse(dialog._webp_quality.isEnabled())
                    dialog._format.setCurrentIndex(dialog._format.findData('webp'))
                    self.assertTrue(dialog._webp_quality.isEnabled())
                    self.assertFalse(dialog._mp4_crf.isEnabled())
                    dialog._webp_quality.setValue(100)
                    dialog._output.setText(str(Path(folder)/'anim.webp'))
                    dialog._paths = [Path(folder)/'a.png', Path(folder)/'b.png']
                    with patch('planetary_tools.ui.animate_dialog._RunWorker') as worker:
                        dialog._run()
                        self.assertEqual(worker.call_args.kwargs['webp_quality'], 100)
                finally:
                    dialog._worker = None
                    dialog.close()


if __name__ == '__main__':
    unittest.main()
