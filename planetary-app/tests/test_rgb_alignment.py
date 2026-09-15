"""RGB alignment shares Derotate/Align estimation and preserves image data."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtWidgets import QApplication
from scipy.ndimage import gaussian_filter, shift

from planetary_tools.core.align import align_channel
from planetary_tools.core.document import ImageDocument
from planetary_tools.core import field_derotate as core
from planetary_tools.ui.field_derotate_dialog import _EstimateWorker
from planetary_tools.ui.main_window import MainWindow
from test_alignment import planet


class RgbAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_exact_parity_with_align_menu_and_render_from_unmasked_source(self):
        reference = planet()
        target = shift(gaussian_filter(reference, 2), (13.38, -20.71), order=3) * .7 + .02
        original = target.copy()
        paths = [Path('green.tif'), Path('red.tif')]
        images = dict(zip(paths, (reference, target)))
        worker = _EstimateWorker(paths, 0, 45, False)
        results, errors = [], []
        worker.finished_ok.connect(results.append)
        worker.failed.connect(errors.append)
        with patch('planetary_tools.ui.field_derotate_dialog.load_image',
                   side_effect=lambda path, **_: ImageDocument(images[path], is_grayscale=True)):
            worker.run()
        self.assertEqual(errors, [])
        expected_match = results[0][1]
        np.testing.assert_allclose([expected_match.dy, expected_match.dx], [-13.38, 20.71], atol=.025)
        with patch.object(core, 'apply_rigid', wraps=core.apply_rigid) as render:
            actual = align_channel(reference, target)
            render.assert_called_once()
            self.assertIs(render.call_args.args[0], target)
            self.assertEqual(render.call_args.args[1], expected_match)
            self.assertFalse(render.call_args.kwargs['expand'])
        np.testing.assert_array_equal(actual, core.apply_rigid(target, expected_match, expand=False))
        np.testing.assert_array_equal(target, original)
        self.assertEqual(actual.shape, reference.shape)
        self.assertAlmostEqual(float(actual[30, 30]), .02, places=6)

    def test_large_fractional_offsets_with_different_seeing(self):
        reference = planet()
        for sigma in (0, 1, 2, 3):
            with self.subTest(sigma=sigma):
                target = shift(gaussian_filter(reference, sigma), (13.38, -20.71), order=3) * .7 + .02
                aligned = align_channel(reference, target)
                expected = shift(target, (-13.38, 20.71), order=3)
                np.testing.assert_allclose(aligned[30:-30, 30:-30], expected[30:-30, 30:-30], atol=.006)

    def test_fixed_canvas_renders_fractional_and_integer_shifts(self):
        target = planet()
        for dy, dx in ((0, 0), (3, -8), (-.47, 12.31)):
            with self.subTest(dy=dy, dx=dx):
                actual = core.apply_rigid(target, core.RigidMatch(0, dy, dx, 1), expand=False)
                order = 0 if dy == int(dy) and dx == int(dx) else 3
                np.testing.assert_array_equal(actual, shift(target, (dy, dx), order=order))
                self.assertEqual(actual.shape, target.shape)

    def test_rgb_action_keeps_green_and_supports_undo(self):
        green = planet()
        data = np.stack([shift(green, (8.38, -11.71), order=3) * .7 + .02,
                         green, shift(gaussian_filter(green, 2), (-6.2, 9.4), order=3)], axis=-1)
        window = MainWindow()
        window._set_document(ImageDocument(data.copy()))
        try:
            expected = np.stack([align_channel(green, data[..., 0]), green,
                                 align_channel(green, data[..., 2])], axis=-1)
            window._run_align_rgb()
            np.testing.assert_array_equal(window._document.data, expected)
            np.testing.assert_array_equal(window._document.data[..., 1], green)
            window._undo_action()
            np.testing.assert_array_equal(window._document.data, data)
            window._redo_action()
            np.testing.assert_array_equal(window._document.data, expected)
        finally:
            window._document.modified = False
            window.close()


if __name__ == '__main__':
    unittest.main()
