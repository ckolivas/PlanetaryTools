"""Align mask controls, worker snapshots, and parity with the approved trial."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.colour import srgb_to_linear
from planetary_tools.core.document import ImageDocument
from planetary_tools.core.field_derotate import (
    IDENTITY_MATCH, mask_alignment_background, pad_to_common,
)
from planetary_tools.ui import field_derotate_dialog as ui


class AlignmentMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def run_worker(self, worker):
        results, errors = [], []
        worker.finished_ok.connect(results.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        return results[0]

    def test_mask_uses_per_image_perceptual_range_and_preserves_source(self):
        for shape in ((2, 2), (2, 2, 3)):
            display = np.array([[.25, .4], [.5, .75]], dtype=np.float32)
            if len(shape) == 3:
                display = np.repeat(display[..., None], 3, axis=2)
            source = srgb_to_linear(display)
            original = source.copy()
            expected = source.copy()
            expected[0, 0] = 0  # .25 + .25 * (.75 - .25) = .375
            np.testing.assert_array_equal(mask_alignment_background(source), expected)
            np.testing.assert_array_equal(mask_alignment_background(source, 0), source)
            np.testing.assert_array_equal(source, original)
            self.assertFalse(np.shares_memory(mask_alignment_background(source), source))
        for invalid in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                mask_alignment_background(source, invalid)

    def test_worker_masks_before_padding_and_export_loads_originals(self):
        a = srgb_to_linear(np.array([[.25, .4], [.5, .75]], np.float32))
        b = np.tile(a, (2, 2))
        paths = [Path('a.tif'), Path('b.tif')]
        arrays = dict(zip(paths, (a, b)))

        def load(path, **_kwargs):
            return ImageDocument(arrays[path], path=path, is_grayscale=True)

        for fraction in (.25, .5, None):
            with self.subTest(fraction=fraction), patch.object(ui, 'load_image', load), \
                    patch.object(ui, 'estimate_rigid', return_value=IDENTITY_MATCH) as estimate:
                matches = self.run_worker(ui._EstimateWorker(paths, 0, 3, False, fraction))
                expected = pad_to_common([
                    mask_alignment_background(p, fraction) if fraction is not None else p
                    for p in (a, b)
                ])
                for got, want in zip(estimate.call_args.args, expected):
                    np.testing.assert_array_equal(got, want)
            with patch.object(ui, 'load_image', load), patch.object(ui, 'derotate_set') as export:
                self.run_worker(ui._RunWorker(paths, matches, Path('unused'), '_test', 16, True, 0))
                for (_, data, _), source in zip(export.call_args.args[0], (a, b)):
                    np.testing.assert_array_equal(data, source)

    def test_controls_invalidate_estimates_and_freeze_during_work(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            'planetary_tools.ui.recent_files._settings',
            lambda: QSettings(str(Path(directory)/'settings.ini'), QSettings.Format.IniFormat),
        ):
            dialog = ui.FieldDerotateDialog()
            self.addCleanup(dialog.close)
            self.assertTrue(dialog._mask_enabled.isChecked())
            self.assertEqual(dialog._mask_percent.value(), 25)
            dialog._rows = [ui._Row(Path('a.tif')), ui._Row(Path('b.tif'))]
            for change in (lambda: dialog._mask_percent.setValue(50),
                           lambda: dialog._mask_enabled.setChecked(False),
                           lambda: dialog._max_angle.setValue(10)):
                dialog._on_estimated([IDENTITY_MATCH, IDENTITY_MATCH])
                self.assertTrue(dialog._run_btn.isEnabled())
                change()
                self.assertFalse(dialog._estimated)
                self.assertTrue(all(row.match is None for row in dialog._rows))
                self.assertFalse(dialog._run_btn.isEnabled())
                self.assertTrue(dialog._est_btn.isEnabled())
            self.assertFalse(dialog._mask_percent.isEnabled())
            for enabled in (False, True):
                dialog._mask_enabled.setChecked(enabled)
                with patch.object(ui, '_EstimateWorker') as worker:
                    worker.return_value.isRunning.return_value = False
                    dialog._estimate()
                    self.assertEqual(worker.call_args.args[-1], .5 if enabled else None)
                    self.assertFalse(dialog._mask_enabled.isEnabled())
                    self.assertFalse(dialog._mask_percent.isEnabled())
                    self.assertFalse(dialog._max_angle.isEnabled())
                    dialog._on_estimated([IDENTITY_MATCH, IDENTITY_MATCH])
                    self.assertTrue(dialog._mask_enabled.isEnabled())
                    self.assertEqual(dialog._mask_percent.isEnabled(), enabled)
                dialog._worker = None

    def test_default_worker_reproduces_approved_25_percent_trial(self):
        directory = Path(__file__).resolve().parents[2] / 'realign'
        report = directory / 'masked-image-range-25pct' / 'matches.json'
        if not report.is_file():
            self.skipTest('Local approved mask trial is not installed')
        expected = json.loads(report.read_text())['frames']
        paths = [directory / item['file'] for item in expected]
        actual = self.run_worker(ui._EstimateWorker(paths, 0, 45, True))
        for match, item in zip(actual, expected):
            with self.subTest(file=item['file']):
                for attribute in ('angle_deg', 'dy', 'dx', 'score'):
                    self.assertAlmostEqual(getattr(match, attribute), item['masked'][attribute], delta=1e-9)


if __name__ == '__main__':
    unittest.main()
