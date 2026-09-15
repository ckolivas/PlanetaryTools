"""Timestamp rounding, real motion interpolation, export, and dialog wiring."""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication
from scipy.ndimage import shift

from planetary_tools.core import animate
from planetary_tools.core import animation_interpolation as motion
from planetary_tools.ui.animate_dialog import AnimateDialog, _RunWorker


class TimestampTests(unittest.TestCase):
    def test_winjupos_and_pvol_fractional_minutes_and_seconds(self):
        for name, second in (
            ('2026-09-11-1506_4-CK-L3-Sat_processed.png', 24),
            ('2026-09-11-1506.4-CK.png', 24),
            ('20260911_1506.25_Observer.tif', 15),
            ('s2026-09-11_15-06-24_rgb_ck.png', 24),
            ('j2026-09-11_15-06_rgb_ck.png', 0),
            ('2026-09-11-1506-CK.png', 0),
        ):
            with self.subTest(name=name):
                self.assertEqual(motion.filename_timestamp(Path('/unrelated/2025-01-01-0000')/name),
                                 datetime(2026, 9, 11, 15, 6, second, tzinfo=timezone.utc))
        for name in ('image.png', '2026-02-30-1506_4.png', 's2026-09-11_25-00.png',
                     '2026-09-11-1560_1.png', 's2026-09-11_15-06-60.png'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                motion.filename_timestamp(name)

    def test_rounding_half_up_midnight_and_fractional_interval(self):
        for name, interval, expected in (
            ('2026-09-11-2359_4.png', 1, '2026-09-11T23:59:00+00:00'),
            ('2026-09-11-2359_5.png', 1, '2026-09-12T00:00:00+00:00'),
            ('2026-09-11-1506_5.png', 1, '2026-09-11T15:07:00+00:00'),
            ('2026-09-11-1506_4.png', .5, '2026-09-11T15:06:30+00:00'),
            ('2026-09-11-1506_4.png', 5, '2026-09-11T15:05:00+00:00'),
        ):
            with self.subTest(name=name, interval=interval):
                self.assertEqual(motion.rounded_timestamp(motion.filename_timestamp(name), interval).isoformat(), expected)

    def test_chronological_irregular_gaps_collisions_and_limits(self):
        paths = [Path(name) for name in ('s2026-09-12_00-05.png', '2026-09-11-2358_4.png',
                                        '2026-09-12-0000_3.png')]
        plan = motion.build_timeline(paths)
        self.assertEqual([f.path for f in plan.sources], [paths[1], paths[2], paths[0]])
        self.assertEqual(plan.gaps, (2, 5))
        self.assertEqual(plan.frame_count, 8)
        with self.assertRaisesRegex(ValueError, 'both round'):
            motion.build_timeline([Path('2026-09-11-1506_1.png'), Path('s2026-09-11_15-06-24.png')])
        with self.assertRaisesRegex(ValueError, 'maximum'):
            motion.build_timeline([Path('2025-09-11-1506.png'), Path('2026-09-11-1506.png')])
        for interval in (0, -1, float('nan'), float('inf'), 1441):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                motion.build_timeline(paths, interval)


@unittest.skipUnless(shutil.which('ffmpeg'), 'Motion interpolation requires FFmpeg')
class MotionTests(unittest.TestCase):
    def frames(self):
        rng = np.random.default_rng(31)
        first = np.zeros((97, 129, 3), dtype=np.uint8)
        first[24:64, 24:64] = rng.integers(50, 240, (40, 40, 1), dtype=np.uint8)
        return first, shift(first, (0, 12, 0), order=0)

    def test_motion_moves_texture_instead_of_crossfading(self):
        first, last = self.frames()
        before = first.copy(), last.copy()
        generated = motion.interpolate_pair(first, last, 4)
        self.assertEqual(len(generated), 3)
        for index, frame in enumerate(generated, 1):
            expected = shift(first, (0, index*3, 0), order=0)
            blend = (1-index/4)*first + index/4*last
            error = np.mean((frame.astype(float)-expected)**2)
            self.assertLess(error, 1)
            self.assertLess(error, np.mean((blend-expected)**2) * .01)
        np.testing.assert_array_equal(first, before[0])
        np.testing.assert_array_equal(last, before[1])

    def test_real_apng_export_timestamp_order_endpoints_and_pingpong(self):
        first, last = self.frames()
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/'2026-09-11-1500_2.png', Path(folder)/'s2026-09-11_15-04-12.png']
            for path, frame in zip(paths, (first, last)):
                Image.fromarray(frame).save(path)
            progress = []
            result = animate.write_animation(paths[::-1], Path(folder)/'out.png', fps=10, fmt='apng',
                                             motion_interpolation=True,
                                             on_progress=lambda *args: progress.append(args))
            self.assertEqual(result.frames, 8)
            with Image.open(result.path) as image:
                self.assertEqual(image.n_frames, 8)
                image.seek(0)
                np.testing.assert_array_equal(np.asarray(image.convert('RGB')), first)
                image.seek(2)
                middle = np.asarray(image.convert('RGB'))
                self.assertLess(np.mean((middle.astype(float)-shift(first, (0, 6, 0), order=0))**2), 1)
                image.seek(4)
                np.testing.assert_array_equal(np.asarray(image.convert('RGB')), last)
                image.seek(6)
                np.testing.assert_array_equal(np.asarray(image.convert('RGB')), middle)
            self.assertEqual(progress[-1][:2], (4, 4))
            self.assertTrue(any('Interpolating' in p[2] for p in progress))

    def test_small_images_missing_ffmpeg_and_failed_process_cleanup(self):
        first = np.zeros((9, 11, 3), dtype=np.uint8)
        result = motion.interpolate_pair(first, first, 2)
        self.assertEqual(result[0].shape, first.shape)
        self.assertEqual(motion.interpolate_pair(first, first, 1), [])
        with patch.object(motion.shutil, 'which', return_value=None), self.assertRaisesRegex(RuntimeError, 'requires FFmpeg'):
            motion.interpolate_pair(first, first, 2)
        popen = subprocess.Popen
        children = []
        def fail(_command, **kwargs):
            child = popen([sys.executable, '-c', 'import sys; sys.exit(1)'], **kwargs)
            children.append(child)
            return child
        with patch.object(motion.subprocess, 'Popen', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'failed'):
            motion.interpolate_pair(first, first, 2)
        self.assertEqual(children[0].poll(), 1)


class InterpolationDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_defaults_table_rounding_controls_and_worker_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder)/'prefs.ini'), QSettings.Format.IniFormat)
            with patch('planetary_tools.ui.recent_files._settings', return_value=settings):
                dialog = AnimateDialog()
                try:
                    self.assertFalse(dialog._motion_interpolation.isChecked())
                    self.assertEqual(dialog._frame_interval.value(), 1)
                    self.assertFalse(dialog._frame_interval.isEnabled())
                    dialog._append_paths([Path('s2026-09-11_15-04-12.png'), Path('2026-09-11-1500_2.png')])
                    dialog._motion_interpolation.setChecked(True)
                    self.assertTrue(dialog._frame_interval.isEnabled())
                    self.assertIn('5 forward frames', dialog._timing_hint.text())
                    self.assertEqual(dialog._table.item(0, 1).text(), '2026-09-11 15:00:12')
                    self.assertEqual(dialog._table.item(0, 2).text(), '2026-09-11 15:00:00')
                    self.assertTrue(all(not b.isEnabled() for b in dialog._order_buttons))
                    dialog._frame_interval.setValue(2)
                    self.assertIn('3 forward frames', dialog._timing_hint.text())
                    dialog._output.setText(str(Path(folder)/'out.gif'))
                    with patch('planetary_tools.ui.animate_dialog._RunWorker') as worker:
                        dialog._run()
                        self.assertEqual(worker.call_args.kwargs,
                                         {'motion_interpolation': True, 'frame_interval_minutes': 2.0})
                        self.assertFalse(dialog._frame_interval.isEnabled())
                    dialog._worker = None
                    dialog._set_running(False)
                    dialog._motion_interpolation.setChecked(False)
                    self.assertTrue(dialog._table.isColumnHidden(1))
                    self.assertTrue(all(b.isEnabled() for b in dialog._order_buttons))
                finally:
                    dialog._worker = None
                    dialog.close()

    def test_worker_forwards_interpolation_and_invalid_names_fail_before_loading(self):
        paths = [Path('a.png'), Path('b.png')]
        worker = _RunWorker(paths, Path('out.gif'), 10, 'gif', 'best', True,
                            motion_interpolation=True, frame_interval_minutes=.5)
        with patch('planetary_tools.ui.animate_dialog.write_animation') as write:
            worker.run()
            self.assertTrue(write.call_args.kwargs['motion_interpolation'])
            self.assertEqual(write.call_args.kwargs['frame_interval_minutes'], .5)
        with patch.object(animate, 'load_image') as load, self.assertRaisesRegex(ValueError, 'timestamp'):
            animate.write_animation(paths, 'out.gif', fps=10, fmt='gif', motion_interpolation=True)
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
