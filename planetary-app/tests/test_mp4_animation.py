"""Real MP4 encode/decode checks plus export failure and UI coverage."""
import json
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

from planetary_tools.core import animate
from planetary_tools.ui.animate_dialog import AnimateDialog, _RunWorker


class Mp4Tests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        rng = np.random.default_rng(314)
        self.frames = [rng.integers(0, 256, (17, 23, 3), dtype=np.uint8) for _ in range(5)]
        self.frames[2] = self.frames[2][:, ::-1]

    def require_encoder(self):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            self.skipTest('FFmpeg/ffprobe unavailable')
        encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], stderr=subprocess.DEVNULL)
        if b'libx264rgb' not in encoders:
            self.skipTest('FFmpeg lacks libx264rgb')

    def decode(self, path, height=17, width=23):
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                       '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
        return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)

    def test_lossless_order_odd_dimensions_and_exact_frame_rate(self):
        self.require_encoder()
        originals = [frame.copy() for frame in self.frames]
        for count, reverse in ((2, True), (3, False), (5, True)):
            with self.subTest(count=count, reverse=reverse):
                frames = self.frames[:count]
                path = self.root/'video.mp4'
                result = animate.encode_frames(frames, path, fps=29.9, fmt='mp4', back_and_forth=reverse)
                expected = animate.expand_back_and_forth(frames) if reverse else frames
                np.testing.assert_array_equal(self.decode(path), expected)
                probe = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
                self.assertEqual(probe['codec_name'], 'h264')
                self.assertEqual(probe['avg_frame_rate'], '299/10')
                self.assertEqual(int(probe['nb_frames']), len(expected))
                self.assertAlmostEqual(float(probe['duration']), len(expected)/29.9, places=5)
                self.assertEqual((result.width, result.height, result.frames), (23, 17, len(expected)))
        for actual, original in zip(self.frames, originals):
            np.testing.assert_array_equal(actual, original)

    def test_constant_quality_changes_compression(self):
        self.require_encoder()
        sizes = []
        for crf in (0, 35, 51):
            path = self.root/f'quality-{crf}.mp4'
            animate.encode_frames(self.frames, path, fps=10, fmt='mp4', mp4_crf=crf, back_and_forth=False)
            actual = self.decode(path)
            self.assertEqual(actual.shape, (5, 17, 23, 3))
            sizes.append(path.stat().st_size)
            if crf:
                self.assertFalse(np.array_equal(actual, self.frames))
        self.assertLess(sizes[1], sizes[0])
        self.assertLess(sizes[2], sizes[1])

    def test_loader_padding_and_worker_forward_quality(self):
        self.require_encoder()
        paths = []
        frames = [self.frames[0], self.frames[1][:11, :15], self.frames[2]]
        for i, frame in enumerate(frames):
            path = self.root/f'frame{i}.png'
            Image.fromarray(frame).save(path)
            paths.append(path)
        progress = []
        result = animate.write_animation(paths, self.root/'padded.mp4', fps=12.5, fmt='mp4',
                                          on_progress=lambda *args: progress.append(args))
        expected = animate.expand_back_and_forth(animate.pad_frames(frames))
        np.testing.assert_array_equal(self.decode(result.path), expected)
        self.assertEqual(progress[-1], (4, 4, 'Done'))
        worker = _RunWorker(paths, result.path, 12.5, 'mp4', 'best', False, 23)
        with patch('planetary_tools.ui.animate_dialog.write_animation', return_value=result) as write:
            worker.run()
        self.assertEqual(write.call_args.kwargs['mp4_crf'], 23)
        self.assertFalse(write.call_args.kwargs['back_and_forth'])

    def test_validation_and_missing_encoder_preserve_existing_output(self):
        path = self.root/'keep.mp4'
        path.write_bytes(b'previous video')
        for crf in (-1, 52, 0.5, '0'):
            with self.subTest(crf=crf), self.assertRaisesRegex(ValueError, 'constant quality'):
                animate.encode_frames(self.frames, path, fps=10, fmt='mp4', mp4_crf=crf)
        for fps in (0, -1, float('nan'), float('inf')):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, 'Frame rate'):
                animate.encode_frames(self.frames, path, fps=fps, fmt='mp4')
        with patch.object(animate.shutil, 'which', return_value=None), self.assertRaisesRegex(RuntimeError, 'requires FFmpeg'):
            animate.encode_frames(self.frames, path, fps=10, fmt='mp4')
        self.assertEqual(path.read_bytes(), b'previous video')

    def test_failed_encoder_is_reaped_and_partial_file_removed(self):
        path = self.root/'keep.mp4'
        path.write_bytes(b'previous video')
        popen = subprocess.Popen
        for code in (0, 1):
            children = []
            def spawn(_args, **kwargs):
                child = popen([sys.executable, '-c', f'import sys; sys.exit({code})'], **kwargs)
                children.append(child)
                return child
            with patch.object(animate.shutil, 'which', return_value=sys.executable), \
                 patch.object(animate.subprocess, 'Popen', side_effect=spawn), \
                 self.assertRaisesRegex(RuntimeError, 'MP4 export failed'):
                animate.encode_frames(self.frames, path, fps=10, fmt='mp4')
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0].poll(), code)
            self.assertTrue(children[0].stdin.closed)
            self.assertEqual(path.read_bytes(), b'previous video')
            self.assertEqual(list(self.root.iterdir()), [path])


class Mp4DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_controls_picker_persistence_and_busy_close(self):
        with tempfile.TemporaryDirectory() as folder:
            def settings():
                return QSettings(str(Path(folder)/'prefs.ini'), QSettings.Format.IniFormat)
            with patch('planetary_tools.ui.recent_files._settings', settings):
                dialog = AnimateDialog()
                try:
                    self.assertEqual(dialog._mp4_crf.value(), 0)
                    self.assertFalse(dialog._mp4_crf.isEnabled())
                    with patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName',
                               return_value=(str(Path(folder)/'video.gif'), 'MP4 video (*.mp4)')):
                        dialog._browse_output()
                    self.assertEqual(dialog._fmt(), 'mp4')
                    self.assertTrue(dialog._mp4_crf.isEnabled())
                    self.assertFalse(dialog._gif_quality.isEnabled())
                    self.assertTrue(dialog._output.text().endswith('.mp4'))
                    dialog._mp4_crf.setValue(23)
                    dialog._paths = [Path(folder)/'a.png', Path(folder)/'b.png']
                    with patch('planetary_tools.ui.animate_dialog._RunWorker') as worker:
                        dialog._run()
                        self.assertEqual(worker.call_args.args[-1], 23)
                        worker.return_value.isRunning.return_value = True
                        with patch('PyQt6.QtWidgets.QMessageBox.warning') as warning:
                            dialog.reject()
                            warning.assert_called_once()
                    dialog._worker = None
                    reopened = AnimateDialog()
                    self.assertEqual(reopened._fmt(), 'mp4')
                    self.assertEqual(reopened._mp4_crf.value(), 0)
                    reopened.close()
                finally:
                    dialog._worker = None
                    dialog.close()


if __name__ == '__main__':
    unittest.main()
