"""Pixel-only loading must preserve worker outputs and editing noise context."""
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile
from scipy.ndimage import shift

from planetary_tools.batch import pipeline
from planetary_tools.core import animate
from planetary_tools.core.document import ImageDocument
from planetary_tools.io import loader
from planetary_tools.ui import field_derotate_dialog as alignment
from test_alignment import planet


def eager_load(path, **_kwargs):
    return loader.load_image(path)


class WorkerLoadingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        source = planet((97, 113))
        self.paths = []
        for i in range(3):
            path = self.root / f'frame_{i}.tif'
            tifffile.imwrite(path, shift(source, (.37*i,-.64*i), order=3))
            self.paths.append(path)

    def no_pinning(self):
        return patch.object(ImageDocument, 'pin_noise_context', side_effect=AssertionError('Worker pinned unused noise context'))

    def test_default_loader_still_pins_source_and_optional_context_is_lazy(self):
        for path in self.paths:
            expected = loader.load_image(path)
            self.assertIsNotNone(expected.noise_texture_scale)
            self.assertIsNotNone(expected.noise_chromatic)
            with self.no_pinning():
                actual = loader.load_image(path, pin_noise=False)
            np.testing.assert_array_equal(actual.data, expected.data)
            for attribute in ('path', 'is_grayscale', 'modified', 'storage_bits'):
                self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
            self.assertIsNone(actual.noise_texture_scale)
            self.assertIsNone(actual.noise_chromatic)
            self.assertEqual(actual.noise_context(), expected.noise_context())
            pinned = expected.noise_context()
            expected.set_data(expected.data*.5)
            self.assertEqual(expected.noise_context(), pinned)

    def test_batch_auto_outputs_and_progress_match_eager_loading(self):
        def run():
            progress = []
            result = pipeline.run_batch(self.paths, self.root/'batch', [pipeline.PipelineStep(
                'adaptive_deconv', {'auto':True, 'target_noise':8., 'target_contrast':15.},
            )], bit_depth=32, on_progress=lambda *values: progress.append(values))
            self.assertEqual(result.processed, 3)
            self.assertEqual(result.failed, [])
            files = [path.read_bytes() for path in sorted((self.root/'batch').glob('*.tif'))]
            return asdict(result), files, progress
        with patch.object(pipeline, 'load_image', side_effect=eager_load):
            expected = run()
        with self.no_pinning():
            self.assertEqual(run(), expected)

    def test_animation_files_and_progress_match_eager_loading(self):
        for fmt in animate.FORMATS:
            if fmt == 'mp4' and shutil.which('ffmpeg') is None:
                continue  # Optional encoder; MP4 integration tests report the skip.
            def run():
                progress = []
                result = animate.write_animation(self.paths, self.root/f'animation.{fmt}',
                                                  fps=12, fmt=fmt, on_progress=lambda *values: progress.append(values))
                return asdict(result), result.path.read_bytes(), progress
            with patch.object(animate, 'load_image', side_effect=eager_load):
                expected = run()
            with self.no_pinning():
                self.assertEqual(run(), expected)

    def test_alignment_workers_keep_matches_exports_and_progress(self):
        def run_worker(worker):
            completed, errors, progress = [], [], []
            worker.finished_ok.connect(completed.append)
            worker.failed.connect(errors.append)
            worker.progress.connect(lambda *values: progress.append(values))
            worker.run()
            self.assertEqual(errors, [])
            self.assertEqual(len(completed), 1)
            return completed[0], progress
        def run():
            matches, estimate_progress = run_worker(alignment._EstimateWorker(self.paths, 0, 3., False))
            result, export_progress = run_worker(alignment._RunWorker(
                self.paths, matches, self.root/'align', '_aligned', 32, True, 0,
            ))
            self.assertEqual(result.processed, 3)
            self.assertEqual(result.failed, [])
            files = [path.read_bytes() for path in sorted((self.root/'align').glob('*.tif'))]
            return [asdict(match) for match in matches], asdict(result), files, estimate_progress, export_progress
        with patch.object(alignment, 'load_image', side_effect=eager_load):
            expected = run()
        with self.no_pinning():
            self.assertEqual(run(), expected)


if __name__ == '__main__':
    unittest.main()
