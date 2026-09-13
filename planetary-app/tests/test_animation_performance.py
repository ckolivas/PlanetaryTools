"""Reusing return-trip frames must preserve complete animation files."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from planetary_tools.core import animate


class AnimationPerformanceTests(unittest.TestCase):
    def test_encoded_bytes_match_preexpanded_frames(self):
        rng = np.random.default_rng(48)
        sources = [rng.integers(0, 256, (17, 23, 3), dtype=np.uint8) for _ in range(5)]
        # Include a repeated frame and non-contiguous inputs.
        sources[2] = sources[1]
        sources[3] = sources[3][:, ::-1]
        before = [frame.copy() for frame in sources]
        for frame in sources:
            frame.setflags(write=False)
        with tempfile.TemporaryDirectory() as folder:
            for count in (2, 3, 5):
                for fmt in animate.FORMATS:
                    for quality in (animate.GIF_QUALITIES if fmt == 'gif' else ('best',)):
                        for reverse in (False, True):
                            with self.subTest(count=count, fmt=fmt, quality=quality, reverse=reverse):
                                frames = sources[:count]
                                expanded = animate.expand_back_and_forth(frames) if reverse else frames
                                path = Path(folder) / 'animation'
                                expected = animate.encode_frames(expanded, path, fps=17, fmt=fmt,
                                                                 gif_quality=quality, back_and_forth=False)
                                expected_bytes = path.read_bytes()
                                actual = animate.encode_frames(frames, path, fps=17, fmt=fmt,
                                                               gif_quality=quality, back_and_forth=reverse)
                                self.assertEqual(asdict(actual), asdict(expected))
                                self.assertEqual(path.read_bytes(), expected_bytes)
        for frame, original in zip(sources, before):
            np.testing.assert_array_equal(frame, original)

    def test_return_trip_quantizes_only_source_frames(self):
        frames = [np.full((8, 11, 3), i*40, dtype=np.uint8) for i in range(5)]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(animate, '_quantize_gif', wraps=animate._quantize_gif) as quantize:
                result = animate.encode_frames(frames, Path(folder)/'output.gif', fps=10, fmt='gif')
            self.assertEqual(quantize.call_count, 5)
            self.assertEqual(result.frames, 8)
            with Image.open(result.path) as image:
                self.assertEqual(image.n_frames, 8)
                self.assertEqual(image.info['loop'], 0)
                for index, level in enumerate((0, 40, 80, 120, 160, 120, 80, 40)):
                    image.seek(index)
                    self.assertEqual(image.info['duration'], 100)
                    np.testing.assert_array_equal(np.asarray(image.convert('RGB')), np.full((8,11,3), level))


if __name__ == '__main__':
    unittest.main()
