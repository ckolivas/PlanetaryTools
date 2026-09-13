"""Multi-folder batches must not overwrite another input's output."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from planetary_tools.batch.pipeline import PipelineStep, planned_output_paths, run_batch
from planetary_tools.core.document import ImageDocument
from planetary_tools.io.loader import load_image, save_image


class BatchOutputTests(unittest.TestCase):
    def test_colliding_names_export_distinct_images_using_planned_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root/'a'/'saturn.tif', root/'b'/'saturn.tif', root/'b'/'saturn_2.tif']
            for index, path in enumerate(inputs):
                path.parent.mkdir(exist_ok=True)
                save_image(ImageDocument(np.full((8, 8), .1*(index+1), dtype=np.float32),
                                         is_grayscale=True), path, bit_depth=32)
            outputs = planned_output_paths(inputs, root/'out', suffix='')
            self.assertEqual([p.name for p in outputs],
                             ['saturn.tif', 'saturn_3.tif', 'saturn_2.tif'])
            with patch('planetary_tools.batch.pipeline.apply_pipeline', side_effect=lambda data, *_: data):
                result = run_batch(inputs, root/'out', [PipelineStep('curves')], suffix='')
            self.assertEqual(result.processed, 3)
            self.assertEqual(result.failed, [])
            for index, path in enumerate(outputs):
                np.testing.assert_allclose(load_image(path).data, .1*(index+1), atol=1e-6)

    def test_preserved_folders_keep_their_original_filenames(self):
        inputs = [Path('/input/a/saturn.png'), Path('/input/b/saturn.png')]
        self.assertEqual(
            planned_output_paths(inputs, Path('/output'), preserve_tree=True, input_root=Path('/input')),
            [Path('/output/a/saturn_processed.png'), Path('/output/b/saturn_processed.png')],
        )


if __name__ == '__main__':
    unittest.main()
