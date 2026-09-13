"""Normalization and TIFF writes retain exact samples and source ownership."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from planetary_tools.core.document import ImageDocument
from planetary_tools.io import loader


def original_normalize(source, path):
    arr = np.asarray(source)
    gray = arr.ndim == 2 or arr.shape[2] == 1
    if arr.ndim == 3:
        arr = arr[...,0] if gray else arr[...,:3]
    bits = loader._storage_bits(arr, path)
    linear = loader._is_probably_linear(path, arr)
    if arr.dtype == np.uint8:
        f = arr.astype(np.float32)/255.
        if not linear:
            f = loader.srgb_to_linear(f)
    elif arr.dtype == np.uint16:
        f = arr.astype(np.float32)/65535.
        if not linear:
            f = loader.srgb_to_linear(f)
    elif arr.dtype in (np.float32, np.float64):
        f = arr.astype(np.float32)
        if f.max() > 1.5:
            f = f/65535.
    else:
        f = arr.astype(np.float32)
        if f.max() > 1.:
            f = f/f.max()
    f = np.clip(f, 0., None).astype(np.float32)
    if gray:
        f = np.stack([f,f,f], axis=-1)
    return f, False, bits


class IoBufferPerformanceTests(unittest.TestCase):
    def test_normalization_exact_bits_layouts_dtypes_and_owned_output(self):
        rng = np.random.default_rng(16)
        cases = [np.arange(65536, dtype=np.uint16).reshape(256,256),
                 np.arange(256, dtype=np.uint8).reshape(16,16),
                 rng.integers(-200,1000,(19,23,4),dtype=np.int16),
                 rng.uniform(-.1,1.4,(19,23,3)).astype(np.float32),
                 rng.uniform(-100,65535,(19,23,3)),
                 rng.integers(0,100000,(19,23,1),dtype=np.uint32)]
        for raw in cases:
            for source in (raw, raw[::-1, ::2], np.asfortranarray(raw)):
                before = source.copy()
                source.setflags(write=False)
                for path in (Path('fixture.png'), Path('fixture.fits')):
                    expected, gray, bits = original_normalize(source,path)
                    actual, new_gray, new_bits = loader._normalize_array(source,path)
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                    self.assertEqual((new_gray,new_bits),(gray,bits))
                    self.assertFalse(np.shares_memory(actual,source))
                    self.assertTrue(actual.flags.writeable)
                    np.testing.assert_array_equal(source,before)

    def test_normalization_thresholds_and_nonfinite_bits(self):
        for peak in (np.nextafter(np.float32(1.5),np.float32(0)), np.float32(1.5),
                     np.nextafter(np.float32(1.5),np.float32(2)), np.inf, np.nan):
            source = np.array([[-0., 0., -.01, .5, peak]],dtype=np.float32)
            with np.errstate(invalid='ignore'):
                expected = original_normalize(source,Path('float.tif'))[0]
                actual = loader._normalize_array(source,Path('float.tif'))[0]
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

    def test_float_tiff_bytes_match_copying_writer_for_strided_and_hdr_data(self):
        rng = np.random.default_rng(23)
        rgb = rng.uniform(-.2,2.,(19,23,3)).astype(np.float32)
        rgb[0,0] = [np.nan, np.inf, -0.]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'output.tif'
            for source in (rgb, rgb[::-1,::2], np.asfortranarray(rgb), rgb.astype(np.float64), rgb[...,0]):
                before = source.copy()
                source.setflags(write=False)
                gray = source.ndim == 2
                options = {'photometric':'minisblack'} if gray else {}
                tifffile.imwrite(path,source.astype(np.float32),**options)
                expected = path.read_bytes()
                doc = ImageDocument(source,is_grayscale=gray,modified=True,storage_bits=16)
                loader.save_image(doc,path,bit_depth=32)
                self.assertEqual(path.read_bytes(),expected)
                self.assertEqual((doc.path,doc.modified,doc.storage_bits),(path,False,32))
                np.testing.assert_array_equal(source,before)

    def test_native_float_tiff_write_uses_source_and_failure_keeps_metadata(self):
        source = np.ones((9,11,3),dtype=np.float32)
        source.setflags(write=False)
        doc = ImageDocument(source,path=Path('original.tif'),modified=True,storage_bits=16)
        with patch.object(loader.tifffile,'imwrite',side_effect=OSError('write failed')) as write:
            with self.assertRaisesRegex(OSError,'write failed'):
                loader.save_image(doc,Path('new.tif'),bit_depth=32)
            self.assertIs(write.call_args.args[1],source)
        self.assertEqual((doc.path,doc.modified,doc.storage_bits),(Path('original.tif'),True,16))


if __name__ == '__main__':
    unittest.main()
