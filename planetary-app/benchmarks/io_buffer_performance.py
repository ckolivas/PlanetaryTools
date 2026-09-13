"""Measure normalization, complete loads and float TIFF writes against Git."""
import argparse
import json
from pathlib import Path
import tempfile

import tifffile

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.core.document import ImageDocument
from planetary_tools.io import loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='1245038')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old = load_baseline(args.baseline, 'io/loader')
    raw = loader._load_array(args.image)
    data = loader.load_image(args.image, pin_noise=False).data
    report = {'image': str(args.image), 'shape': list(data.shape), 'raw_dtype': str(raw.dtype),
              'baseline': args.baseline, 'repeats': args.repeats}
    def record(name, before, after):
        report[name] = compare(before, after, args.repeats)
        report[name].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    record('normalize_encoded',lambda: old._normalize_array(raw,args.image),lambda: loader._normalize_array(raw,args.image))
    def loaded(module, path, pin_noise):
        doc = module.load_image(path,pin_noise=pin_noise)
        return doc.data, (doc.path,doc.is_grayscale,doc.storage_bits,doc.modified,
                          doc.noise_texture_scale,doc.noise_chromatic)
    record('load_png_editing',lambda: loaded(old,args.image,True),lambda: loaded(loader,args.image,True))
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder)/'source.tif', Path(folder)/'output.tif'
        tifffile.imwrite(source,data,photometric='rgb' if data.ndim == 3 else 'minisblack')
        record('normalize_float32',lambda: old._normalize_array(data,source),lambda: loader._normalize_array(data,source))
        record('load_float_tiff_pixels',lambda: loaded(old,source,False),lambda: loaded(loader,source,False))
        def write(module):
            doc = ImageDocument(data,is_grayscale=data.ndim == 2,modified=True)
            module.save_image(doc,target,bit_depth=32)
            return doc.path,doc.modified,doc.storage_bits
        # Compare complete file bytes outside the timed writer, so readback
        # does not dominate the allocation peak saved by avoiding a copy.
        write(old)
        expected = target.read_bytes()
        write(loader)
        assert target.read_bytes() == expected
        del expected
        record('write_float_tiff',lambda: write(old),lambda: write(loader))
    report['parity'] = 'normalized/loaded pixels, metadata, pinned context and complete TIFF bytes identical'
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
