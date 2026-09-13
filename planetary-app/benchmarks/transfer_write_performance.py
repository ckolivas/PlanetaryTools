"""Compare float transfers and byte-identical PNG writing against Git.

Input creation and PNG verification are excluded from timed operations.
Conversion peaks include output buffers; existing inputs are excluded.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.core import colour
from planetary_tools.io import png_read, png_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='91aeb13')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old_colour = load_baseline(args.baseline, 'core/colour')
    old_writer = load_baseline(args.baseline, 'io/png_write')
    samples = png_read.read_png_rgb16(args.image)
    encoded = samples.astype(np.float32) / 65535
    linear = colour.srgb_to_linear(encoded)
    report = {'image': str(args.image), 'shape': list(samples.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    for name, data in (('srgb_to_linear', encoded), ('linear_to_srgb', linear)):
        for clamp in (True, False):
            before = lambda: getattr(old_colour, name)(data, clamp=clamp)
            after = lambda: getattr(colour, name)(data, clamp=clamp)
            key = f'{name}_clamp_{clamp}'
            report[key] = compare(before, after, args.repeats)
            report[key].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    with tempfile.TemporaryDirectory() as directory:
        before_path, after_path = Path(directory)/'before.png', Path(directory)/'after.png'
        before = lambda: old_writer.write_png_rgb16(before_path, samples)
        after = lambda: png_write.write_png_rgb16(after_path, samples)
        report['png_rgb16_write'] = compare(before, after, args.repeats)
        report['png_rgb16_write'].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
        assert before_path.read_bytes() == after_path.read_bytes()
    report['parity'] = 'all transfer pixels and complete PNG bytes identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
