"""Compare pixel-only loading and complete workers with eager-load Git code."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np
import tifffile

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.batch import pipeline
from planetary_tools.core import animate
from planetary_tools.core.field_derotate import estimate_rigid
from planetary_tools.io import loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='0c2c7f2')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    old_loader = load_baseline(args.baseline, 'io/loader')
    old_batch = load_baseline(args.baseline, 'batch/pipeline')
    old_animate = load_baseline(args.baseline, 'core/animate')
    old_batch.load_image = old_animate.load_image = old_loader.load_image
    fast_load = lambda path: loader.load_image(path, pin_noise=False)
    data = fast_load(args.image).data
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    def loaded(function):
        doc = function(args.image)
        return doc.data, (doc.path, doc.is_grayscale, doc.storage_bits, doc.modified)
    before, after = lambda: loaded(old_loader.load_image), lambda: loaded(fast_load)
    report['pixel_load'] = compare(before, after, args.repeats)
    report['pixel_load'].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        paths = [root/'one.tif', root/'two.tif']
        for path, frame in zip(paths, (data, np.roll(data, 2, axis=1))):
            tifffile.imwrite(path, frame, photometric='rgb' if frame.ndim == 3 else 'minisblack')
        steps = [pipeline.PipelineStep('colour_matrix', {'matrix': [[1.05,-.03,-.02], [-.02,1.04,-.02], [-.01,-.02,1.03]]})]
        def batch(module):
            result = module.run_batch(paths, root/'batch', steps, bit_depth=32)
            assert result.processed == 2 and not result.failed, result
            outputs = tuple(tifffile.imread(path) for path in sorted((root/'batch').glob('*.tif')))
            return asdict(result), outputs
        report['two_frame_tiff_batch'] = compare(lambda: batch(old_batch), lambda: batch(pipeline), args.repeats)
        def animation(module):
            result = module.write_animation(paths, root/'animation.gif', fps=10, fmt='gif')
            return asdict(result), result.path.read_bytes()
        report['two_frame_tiff_to_gif'] = compare(lambda: animation(old_animate), lambda: animation(animate), args.repeats)
        def alignment(function):
            reference, target = (function(path).data for path in paths)
            return asdict(estimate_rigid(reference, target, rotate=False))
        report['load_and_match_shift_only'] = compare(lambda: alignment(old_loader.load_image), lambda: alignment(fast_load), args.repeats)
    report['parity'] = 'loaded pixels/metadata, batch pixels, GIF bytes and alignment matches identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
