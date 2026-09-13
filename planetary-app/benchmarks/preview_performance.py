"""Compare preview evaluation and auto-sharpen trials with a baseline revision.

PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
    planetary-app/benchmarks/preview_performance.py 3moons.png
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import numpy as np

from planetary_tools.filters.registry import apply_filter, output_filter_stats, apply_filter_and_output_stats
from planetary_tools.filters.wavelet_auto import _SharpenTrialEngine
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='cc5c12d')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    doc = load_image(args.image)
    image, gray = doc.data, doc.is_grayscale
    report = {'image': str(args.image), 'shape': list(image.shape), 'repeats': args.repeats}
    for name in ('deglow', 'wavelet_sharpen', 'adaptive_deconv'):
        before, after = [], []
        for _ in range(args.repeats):
            start = time.perf_counter()
            expected = apply_filter(name, image, gray)
            stats = output_filter_stats(name, image, gray, texture_scale=2, chromatic=False)
            before.append(time.perf_counter()-start)
            start = time.perf_counter()
            result, combined = apply_filter_and_output_stats(name, image, gray, texture_scale=2, chromatic=False)
            after.append(time.perf_counter()-start)
            np.testing.assert_array_equal(expected, result)
            assert combined == stats
        report[name] = {'before_seconds': statistics.median(before), 'after_seconds': statistics.median(after)}
    # Load only the old trial engine; its imported image operations still use
    # current code, whose unsharp arithmetic has separate exact-parity tests.
    old = types.ModuleType('_benchmark_old_wavelet_auto')
    sys.modules[old.__name__] = old
    source = subprocess.check_output(
        ['git', 'show', f'{args.baseline}:planetary-app/planetary_tools/filters/wavelet_auto.py'],
        text=True,
    )
    exec(compile(source, '<baseline wavelet_auto>', 'exec'), old.__dict__)
    before, after = [], []
    for _ in range(args.repeats):
        outputs = []
        for cls, timings in ((old._SharpenTrialEngine, before), (_SharpenTrialEngine, after)):
            engine = cls(image, gray, texture_scale=2, chromatic=False)
            start = time.perf_counter()
            outputs.append([engine.apply(fine,8,1,0) for fine in (2,4,6,8,10,12,14,16)])
            timings.append(time.perf_counter()-start)
            del engine
        for expected, result in zip(*outputs):
            np.testing.assert_array_equal(expected, result)
    report['eight_auto_trials'] = {'before_seconds': statistics.median(before), 'after_seconds': statistics.median(after)}
    for value in report.values():
        if isinstance(value, dict):
            value['speedup'] = value['before_seconds']/value['after_seconds']
    report['parity'] = 'all pixels and statistics identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
