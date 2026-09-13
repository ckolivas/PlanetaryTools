"""Compare image measurements, histograms and detail merging against Git.

Run with PYTHONPATH=planetary-app and the app's Python environment.
Timings exclude file loading and assert exact result parity.
"""
import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import numpy as np

from planetary_tools.core import brightness
from planetary_tools.filters import wavelet
from planetary_tools.filters.registry import _near_identity
from planetary_tools.io.loader import load_image
from planetary_tools.ui import histogram


def load_baseline(revision, path):
    module = types.ModuleType('_benchmark_' + path.replace('/', '_').replace('.', '_'))
    sys.modules[module.__name__] = module
    source = subprocess.check_output(['git', 'show', f'{revision}:planetary-app/planetary_tools/{path}.py'], text=True)
    exec(compile(source, f'<baseline {path}>', 'exec'), module.__dict__)
    return module


def assert_equal(expected, actual):
    if isinstance(expected, tuple):
        for left, right in zip(expected, actual):
            assert_equal(left, right)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual)
    elif is_dataclass(expected):
        assert asdict(expected) == asdict(actual)
    else:
        assert expected == actual


def compare(before, after, repeats):
    assert_equal(before(), after())  # Warm both paths and check output first.
    times = [[], []]
    for repeat in range(repeats):
        # Alternate order to reduce warm-cache/order bias.
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            result = (before, after)[index]()
            times[index].append(time.perf_counter() - start)
            del result
    first, second = map(statistics.median, times)
    return {'before_seconds': first, 'after_seconds': second, 'speedup': first / second}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='312545a')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old_brightness = load_baseline(args.baseline, 'core/brightness')
    old_histogram = load_baseline(args.baseline, 'ui/histogram')
    old_wavelet = load_baseline(args.baseline, 'filters/wavelet')
    doc = load_image(args.image)
    data, gray = doc.data, doc.is_grayscale
    output = data * np.float32(1.15)
    secondary = data[::2, ::2]
    def levels(module):
        return module.measure_brightness(output, gray), module.brightness_increase_pct(data, output, gray)
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    report['brightness_and_increase'] = compare(lambda: levels(old_brightness), lambda: levels(brightness), args.repeats)
    def old_identity(raw):
        raw_f = np.asarray(raw, dtype=np.float64)
        source_f = np.asarray(data, dtype=np.float64)
        return float(np.max(np.abs(raw_f - source_f))) < 1e-5
    for name, raw in (('unchanged', data), ('changed', output)):
        report[f'identity_check_{name}'] = compare(
            lambda: old_identity(raw), lambda: _near_identity(raw, data), args.repeats)
    for perceptual in (False, True):
        report['perceptual_histogram' if perceptual else 'linear_histogram'] = compare(
            lambda: old_histogram.compute_rgb_histograms(data, perceptual=perceptual),
            lambda: histogram.compute_rgb_histograms(data, perceptual=perceptual), args.repeats)
    report['merge_detail_with_resize'] = compare(
        lambda: old_wavelet.merge_wavelet_detail(data, secondary, gray),
        lambda: wavelet.merge_wavelet_detail(data, secondary, gray), args.repeats)
    report['parity'] = 'all pixels, histogram bins and brightness statistics identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
