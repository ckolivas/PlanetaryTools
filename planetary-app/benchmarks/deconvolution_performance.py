"""Compare deconvolution and complete Auto searches with a baseline revision.

Run with PYTHONPATH=planetary-app and the app's Python environment.
Includes preparation costs, checks exact pixels, selected parameters and all
progress values, and excludes file loading. No global image cache is used.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from image_performance import compare, load_baseline
from planetary_tools.filters.adaptive_deconv import adaptive_deconvolution
from planetary_tools.filters.adaptive_deconv_auto import auto_adaptive_deconv_params
from planetary_tools.filters.wiener_deconv import wiener_deconvolution
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='20c0c16')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--target-noise', type=float, default=8.)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old = load_baseline(args.baseline, 'filters/adaptive_deconv')
    old_auto = load_baseline(args.baseline, 'filters/adaptive_deconv_auto')
    # Its import would otherwise use today's implementation.
    old_auto.adaptive_deconvolution = old.adaptive_deconvolution
    old_wiener = load_baseline(args.baseline, 'filters/wiener_deconv')
    old_wiener._std_windowed = old._std_windowed
    doc = load_image(args.image)
    data, gray = doc.data, doc.is_grayscale
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats, 'target_noise': args.target_noise, 'target_contrast': 15.}
    for luminance in (True, False):
        mode = 'luminance' if luminance else 'rgb'
        for adaptive in (True, False):
            key = f'deconvolution_{mode}_adaptive_{adaptive}'
            report[key] = compare(
                lambda: old.adaptive_deconvolution(data, gray, 10, adaptive, luminance),
                lambda: adaptive_deconvolution(data, gray, 10, adaptive, luminance), args.repeats)
        def search(function):
            progress = []
            result = function(data, gray, target_noise=args.target_noise, target_contrast=15., adaptive=True,
                              luminance=luminance, texture_scale=2, chromatic=False,
                              progress=lambda *values: progress.append(values))
            return asdict(result), progress
        report[f'auto_{mode}'] = compare(
            lambda: search(old_auto.auto_adaptive_deconv_params),
            lambda: search(auto_adaptive_deconv_params), args.repeats)
        result, progress = search(auto_adaptive_deconv_params)
        report[f'auto_{mode}'].update(result=result, progress_events=len(progress))
    for oklab in (False, True):
        report['wiener_oklab' if oklab else 'wiener_rgb'] = compare(
            lambda: old_wiener.wiener_deconvolution(data, gray, 10, True, oklab),
            lambda: wiener_deconvolution(data, gray, 10, True, oklab), args.repeats)
    report['parity'] = 'all pixels, Auto results and progress values identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
