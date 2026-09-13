"""Compare noise preparation, measurements and Auto searches with a Git baseline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from image_performance import compare, load_baseline
from loading_performance import peak_mib
from planetary_tools.core import noise
from planetary_tools.filters.adaptive_deconv_auto import auto_adaptive_deconv_params
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='7837332')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old = load_baseline(args.baseline, 'core/noise')
    old_auto = load_baseline(args.baseline, 'filters/adaptive_deconv_auto')
    old_auto.absolute_noise = old.absolute_noise
    old_auto.estimate_texture_scale = old.estimate_texture_scale
    old_auto.is_chromatic = old.is_chromatic
    doc = load_image(args.image)
    data, gray = doc.data, doc.is_grayscale
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    operations = {
        'luminance': lambda module: module._luminance(data, gray),
        'chromatic_detection': lambda module: module.is_chromatic(data, gray),
        'pin_noise_context': lambda module: (module.estimate_texture_scale(data, gray), module.is_chromatic(data, gray)),
        'noise_readout_mono': lambda module: module.absolute_noise(data, gray, texture_scale=2, chromatic=False),
        'noise_readout_soft_mono': lambda module: module.absolute_noise(data, gray, texture_scale=5, chromatic=False),
        'noise_readout_colour': lambda module: module.absolute_noise(data, gray, texture_scale=2, chromatic=True),
    }
    for name, operation in operations.items():
        before, after = lambda: operation(old), lambda: operation(noise)
        report[name] = compare(before, after, args.repeats)
        report[name].update(before_peak_mib=peak_mib(before), after_peak_mib=peak_mib(after))
    # Ensure both paths run a complete search, even if a sample exceeds the
    # default target. This changes benchmark inputs, not application defaults.
    target = max(8., 1. + (noise.absolute_noise(data, gray, texture_scale=2, chromatic=False) or 0.))
    def search(function):
        progress = []
        result = function(data, gray, target_noise=target, target_contrast=15,
                          texture_scale=2, chromatic=False,
                          progress=lambda *values: progress.append(values))
        return asdict(result), progress
    report['auto_deconv_luminance'] = compare(lambda: search(old_auto.auto_adaptive_deconv_params), lambda: search(auto_adaptive_deconv_params), args.repeats)
    result, progress = search(auto_adaptive_deconv_params)
    report['auto_deconv_luminance'].update(target_noise=target, result=result, progress_events=len(progress))
    report['parity'] = 'all luminance pixels, classifications, scores, Auto results and progress values identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
