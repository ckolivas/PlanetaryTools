"""Compare native PNG loading and single-buffer clipping with a Git baseline.

Timings include file reads for decoding and exclude fixture preparation for
clipping. Native Qt imports are warmed as they already are in the GUI app.
Peak traced clipping allocations exclude the input image.
"""
import argparse
import json
from pathlib import Path
import tracemalloc

from image_performance import compare, load_baseline
from planetary_tools.core import brightness
from planetary_tools.io import png_read
from planetary_tools.io import loader


def peak_mib(function):
    tracemalloc.start()
    try:
        result = function()
        return tracemalloc.get_traced_memory()[1] / 1024**2
    finally:
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='bd529d5')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old_png = load_baseline(args.baseline, 'io/png_read')
    old_brightness = load_baseline(args.baseline, 'core/brightness')
    old_loader = load_baseline(args.baseline, 'io/loader')
    old_loader.read_png_ihdr = old_png.read_png_ihdr
    old_loader.read_png_rgb16 = old_png.read_png_rgb16
    doc = loader.load_image(args.image)
    # Normalize the peak so both undershoot and highlight clamping are exercised.
    data = doc.data / max(float(doc.data.max()), 1e-6) * 1.4 - .02
    report = {'image': str(args.image), 'shape': list(data.shape), 'baseline': args.baseline, 'repeats': args.repeats}
    report['png_rgb16_decode'] = compare(
        lambda: old_png.read_png_rgb16(args.image), lambda: png_read.read_png_rgb16(args.image), args.repeats)
    report['png_header'] = compare(
        lambda: old_png.read_png_ihdr(args.image), lambda: png_read.read_png_ihdr(args.image), args.repeats)
    def loaded(module):
        image = module.load_image(args.image)
        return image.data, image.is_grayscale, image.storage_bits, image.noise_texture_scale, image.noise_chromatic
    report['complete_image_load'] = compare(lambda: loaded(old_loader), lambda: loaded(loader), args.repeats)
    for black, low in ((True, False), (False, True)):
        key = 'clip_black_and_clamp_high' if black else 'clamp_full_range'
        options = dict(clip_black=black, clamp_high=True, clamp_low=low)
        before = lambda: old_brightness.apply_channel_post_process(data, doc.is_grayscale, **options)
        after = lambda: brightness.apply_channel_post_process(data, doc.is_grayscale, **options)
        report[key] = compare(before, after, args.repeats)
        report[key]['before_peak_mib'] = peak_mib(before)
        report[key]['after_peak_mib'] = peak_mib(after)
    report['parity'] = 'all uint16 decoded samples and float32 postprocessed pixels identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
