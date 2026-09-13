"""Compare alignment with the bottom of the histogram range masked to black.

Run with PYTHONPATH=planetary-app and the app's Python environment:
  alignment_mask_experiment.py realign realign/masked-25pct

By default the cutoff is a fraction of each image's observed min/max brightness
range. Use --range-mode full for a fixed cutoff on the full 0..1 histogram axis.
Neither mode uses percentiles of the pixel population.
Perceptual sRGB matches the histogram's default encoding.
Only registration copies are masked. Exports resample the original pixels once.
This experiment does not change application defaults.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from planetary_tools.core.colour import linear_to_srgb
from planetary_tools.core.field_derotate import (
    IDENTITY_MATCH, _render_rigid, derotate_set, estimate_rigid, luma,
    mask_alignment_background, pad_to_common,
)
from planetary_tools.io.loader import load_image


def registration_copy(data, cutoff, space, range_mode='image'):
    encoded = linear_to_srgb(data) if space == 'perceptual' else data
    brightness = luma(encoded)
    low, high = float(brightness.min()), float(brightness.max())
    threshold = cutoff if range_mode == 'full' else low + cutoff * (high - low)
    keep = brightness >= threshold
    masked = (mask_alignment_background(data, cutoff)
              if space == 'perceptual' and range_mode == 'image' else
              np.where(keep[..., None] if data.ndim == 3 else keep, data, 0))
    return masked, float(np.mean(keep)), dict(minimum=low, maximum=high, threshold=threshold)


def display(data):
    rgb = np.clip(linear_to_srgb(data), 0, 1)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    return Image.fromarray((rgb[..., :3] * 255 + .5).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory of PNG captures')
    parser.add_argument('output', type=Path, help='New experiment directory')
    parser.add_argument('--cutoff', type=float, default=.25)
    parser.add_argument('--space', choices=['perceptual', 'linear'], default='perceptual')
    parser.add_argument('--range-mode', choices=['image', 'full'], default='image')
    parser.add_argument('--max-angle', type=float, default=45.)
    args = parser.parse_args()
    if not 0 <= args.cutoff < 1:
        parser.error('cutoff must be in [0, 1)')
    if not np.isfinite(args.max_angle) or args.max_angle < 0:
        parser.error('max-angle must be finite and nonnegative')
    paths = sorted(args.input.glob('*.png'))
    if len(paths) < 2:
        parser.error('at least two PNG captures are required')
    if args.output.exists():
        parser.error('choose a new output directory to preserve earlier results')
    loaded = [load_image(p, pin_noise=False).data for p in paths]
    # Measure each native image before common-canvas padding can lower its min.
    copies, fractions, ranges = zip(*(registration_copy(
        a, args.cutoff, args.space, args.range_mode,
    ) for a in loaded))
    originals = pad_to_common(loaded)
    copies = pad_to_common(list(copies))
    if any(f == 0 for f in fractions):
        parser.error('the cutoff masks every pixel in at least one capture')
    args.output.mkdir(parents=True)
    items, preview, report = [], [], []
    h, w = originals[0].shape[:2]
    reference_panel = display(originals[0])
    for i, (path, original, masked) in enumerate(zip(paths, originals, copies)):
        current = estimate_rigid(originals[0], original, max_angle=args.max_angle) if i else IDENTITY_MATCH
        match = estimate_rigid(copies[0], masked, max_angle=args.max_angle) if i else IDENTITY_MATCH
        items.append((path, original, match))
        report.append(dict(file=path.name, brightness=ranges[i], retained_fraction=fractions[i],
                           current=asdict(current), masked=asdict(match),
                           delta_dy=match.dy-current.dy, delta_dx=match.dx-current.dx))
        print(json.dumps(report[-1]), flush=True)
        frame = Image.new('RGB', (3*w, h+24))
        panels = [reference_panel] + [display(_render_rigid(
            original, transform, (h, w), np.zeros(2),
        )) for transform in (current, match)]
        labels = ['Reference', 'Current', f'Masked {args.cutoff:.0%} of {args.range_mode} range']
        for column, (panel, label) in enumerate(zip(panels, labels)):
            frame.paste(panel, (column*w, 24))
            ImageDraw.Draw(frame).text((column*w+10, 5), label, fill='white')
        preview.append(frame)
        frame.save(args.output / f'{path.stem}_comparison.png')
        display(masked).save(args.output / f'{path.stem}_registration_mask.png')

    result = derotate_set(items, args.output / 'frames', suffix='_masked',
                          bit_depth=16, subpixel=True, ref_index=0)
    if result.failed:
        raise RuntimeError(result.failed)
    sequence = preview + preview[-2:0:-1]
    sequence[0].save(args.output / 'comparison.webp', save_all=True,
                     append_images=sequence[1:], duration=250, loop=0, lossless=True)
    (args.output / 'matches.json').write_text(json.dumps(dict(
        reference=paths[0].name, cutoff=args.cutoff, space=args.space, range_mode=args.range_mode,
        mask=('Set pixels below the fixed cutoff to zero in registration copies only'
              if args.range_mode == 'full' else
              'Set pixels below min + cutoff * (max - min) to zero in registration copies only'),
        frames=report, canvas_size=result.canvas_size,
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
