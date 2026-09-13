"""Measure back-and-forth GIF encoding against Git; require byte equality."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from image_performance import compare, load_baseline
from planetary_tools.core import animate
from planetary_tools.io.loader import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--baseline', default='f64f351')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--frames', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.frames < 2:
        parser.error('repeats must be positive and frames at least two')
    old = load_baseline(args.baseline, 'core/animate')
    frame = animate._to_uint8_srgb(load_image(args.image).data)
    frames = [np.roll(frame, i*2, axis=1) for i in range(args.frames)]
    report = {'image': str(args.image), 'shape': list(frame.shape), 'source_frames': len(frames),
              'baseline': args.baseline, 'repeats': args.repeats}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'output.gif'
        for quality in ('best', 'low'):
            def encode(module):
                result = module.encode_frames(frames, path, fps=10, fmt='gif', gif_quality=quality)
                return asdict(result), path.read_bytes()
            report[quality] = compare(lambda: encode(old), lambda: encode(animate), args.repeats)
    report['parity'] = 'complete GIF bytes and result metadata identical'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
