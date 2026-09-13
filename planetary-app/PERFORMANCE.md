# Performance improvements

## Preview and Apply

A filter now produces preview pixels and pre-clamp brightness/noise readouts
in one background job. Previously, calculating the readouts ran the filter
again on the UI thread. The last completed image remains visible during an
update. Toggling preview does not invalidate the calculation.

Apply reuses a completed result only when its source/settings generation is
still current, including a worker result awaiting Qt signal delivery. Changed
settings invalidate it immediately, before the debounce timer fires. Running
jobs keep immutable input/function snapshots; a pending update runs after the
worker finishes. Cancel and later sessions ignore stale results and failures.

## Auto wavelet sharpening

The Gaussian blur and linear-light conversion of a wavelet scale do not depend
on sharpening amount. Auto search prepares them once per channel/scale and
reuses the resulting detail. Only the latest sharpened layer per channel/scale
is retained, so testing additional amounts no longer grows the image cache
without bound. Unsharp arithmetic and pixel precision are unchanged.

## Measurements

Measured locally on `3moons.png` (704 × 464 RGB), median of three paired runs.
Preview timings cover filter evaluation and output statistics, excluding file
I/O, display conversion and debounce. Noise context was fixed in both paths.
Auto timings cover eight fine-amount trials after decomposition, with medium,
coarse and chunky held at 8, 1 and 0. Actual speedups depend on image and settings.

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| Deglow preview + readouts | 0.324 s | 0.167 s | 1.94× |
| Wavelet Sharpen preview + readouts | 0.609 s | 0.363 s | 1.68× |
| Adaptive Deconvolution preview + readouts | 0.068 s | 0.045 s | 1.51× |
| Eight auto-sharpen trials | 0.783 s | 0.419 s | 1.87× |

All benchmark pixels and statistics compare exactly. Tests also exercise
clamping, RGB/grayscale/luminance sharpening, cache bounds, stale jobs, Apply
during debounce or processing, Cancel/reopen, and the actual main-window
preview/readout/toggle/Apply path.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/preview_performance.py 3moons.png --baseline cc5c12d
QT_QPA_PLATFORM=offscreen PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  -m unittest discover -s planetary-app/tests -p 'test_*performance.py' -v
```

The benchmark loads the baseline auto-trial engine from Git. For preview it
compares the old two-evaluation path with the new combined path using the same
filter implementations, and asserts result parity before reporting timings.

## Image measurements, histograms and detail merging

Brightness measurements now reduce the existing float32/float64 image buffer,
convert the resulting scalars to Python floats, and reuse the extrema for the
clipping flag. Peak-increase readouts no longer calculate unused minima.
Other input dtypes retain their previous float64 interpretation.

Histograms select the same evenly spaced pixels before applying the display
transfer function. This avoids converting pixels that will not be counted.
Monochrome images calculate one histogram and share the counts across RGB.
Sampling positions, bin boundaries, population scaling and log heights remain
unchanged, including for cropped/strided arrays and values outside [0, 1].

The noise-readout identity check compares float64 values in blocks of at most
65,536 samples, stopping as soon as a block fails the original strict 1e-5
threshold. Unchanged images still check every pixel. Shape mismatches and
nonfinite differences cannot reuse the source noise. On a 3088 × 1600 RGB
buffer, an unchanged-image probe reduced peak traced temporary allocations
from 452.3 MiB to 2.0 MiB (Python `tracemalloc`, excluding the existing images).

Wavelet detail merging now resizes its secondary luminance channel once,
instead of copying it into RGB and resizing three identical channels. Choosing
zero secondary scales also skips all secondary-image processing.

### Second-pass measurements

Median of five paired runs against `312545a`, with warmup, alternating timing
order and exact output checks. These timings measure the named operations,
not the entire application. For merging, the secondary image is half the
main image's width and height. The changed-image identity probe multiplies the
source by 1.15; its early-exit benefit depends on where changed pixels occur.

| Operation | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| Brightness + peak increase | 103.11 → 9.37 ms | 11.01× | 2.26 → 0.29 ms | 7.83× |
| Linear histogram | 14.09 → 5.45 ms | 2.59× | 3.73 → 3.58 ms | 1.04× |
| Perceptual histogram | 213.75 → 13.58 ms | 15.74× | 16.34 → 12.88 ms | 1.27× |
| Noise identity check, unchanged | 69.81 → 15.33 ms | 4.55× | 4.89 → 1.14 ms | 4.28× |
| Noise identity check, changed | 70.57 → 0.21 ms | 339.14× | 4.89 → 0.18 ms | 27.63× |
| Detail merge including resize | 2224.35 → 2053.70 ms | 1.08× | 78.44 → 75.22 ms | 1.04× |

The eight new regression tests cover exact measurement/threshold behavior,
histogram sampling and grayscale counts, one-channel resize parity, bounded
comparison buffers, late changed pixels, and source-noise reuse. The full
suite passes all 88 tests.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/image_performance.py widefield.png --baseline 312545a
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/image_performance.py 3moons.png --baseline 312545a
```

The benchmark loads the previous brightness, histogram and wavelet modules
from Git, and reproduces the previous full-buffer identity expression for
comparison. Outputs, histogram bins and brightness statistics match exactly.
