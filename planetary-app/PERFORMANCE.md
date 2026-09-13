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

## Deconvolution preparation reuse

Adaptive deconvolution Auto now prepares the local-contrast map, source
channels/luminance and Moffat correction fields once per search. Trial amounts
reuse those fields with the original multiplication order and float32
rounding. Prepared fields live only for that search; trials do not accumulate
cached result images. The search limits, rounding, progress events and chosen
parameters are unchanged.

Single-filter runs also avoid redundant channel copies and luminance work.
Disabling Adaptive skips the unused local-contrast calculation. Wiener RGB
denoising computes its frequency-domain kernel gain once per image, sharing
it across the three channels (four forward FFTs instead of six). The gain is
released with the filter call, so processing more files does not grow a cache.

### Third-pass measurements

Compared against `20c0c16`, median of seven paired runs after warmup, alternating
timing order. Timings include preparation, and Auto includes all trial noise
and brightness calculations. File loading and UI display are excluded.
The contrast target is 15%, texture scale is pinned at 2 and chromatic noise
is disabled. Noise targets are 8 for `3moons.png` and 10 for `4moons.png`, so
both samples exercise a complete search. These are benchmark settings; the
application's defaults are unchanged.

| Operation | 704 × 464 before → after | Speedup | 1920 × 320 before → after | Speedup |
|---|---:|---:|---:|---:|
| Auto, luminance | 432.99 → 178.89 ms | 2.42× | 813.32 → 276.78 ms | 2.94× |
| Auto, RGB | 717.92 → 256.43 ms | 2.80× | 1280.74 → 364.33 ms | 3.52× |
| Single luminance, Adaptive on | 22.90 → 20.34 ms | 1.13× | 53.93 → 41.08 ms | 1.31× |
| Single RGB, Adaptive on | 41.32 → 39.88 ms | 1.04× | 83.08 → 86.87 ms | 0.96× |
| Single luminance, Adaptive off | 33.65 → 17.86 ms | 1.88× | 42.42 → 23.04 ms | 1.84× |
| Single RGB, Adaptive off | 41.49 → 30.64 ms | 1.35× | 96.66 → 64.16 ms | 1.51× |
| Wiener RGB | 49.64 → 39.01 ms | 1.27× | 104.81 → 86.18 ms | 1.22× |
| Wiener OKLab | 72.58 → 76.31 ms | 0.95× | 141.24 → 139.45 ms | 1.01× |

Pixel arrays, Auto results and every progress event compare exactly against
the previous implementation. Six new regression tests cover RGB/grayscale,
luminance/independent channels, Adaptive on/off, zero and high amounts, HDR
values, strided input, bounded preparation state, repeated trials, an input
already over the noise limit, zero search range and shared kernel transforms.
The complete suite passes all 94 tests.

A sample already exceeding Auto's noise target still returns after the
initial trial and gets no multi-trial reuse benefit. Single adaptive RGB
runs showed variable timings around their previous performance; the main
gain is eliminating preparation from subsequent Auto trials.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/deconvolution_performance.py 3moons.png --repeats 7
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/deconvolution_performance.py 4moons.png --target-noise 10 --repeats 7
```

The benchmark loads the previous filter and search modules from Git, including
the previous local-contrast helper, to compare the complete implementations.

## PNG loading and clipping buffers

16-bit RGB PNGs now use Qt's native decoder when it supplies opaque RGBX64
pixels. The uint16 channels are copied directly, without an 8-bit intermediate,
colour conversion or automatic orientation change. Qt is already an app
dependency. Other native formats and decode failures retain the original
Python decoder. This also protects RGB PNGs with a transparent-colour key:
a regression fixture exposed incorrect native sample expansion for that case.
Header probes read just the 33-byte PNG signature/IHDR instead of the full file.

Clip-black, highlight clamping and full-range clamping now share a single
owned float32 result buffer. Operations preserve the old arithmetic order,
thresholds and grayscale-channel selection, including nonfinite values. Even
a no-op returns independent storage, so document/preview/undo pixels remain
unchanged when callers modify the result.

### Fourth-pass measurements

Compared against `bd529d5`, median of three paired runs for `widefield.png`
(3088 × 1600) and five for `3moons.png` (704 × 464), with warmup and alternating
order. Decode timings include file reads; complete loading additionally
includes linear-light conversion and pinning the document's noise context.
Qt imports are warmed as they already are in the GUI. Clipping uses the sample
normalized to a 1.4 peak and shifted by −0.02 to exercise both bounds.

| Operation | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| 16-bit PNG decode | 6157.71 → 117.95 ms | 52.21× | 8.22 → 11.54 ms | 0.71× |
| Complete image load | 6970.34 → 926.35 ms | 7.52× | 71.46 → 68.85 ms | 1.04× |
| PNG header probe | 5.51 → 0.06 ms | 97.79× | 0.05 → 0.01 ms | 6.98× |
| Clip black + clamp high | 94.78 → 47.22 ms | 2.01× | 1.72 → 0.92 ms | 1.87× |
| Clamp full range | 60.11 → 54.39 ms | 1.11× | 1.12 → 0.69 ms | 1.62× |

On the larger sample, peak traced allocations for clip-black plus clamp-high
fell from 169.6 MiB to 56.5 MiB, and full-range clamping from
113.1 MiB to 56.5 MiB. These `tracemalloc` measurements
exclude the existing input image and include the returned output buffer.

The largest PNG gain is for files with predictive scanline filters, which the
old implementation reconstructed byte by byte in Python. Small PNGs with
unfiltered rows already decoded cheaply and do not show the same benefit.
The native fast path is conditional on Qt's full-precision opaque output;
transparent-key PNGs and unsupported native formats retain the prior path.

Ten new regression tests cover every PNG predictor, mixed row filters, split
IDAT chunks, gamma/ICC/significant-bit metadata, transparency fallback, exact
linear pixels and pinned noise context, bounded header reads, all clipping
options, threshold/extreme values, readonly and strided input, and result
ownership. All 104 tests pass.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/loading_performance.py widefield.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/loading_performance.py 3moons.png --repeats 5
```

The benchmark loads the previous PNG reader, loader and brightness functions
from Git. It checks decoded uint16 samples, linear document pixels, storage
precision, noise context and postprocessed float32 pixels for exact equality.
