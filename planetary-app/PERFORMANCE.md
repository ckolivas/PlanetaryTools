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

## Bounded colour transfers and PNG output buffers

sRGB-to-linear and linear-to-sRGB transfers now process large arrays in blocks
of at most 65,536 channel samples. Conversion and arithmetic remain float64,
with the same final float32 rounding, clamping and signed HDR extension. Small
arrays retain direct evaluation. Buffered traversal handles strided inputs
without creating a full-frame float64 copy; the full-sized allocation is the
float32 output buffer.

The 16-bit PNG writer now prepares one contiguous scanline buffer, frees it
after compression, and writes the existing compressed payload directly into
its IDAT chunk. This removes the retained row list and whole-PNG concatenation.
It retains the same row filters, compression level, chunk layout and CRCs,
producing byte-identical files. Validation and compression still finish before
the destination is opened.

### Fifth-pass measurements

Compared against `91aeb13`, medians of three paired runs on `widefield.png`
(3088 × 1600 RGB) and five on `3moons.png` (704 × 464 RGB). Timing order alternates
after warmup. Conversion inputs are normalized encoded samples or their linear
counterparts; fixture creation and verification are excluded. PNG-write timings
include compression and temporary-file writes. Peak allocations use
`tracemalloc`, exclude existing inputs, and include returned output buffers.

| Operation | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| sRGB → linear, clamped | 319.80 → 182.98 ms | 1.75× | 17.20 → 11.16 ms | 1.54× |
| sRGB → linear, HDR allowed | 374.94 → 171.06 ms | 2.19× | 20.26 → 12.14 ms | 1.67× |
| Linear → sRGB, clamped | 281.41 → 146.14 ms | 1.93× | 16.42 → 9.61 ms | 1.71× |
| Linear → sRGB, HDR allowed | 341.60 → 173.84 ms | 1.97× | 20.09 → 11.11 ms | 1.81× |
| 16-bit RGB PNG write | 1895.63 → 1816.47 ms | 1.04× | 86.27 → 84.39 ms | 1.02× |

On the larger image, clamped-transfer peak allocations fell from
466.5 MiB to 59.1 MiB; transfers allowing HDR fell from
692.7 MiB to 59.6 MiB. PNG-write peak allocations fell from
89.3 MiB to 52.5 MiB.

PNG compression remains the main cost of writing; the measured speed change
was small. These measurements describe individual transfers and PNG encoding,
not the total time of every filter or save workflow.

Eight new tests verify exact float32 bits across every 16-bit level, transfer
breakpoints, near-halfway rounding cases, HDR/nonfinite values, awkward block
boundaries, readonly/strided/Fortran arrays and large wavelet filter results.
They also check PNG byte equality for RGB/grayscale and different input dtypes,
scanline samples/CRCs, and preservation of an existing file if validation or
compression fails. All 112 tests pass.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/transfer_write_performance.py widefield.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/transfer_write_performance.py 3moons.png --repeats 5
```

The benchmark loads the previous colour-transfer and PNG-writing modules from
Git, comparing all converted pixels and complete PNG files for exact equality.

## Noise preparation and readouts

Noise luminance preparation now converts RGB in blocks of at most 65,536
pixels, preserving float64 arithmetic and the original intermediate float32
rounding. Grayscale inputs convert only the selected channel. Subject bounds
come from row/column masks instead of coordinates for every signal pixel;
the threshold, minimum sample count and crop padding remain unchanged.

Colour detection computes extrema in the source floating dtype, then uses
float64 for threshold comparisons and saturation of the selected signal pixels.
It avoids converting the entire RGB image to float64. The hybrid noise score
also skips the band-tail percentile in branches that do not use it.

### Sixth-pass measurements

Compared against `7837332`, medians of three paired runs on `widefield.png`
(3088 × 1600 RGB) and five on `3moons.png` (704 × 464 RGB), alternating timing
order after warmup. Loading, fixture creation and verification are excluded.

| Operation | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| Noise luminance preparation | 64.29 → 17.58 ms | 3.66× | 1.08 → 0.89 ms | 1.21× |
| Colour detection | 124.71 → 40.78 ms | 3.06× | 3.74 → 0.98 ms | 3.80× |
| Pin noise context | 193.74 → 95.35 ms | 2.03× | 29.33 → 25.22 ms | 1.16× |
| Noise readout, mono scoring | 100.68 → 58.35 ms | 1.73× | 6.99 → 6.43 ms | 1.09× |
| Noise readout, soft mono scoring | 99.79 → 55.62 ms | 1.79× | 8.79 → 7.80 ms | 1.13× |
| Noise readout, colour scoring | 99.08 → 55.47 ms | 1.79× | 11.91 → 10.04 ms | 1.19× |
| Complete luminance deconvolution Auto search | 1693.95 → 1283.57 ms | 1.32× | 119.88 → 106.85 ms | 1.12× |

On the larger image, traced peak allocations fell from 188.5 to 40.4 MiB for
luminance preparation, 268.6 to 80.1 MiB for pinning the noise context, and
188.5 to 75.4 MiB for noise readouts. These measurements exclude existing
source buffers and include returned outputs.

Readout fixtures explicitly exercise each scoring branch on the same RGB
source. Auto uses texture scale 2, monochrome scoring and contrast target 15;
its noise target is the larger of 8 or source noise plus 1, ensuring a complete
search on both samples. Application defaults are unchanged. Both searches
produce identical settings, scores and all 12 progress events.

Five new tests cover exact luminance bits with bounded conversion blocks,
readonly/strided/Fortran inputs, crop boundaries and minimum sample counts,
colour thresholds, complete noise/texture scores and conditional percentile
evaluation. All 117 tests pass.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/noise_performance.py widefield.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/noise_performance.py 3moons.png --repeats 5
```

The benchmark loads the previous noise and Auto modules from Git, checking
luminance pixels, classifications, scores, Auto settings and progress values
for exact equality.

## Back-and-forth animation preparation

Return-trip animation frames now reuse the prepared Pillow images. GIF palette
quantization runs once per source frame instead of again for each returning
frame. APNG and WebP similarly reuse RGB images. The encoder still receives
the same expanded sequence, durations, loop/disposal options and quality.
There is no persistent cache or change to forward-only exports.

### Seventh-pass measurements

Compared against `f64f351`, medians of three paired runs, alternating order
after warmup. Five source frames produce eight displayed frames. Fixtures
use the loaded sample shifted horizontally by two pixels per frame, prepared
before timing. Measurements include complete encoding, temporary-file writes
and reading the bytes for equality; source loading and conversion are excluded.

| GIF quality | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| Best | 501.49 → 332.36 ms | 1.51× | 161.64 → 103.90 ms | 1.56× |
| Low | 1081.79 → 921.34 ms | 1.17× | 113.82 → 89.98 ms | 1.26× |

Complete GIF files and result metadata match exactly. Two new tests also
compare bytes across GIF's four qualities, APNG and WebP; two, three and five
source frames; forward-only/return-trip sequences; repeated, readonly and
strided frames. Decoded GIF frame order, duration and looping are checked,
and a five-frame return trip requires five quantizations rather than eight.
All 119 tests pass.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/animation_performance.py widefield.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/animation_performance.py 3moons.png
```

### Timed audit continuation

The user authorized continued work until 2026-09-13 22:45:15 UTC, with an early
stop when no obvious worthwhile improvements remain. Follow-ups use the
`planetarytools-performance-audit-10-hours` heartbeat in the existing task.
Preserve the separate user Save As edit in `ui/main_window.py` and untracked
sample/configuration files. Benchmark serially; retain measured improvements
with exact output parity and commit each completed step.

Remaining candidates to evaluate, not established gains:

- Levels channel selection and joint percentiles: completed in the eighth
  pass below.
- Alignment reference FFT reuse: completed in the ninth pass below.
- Alignment refinement allocates coordinates for every signal pixel to find
  the bounding box; row/column masks may save allocation while preserving it.
- Curves mapping and noise median/percentile temporaries may have redundant
  copies; evaluate complete operations before retaining small local changes.
- Deglow calculates the same full-resolution median twice for the first
  grayscale channel; consider retaining that model until its channel is used.
- OKLab matrix arithmetic changes can alter rounding. Avoid changing its
  equations or claiming equivalence without exact pixel verification.

## Levels channel measurements and auto-balance

RGB Levels histograms now select the requested channel before converting it
to sRGB. This avoids processing all three channels for each histogram and
avoids creating three copies of grayscale images. The OKLab L path is
unchanged. Auto input levels compute the same 2nd and 98th percentiles together,
sharing the partition work. Clamping, precision, interpolation, output limits
and application order remain unchanged.

### Eighth-pass measurements

Compared against `a8b5ee1`, medians of three paired runs on `widefield.png`
(3088 × 1600 RGB) and five on `3moons.png` (704 × 464 RGB), with alternating
order after warmup. Loading and fixture creation are excluded. The percentile
benchmark uses precomputed R-channel samples; the auto-balance and combined
application benchmarks include channel conversion.

| Operation | 3088 × 1600 before → after | Speedup | 704 × 464 before → after | Speedup |
|---|---:|---:|---:|---:|
| Red input peak | 142.17 → 51.74 ms | 2.75× | 8.60 → 2.92 ms | 2.95× |
| Channel percentiles | 82.90 → 57.30 ms | 1.45× | 2.57 → 1.74 ms | 1.48× |
| RGB auto-balance | 639.15 → 279.91 ms | 2.28× | 34.37 → 14.65 ms | 2.35× |
| Auto-balance and apply | 1338.22 → 1029.32 ms | 1.30× | 79.75 → 58.18 ms | 1.37× |

The large-image peak measurement's traced allocations fell from 75.4 to
21.4 MiB. Whole auto-balance and application peaks remain essentially unchanged,
since their percentile/application buffers dominate. Existing input buffers
are excluded from these allocation measurements.

Four new tests preserve exact histogram float32 bits, percentile limits,
auto-balance settings, applied pixels and peak defaults across grayscale/RGB,
readonly/strided/Fortran inputs, empty/constant samples, nonfinite values and
percentile/identity boundaries. All 123 tests pass. The paired benchmark also
compares parameters and complete applied images against the previous Git code.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/levels_performance.py widefield.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/levels_performance.py 3moons.png --repeats 5
```

## Alignment reference FFT reuse

Each angle sweep prepares the unchanged reference mean, norm and FFT once.
All candidate rotations, target FFTs, correlation normalization and peak/tie
selection use the original arithmetic and ordering. Preparation is local to
the sweep, holding one complex spectrum and scalar/shape metadata; no frame
cache persists across calls. Subpixel refinement and final resampling remain
unchanged.

### Ninth-pass measurements

Compared against `48fc394`, medians of three paired runs with alternating
order after warmup. Fixtures apply a 1.37-degree rotation and (0.38, -0.71)
pixel shift to a loaded image before timing. Both paths use the default ±45°
search. Loading and fixture creation are excluded; complete matching includes
luminance, structure preparation, search and native-resolution refinement.

| Operation | Saturn 917 × 556 before → after | Speedup | Widefield 3088 × 1600 before → after | Speedup |
|---|---:|---:|---:|---:|
| Angle search | 1807.94 → 1381.78 ms | 1.31× | 1376.53 → 1102.98 ms | 1.25× |
| Complete match | 2044.99 → 1632.53 ms | 1.25× | 3280.37 → 2996.35 ms | 1.09× |

The search planes are 422 × 256 and 494 × 256 respectively. The paired
benchmark verifies identical angle searches, full match records and rendered
pixels. These timings establish preserved behavior, not increased alignment
accuracy: Saturn recovers -1.3699946°, while both old and new widefield
searches return the same alternate -3.15° match.

Four new tests check exact correlation scores and shifts for odd/even shapes,
wrap boundaries, weak signals and readonly/strided/Fortran inputs; reference
FFT counts; rival-angle ordering; and complete matches/rendering with rotation,
shift-only and search-limit settings. A nine-angle sweep now uses ten forward
FFTs instead of eighteen. All 127 tests pass, including the existing real
Saturn fixture and alignment quality regressions.

Reproduce from the repository root:

```sh
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/alignment_performance.py \
  align/2026-09-11-1552_0-CK-L3-Sat-planetrecon_processed_derot.png
PYTHONPATH=planetary-app planetary-app/.venv/bin/python \
  planetary-app/benchmarks/alignment_performance.py widefield.png
```
