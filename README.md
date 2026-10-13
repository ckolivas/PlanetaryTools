# Planetary Tools

Planetary Tools (PT) is a standalone desktop application for processing stacked
planetary images. Sharpen detail, reduce noise, correct colour, enhance moons,
align frames and create animations in one application.

**[Download the latest release](https://github.com/ckolivas/PlanetaryTools/releases/latest)**
 · **[Read the illustrated manual](docs/manual.md)**
 · **[Download the PDF manual](Planetary-Tools-Manual.pdf)**

![Planetary Tools showing Saturn with the Levels panel](docs/images/workspace_dock.png)

## Get started

1. Download the executable for your operating system from
   [Releases](https://github.com/ckolivas/PlanetaryTools/releases). Builds are
   provided for Linux, Windows and Apple Silicon macOS. On Linux and macOS,
   make the downloaded file executable if needed.
2. Launch PT and open a stacked image using **File > Open**.
3. Choose a tool from **Enhance** or **Colours**, compare its preview, then
   click **OK** to apply it.
4. Use **File > Save As** to save a processed copy in your chosen format and
   bit depth.

The standalone binaries include the Python runtime and application dependencies.
MP4 export and motion interpolation use bundled FFmpeg libraries through PyAV;
a separate FFmpeg executable is not required.

The manual covers every tool with screenshots, including the
[author's suggested sharpening workflow](docs/manual.md#authors-sharpening-workflow)
for one-shot colour images.

## Tools

- **Sharpen and denoise:** Wavelet Sharpen, Adaptive Deconvolution, Wavelet
  Denoise and post-sharpening FFT Denoise with automatic starting settings.
- **Planet and moons:** Moon Enhance, Deglow and Merge Wavelet Detail.
- **Colour:** Align RGB, camera colour-correction presets, Curves, Levels,
  Saturation & Vibrance, Extract Component, and RGB composition/decomposition.
- **Geometry:** scale, rotate, flip, crop and expand the image canvas.
- **Sequences:** Derotate/Align, batch processing and animation with optional
  timestamp-based motion interpolation. Export GIF, APNG, WebP or MP4.

Processing uses floating-point linear image data. Tools provide previews,
operation-labelled undo/redo and reusable presets where applicable.

## Development

The application, tests, benchmarks and packaging configuration live in
[`planetary-app/`](planetary-app/). See the
[development guide](docs/development.md) for source setup, tests and standalone
build commands.

Documentation describes the current development tree. A published release may
not yet include every feature shown in the manual.

## License and author

Planetary Tools is free software under the GNU General Public License;
see [LICENSE](LICENSE) and [COPYING](COPYING).

Con Kolivas · <kernel@kolivas.org>
