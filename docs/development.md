# Developing Planetary Tools

[Project overview](../README.md) · [User manual](manual.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `planetary-app/planetary_tools/` | Application UI, filters, image I/O and batch processing |
| `planetary-app/tests/` | Automated regression tests |
| `planetary-app/benchmarks/` | Image-processing and performance experiments |
| `planetary-app/run.py` | Source and packaged application entry point |
| `planetary-app/planetary-tools.spec` | PyInstaller standalone executable configuration |
| `.github/workflows/build-planetary-tools.yml` | Linux, Windows and macOS release builds |
| `docs/manual.md`, `docs/images/` | Illustrated browser-readable user manual |
| `Planetary-Tools-Manual.pdf` | Downloadable manual |

## Run from source

Use Python 3.10 or newer. The release workflow uses Python 3.12.
From the repository root on Linux or macOS:

```sh
cd planetary-app
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=av -r requirements.txt
python run.py
```

On Windows PowerShell, use `py -m venv .venv` and
`.\.venv\Scripts\Activate.ps1` to create and activate the environment;
the remaining Python commands are the same.

Application dependencies are listed in `requirements.txt` and `pyproject.toml`.
PyAV binary wheels supply the FFmpeg libraries used for motion interpolation
and MP4. The application does not invoke an external FFmpeg executable.

The Qt GUI needs the platform's graphics libraries. The Linux release build
installs `libegl1` before testing its packaged application.

## Tests

With the virtual environment activated, run from `planetary-app/`:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

For a headless Linux run:

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p 'test_*.py' -v
```

To run a focused group, replace the pattern, for example with
`test_fft_denoise.py`. Some tests use optional local image fixtures or reference
source trees; missing optional inputs are skipped. Keep personal captures out
of commits unless they are deliberately being added as test fixtures.

[Performance notes](../planetary-app/PERFORMANCE.md) describe measurements and
reproduction commands. [Curves](../planetary-app/CURVES.md) and
[Deglow](../planetary-app/DEGLOW.md) have additional implementation notes.

## Build a standalone executable

Build on the target operating system. With the environment activated, run from
`planetary-app/`:

```sh
python -m pip install pyinstaller
pyinstaller --clean planetary-tools.spec
python scripts/check_packaged_media.py dist/planetary-tools
```

The executable is named `planetary-tools` (`planetary-tools.exe` on Windows).
The source directory remains `planetary-app/`.

On Windows, the last argument is `dist/planetary-tools.exe`. The smoke test checks
the packaged GUI and media support with an empty `PATH`, including motion
interpolation and lossless MP4 encode/decode. Run it against the newly built
binary before publishing.

The [release workflow](../.github/workflows/build-planetary-tools.yml) builds Linux,
Windows and Apple Silicon macOS executables on `v*` tag pushes. Manual workflow
runs can build without publishing, or rebuild an explicitly selected release
with its existing tag.

## Documentation changes

Edit [the manual](manual.md) and keep its screenshots in `docs/images/`.
Use relative links so documentation works on GitHub and in a local checkout.
When changing user-facing guidance, update the downloadable PDF edition too.
Keep the manual's build identification accurate: development documentation can
include features that are not in the latest published binary.
