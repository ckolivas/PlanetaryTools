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
pyinstaller --clean --noconfirm planetary-tools.spec
```

Linux and Windows produce the single-file executable `planetary-tools` or
`planetary-tools.exe`. On macOS, the spec uses an onedir build wrapped in
`dist/Planetary Tools.app`, with its Python, Qt and media libraries inside the
bundle. Its display name and version are recorded in `Info.plist`.

Test the Linux executable with:

```sh
python scripts/check_packaged_media.py dist/planetary-tools
```

On Windows, use `dist/planetary-tools.exe`. The smoke test checks the packaged
GUI and media support with an empty `PATH`, including motion interpolation and
lossless MP4 encode/decode.

### macOS ZIP and Finder launch

Build on a Mac of the intended architecture. Package the complete `.app` using
`ditto` to retain executable permissions, framework symlinks and metadata:

```sh
ditto -c -k --sequesterRsrc --keepParent "dist/Planetary Tools.app" dist/planetary-tools-macos.zip
python scripts/check_packaged_macos.py dist/planetary-tools-macos.zip --architecture "$(uname -m)"
```

The validator extracts the actual ZIP, checks bundle identity/version,
permissions, symlinks, native architecture and its code signature, then runs
the media test with an empty `PATH`. It also launches the extracted `.app`
through macOS Launch Services (`open`), exercising the same launch mechanism
as Finder with native Cocoa GUI support.

PyInstaller applies ad-hoc signing. This is not Developer ID signing or Apple
notarization; downloaded builds can still require the first-launch approval
described in the [README](../README.md).

### Release builds

The [release workflow](../.github/workflows/build-planetary-tools.yml) builds:

| Runner | Release asset |
| --- | --- |
| `ubuntu-latest` | `planetary-tools-linux` |
| `windows-latest` | `planetary-tools-windows.exe` |
| `macos-15` (Apple Silicon) | `planetary-tools-macos-arm64.zip` |
| `macos-15-intel` (Intel) | `planetary-tools-macos-intel.zip` |

The Mac builds use native Python and dependency wheels on their respective
architectures. See [GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [PyInstaller's app-bundle documentation](https://pyinstaller.org/en/stable/usage.html#building-macos-app-bundles).

`v*` tag pushes publish the tested assets. Manual runs with no release tag
produce downloadable workflow artifacts without publishing a release. For an
existing release, supply its tag: the workflow uses that tag's application code
and version with the current workflow's packaging spec and validation scripts.
This also allows older releases with the original spec name to receive the new
Mac ZIP packaging without moving their tags. Existing release assets with other
names are retained; the new Mac ZIPs have distinct architecture-specific names.

## Documentation changes

Edit [the manual](manual.md) and keep its screenshots in `docs/images/`.
Use relative links so documentation works on GitHub and in a local checkout.
When changing user-facing guidance, update the downloadable PDF edition too.
Keep the manual's build identification accurate: development documentation can
include features that are not in the latest published binary.
