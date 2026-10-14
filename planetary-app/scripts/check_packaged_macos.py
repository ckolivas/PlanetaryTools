"""Validate the distributed Mac ZIP and launch its app through Launch Services."""
import argparse
import os
from pathlib import Path
import platform
import plistlib
import runpy
import subprocess
import sys
import tempfile

from check_packaged_media import check_binary, read_report

APP_NAME = 'Planetary Tools.app'
BUNDLE_ID = 'org.kolivas.planetary-tools'


def check_bundle(bundle: Path, version: str) -> Path:
    """Catch malformed metadata or ZIP damage before trying to launch the app."""
    with (bundle / 'Contents' / 'Info.plist').open('rb') as stream:
        info = plistlib.load(stream)
    expected = {
        'CFBundleExecutable': 'planetary-tools',
        'CFBundleIdentifier': BUNDLE_ID,
        'CFBundleName': 'Planetary Tools',
        'CFBundleDisplayName': 'Planetary Tools',
        'CFBundleShortVersionString': version,
        'CFBundleVersion': version,
        'CFBundlePackageType': 'APPL',
    }
    for key, value in expected.items():
        if info.get(key) != value:
            raise RuntimeError(f'Incorrect {key}: {info.get(key)!r}; expected {value!r}')
    binary = bundle / 'Contents' / 'MacOS' / info['CFBundleExecutable']
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError('The app executable is missing or lost its execute permission.')
    for path in bundle.rglob('*'):
        if path.is_symlink() and not path.exists():
            raise RuntimeError(f'Broken bundle symlink: {path}')
    return binary


def check_archive(archive: Path, architecture: str) -> None:
    if sys.platform != 'darwin':
        raise RuntimeError('The Mac archive must be checked on macOS.')
    if platform.machine() != architecture:
        raise RuntimeError(f'Expected native {architecture} runner, got {platform.machine()}')
    version = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'planetary_tools' / '__init__.py'))['__version__']
    with tempfile.TemporaryDirectory(prefix='pt-release-') as folder:
        root = Path(folder)
        # ditto preserves the permissions, symlinks and metadata Finder needs.
        subprocess.run(['/usr/bin/ditto', '-x', '-k', str(archive.resolve()), str(root)], check=True)
        bundle = root / APP_NAME
        binary = check_bundle(bundle, version)
        arches = subprocess.check_output(['/usr/bin/lipo', '-archs', str(binary)], text=True).split()
        if arches != [architecture]:
            raise RuntimeError(f'Unexpected executable architectures: {arches}')
        subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(bundle)], check=True)
        # Check all media dependencies without source paths or external FFmpeg.
        check_binary(binary)
        # Launch Services is the same app-launch mechanism Finder uses. Run a
        # new instance and wait for the self-test to exit, using native Cocoa.
        report = root / 'finder-media-check.json'
        env = dict(os.environ)
        env.pop('QT_QPA_PLATFORM', None)
        subprocess.run(['/usr/bin/open', '-n', '-W', str(bundle), '--args',
                        '--self-test-media', str(report)], cwd=root, env=env,
                       check=True, timeout=120)
        data = read_report(report)
        # Older tagged app sources predate qt_platform reporting. Their normal
        # QApplication still uses Cocoa here, with the override removed above.
        if data.get('qt_platform', 'cocoa') != 'cocoa':
            raise RuntimeError('Finder launch did not use the native Cocoa Qt platform.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--architecture', required=True, choices=('arm64', 'x86_64'))
    args = parser.parse_args()
    check_archive(args.archive, args.architecture)
