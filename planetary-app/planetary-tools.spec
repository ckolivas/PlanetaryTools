# -*- mode: python ; coding: utf-8 -*-
import sys
import runpy
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata

datas = []
datas += copy_metadata('imageio')
datas += copy_metadata('av')

# Bundle every planetary_tools submodule (png_read, recent_files, etc.).
hiddenimports = collect_submodules('planetary_tools')
hiddenimports += collect_submodules('av')

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=collect_dynamic_libs('av'),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
is_macos = sys.platform == 'darwin'

# Bootloader-level splash screen shown while the onefile archive extracts and
# Python starts up. Requires Tcl/Tk at build time, so only enabled on Windows
# where startup is slowest; dismissed in run.py via pyi_splash.
splash = None
if sys.platform == 'win32':
    splash = Splash(
        'splash.png',
        binaries=a.binaries,
        datas=a.datas,
        always_on_top=False,
    )

exe = EXE(
    pyz,
    a.scripts,
    *([splash, splash.binaries] if splash else []),
    *([] if is_macos else [a.binaries, a.datas]),
    [],
    exclude_binaries=is_macos,
    name='planetary-tools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if is_macos:
    # Keep dependencies in the app bundle so Finder launches do not need to
    # unpack a onefile archive on every run.
    coll = COLLECT(
        exe, a.binaries, a.datas,
        strip=False, upx=True, upx_exclude=[], name='planetary-tools',
    )
    version = runpy.run_path(str(Path(SPECPATH) / 'planetary_tools' / '__init__.py'))['__version__']
    app = BUNDLE(
        coll,
        name='Planetary Tools.app',
        bundle_identifier='org.kolivas.planetary-tools',
        version=version,
        info_plist={
            'CFBundleName': 'Planetary Tools',
            'CFBundleDisplayName': 'Planetary Tools',
            'CFBundleVersion': version,
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': True,
        },
    )
