#!/usr/bin/env python3
"""Planetary Tools by Con Kolivas <kernel@kolivas.org>"""

# Imported for PyInstaller: ensure onefile builds bundle these submodules.
import planetary_tools.io.png_read  # noqa: F401
import planetary_tools.io.png_write  # noqa: F401
import planetary_tools.ui.recent_files  # noqa: F401

from planetary_tools.ui.main_window import run_app

if __name__ == "__main__":
    raise SystemExit(run_app())