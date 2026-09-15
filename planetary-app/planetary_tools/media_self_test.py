"""Exercise bundled media libraries and Qt in the actual release executable."""

import json
from pathlib import Path
import shutil
import sys
import tempfile

import av
import numpy as np
from PyQt6.QtWidgets import QApplication

from planetary_tools.core.animate import encode_frames
from planetary_tools.core.animation_interpolation import interpolate_pair
from planetary_tools.core.document import ImageDocument
from planetary_tools.ui.main_window import MainWindow


def run(report_path: str) -> int:
    report = {'frozen': bool(getattr(sys, 'frozen', False)),
              'pyav': av.__version__, 'libraries': av.library_versions,
              'external_ffmpeg': shutil.which('ffmpeg')}
    try:
        rng = np.random.default_rng(31)
        first = np.zeros((97, 129, 3), dtype=np.uint8)
        first[24:64, 24:64] = rng.integers(50, 240, (40, 40, 1), dtype=np.uint8)
        last = np.zeros_like(first)
        last[:, 12:] = first[:, :-12]
        generated = interpolate_pair(first, last, 4)
        expected_mid = np.zeros_like(first)
        expected_mid[:, 6:] = first[:, :-6]
        assert len(generated) == 3
        assert np.mean((generated[1].astype(float)-expected_mid)**2) < 1
        frames = [first, *generated, last]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'test.mp4'
            encode_frames(frames, output, fps=29.9, fmt='mp4', back_and_forth=False)
            with av.open(str(output)) as container:
                decoded = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
            np.testing.assert_array_equal(decoded, frames)
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            window._set_document(ImageDocument(first.astype(np.float32)/255))
            window.show()
            app.processEvents()
            assert not window.grab().isNull()
        finally:
            window._document.modified = False
            window.close()
        report['ok'] = True
    except Exception as exc:
        report.update(ok=False, error=str(exc))
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['ok'] else 1
