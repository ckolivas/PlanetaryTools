"""Run the release executable away from the checkout, with no FFmpeg on PATH."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as folder:
    report = Path(folder)/'media-check.json'
    env = dict(os.environ, PATH='', PYTHONPATH='', QT_QPA_PLATFORM='offscreen', OPENBLAS_NUM_THREADS='1')
    env.pop('PYTHONHOME', None)
    result = subprocess.run([str(binary), '--self-test-media', str(report)],
                            cwd=folder, env=env, timeout=120)
    if report.exists():
        data = json.loads(report.read_text(encoding='utf-8'))
        print(json.dumps(data, indent=2))
        assert data['frozen'] and data['ok'] and data['external_ffmpeg'] is None
    else:
        # Surface startup failures (for example, a missing shared library)
        # before reporting a missing self-test report.
        result.check_returncode()
        raise RuntimeError('Packaged application did not write its media test report.')
    result.check_returncode()
