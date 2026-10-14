"""Run the release executable away from the checkout, with no FFmpeg on PATH."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def read_report(report: Path) -> dict:
    if not report.exists():
        raise RuntimeError('Packaged application did not write its media test report.')
    data = json.loads(report.read_text(encoding='utf-8'))
    print(json.dumps(data, indent=2))
    if not data.get('frozen') or not data.get('ok'):
        raise RuntimeError(f'Packaged media self-test failed: {data}')
    return data


def check_binary(binary: Path) -> dict:
    binary = binary.resolve()
    with tempfile.TemporaryDirectory() as folder:
        report = Path(folder)/'media-check.json'
        env = dict(os.environ, PATH='', PYTHONPATH='', QT_QPA_PLATFORM='offscreen', OPENBLAS_NUM_THREADS='1')
        env.pop('PYTHONHOME', None)
        result = subprocess.run([str(binary), '--self-test-media', str(report)],
                                cwd=folder, env=env, timeout=120)
        if not report.exists():
            # Surface startup failures before reporting a missing report.
            result.check_returncode()
        data = read_report(report)
        result.check_returncode()
        if data.get('external_ffmpeg') is not None:
            raise RuntimeError('Media test unexpectedly found an external FFmpeg executable.')
        return data


if __name__ == '__main__':
    check_binary(Path(sys.argv[1]))
