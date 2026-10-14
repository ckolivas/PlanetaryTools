"""Cross-platform packaging decisions and damaged Mac release validation."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

APP = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, APP / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


media = load_script('check_packaged_media')
with patch.dict(sys.modules, {'check_packaged_media': media}):
    macos = load_script('check_packaged_macos')


class SpecTests(unittest.TestCase):
    def test_native_mac_bundle_and_other_platform_single_file_outputs(self):
        hooks = types.ModuleType('PyInstaller.utils.hooks')
        for name in ('collect_dynamic_libs', 'collect_submodules', 'copy_metadata'):
            setattr(hooks, name, MagicMock(return_value=[]))
        for platform in ('darwin', 'linux', 'win32'):
            with self.subTest(platform=platform):
                analysis = types.SimpleNamespace(scripts=['run.py'], pure=[],
                                                 binaries=['library'], datas=['resource'])
                calls = {name: MagicMock(name=name) for name in ('Analysis', 'PYZ', 'EXE', 'COLLECT', 'BUNDLE', 'Splash')}
                calls['Analysis'].return_value = analysis
                with patch.dict(sys.modules, {'PyInstaller.utils.hooks': hooks}), patch.object(sys, 'platform', platform):
                    runpy.run_path(str(APP/'planetary-tools.spec'),
                                   init_globals=dict(calls, SPECPATH=str(APP)))
                exe = calls['EXE'].call_args
                self.assertEqual(exe.kwargs['name'], 'planetary-tools')
                self.assertFalse(exe.kwargs['console'])
                self.assertEqual(calls['Splash'].call_count, int(platform == 'win32'))
                if platform == 'darwin':
                    self.assertTrue(exe.kwargs['exclude_binaries'])
                    self.assertNotIn(analysis.binaries, exe.args)
                    self.assertIn(analysis.binaries, calls['COLLECT'].call_args.args)
                    self.assertIn(analysis.datas, calls['COLLECT'].call_args.args)
                    bundle = calls['BUNDLE'].call_args
                    self.assertEqual(bundle.args, (calls['COLLECT'].return_value,))
                    self.assertEqual(bundle.kwargs['name'], 'Planetary Tools.app')
                    self.assertEqual(bundle.kwargs['bundle_identifier'], macos.BUNDLE_ID)
                    version = runpy.run_path(str(APP/'planetary_tools/__init__.py'))['__version__']
                    self.assertEqual(bundle.kwargs['version'], version)
                    self.assertEqual(bundle.kwargs['info_plist']['CFBundleVersion'], version)
                else:
                    self.assertFalse(exe.kwargs['exclude_binaries'])
                    self.assertIn(analysis.binaries, exe.args)
                    self.assertIn(analysis.datas, exe.args)
                    calls['COLLECT'].assert_not_called()
                    calls['BUNDLE'].assert_not_called()


class MacBundleValidationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.bundle = Path(self.folder.name) / macos.APP_NAME
        self.binary = self.bundle/'Contents/MacOS/planetary-tools'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_text('fixture')
        self.binary.chmod(0o755)
        self.info = {
            'CFBundleExecutable': 'planetary-tools',
            'CFBundleIdentifier': macos.BUNDLE_ID,
            'CFBundleName': 'Planetary Tools',
            'CFBundleDisplayName': 'Planetary Tools',
            'CFBundleShortVersionString': '1.2.0',
            'CFBundleVersion': '1.2.0',
            'CFBundlePackageType': 'APPL',
        }
        self.save_info()

    def save_info(self):
        with (self.bundle/'Contents/Info.plist').open('wb') as stream:
            plistlib.dump(self.info, stream)

    def test_bundle_identity_version_and_executable(self):
        self.assertEqual(macos.check_bundle(self.bundle, '1.2.0'), self.binary)
        with self.assertRaisesRegex(RuntimeError, 'CFBundleShortVersionString'):
            macos.check_bundle(self.bundle, '1.3.0')
        self.info['CFBundleExecutable'] = '../../wrong-executable'
        self.save_info()
        with self.assertRaisesRegex(RuntimeError, 'CFBundleExecutable'):
            macos.check_bundle(self.bundle, '1.2.0')

    @unittest.skipIf(os.name == 'nt', 'POSIX execute permissions')
    def test_archive_lost_execute_permission(self):
        self.binary.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, 'execute permission'):
            macos.check_bundle(self.bundle, '1.2.0')

    def test_archive_missing_executable(self):
        self.binary.unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            macos.check_bundle(self.bundle, '1.2.0')

    @unittest.skipIf(os.name == 'nt', 'Symlinks require additional Windows privileges')
    def test_archive_broken_framework_link(self):
        link = self.bundle/'Contents/Frameworks'
        link.symlink_to('MacOS', target_is_directory=True)
        macos.check_bundle(self.bundle, '1.2.0')
        link.unlink()
        link.symlink_to('missing', target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, 'Broken bundle symlink'):
            macos.check_bundle(self.bundle, '1.2.0')


class MediaValidationTests(unittest.TestCase):
    def test_binary_runs_away_from_checkout_with_empty_paths(self):
        def fake_run(args, **kwargs):
            self.assertEqual(kwargs['env']['PATH'], '')
            self.assertEqual(kwargs['env']['PYTHONPATH'], '')
            self.assertEqual(kwargs['env']['QT_QPA_PLATFORM'], 'offscreen')
            self.assertNotEqual(Path(kwargs['cwd']).resolve(), Path.cwd())
            Path(args[-1]).write_text(json.dumps({'frozen': True, 'ok': True, 'external_ffmpeg': None}))
            return subprocess.CompletedProcess(args, 0)
        with patch.object(media.subprocess, 'run', side_effect=fake_run):
            self.assertTrue(media.check_binary(Path('example-binary'))['ok'])

    def test_missing_report_preserves_startup_failure(self):
        with patch.object(media.subprocess, 'run', return_value=subprocess.CompletedProcess([], 3)):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                media.check_binary(Path('example-binary'))
            self.assertEqual(error.exception.returncode, 3)

    def test_failed_or_nonfrozen_reports_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder)/'report.json'
            with self.assertRaisesRegex(RuntimeError, 'did not write'):
                media.read_report(report)
            for data in ({'ok': True, 'frozen': False}, {'ok': False, 'frozen': True}):
                report.write_text(json.dumps(data))
                with self.assertRaisesRegex(RuntimeError, 'self-test failed'):
                    media.read_report(report)


if __name__ == '__main__':
    unittest.main()
