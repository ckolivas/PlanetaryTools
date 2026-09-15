"""Update checks compare versions and handle network results without blocking."""
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtCore import QByteArray, QObject, pyqtSignal
from PyQt6.QtNetwork import QNetworkReply
from PyQt6.QtWidgets import QApplication

from planetary_tools.ui import update_dialog as updates
from planetary_tools.ui.main_window import MainWindow


class Reply(QObject):
    finished = pyqtSignal()

    def __init__(self, status=200, body=None, error=QNetworkReply.NetworkError.NoError):
        super().__init__()
        self.status = status
        self.body = json.dumps({'tag_name': 'v0.5.2', 'draft': False, 'prerelease': False}).encode() if body is None else body
        self.network_error = error
        self.aborted = False

    def attribute(self, _):
        return self.status

    def error(self):
        return self.network_error

    def errorString(self):
        return 'Network unavailable'

    def readAll(self):
        return QByteArray(self.body)

    def abort(self):
        self.aborted = True
        self.finished.emit()


class UpdateCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        version = patch.object(updates, '__version__', '0.5.1')
        version.start()
        self.addCleanup(version.stop)

    def test_numeric_versions_and_invalid_releases(self):
        for tag, current, newer in (('v0.5.2', '0.5.1', True), ('v0.10.0', '0.9.9', True),
                                    ('v0.5.1', '0.5.1', False), ('v0.5.0', '0.5.1', False),
                                    ('0.5.1.0', '0.5.1', False), ('v1.0.0', '0.99.9', True)):
            with self.subTest(tag=tag, current=current):
                self.assertEqual(updates.release_version(json.dumps({'tag_name': tag}).encode(), current),
                                 (tag, newer))
        for payload in (b'not json', b'[]', b'{}', b'{"tag_name": 5}',
                        b'{"tag_name": "v0.6.0-rc1"}',
                        b'{"tag_name": "v0.6.0", "prerelease": true}',
                        b'{"tag_name": "v0.6.0", "draft": true}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                updates.release_version(payload, '0.5.1')

    def dialog(self, reply):
        manager = patch.object(updates, 'QNetworkAccessManager')
        mocked = manager.start()
        self.addCleanup(manager.stop)
        mocked.return_value.get.return_value = reply
        dialog = updates.UpdateCheckDialog()
        self.addCleanup(dialog.close)
        self.assertEqual(mocked.return_value.get.call_args.args[0].url().toString(), updates.LATEST_RELEASE_API)
        return dialog

    def test_new_release_and_browser_action(self):
        reply = Reply()
        dialog = self.dialog(reply)
        self.assertIn('Checking', dialog._status.text())
        self.assertTrue(dialog._timer.isActive())
        self.assertFalse(dialog._view_release.isEnabled())
        reply.finished.emit()
        self.assertIn('0.5.2 is available', dialog._status.text())
        self.assertFalse(dialog._timer.isActive())
        with patch.object(updates.QDesktopServices, 'openUrl', return_value=True) as browser:
            dialog._view_release.click()
            self.assertEqual(browser.call_args.args[0].toString(), updates.RELEASES_URL + '/tag/v0.5.2')

    def test_current_missing_rate_limited_offline_and_invalid_response(self):
        cases = [
            (Reply(body=b'{"tag_name": "v0.5.1"}'), 'No newer release', True),
            (Reply(status=404), 'No published stable release', False),
            (Reply(status=403), 'request limit', False),
            (Reply(status=429), 'request limit', False),
            (Reply(status=None, error=QNetworkReply.NetworkError.HostNotFoundError), 'Network unavailable', False),
            (Reply(status=500), 'HTTP 500', False),
            (Reply(body=b'bad json'), 'Could not read', False),
        ]
        for reply, message, link in cases:
            with self.subTest(message=message):
                dialog = self.dialog(reply)
                reply.finished.emit()
                self.assertIn(message, dialog._status.text())
                self.assertEqual(dialog._view_release.isEnabled(), link)
                self.assertIsNone(dialog._reply)

    def test_timeout_and_close_abort_active_request(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                reply = Reply()
                dialog = self.dialog(reply)
                if timeout:
                    dialog._timer.timeout.emit()
                    self.assertIn('timed out', dialog._status.text())
                else:
                    dialog.reject()
                self.assertTrue(reply.aborted)
                self.assertIsNone(dialog._reply)
                self.assertFalse(dialog._timer.isActive())

    def test_help_menu_without_image_and_reuses_open_dialog(self):
        with patch.object(updates, 'QNetworkAccessManager') as manager:
            reply = Reply()
            manager.return_value.get.return_value = reply
            window = MainWindow()
            try:
                menu = next(a.menu() for a in window.menuBar().actions() if a.text() == '&Help')
                actions = menu.actions()
                self.assertEqual([a.text().replace('&', '') for a in actions],
                                 ['About Planetary Tools', 'Check for Updates…'])
                actions[1].trigger()
                first = window.findChild(updates.UpdateCheckDialog)
                self.assertTrue(first.isVisible())
                actions[1].trigger()
                manager.return_value.get.assert_called_once()
                self.assertIs(window.findChild(updates.UpdateCheckDialog), first)
                first.reject()
                self.assertTrue(reply.aborted)
            finally:
                window.close()


if __name__ == '__main__':
    unittest.main()
