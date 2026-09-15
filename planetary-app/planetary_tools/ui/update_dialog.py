"""On-demand GitHub release checks without blocking the image editor."""

from __future__ import annotations

import json
import re

from PyQt6.QtCore import Qt, QTimer, QUrl, QVersionNumber
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout

from planetary_tools import __version__

RELEASES_URL = "https://github.com/ckolivas/PlanetaryTools/releases"
LATEST_RELEASE_API = "https://api.github.com/repos/ckolivas/PlanetaryTools/releases/latest"
CHECK_TIMEOUT_MS = 15_000


def release_version(payload: bytes, current: str) -> tuple[str, bool]:
    """Validate the stable release tag and compare numeric version sections."""
    release = json.loads(payload)
    if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
        raise ValueError("GitHub did not return a stable release.")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or not re.fullmatch(r"v?\d+(?:\.\d+)+", tag):
        raise ValueError("The release tag does not contain a recognised version number.")
    if not re.fullmatch(r"v?\d+(?:\.\d+)+", current):
        raise ValueError("The installed version number could not be compared.")
    latest, _ = QVersionNumber.fromString(tag.removeprefix("v"))
    installed, _ = QVersionNumber.fromString(current.removeprefix("v"))
    return tag, QVersionNumber.compare(latest.normalized(), installed.normalized()) > 0


class UpdateCheckDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Check for Updates")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(380, 150)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Installed version: {__version__}"))
        self._status = QLabel("Checking GitHub releases…")
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self._view_release = QPushButton("View Release")
        self._view_release.setEnabled(False)
        self._view_release.clicked.connect(self._open_release)
        buttons.addButton(self._view_release, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)
        self._release_url = RELEASES_URL

        self._network = QNetworkAccessManager(self)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._timeout)
        self.finished.connect(self._cancel_request)
        request = QNetworkRequest(QUrl(LATEST_RELEASE_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", f"PlanetaryTools/{__version__}".encode("ascii"))
        request.setAttribute(QNetworkRequest.Attribute.CacheLoadControlAttribute,
                             QNetworkRequest.CacheLoadControl.AlwaysNetwork)
        self._reply = self._network.get(request)
        self._reply.finished.connect(self._complete)
        self._timer.start(CHECK_TIMEOUT_MS)

    def _cancel_request(self) -> None:
        self._timer.stop()
        if self._reply is not None:
            reply, self._reply = self._reply, None
            reply.abort()
            reply.deleteLater()

    def _timeout(self) -> None:
        self._cancel_request()
        self._status.setText("The update check timed out. Please try again later.")

    def _complete(self) -> None:
        if self._reply is None:
            return
        self._timer.stop()
        reply, self._reply = self._reply, None
        try:
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if status == 404:
                self._status.setText("No published stable release was found on GitHub.")
            elif status in (403, 429):
                self._status.setText("GitHub refused the update check or its request limit was reached. "
                                     "Please try again later.")
            elif reply.error() != QNetworkReply.NetworkError.NoError:
                self._status.setText(f"Could not check for updates: {reply.errorString()}")
            elif status != 200:
                self._status.setText(f"Could not check for updates (HTTP {status}).")
            else:
                try:
                    tag, newer = release_version(bytes(reply.readAll()), __version__)
                except (ValueError, UnicodeError) as exc:
                    self._status.setText(f"Could not read the GitHub release: {exc}")
                    return
                self._release_url = f"{RELEASES_URL}/tag/{tag}"
                self._view_release.setEnabled(True)
                if newer:
                    self._status.setText(f"Planetary Tools {tag.removeprefix('v')} is available. "
                                         "Open the release page for downloads and release notes.")
                else:
                    self._status.setText(f"No newer release is available. "
                                         f"Latest GitHub release: {tag}.")
        finally:
            reply.deleteLater()

    def _open_release(self) -> None:
        if not QDesktopServices.openUrl(QUrl(self._release_url)):
            self._status.setText(f"Could not open your browser. Visit {self._release_url}")
