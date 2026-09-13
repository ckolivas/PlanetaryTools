"""RGB Compose from Files — assign channel images to R/G/B."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from planetary_tools.ui.file_filters import image_file_filters
from planetary_tools.ui.recent_files import last_open_directory, remember_open_path

_CHANNELS = ("Red", "Green", "Blue")
_TOKEN_CHANNEL = {
    "r": "Red",
    "red": "Red",
    "g": "Green",
    "green": "Green",
    "b": "Blue",
    "blue": "Blue",
}


def detect_channel(path: Path) -> str | None:
    """Guess R/G/B from a filename, e.g. 'jupiter-red.tif' or 'jupiter_G.png'."""
    stem = path.stem
    for token in re.split(r"[^a-zA-Z]+", stem):
        channel = _TOKEN_CHANNEL.get(token.lower())
        if channel:
            return channel
    lowered = stem.lower()
    for word in ("red", "green", "blue"):
        if word in lowered:
            return word.capitalize()
    return None


class RGBComposeDialog(QDialog):
    """Pick Red/Green/Blue files from any directories, then compose."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RGB Compose from Files")
        self.setMinimumWidth(560)
        self._paths: dict[str, Path | None] = {ch: None for ch in _CHANNELS}
        self._edits: dict[str, QLineEdit] = {}
        self._browse_buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Choose two or three channel images. Files may be in different "
                "folders. A missing third channel is calculated from the other two."
            )
        )

        add_multi = QPushButton("Add files…")
        add_multi.setToolTip(
            "Add channel images from any folder, keeping assigned channels. "
            "Channel assignment is guessed from filenames (e.g. jupiter_R.tif)."
        )
        add_multi.clicked.connect(self._add_multiple_files)
        layout.addWidget(add_multi)

        form = QFormLayout()
        for channel in _CHANNELS:
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("None")
            self._edits[channel] = edit
            row.addWidget(edit, stretch=1)

            browse = QPushButton("Add…")
            self._browse_buttons[channel] = browse
            browse.setToolTip(f"Select the {channel} channel image from any folder.")
            browse.clicked.connect(lambda _=False, ch=channel: self._browse(ch))
            row.addWidget(browse)

            clear = QPushButton("Clear")
            clear.setToolTip(f"Remove the {channel} channel file.")
            clear.clicked.connect(lambda _=False, ch=channel: self._set_channel(ch, None))
            row.addWidget(clear)
            form.addRow(f"{channel}", row)
        layout.addLayout(form)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._align_check = QCheckBox("Align channels")
        self._align_check.setToolTip(
            "Align channel detail with fractional-pixel precision before combining.\n"
            "Corrects up to five pixels of misregistration between separately\n"
            "captured channels, with one resampling step."
        )
        layout.addWidget(self._align_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_hint()

    def _set_channel(self, channel: str, path: Path | None) -> None:
        if path is not None:
            resolved = path.expanduser().resolve()
            for other, existing in list(self._paths.items()):
                if other != channel and existing is not None:
                    if existing.expanduser().resolve() == resolved:
                        self._paths[other] = None
                        self._refresh_row(other)
            self._paths[channel] = path
        else:
            self._paths[channel] = None
        self._refresh_row(channel)
        self._update_hint()

    def _refresh_row(self, channel: str) -> None:
        path = self._paths[channel]
        edit = self._edits[channel]
        self._browse_buttons[channel].setText("Change…" if path is not None else "Add…")
        if path is None:
            edit.clear()
            edit.setToolTip("")
        else:
            edit.setText(str(path))
            edit.setToolTip(str(path))

    def _browse_start_dir(self, channel: str | None = None) -> str:
        if channel is not None:
            current = self._paths[channel]
            if current is not None and current.parent.is_dir():
                return str(current.parent)
        for path in self._paths.values():
            if path is not None and path.parent.is_dir():
                return str(path.parent)
        return last_open_directory()

    def _browse(self, channel: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {channel} channel image",
            self._browse_start_dir(channel),
            image_file_filters(),
        )
        if not path:
            return
        remember_open_path(path)
        self._set_channel(channel, Path(path))

    def _add_multiple_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add channel files",
            self._browse_start_dir(),
            image_file_filters(),
        )
        if not paths:
            return
        remember_open_path(paths[0])
        seen = {p.resolve() for p in self._assigned().values()}
        claimed: dict[str, Path] = {}
        leftover: list[Path] = []
        skipped: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            guess = detect_channel(path)
            if guess is not None and (guess in claimed or self._paths[guess] is not None):
                skipped.append(path)
            elif guess is not None:
                claimed[guess] = path
            else:
                leftover.append(path)
        for channel, path in claimed.items():
            self._set_channel(channel, path)
        for path in leftover:
            empty = next((ch for ch in _CHANNELS if self._paths[ch] is None), None)
            if empty is None:
                skipped.append(path)
            else:
                self._set_channel(empty, path)
        if skipped:
            QMessageBox.information(
                self, "RGB Compose from Files",
                "These files could not be added because their channel is already assigned "
                "or all three channels are filled. Use Change or Clear on a channel to replace it.\n\n"
                + "\n".join(str(path) for path in skipped),
            )

    def _assigned(self) -> dict[str, Path]:
        return {ch: path for ch, path in self._paths.items() if path is not None}

    def _update_hint(self) -> None:
        assigned = self._assigned()
        if len(assigned) == 2:
            missing = next(ch for ch in _CHANNELS if ch not in assigned)
            self._hint.setText(
                f"Two files selected — the missing {missing} channel will be "
                "calculated from the other two."
            )
        elif len(assigned) >= 3:
            self._hint.setText("All three channels assigned.")
        else:
            self._hint.setText("")

    def _on_accept(self) -> None:
        assigned = self._assigned()
        if len(assigned) not in (2, 3):
            QMessageBox.warning(
                self,
                "RGB Compose from Files",
                "Select either two or three channel images.",
            )
            return
        missing = [ch for ch, path in assigned.items() if not path.is_file()]
        if missing:
            QMessageBox.warning(
                self,
                "RGB Compose from Files",
                "These file(s) could not be found:\n"
                + "\n".join(str(assigned[ch]) for ch in missing),
            )
            return
        self.accept()

    def channel_assignment(self) -> dict[str, Path]:
        """Return e.g. {'Red': path, 'Green': path, 'Blue': path}."""
        return self._assigned()

    def align_channels(self) -> bool:
        return self._align_check.isChecked()
