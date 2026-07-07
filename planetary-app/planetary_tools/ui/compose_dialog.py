"""RGB Compose from Files — assign loaded channel images to R/G/B."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

_CHANNELS = ["Red", "Green", "Blue"]
_TOKEN_CHANNEL = {"r": "Red", "red": "Red", "g": "Green", "green": "Green", "b": "Blue", "blue": "Blue"}


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
    """Let the user confirm/adjust which file maps to which RGB channel."""

    def __init__(self, paths: list[Path], guesses: list[str | None], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RGB Compose from Files")
        self._paths = paths
        self._combos: list[QComboBox] = []

        layout = QVBoxLayout(self)
        if len(paths) == 2:
            layout.addWidget(
                QLabel(
                    "Only two files selected — the missing channel will be\n"
                    "calculated from the other two."
                )
            )

        form = QFormLayout()
        used: set[str] = set()
        for path, guess in zip(paths, guesses):
            combo = QComboBox()
            combo.addItems(_CHANNELS)
            if guess and guess not in used:
                combo.setCurrentText(guess)
                used.add(guess)
            form.addRow(path.name, combo)
            self._combos.append(combo)
        layout.addLayout(form)

        self._align_check = QCheckBox("Align channels")
        self._align_check.setToolTip(
            "Enlarge each channel 3×, align them by best luma match, then\n"
            "resize back down before combining. Corrects small misregistration\n"
            "between separately captured channels."
        )
        layout.addWidget(self._align_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        choices = [combo.currentText() for combo in self._combos]
        if len(set(choices)) != len(choices):
            QMessageBox.warning(
                self,
                "RGB Compose from Files",
                "Each file must be assigned a different channel.",
            )
            return
        self.accept()

    def channel_assignment(self) -> dict[str, Path]:
        """Return e.g. {'Red': path, 'Green': path, 'Blue': path}."""
        return {combo.currentText(): path for path, combo in zip(self._paths, self._combos)}

    def align_channels(self) -> bool:
        return self._align_check.isChecked()
