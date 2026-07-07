"""Persist and query recently opened image paths."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings

_MAX_RECENT = 10


def _settings() -> QSettings:
    return QSettings()


def _normalize(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def list_recent(*, existing_only: bool = False) -> list[str]:
    """Return stored recent paths, most recent first."""
    raw = _settings().value("recentFiles")
    if not raw:
        return []
    if isinstance(raw, str):
        paths = [raw]
    else:
        paths = [str(p) for p in raw]
    if not existing_only:
        return paths[:_MAX_RECENT]
    return [p for p in paths if Path(p).is_file()][:_MAX_RECENT]


def last_open_directory() -> str:
    """Return the last directory used for Open, or a sensible fallback."""
    raw = _settings().value("lastOpenDir")
    if raw:
        directory = Path(str(raw)).expanduser()
        if directory.is_dir():
            return str(directory.resolve())
    for path in list_recent():
        parent = Path(path).parent
        if parent.is_dir():
            return str(parent.resolve())
    return str(Path.home())


def remember_open_path(path: str | Path) -> None:
    """Store the parent directory of a file opened successfully."""
    resolved = Path(path).expanduser().resolve()
    directory = resolved.parent if resolved.is_file() else resolved
    if directory.is_dir():
        _settings().setValue("lastOpenDir", str(directory))


def last_save_directory() -> str:
    """Return the last directory used for Save As, or a sensible fallback."""
    raw = _settings().value("lastSaveDir")
    if raw:
        directory = Path(str(raw)).expanduser()
        if directory.is_dir():
            return str(directory.resolve())
    return str(Path.home())


def remember_save_path(path: str | Path) -> None:
    """Store the parent directory of a file saved successfully."""
    resolved = Path(path).expanduser().resolve()
    directory = resolved.parent
    if directory.is_dir():
        _settings().setValue("lastSaveDir", str(directory))


def add_recent(path: str | Path) -> None:
    """Record a successfully opened file at the front of the list."""
    resolved = _normalize(path)
    paths = [p for p in list_recent() if p != resolved]
    paths.insert(0, resolved)
    _settings().setValue("recentFiles", paths[:_MAX_RECENT])
    remember_open_path(resolved)


def remove_recent(path: str | Path) -> None:
    resolved = _normalize(path)
    paths = [p for p in list_recent() if p != resolved]
    _settings().setValue("recentFiles", paths)