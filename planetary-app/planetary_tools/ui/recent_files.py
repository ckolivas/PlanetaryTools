"""Persist and query recently opened image paths."""

from __future__ import annotations

import math
from pathlib import Path

from PyQt6.QtCore import QSettings

_MAX_RECENT = 10
_MAX_RECENT_DIRECTORIES = 10


def _settings() -> QSettings:
    return QSettings()


def _normalize(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def list_recent_directories() -> list[str]:
    """Existing directories used by any file picker, newest first."""
    settings = _settings()
    raw = settings.value("recentDirectories")
    if raw is None:
        # Seed the new sidebar from the file history of earlier versions.
        paths = [settings.value("lastOpenDir"), settings.value("lastSaveDir")]
        paths.extend(str(Path(path).parent) for path in list_recent())
    else:
        paths = [raw] if isinstance(raw, str) else raw
    directories: list[str] = []
    for path in paths:
        if not path:
            continue
        directory = Path(str(path)).expanduser()
        if directory.is_dir():
            normalized = _normalize(directory)
            if normalized not in directories:
                directories.append(normalized)
    return directories[:_MAX_RECENT_DIRECTORIES]


def remember_directory(path: str | Path) -> None:
    """Record a confirmed picker directory independently of recent image files."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        return
    normalized = _normalize(directory)
    paths = [normalized] + [p for p in list_recent_directories() if p != normalized]
    settings = _settings()
    settings.setValue("recentDirectories", paths[:_MAX_RECENT_DIRECTORIES])
    settings.sync()


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
        remember_directory(directory)
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
        remember_directory(directory)
        _settings().setValue("lastSaveDir", str(directory))


def last_save_filter() -> str | None:
    """Return the last confirmed image file-type choice, if any."""
    raw = _settings().value("lastSaveFilter")
    return str(raw) if raw else None


def remember_save_filter(filter_str: str) -> None:
    """Store the last confirmed file-type choice, including across restarts."""
    settings = _settings()
    settings.setValue("lastSaveFilter", filter_str)
    settings.sync()


def last_output_option(key: str, default: str, choices: tuple[str, ...]) -> str:
    """Restore an export option, falling back if a stored value is invalid."""
    value = str(_settings().value(f"outputOptions/{key}", default))
    return value if value in choices else default


def last_output_number(key: str, default: float, minimum: float, maximum: float) -> float:
    """Restore a finite export setting within its supported range."""
    try:
        value = float(_settings().value(f"outputOptions/{key}", default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and minimum <= value <= maximum else default


def remember_output_option(key: str, value: str | int | float) -> None:
    settings = _settings()
    settings.setValue(f"outputOptions/{key}", str(value))
    settings.sync()


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
