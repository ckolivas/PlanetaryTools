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


def add_recent(path: str | Path) -> None:
    """Record a successfully opened file at the front of the list."""
    resolved = _normalize(path)
    paths = [p for p in list_recent() if p != resolved]
    paths.insert(0, resolved)
    _settings().setValue("recentFiles", paths[:_MAX_RECENT])


def remove_recent(path: str | Path) -> None:
    resolved = _normalize(path)
    paths = [p for p in list_recent() if p != resolved]
    _settings().setValue("recentFiles", paths)