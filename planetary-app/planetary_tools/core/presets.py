"""Per-filter preset load/save."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

PRESET_DIR = Path(os.path.expanduser("~/.config/planetary-tools/presets"))

RESERVED = frozenset({"Default", "Last"})

_BUILTIN_PRESET_BUILDERS: dict[str, Callable[[dict[str, Any]], dict[str, dict[str, Any]]]] = {}
_BUILTIN_RESERVED: dict[str, frozenset[str]] = {}


def register_builtin_presets(
    filter_id: str,
    builder: Callable[[dict[str, Any]], dict[str, dict[str, Any]]],
    *,
    reserved_names: frozenset[str] = frozenset(),
) -> None:
    _BUILTIN_PRESET_BUILDERS[filter_id] = builder
    if reserved_names:
        _BUILTIN_RESERVED[filter_id] = reserved_names


def reserved_preset_names(filter_id: str) -> frozenset[str]:
    return RESERVED | _BUILTIN_RESERVED.get(filter_id, frozenset())

_LEGACY_PRESET_IDS = {"colour_matrix": "color_matrix"}


def _preset_path(filter_id: str) -> Path:
    return PRESET_DIR / f"{filter_id}.json"


def _preset_load_paths(filter_id: str) -> list[Path]:
    paths = [_preset_path(filter_id)]
    legacy = _LEGACY_PRESET_IDS.get(filter_id)
    if legacy is not None:
        paths.append(_preset_path(legacy))
    return paths


def load_presets(filter_id: str, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for path in _preset_load_paths(filter_id):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    return {}


def save_presets(filter_id: str, presets: dict[str, dict[str, Any]]) -> None:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = _preset_path(filter_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2)


def ensure_builtin_presets(
    filter_id: str,
    default_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    presets = load_presets(filter_id, default_params)
    if "Default" not in presets:
        presets["Default"] = deepcopy(default_params)
    if "Last" not in presets:
        presets["Last"] = deepcopy(presets["Default"])
    builder = _BUILTIN_PRESET_BUILDERS.get(filter_id)
    if builder is not None:
        for name, params in builder(default_params).items():
            if name not in presets:
                presets[name] = deepcopy(params)
    save_presets(filter_id, presets)
    return presets


def _register_filter_builtin_presets() -> None:
    from planetary_tools.filters.colour_matrix import (
        COLOUR_MATRIX_SENSOR_NAMES,
        colour_matrix_sensor_presets,
    )

    register_builtin_presets(
        "colour_matrix",
        colour_matrix_sensor_presets,
        reserved_names=COLOUR_MATRIX_SENSOR_NAMES,
    )


_register_filter_builtin_presets()