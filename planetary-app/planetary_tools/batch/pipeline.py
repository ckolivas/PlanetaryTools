"""Batch image processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from planetary_tools.core.document import ImageDocument
from planetary_tools.filters.registry import FILTERS, apply_filter
from planetary_tools.io.loader import load_image, save_image, supported_extensions

_PROGRESS = Callable[[int, int, str], None]  # current, total, message


@dataclass
class PipelineStep:
    filter_id: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        from planetary_tools.filters.registry import FILTERS
        return FILTERS[self.filter_id].label


@dataclass
class BatchResult:
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def _image_paths(folder: Path, recursive: bool) -> list[Path]:
    exts = {e.lower() for e in supported_extensions()}
    paths: list[Path] = []
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    for p in sorted(iterator):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return paths


def apply_pipeline(
    data: np.ndarray,
    is_grayscale: bool,
    steps: list[PipelineStep],
) -> np.ndarray:
    out = data
    gray = is_grayscale
    for step in steps:
        merged = {**FILTERS[step.filter_id].default_params, **step.params}
        out = apply_filter(step.filter_id, out, gray, merged)
        # if step.filter_id == "oklab_luminance":
        #     gray = False
    return out


def run_batch(
    input_paths: list[Path],
    output_dir: Path,
    steps: list[PipelineStep],
    *,
    suffix: str = "_processed",
    bit_depth: int = 32,
    preserve_tree: bool = False,
    input_root: Path | None = None,
    on_progress: _PROGRESS | None = None,
) -> BatchResult:
    if not steps:
        raise ValueError("Pipeline must contain at least one filter step.")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = BatchResult()
    total = len(input_paths)

    for i, in_path in enumerate(input_paths):
        msg = in_path.name
        if on_progress:
            on_progress(i, total, msg)
        try:
            doc = load_image(in_path)
            processed = apply_pipeline(doc.data, doc.is_grayscale, steps)
            doc.set_data(processed)

            if preserve_tree and input_root is not None:
                rel = in_path.relative_to(input_root)
                out_path = output_dir / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = output_dir / f"{in_path.stem}{suffix}{in_path.suffix}"

            save_image(doc, out_path, bit_depth=bit_depth)
            result.processed += 1
        except Exception as exc:
            result.failed.append((str(in_path), str(exc)))

    if on_progress:
        on_progress(total, total, "Done")
    return result


def collect_paths(
    files: list[Path] | None = None,
    folder: Path | None = None,
    recursive: bool = False,
) -> list[Path]:
    if files:
        return list(files)
    if folder:
        return _image_paths(folder, recursive)
    return []