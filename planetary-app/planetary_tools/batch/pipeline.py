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
    # Name of the saved preset used for params, if any (for list display).
    preset_name: str | None = None

    @property
    def label(self) -> str:
        from planetary_tools.filters.registry import FILTERS
        base = FILTERS[self.filter_id].label
        parts: list[str] = []
        if self.preset_name:
            parts.append(self.preset_name)
        if self.params.get("auto"):
            parts.append("Auto")
        if parts:
            return f"{base} — {', '.join(parts)}"
        return base


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


def _resolve_auto_params(
    filter_id: str,
    data: np.ndarray,
    is_grayscale: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    """If auto is enabled, search per-image amounts and merge into params."""
    merged = dict(params)
    if not merged.get("auto"):
        return merged

    if filter_id == "wavelet_sharpen":
        from planetary_tools.filters.wavelet_auto import auto_wavelet_sharpen_params

        result = auto_wavelet_sharpen_params(
            data,
            is_grayscale,
            target_noise=float(merged.get("target_noise", 3.0)),
            target_contrast=float(merged.get("target_contrast", 15.0)),
        )
        merged.update({
            "fine": result.fine,
            "medium": result.medium,
            "coarse": result.coarse,
            "chunky": result.chunky,
        })
    elif filter_id == "adaptive_deconv":
        from planetary_tools.filters.adaptive_deconv_auto import (
            auto_adaptive_deconv_params,
        )

        result = auto_adaptive_deconv_params(
            data,
            is_grayscale,
            target_noise=float(merged.get("target_noise", 3.5)),
            target_contrast=float(merged.get("target_contrast", 15.0)),
            adaptive=bool(merged.get("adaptive", True)),
            oklab=bool(merged.get("oklab", True)),
        )
        merged["amount"] = result.amount
    return merged


def apply_pipeline(
    data: np.ndarray,
    is_grayscale: bool,
    steps: list[PipelineStep],
) -> np.ndarray:
    out = data
    gray = is_grayscale
    for step in steps:
        merged = {**FILTERS[step.filter_id].default_params, **step.params}
        merged = _resolve_auto_params(step.filter_id, out, gray, merged)
        out = apply_filter(step.filter_id, out, gray, merged)
        # if step.filter_id == "oklab_luminance":
        #     gray = False
    return out


def output_path_for(
    in_path: Path,
    output_dir: Path,
    *,
    suffix: str = "_processed",
    preserve_tree: bool = False,
    input_root: Path | None = None,
) -> Path:
    """Return the destination path for one input (same rules as ``run_batch``)."""
    if preserve_tree and input_root is not None:
        rel = in_path.relative_to(input_root)
        return output_dir / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
    return output_dir / f"{in_path.stem}{suffix}{in_path.suffix}"


def planned_output_paths(
    input_paths: list[Path],
    output_dir: Path,
    *,
    suffix: str = "_processed",
    preserve_tree: bool = False,
    input_root: Path | None = None,
) -> list[Path]:
    """Output paths for every input, in the same order as ``input_paths``."""
    return [
        output_path_for(
            p,
            output_dir,
            suffix=suffix,
            preserve_tree=preserve_tree,
            input_root=input_root,
        )
        for p in input_paths
    ]


def existing_output_paths(paths: list[Path]) -> list[Path]:
    """Return the subset of ``paths`` that already exist as files."""
    return [p for p in paths if p.is_file()]


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

            out_path = output_path_for(
                in_path,
                output_dir,
                suffix=suffix,
                preserve_tree=preserve_tree,
                input_root=input_root,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

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