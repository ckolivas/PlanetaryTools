"""Batch image processing pipeline."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from planetary_tools.core.document import ImageDocument
from planetary_tools.filters.registry import FILTERS, apply_filter
from planetary_tools.io.loader import load_image, save_image, supported_extensions

_PROGRESS = Callable[[int, int, str], None]  # current, total, message

WORKFLOW_KIND = "planetary-tools-batch-workflow"
WORKFLOW_VERSION = 1
# JSON file extension for saved batch workflows.
WORKFLOW_EXTENSION = ".ptbatch"


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
class BatchWorkflow:
    """Serializable batch pipeline plus related output options."""

    steps: list[PipelineStep] = field(default_factory=list)
    suffix: str = "_processed"
    bit_depth: int = 32
    preserve_tree: bool = False
    recursive: bool = False
    output_dir: str = ""


@dataclass
class BatchResult:
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def _json_safe(value: Any) -> Any:
    """Convert params to JSON-serialisable forms (numpy scalars → Python)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # bool is a subclass of int — must be checked first.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is None or isinstance(value, str):
        return value
    # Drop non-serialisable blobs (e.g. secondary image arrays).
    raise TypeError(f"Cannot serialise workflow value of type {type(value)!r}")


def workflow_to_dict(workflow: BatchWorkflow) -> dict[str, Any]:
    steps_out: list[dict[str, Any]] = []
    for step in workflow.steps:
        entry: dict[str, Any] = {
            "filter_id": step.filter_id,
            "params": _json_safe(step.params),
        }
        if step.preset_name:
            entry["preset_name"] = step.preset_name
        steps_out.append(entry)
    return {
        "kind": WORKFLOW_KIND,
        "version": WORKFLOW_VERSION,
        "suffix": workflow.suffix,
        "bit_depth": int(workflow.bit_depth),
        "preserve_tree": bool(workflow.preserve_tree),
        "recursive": bool(workflow.recursive),
        "output_dir": str(workflow.output_dir or ""),
        "steps": steps_out,
    }


def workflow_from_dict(data: dict[str, Any]) -> tuple[BatchWorkflow, list[str]]:
    """Parse a workflow dict. Returns (workflow, warning messages)."""
    warnings: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("Workflow file must contain a JSON object.")
    kind = data.get("kind")
    if kind is not None and kind != WORKFLOW_KIND:
        warnings.append(f"Unexpected workflow kind {kind!r}; loading steps anyway.")
    version = data.get("version", 1)
    if version != WORKFLOW_VERSION:
        warnings.append(
            f"Workflow version {version} differs from supported "
            f"{WORKFLOW_VERSION}; attempting load."
        )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Workflow is missing a 'steps' array.")

    steps: list[PipelineStep] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            warnings.append(f"Step {i + 1}: skipped (not an object).")
            continue
        fid = raw.get("filter_id")
        if not isinstance(fid, str) or fid not in FILTERS:
            warnings.append(f"Step {i + 1}: unknown filter {fid!r}; skipped.")
            continue
        fdef = FILTERS[fid]
        if not fdef.batch_enabled:
            warnings.append(
                f"Step {i + 1}: filter {fdef.label!r} is not available in batch; skipped."
            )
            continue
        params_raw = raw.get("params") or {}
        if not isinstance(params_raw, dict):
            warnings.append(f"Step {i + 1}: invalid params; using defaults.")
            params_raw = {}
        try:
            params = {**fdef.default_params, **_json_safe(params_raw)}
        except TypeError:
            warnings.append(f"Step {i + 1}: non-serialisable params; using defaults.")
            params = deepcopy(fdef.default_params)
        preset_name = raw.get("preset_name")
        if preset_name is not None:
            preset_name = str(preset_name)
        steps.append(
            PipelineStep(filter_id=fid, params=params, preset_name=preset_name)
        )

    bit_depth = int(data.get("bit_depth", 32))
    if bit_depth not in (8, 16, 32):
        warnings.append(f"Invalid bit_depth {bit_depth}; using 32.")
        bit_depth = 32

    workflow = BatchWorkflow(
        steps=steps,
        suffix=str(data.get("suffix", "_processed") or "_processed"),
        bit_depth=bit_depth,
        preserve_tree=bool(data.get("preserve_tree", False)),
        recursive=bool(data.get("recursive", False)),
        output_dir=str(data.get("output_dir", "") or ""),
    )
    return workflow, warnings


def save_workflow(path: str | Path, workflow: BatchWorkflow) -> None:
    path = Path(path)
    if path.suffix.lower() != WORKFLOW_EXTENSION:
        path = path.with_suffix(WORKFLOW_EXTENSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(workflow_to_dict(workflow), f, indent=2)
        f.write("\n")


def load_workflow(path: str | Path) -> tuple[BatchWorkflow, list[str]]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return workflow_from_dict(data)


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