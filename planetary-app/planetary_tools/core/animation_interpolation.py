"""Timestamp-driven motion interpolation for planetary still-image animations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import os
from typing import Callable

import numpy as np

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_DATE = r"(?P<year>\d{4})-?(?P<month>\d{2})-?(?P<day>\d{2})"
_WINJUPOS = re.compile(
    r"(?<!\d)" + _DATE + r"[-_](?P<hour>\d{2})(?P<minute>\d{2})"
    r"(?:[._](?P<fraction>\d{1,6}))?(?!\d)"
)
_PVOL = re.compile(
    r"(?<!\d)" + _DATE + r"_(?P<hour>\d{2})-(?P<minute>\d{2})"
    r"(?:-(?P<second>\d{2}))?(?!\d)"
)
MAX_TIMELINE_FRAMES = 10_000


def filename_timestamp(path: str | Path) -> datetime:
    """Read UTC from WinJUPOS YYYY-MM-DD-HHMM_d or PVOL pYYYY-MM-DD_HH-MM-SS.

    WinJUPOS also accepts HHMM.d and compact YYYYMMDD dates. Its fractional
    field is a fraction of a minute, never seconds. PVOL seconds are optional.
    Processing suffixes and planet/observer prefixes are left untouched.
    """
    name = Path(path).stem
    match = _PVOL.search(name) or _WINJUPOS.search(name)
    if match is None:
        raise ValueError(f"No WinJUPOS or PVOL timestamp found in {Path(path).name}.")
    fields = match.groupdict()
    try:
        stamp = datetime(*(int(fields[key]) for key in ('year', 'month', 'day', 'hour', 'minute')),
                         second=int(fields.get('second') or 0), tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in {Path(path).name}: {exc}") from exc
    fraction = fields.get('fraction')
    if fraction:
        stamp += timedelta(microseconds=int(fraction) * 60_000_000 // 10**len(fraction))
    return stamp


def interval_microseconds(minutes: float) -> int:
    if not math.isfinite(minutes) or minutes < .1 or minutes > 1440:
        raise ValueError("Frame interval must be between 0.1 and 1440 minutes.")
    return round(minutes * 60_000_000)


def rounded_timestamp(stamp: datetime, interval_minutes: float) -> datetime:
    """Round to the nearest UTC interval, with exact halfway times rounded up."""
    step = interval_microseconds(interval_minutes)
    elapsed = stamp - _EPOCH
    micros = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
    return _EPOCH + timedelta(microseconds=((micros + step//2) // step) * step)


@dataclass(frozen=True)
class TimedFrame:
    path: Path
    captured: datetime
    rounded: datetime


@dataclass(frozen=True)
class AnimationTimeline:
    sources: tuple[TimedFrame, ...]
    # Number of output intervals between each adjacent pair of source frames.
    gaps: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return 1 + sum(self.gaps)


def build_timeline(paths: list[Path], interval_minutes: float = 1.0) -> AnimationTimeline:
    step = interval_microseconds(interval_minutes)
    if len(paths) < 2:
        raise ValueError("Select at least two images for motion interpolation.")
    sources = []
    for path in paths:
        stamp = filename_timestamp(path)
        sources.append(TimedFrame(Path(path), stamp, rounded_timestamp(stamp, interval_minutes)))
    sources.sort(key=lambda frame: (frame.captured, str(frame.path)))
    gaps = []
    for left, right in zip(sources, sources[1:]):
        if left.rounded == right.rounded:
            raise ValueError(
                f"{left.path.name} and {right.path.name} both round to "
                f"{left.rounded:%Y-%m-%d %H:%M:%S} UTC. "
                "Use a smaller frame interval or remove one of these files."
            )
        delta = right.rounded - left.rounded
        gaps.append(((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) // step)
    timeline = AnimationTimeline(tuple(sources), tuple(gaps))
    if timeline.frame_count > MAX_TIMELINE_FRAMES:
        raise ValueError(f"This interval would produce {timeline.frame_count:,} frames. "
                         f"Use a larger interval or a shorter time span (maximum {MAX_TIMELINE_FRAMES:,}).")
    return timeline


def interpolate_pair(first: np.ndarray, last: np.ndarray, intervals: int) -> list[np.ndarray]:
    """Generate the missing frames between two uint8 RGB endpoints using FFmpeg.

    Repeated endpoints supply the temporal context minterpolate needs. Only
    interior frames pass through FFmpeg, so source pixels remain exact.
    Full-resolution chroma avoids subsampling the generated RGB frames.
    """
    if intervals < 1:
        raise ValueError("Interpolation needs at least one time interval.")
    if intervals == 1:
        return []
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError("Motion interpolation requires FFmpeg with the minterpolate filter on PATH.")
    if first.shape != last.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Motion interpolation needs matching RGB frame sizes.")
    h, w = first.shape[:2]
    # minterpolate needs at least two 8-pixel blocks per dimension. Pad tiny
    # test/cropped images without changing the returned canvas dimensions.
    ph, pw = max(32, h), max(32, w)
    frames = [np.pad(np.asarray(frame, dtype=np.uint8), ((0, ph-h), (0, pw-w), (0, 0)))
              for frame in (first, last)]
    filters = (
        f"format=yuv444p,minterpolate=fps={intervals}:mi_mode=mci:mc_mode=aobmc:"
        "me_mode=bidir:me=umh:mb_size=8:search_param=32:vsbmc=1:scd=none,"
        f"trim=start_frame={intervals+1}:end_frame={2*intervals},format=rgb24"
    )
    # File-backed pipes avoid deadlocks and retaining large serialized input
    # and output copies while FFmpeg is working.
    with tempfile.TemporaryFile() as source, tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        for frame in (frames[0], frames[0], frames[1], frames[1]):
            source.write(frame.tobytes())
        source.seek(0)
        process = subprocess.Popen(
            [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin',
             '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{pw}x{ph}',
             '-framerate', '1', '-i', 'pipe:0', '-vf', filters,
             '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdin=source, stdout=output, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if code:
            errors.seek(0)
            message = errors.read().decode('utf-8', errors='replace').strip()
            raise RuntimeError(f"Motion interpolation failed: {message or f'FFmpeg exited with code {code}'}")
        frame_bytes = ph * pw * 3
        if output.tell() != (intervals-1) * frame_bytes:
            raise RuntimeError("Motion interpolation failed: FFmpeg returned an incomplete frame sequence.")
        output.seek(0)
        generated = []
        for _ in range(intervals-1):
            frame = np.frombuffer(output.read(frame_bytes), dtype=np.uint8).reshape(ph, pw, 3)
            generated.append(frame[:h, :w].copy())
    return generated


def interpolate_timeline(
    frames: list[np.ndarray], timeline: AnimationTimeline,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[np.ndarray]:
    if len(frames) != len(timeline.sources):
        raise ValueError("Frame count does not match the timestamp timeline.")
    result = [frames[0]]
    for i, gap in enumerate(timeline.gaps):
        if on_progress:
            on_progress(i, len(timeline.gaps), f"Interpolating {timeline.sources[i].rounded:%H:%M:%S} → "
                        f"{timeline.sources[i+1].rounded:%H:%M:%S} UTC")
        result.extend(interpolate_pair(frames[i], frames[i+1], gap))
        result.append(frames[i+1])
    return result
