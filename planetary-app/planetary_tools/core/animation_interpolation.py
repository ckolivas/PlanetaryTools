"""Timestamp-driven motion interpolation for planetary still-image animations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
from fractions import Fraction
from typing import Callable

import av
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
    if first.shape != last.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Motion interpolation needs matching RGB frame sizes.")
    h, w = first.shape[:2]
    # minterpolate needs at least two 8-pixel blocks per dimension. Pad tiny
    # test/cropped images without changing the returned canvas dimensions.
    ph, pw = max(32, h), max(32, w)
    frames = [np.pad(np.asarray(frame, dtype=np.uint8), ((0, ph-h), (0, pw-w), (0, 0)))
              for frame in (first, last)]
    graph = av.filter.Graph()
    source = graph.add('buffer', f'video_size={pw}x{ph}:pix_fmt=rgb24:time_base=1/1:frame_rate=1/1:pixel_aspect=1/1')
    graph.link_nodes(
        source,
        graph.add('format', 'pix_fmts=yuv444p'),
        graph.add('minterpolate',
                  f'fps={intervals}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:'
                  'me=umh:mb_size=8:search_param=32:vsbmc=1:scd=none'),
        graph.add('trim', f'start_frame={intervals+1}:end_frame={2*intervals}'),
        graph.add('format', 'pix_fmts=rgb24'),
        graph.add('buffersink'),
    )
    graph.configure()
    generated = []

    def drain() -> None:
        while True:
            try:
                frame = graph.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            generated.append(frame.to_ndarray(format='rgb24')[:h, :w].copy())

    try:
        for index, pixels in enumerate((frames[0], frames[0], frames[1], frames[1])):
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts = index
            frame.time_base = Fraction(1, 1)
            source.push(frame)
            drain()
        source.push(None)
        drain()
    except av.FFmpegError as exc:
        raise RuntimeError(f"Motion interpolation failed: {exc}") from exc
    if len(generated) != intervals-1:
        raise RuntimeError("Motion interpolation failed: FFmpeg returned an incomplete frame sequence.")
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
