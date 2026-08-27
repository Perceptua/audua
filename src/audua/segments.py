"""Clip formation: greedy merging, override windows, and non-destructive flags.

This module is pure — no ffmpeg, no torch, no filesystem beyond reading the
override sidecar. That makes the interesting decisions (where does a clip start
and stop?) cheap to unit-test without any audio.

Three rules drive everything here:

1. **Greedy.** Speech separated by a short silence is one clip, not two. The
   default absorbs gaps up to 5 seconds.
2. **Cautious.** Nothing is ever discarded. Clips that look dubious are
   *flagged* in the sidecar so a downstream workflow (or a human) can decide.
3. **Overrides win.** A supplied time window is always emitted as a clip, with
   the exact bounds requested.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, format_time, parse_time

log = logging.getLogger(__name__)

Interval = tuple[float, float]


class OverrideError(ValueError):
    """Raised when an override sidecar is malformed."""


@dataclass
class Override:
    start: float
    end: float
    label: str | None = None

    def as_interval(self) -> Interval:
        return (self.start, self.end)


@dataclass
class Clip:
    """One output clip. ``index`` is assigned after final ordering."""

    start: float
    end: float
    origin: str = "vad"           # "vad" | "override" | "vad+override"
    labels: list[str] = field(default_factory=list)
    speech_seconds: float = 0.0
    flags: list[str] = field(default_factory=list)
    index: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_ratio(self) -> float:
        return self.speech_seconds / self.duration if self.duration > 0 else 0.0

    @property
    def stem(self) -> str:
        return f"clip_{self.index:04d}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "stem": self.stem,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "start_hms": format_time(self.start),
            "end_hms": format_time(self.end),
            "origin": self.origin,
            "labels": self.labels,
            "speech_seconds": round(self.speech_seconds, 3),
            "speech_ratio": round(self.speech_ratio, 4),
            "flags": self.flags,
        }


# --------------------------------------------------------------------------
# override sidecar
# --------------------------------------------------------------------------

def default_override_path(source: Path) -> Path:
    """``recording.m4a`` -> ``recording.overrides.json`` (alongside the audio)."""
    return source.parent / f"{source.stem}.overrides.json"


def load_overrides(path: Path | None, duration: float) -> list[Override]:
    """Load and validate forced time windows.

    Accepted shapes::

        {"windows": [{"start": "00:12:30", "end": "00:13:45", "label": "intro"}]}
        [{"start": 750, "end": 825}]

    A missing file is not an error — most recordings will not have one.
    """
    if path is None or not Path(path).is_file():
        return []

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OverrideError(f"{path.name} is not valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        entries = payload.get("windows", payload.get("overrides", []))
    elif isinstance(payload, list):
        entries = payload
    else:
        raise OverrideError(f"{path.name}: expected an object or a list at the top level.")

    if not isinstance(entries, list):
        raise OverrideError(f"{path.name}: 'windows' must be a list.")

    overrides: list[Override] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise OverrideError(f"{path.name}: entry #{position} is not an object.")
        if "start" not in entry or "end" not in entry:
            raise OverrideError(f"{path.name}: entry #{position} needs both 'start' and 'end'.")
        try:
            start = parse_time(entry["start"])
            end = parse_time(entry["end"])
        except ValueError as exc:
            raise OverrideError(f"{path.name}: entry #{position}: {exc}") from exc

        if end <= start:
            raise OverrideError(
                f"{path.name}: entry #{position} ends ({end}) at or before it starts ({start})."
            )
        if start >= duration:
            log.warning(
                "%s: override #%d starts at %s, past the end of the recording (%s). Skipping.",
                path.name, position, format_time(start), format_time(duration),
            )
            continue
        if end > duration:
            log.warning(
                "%s: override #%d ends at %s, past the end of the recording (%s). Clamping.",
                path.name, position, format_time(end), format_time(duration),
            )
            end = duration

        label = entry.get("label")
        overrides.append(Override(start=start, end=end, label=str(label) if label else None))

    overrides.sort(key=lambda o: (o.start, o.end))
    return overrides


# --------------------------------------------------------------------------
# interval algebra
# --------------------------------------------------------------------------

def greedy_merge(intervals: list[Interval], gap: float) -> list[Interval]:
    """Collapse intervals separated by <= ``gap`` seconds into single spans.

    This is the rule that turns ``speech -> 3s silence -> speech`` into one clip
    rather than two.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[Interval] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def pad_intervals(intervals: list[Interval], pad: float, duration: float) -> list[Interval]:
    """Widen each interval by ``pad`` on both sides, clamped to the file.

    Padding can make neighbours overlap, so the result is re-merged with a zero
    gap to keep the list disjoint.
    """
    if pad <= 0:
        return list(intervals)
    widened = [
        (max(0.0, start - pad), min(duration, end + pad))
        for start, end in intervals
    ]
    return greedy_merge(widened, 0.0)


def subtract(interval: Interval, holes: list[Interval]) -> list[Interval]:
    """Remove ``holes`` from ``interval``, returning the surviving fragments."""
    fragments = [interval]
    for hole_start, hole_end in sorted(holes):
        next_fragments: list[Interval] = []
        for start, end in fragments:
            if hole_end <= start or hole_start >= end:
                next_fragments.append((start, end))     # no overlap
                continue
            if hole_start > start:
                next_fragments.append((start, hole_start))
            if hole_end < end:
                next_fragments.append((hole_end, end))
        fragments = next_fragments
    return [(s, e) for s, e in fragments if e > s]


def overlap_seconds(interval: Interval, others: list[Interval]) -> float:
    """Total length of ``interval`` covered by ``others`` (assumed disjoint)."""
    start, end = interval
    total = 0.0
    for other_start, other_end in others:
        if other_end <= start:
            continue
        if other_start >= end:
            break
        total += min(end, other_end) - max(start, other_start)
    return total


def split_long(
    interval: Interval,
    max_duration: float,
    speech_regions: list[Interval],
) -> list[Interval]:
    """Break an over-long clip, preferring to cut inside a silence.

    Greedy merging can legitimately produce a very long clip. If the caller has
    set a ceiling, split at the internal silence closest to the ceiling so the
    cut lands between words rather than through one. Falls back to a hard cut
    only when there is no silence to use.
    """
    if max_duration <= 0 or (interval[1] - interval[0]) <= max_duration:
        return [interval]

    start, end = interval
    pieces: list[Interval] = []
    cursor = start

    while end - cursor > max_duration:
        limit = cursor + max_duration
        inside = [r for r in speech_regions if r[1] > cursor and r[0] < end]

        # Best case: the ceiling already lands in a silence. Cut right there and
        # use the whole budget rather than backing off to an earlier gap.
        if not any(start_r < limit < end_r for start_r, end_r in inside):
            cut = limit
        else:
            # Otherwise back off to the latest silence before the ceiling, so the
            # cut falls between words instead of through one.
            candidates = [
                (earlier[1] + later[0]) / 2.0
                for earlier, later in zip(inside, inside[1:])
                if cursor + 0.1 < (earlier[1] + later[0]) / 2.0 <= limit
            ]
            cut = max(candidates) if candidates else limit

        pieces.append((cursor, cut))
        cursor = cut

    if end > cursor:
        pieces.append((cursor, end))
    return pieces


# --------------------------------------------------------------------------
# clip assembly
# --------------------------------------------------------------------------

def build_clips(
    speech_regions: list[Interval],
    overrides: list[Override],
    config: Config,
    duration: float,
) -> list[Clip]:
    """Turn raw VAD regions plus override windows into the final clip list.

    ``override_mode``:

    * ``isolate`` (default) — each override is emitted verbatim as its own clip
      and is carved out of any overlapping VAD clip. You get exactly the bounds
      you asked for, and no audio is duplicated across two clips.
    * ``merge`` — overrides are folded into the greedy merge as mandatory seed
      regions. The window is still guaranteed to be inside a clip, but the clip
      may extend beyond it.
    """
    speech_regions = greedy_merge(speech_regions, 0.0)  # normalise/disjoin
    override_intervals = [o.as_interval() for o in overrides]

    if config.override_mode == "merge":
        seeded = greedy_merge(list(speech_regions) + override_intervals, config.merge_gap)
        padded = pad_intervals(seeded, config.pad, duration)
        # Re-merge with the override windows so padding can never clip one short.
        padded = greedy_merge(padded + override_intervals, 0.0)
        clips = [
            Clip(start=s, end=e, origin="vad")
            for span in padded
            for s, e in split_long(span, config.max_clip_duration, speech_regions)
        ]
        for clip in clips:
            hits = [o for o in overrides if o.start < clip.end and o.end > clip.start]
            if hits:
                clip.origin = "vad+override"
                clip.labels = [o.label for o in hits if o.label]
    else:
        merged = greedy_merge(list(speech_regions), config.merge_gap)
        padded = pad_intervals(merged, config.pad, duration)

        clips = []
        for span in padded:
            for fragment in subtract(span, override_intervals):
                for piece in split_long(fragment, config.max_clip_duration, speech_regions):
                    clips.append(Clip(start=piece[0], end=piece[1], origin="vad"))

        for override in overrides:
            clips.append(
                Clip(
                    start=override.start,
                    end=override.end,
                    origin="override",
                    labels=[override.label] if override.label else [],
                )
            )

    clips.sort(key=lambda c: (c.start, c.end))
    for position, clip in enumerate(clips, start=1):
        clip.index = position
        clip.speech_seconds = overlap_seconds((clip.start, clip.end), speech_regions)
        clip.flags = compute_flags(clip, config)

    _assert_overrides_present(clips, overrides)
    return clips


def compute_flags(clip: Clip, config: Config) -> list[str]:
    """Label a clip's suspicious properties. Never a reason to delete it."""
    flags: list[str] = []

    if clip.duration < config.min_clip_duration:
        flags.append("short")

    if clip.origin == "override":
        # Bounds were dictated by the user; sparseness is expected and fine.
        if clip.speech_seconds <= 0.0:
            flags.append("no_vad_speech")
        return flags

    if clip.speech_seconds <= 0.0:
        flags.append("no_vad_speech")
    elif clip.speech_ratio < config.sparse_ratio:
        # Mostly silence bridged by the greedy merge. Still kept.
        flags.append("sparse")

    return flags


def _assert_overrides_present(clips: list[Clip], overrides: list[Override]) -> None:
    """Hard invariant: every requested window is fully inside some clip.

    A supplied time window must always be saved. If a refactor ever breaks that,
    this fails loudly at segmentation time rather than silently losing audio.
    """
    for override in overrides:
        covered = any(
            clip.start <= override.start + 1e-6 and clip.end >= override.end - 1e-6
            for clip in clips
        )
        if not covered:
            raise AssertionError(
                "Override window "
                f"{format_time(override.start)}–{format_time(override.end)} "
                "was not preserved as a clip. This is a bug; please report it."
            )


def summarise(clips: list[Clip], duration: float) -> dict:
    """Aggregate stats for the manifest and the console summary."""
    kept = sum(c.duration for c in clips)
    flagged = [c for c in clips if c.flags]
    return {
        "clip_count": len(clips),
        "source_duration": round(duration, 3),
        "clip_seconds": round(kept, 3),
        "retained_fraction": round(kept / duration, 4) if duration else 0.0,
        "override_clips": sum(1 for c in clips if "override" in c.origin),
        "flagged_clips": len(flagged),
        "flag_counts": {
            flag: sum(1 for c in clips if flag in c.flags)
            for flag in sorted({f for c in clips for f in c.flags})
        },
    }
