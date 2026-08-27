"""ffmpeg/ffprobe wrappers: probing, bounded decoding, and clip extraction.

Nothing here ever loads a whole file into memory. Decoding is done in windows
streamed off ffmpeg's stdout, so a four-hour recording costs the same RAM as a
four-minute one.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import SAMPLE_RATE, STREAM_COPY_SAFE, find_ffmpeg, find_ffprobe

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Raised when ffmpeg/ffprobe fails on a file."""


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration: float
    sample_rate: int
    channels: int
    codec: str
    size_bytes: int
    mtime_ns: int

    def fingerprint(self) -> str:
        """Cheap identity check for resumability: size + mtime + duration.

        Deliberately avoids hashing file contents — these files are large and
        a full hash would defeat the point of a fast resume.
        """
        return f"{self.size_bytes}:{self.mtime_ns}:{self.duration:.3f}"


def _run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    log.debug("exec: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=capture,
        check=False,
        # never shell=True: keeps quoting sane and identical across Windows/Linux
    )


def probe(path: Path) -> AudioInfo:
    """Read duration/codec metadata for an audio file."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"Not a file: {path}")

    cmd = [
        find_ffprobe(),
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed on {path.name}: {proc.stderr.decode('utf-8', 'replace').strip()}")

    try:
        payload = json.loads(proc.stdout.decode("utf-8", "replace"))
        streams = payload.get("streams") or []
        if not streams:
            raise AudioError(f"No audio stream found in {path.name}.")
        stream = streams[0]
        duration = float(payload.get("format", {}).get("duration") or 0.0)
    except (ValueError, KeyError) as exc:  # pragma: no cover - malformed ffprobe output
        raise AudioError(f"Could not parse ffprobe output for {path.name}: {exc}") from exc

    if duration <= 0:
        raise AudioError(f"{path.name} reports a non-positive duration; refusing to process.")

    stat = path.stat()
    return AudioInfo(
        path=path,
        duration=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or "unknown"),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def iter_windows(
    path: Path,
    duration: float,
    window_seconds: float,
    *,
    overlap: float = 0.5,
    audio_filter: str | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(window_start_seconds, float32_mono_16k_samples)`` pairs.

    A small overlap keeps a word straddling a window edge from being missed by
    the VAD; duplicate detections across the seam are harmless because the
    greedy merge collapses them.

    ``audio_filter`` is an optional ffmpeg ``-af`` chain applied to this
    decode only. It exists so VAD can see a cleaned-up signal (e.g. wind
    noise knocked down) without altering the bytes that end up in a saved
    clip, which are always cut from the untouched source.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    ffmpeg = find_ffmpeg()
    start = 0.0
    step = window_seconds

    while start < duration:
        length = min(window_seconds + overlap, duration - start + overlap)
        cmd = [
            ffmpeg,
            "-v", "error",
            "-nostdin",
            # -ss before -i seeks; -t must come *after* -i or it is measured on
            # the pre-seek timeline and yields an empty read for any start > 0.
            "-ss", f"{start:.6f}",
            "-i", str(path),
            "-t", f"{length:.6f}",
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            *(["-af", audio_filter] if audio_filter else []),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-",
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            raise AudioError(
                f"ffmpeg decode failed at {start:.2f}s of {path.name}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )

        raw = proc.stdout
        if not raw:
            # Nothing decoded; either EOF or a gap. Advance rather than spin.
            start += step
            continue

        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        yield start, samples
        start += step


def extract_clip(
    source: Path,
    dest: Path,
    start: float,
    end: float,
    *,
    allow_reencode: bool = True,
) -> dict:
    """Cut ``[start, end)`` out of ``source`` into ``dest``.

    Tries a stream copy first (no re-encode, original codec preserved). Cut
    points snap outward to the nearest codec frame boundary, which errs toward
    keeping audio rather than shaving it.

    Returns a dict describing what actually happened, for the sidecar.
    """
    if end <= start:
        raise ValueError(f"Clip end ({end}) must be after start ({start}).")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real extension last: ffmpeg picks its output format from the
    # filename suffix, so "clip.wav.partial" would leave it unable to guess.
    tmp = dest.with_name(f"{dest.stem}.partial{dest.suffix}")
    if tmp.exists():
        tmp.unlink()

    ffmpeg = find_ffmpeg()
    duration = end - start
    suffix = dest.suffix.lower()
    notes: list[str] = []

    def _attempt(copy: bool) -> subprocess.CompletedProcess:
        codec_args = ["-c:a", "copy"] if copy else ["-c:a", "pcm_s16le" if suffix == ".wav" else "libmp3lame" if suffix == ".mp3" else "aac"]
        cmd = [
            ffmpeg,
            "-v", "error",
            "-nostdin",
            "-y",
            # input-side seek: fast, and for audio-only streams every frame is a
            # keyframe so accuracy is bounded by the codec frame size (~25ms).
            # -t stays on the output side: as an input option it is measured
            # against the pre-seek timeline and silently produces nothing.
            "-ss", f"{start:.6f}",
            "-i", str(source),
            "-t", f"{duration:.6f}",
            "-map", "0:a:0",
            "-vn",
            *codec_args,
            "-avoid_negative_ts", "make_zero",
            str(tmp),
        ]
        return _run(cmd)

    can_copy = suffix in STREAM_COPY_SAFE
    if not can_copy:
        notes.append(f"container {suffix or '(none)'} not stream-copy safe")

    proc = _attempt(copy=can_copy) if can_copy else None
    mode = "copy"

    if proc is None or proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if proc is not None and proc.returncode != 0:
            notes.append("stream copy failed: " + proc.stderr.decode("utf-8", "replace").strip()[:200])
        if not allow_reencode:
            raise AudioError(f"Stream copy failed for {dest.name} and re-encode is disabled.")
        if tmp.exists():
            tmp.unlink()
        proc = _attempt(copy=False)
        mode = "reencode"
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            if tmp.exists():
                tmp.unlink()
            raise AudioError(
                f"ffmpeg could not extract {dest.name}: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )

    # Atomic-ish publish: the .partial name means an interrupted run never
    # leaves a half-written clip that a resume would mistake for finished work.
    tmp.replace(dest)

    actual = probe(dest)
    return {
        "mode": mode,
        "notes": notes,
        "bytes": actual.size_bytes,
        "actual_duration": actual.duration,
        "requested_duration": duration,
        "codec": actual.codec,
    }
