"""Voice activity detection.

Silero VAD is the engine. It runs locally, is robust to non-stationary
background noise, and is fast enough on CPU that a long recording is I/O-bound
rather than model-bound.

The file is processed in windows (see :func:`audua.audio.iter_windows`) so peak
memory does not scale with recording length. Regions are returned in absolute
source-file seconds; stitching across window seams is left to the greedy merge
in :mod:`audua.segments`, which already collapses zero-length gaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import AudioInfo, iter_windows
from .config import SAMPLE_RATE, Config

log = logging.getLogger(__name__)


class VadUnavailable(RuntimeError):
    """Raised when Silero/torch is not installed."""


class SpeechDetector(Protocol):
    """Minimal interface a detector must satisfy (lets tests inject a fake)."""

    def detect(self, samples: np.ndarray) -> list[tuple[float, float]]:
        """Return ``[(start_s, end_s), ...]`` relative to the given buffer."""
        ...


class SileroDetector:
    """Silero VAD wrapper. The model is loaded once and reused across windows."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None
        self._get_speech_timestamps = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise VadUnavailable(
                "Silero VAD is not installed. Run `pip install silero-vad` "
                "(it pulls in torch). See INSTALL-linux.md / INSTALL-windows.md."
            ) from exc

        log.info("Loading Silero VAD model...")
        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps

    def detect(self, samples: np.ndarray) -> list[tuple[float, float]]:
        self._ensure_loaded()
        import torch

        if samples.size < SAMPLE_RATE // 32:
            # Too short for the model to say anything meaningful.
            return []

        tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

        # Silero is stateful across calls; each window is an independent buffer.
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

        stamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.config.vad_threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=False,
        )
        return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def _vad_filter_chain(config: Config) -> str | None:
    """ffmpeg ``-af`` chain for VAD's decode only; saved clips never see it.

    Wind and handling noise pushes Silero's speech probability toward zero
    even when the underlying speech is loud and clean — a highpass strips the
    low-frequency rumble wind buffeting concentrates in, and ``afftdn`` knocks
    down what broadband hiss remains. Both are needed: highpass alone still
    leaves enough noise floor to mask speech in gusty recordings.
    """
    parts = []
    if config.vad_highpass_hz > 0:
        parts.append(f"highpass=f={config.vad_highpass_hz}")
    if config.vad_denoise:
        parts.append("afftdn=nf=-25")
    return ",".join(parts) if parts else None


def detect_speech_regions(
    info: AudioInfo,
    config: Config,
    detector: SpeechDetector | None = None,
) -> list[tuple[float, float]]:
    """Run VAD over an entire file, returning absolute-time speech regions.

    Regions are raw detections: no merging, no padding, no override handling.
    Those are separate, independently testable steps.
    """
    detector = detector or SileroDetector(config)
    regions: list[tuple[float, float]] = []
    audio_filter = _vad_filter_chain(config)

    windows = 0
    for offset, samples in iter_windows(
        info.path, info.duration, config.vad_window_seconds, audio_filter=audio_filter
    ):
        windows += 1
        for rel_start, rel_end in detector.detect(samples):
            start = offset + rel_start
            end = min(offset + rel_end, info.duration)
            if end > start:
                regions.append((start, end))
        log.debug("window @%.1fs -> %d regions so far", offset, len(regions))

    regions.sort()
    deduped = _dedupe_overlaps(regions)
    log.info(
        "VAD: %d raw regions across %d window(s), %.1f%% of %.1fs is speech",
        len(deduped),
        windows,
        100.0 * sum(e - s for s, e in deduped) / info.duration if info.duration else 0.0,
        info.duration,
    )
    return deduped


def _dedupe_overlaps(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse regions that overlap or touch.

    Window overlap means the same speech can be detected twice at a seam; this
    folds those duplicates together before the greedy merge sees them.
    """
    merged: list[tuple[float, float]] = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
