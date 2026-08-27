"""Transcription stage: one clip in, one .txt and one .json out.

The 1:1 mapping is enforced structurally — :func:`transcribe_clip` always
writes both files, including when the model returns nothing. An empty
transcript is a real result (it tells you the VAD produced a false positive at
that timestamp) and is kept rather than deleted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config, format_time
from .segments import Clip

log = logging.getLogger(__name__)


class TranscriberUnavailable(RuntimeError):
    """Raised when faster-whisper is not installed."""


class Transcriber:
    """Lazy wrapper around a single faster-whisper model instance."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._model = None

    def _resolve_runtime(self) -> tuple[str, str]:
        """Pick device and compute type, honouring explicit user settings."""
        device = self.config.device
        compute_type = self.config.compute_type

        if device == "auto":
            device = "cpu"
            try:
                import torch  # optional; faster-whisper does not require it

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:  # pragma: no cover - torch absent or broken
                pass

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        return device, compute_type

    @property
    def model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise TranscriberUnavailable(
                    "faster-whisper is not installed. Run `pip install faster-whisper`. "
                    "See INSTALL-linux.md / INSTALL-windows.md."
                ) from exc

            device, compute_type = self._resolve_runtime()
            log.info(
                "Loading Whisper model %r on %s (%s)...",
                self.config.model, device, compute_type,
            )
            self._model = WhisperModel(
                self.config.model, device=device, compute_type=compute_type
            )
            self.config.extras["device"] = device
            self.config.extras["compute_type"] = compute_type
        return self._model

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> dict:
        """Run the model on one clip and return a plain-dict result."""
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            condition_on_previous_text=self.config.condition_on_previous_text,
            # VAD already happened upstream; running it again here would risk
            # discarding audio we deliberately chose to keep.
            vad_filter=False,
            word_timestamps=False,
        )

        segments = []
        for segment in segments_iter:
            segments.append(
                {
                    "id": segment.id,
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": segment.text.strip(),
                    "avg_logprob": round(float(segment.avg_logprob), 4),
                    "no_speech_prob": round(float(segment.no_speech_prob), 4),
                    "compression_ratio": round(float(segment.compression_ratio), 4),
                }
            )

        return {
            "segments": segments,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "duration": round(float(info.duration), 3),
        }


def confidence_flags(result: dict, config: Config) -> list[str]:
    """Derive quality flags from a transcription result.

    These are advisory only. Consistent with the rest of the pipeline, a bad
    score never causes a file to be removed.
    """
    flags: list[str] = []
    segments = result.get("segments") or []

    text = " ".join(s["text"] for s in segments).strip()
    if not text:
        flags.append("empty_transcript")
        return flags

    mean_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
    mean_no_speech = sum(s["no_speech_prob"] for s in segments) / len(segments)

    if mean_logprob < config.low_logprob:
        flags.append("low_confidence")
    if mean_no_speech > config.high_no_speech:
        flags.append("high_no_speech")
    if any(s["compression_ratio"] > 2.4 for s in segments):
        # Classic Whisper failure mode: the same phrase repeated forever.
        flags.append("repetitive")

    return flags


def transcribe_clip(
    transcriber: Transcriber,
    clip: Clip,
    audio_path: Path,
    text_path: Path,
    sidecar_path: Path,
    config: Config,
    extraction: dict | None = None,
) -> dict:
    """Transcribe one clip and write its paired .txt and .json.

    Both files are always written, so ``clip_0007.wav`` never exists without
    ``clip_0007.txt`` and ``clip_0007.json`` beside it.
    """
    result = transcriber.transcribe(audio_path)
    text = "\n".join(s["text"] for s in result["segments"] if s["text"]).strip()

    flags = list(dict.fromkeys(clip.flags + confidence_flags(result, config)))

    sidecar = {
        "schema": "audua/clip/1",
        "clip": {**clip.to_dict(), "flags": flags},
        "audio_file": audio_path.name,
        "text_file": text_path.name,
        "transcript": {
            "text": text,
            "language": result["language"],
            "language_probability": result["language_probability"],
            "segments": [
                {
                    **segment,
                    # global_* place the segment on the original recording's
                    # timeline, so downstream workflows never need the manifest
                    # to locate a quote in the source audio.
                    "global_start": round(clip.start + segment["start"], 3),
                    "global_end": round(clip.start + segment["end"], 3),
                    "global_start_hms": format_time(clip.start + segment["start"]),
                }
                for segment in result["segments"]
            ],
        },
        "extraction": extraction or {},
        "model": {
            "name": config.model,
            "beam_size": config.beam_size,
            "device": config.extras.get("device"),
            "compute_type": config.extras.get("compute_type"),
        },
        "flags": flags,
    }

    # Write text first, then the sidecar. The pipeline treats the sidecar as the
    # completion marker, so a crash between the two is detected as unfinished
    # and simply redone on the next run.
    text_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return sidecar
