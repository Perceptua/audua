"""Configuration, time parsing, and external-tool discovery.

Everything OS-specific is confined to :func:`find_ffmpeg` / :func:`find_ffprobe`
so the rest of the pipeline stays platform-neutral.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Audio extensions we will pick up when handed a directory.
AUDIO_EXTENSIONS = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif", ".m4b", ".webm"}
)

# Containers that reliably accept a stream-copied audio elementary stream.
# Anything outside this set falls back to re-encode (and gets flagged).
STREAM_COPY_SAFE = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".m4b", ".aiff", ".aif"}
)

SAMPLE_RATE = 16_000  # Silero VAD and Whisper both want 16 kHz mono.

# Documents derived from a finished run and written beside its clips: the
# digest of every clip, and the prose summary built from that digest. Named
# here because verify_pairing has to know they are expected, not orphans.
DIGEST_NAME = "transcript_digest.md"
SUMMARY_NAME = "summary.md"


class ConfigError(RuntimeError):
    """Raised when configuration or the environment is unusable."""


# --------------------------------------------------------------------------
# external tools
# --------------------------------------------------------------------------

def _find_tool(name: str, env_var: str) -> str:
    """Locate an external binary, honouring an env-var override.

    Works identically on Windows and Linux: ``shutil.which`` appends the
    platform's executable extensions automatically.
    """
    override = os.environ.get(env_var)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(override)
        if found:
            return found
        raise ConfigError(f"{env_var} is set to {override!r} but no such executable was found.")

    found = shutil.which(name)
    if found:
        return found
    raise ConfigError(
        f"Could not find {name!r} on PATH. Install ffmpeg (see INSTALL-linux.md / "
        f"INSTALL-windows.md) or set the {env_var} environment variable to its full path."
    )


def find_ffmpeg() -> str:
    return _find_tool("ffmpeg", "AUDUA_FFMPEG")


def find_ffprobe() -> str:
    return _find_tool("ffprobe", "AUDUA_FFPROBE")


# --------------------------------------------------------------------------
# time parsing
# --------------------------------------------------------------------------

_CLOCK_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")
_PLAIN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)


def parse_time(value: Any) -> float:
    """Parse a timestamp into seconds.

    Accepts ``12.5`` (number), ``"12.5"``, ``"12.5s"``, ``"MM:SS"``,
    ``"MM:SS.mmm"``, ``"HH:MM:SS"`` and ``"HH:MM:SS.mmm"``.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"Cannot parse {value!r} as a timestamp.")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Timestamp may not be negative: {value!r}")
        return seconds
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse {value!r} as a timestamp.")

    match = _CLOCK_RE.match(value)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        secs = float(match.group(3))
        if minutes >= 60 or secs >= 60:
            raise ValueError(f"Minutes and seconds must each be < 60: {value!r}")
        return hours * 3600 + minutes * 60 + secs

    match = _PLAIN_RE.match(value)
    if match:
        return float(match.group(1))

    raise ValueError(
        f"Unrecognised timestamp {value!r}. Use seconds (12.5), MM:SS, or HH:MM:SS(.mmm)."
    )


def format_time(seconds: float) -> str:
    """Render seconds as ``HH:MM:SS.mmm`` for logs and manifests."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def stamp_for_filename(seconds: float) -> str:
    """Compact, filesystem-safe timestamp (``HHMMSSmmm``)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}{int(minutes):02d}{secs:06.3f}".replace(".", "")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class Config:
    """All tunables for a pipeline run.

    Defaults encode the project's stated bias: greedy clip formation and
    err-on-the-side-of-keeping-it. Nothing here ever deletes audio.
    """

    # --- input / output -----------------------------------------------
    output_root: Path = Path("output")
    overrides_path: Path | None = None  # explicit sidecar; default is <stem>.overrides.json

    # --- voice activity detection --------------------------------------
    vad_threshold: float = 0.35            # Silero speech probability. Lower = more permissive.
    min_speech_ms: int = 150               # Ignore blips shorter than this as speech onsets.
    min_silence_ms: int = 400              # Silence needed before Silero closes a region.
    speech_pad_ms: int = 150               # Silero's own internal padding around regions.
    vad_window_seconds: float = 600.0      # Decode/VAD window; bounds memory on huge files.
    vad_highpass_hz: float = 100.0         # High-pass before VAD only; cuts wind/handling rumble. 0 disables.
    vad_denoise: bool = True               # FFT noise reduction before VAD only. Saved clips are never touched.

    # --- clip formation -------------------------------------------------
    merge_gap: float = 10.0                # Silences <= this are absorbed into one clip.
    pad: float = 0.5                       # Extra lead-in/tail-out added to each VAD clip.
    min_clip_duration: float = 0.5         # Below this a clip is *flagged*, never dropped.
    max_clip_duration: float = 0.0         # 0 = unlimited. Otherwise split greedily at silence.
    sparse_ratio: float = 0.35             # speech/duration below this flags the clip "sparse".

    # --- overrides -------------------------------------------------------
    override_mode: str = "isolate"         # "isolate" | "merge"

    # --- transcription ----------------------------------------------------
    model: str = "large-v3"
    device: str = "auto"                   # "auto" | "cpu" | "cuda"
    compute_type: str = "auto"             # e.g. "int8", "float16", "int8_float16"
    language: str | None = None            # None = autodetect per clip
    beam_size: int = 5
    condition_on_previous_text: bool = False   # clips are independent; avoids drift
    low_logprob: float = -1.0              # mean avg_logprob below this -> "low_confidence"
    high_no_speech: float = 0.6            # mean no_speech_prob above this -> "high_no_speech"

    # --- run control -------------------------------------------------------
    force: bool = False                    # ignore cached work and redo everything
    segment_only: bool = False             # stop after clip extraction
    dry_run: bool = False                  # plan clips, write nothing but a plan file
    verbose: bool = False

    # populated at runtime, not user-facing
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.merge_gap < 0:
            raise ConfigError("--merge-gap must be >= 0.")
        if self.pad < 0:
            raise ConfigError("--pad must be >= 0.")
        if not 0.0 < self.vad_threshold < 1.0:
            raise ConfigError("--vad-threshold must be strictly between 0 and 1.")
        if self.vad_window_seconds < 30:
            raise ConfigError("--vad-window must be at least 30 seconds.")
        if self.vad_highpass_hz < 0:
            raise ConfigError("--vad-highpass must be >= 0.")
        if self.override_mode not in {"isolate", "merge"}:
            raise ConfigError("--override-mode must be 'isolate' or 'merge'.")
        if self.max_clip_duration and self.max_clip_duration <= self.merge_gap:
            raise ConfigError("--max-clip-duration must exceed --merge-gap.")

    # ------------------------------------------------------------------
    def segmentation_fingerprint(self) -> str:
        """Hash of every knob that affects *where clips are cut*.

        Used to decide whether cached segmentation can be reused on re-run.
        Transcription-only settings are deliberately excluded so that changing
        the Whisper model does not force a re-cut of the audio.
        """
        relevant = {
            "vad_threshold": self.vad_threshold,
            "min_speech_ms": self.min_speech_ms,
            "min_silence_ms": self.min_silence_ms,
            "speech_pad_ms": self.speech_pad_ms,
            "vad_window_seconds": self.vad_window_seconds,
            "vad_highpass_hz": self.vad_highpass_hz,
            "vad_denoise": self.vad_denoise,
            "merge_gap": self.merge_gap,
            "pad": self.pad,
            "max_clip_duration": self.max_clip_duration,
            "override_mode": self.override_mode,
        }
        blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def transcription_fingerprint(self) -> str:
        relevant = {
            "model": self.model,
            "language": self.language,
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["overrides_path"] = str(self.overrides_path) if self.overrides_path else None
        data.pop("extras", None)
        return data
