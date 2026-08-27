"""Inbox batch runner: watch a folder, process what is in it, file the results.

:mod:`audua.pipeline` is the engine — one audio file in, a folder of clips and
transcripts out. This module is the conveyor belt around it, and its decisions
are deliberate:

* **The inbox scan is flat.** ``discover_sources`` recurses on purpose, which is
  right for ``audua run ./recordings`` and wrong for an inbox: recursing would
  drag everything already sitting in ``processed/`` and ``failed/`` back through
  the pipeline on every run.
* **Sources are filed, never deleted.** A source leaves ``raw/`` for
  ``raw/processed/`` or ``raw/failed/`` and nothing else ever happens to it. A
  name collision gets a numeric suffix instead of an overwrite, and an override
  sidecar travels with its audio so a re-run from the archive still finds it.
* **A run that half-worked is a failure.** Failed clips, a broken 1:1 pairing,
  or no clips at all send the source to ``failed/``. Quality *flags* do not:
  the pipeline flags rather than filters precisely because the caller decides
  what counts as signal, and a sparse or low-confidence clip is still a result.
* **The digest is the summary's input.** ``transcript_digest.md`` collects every
  clip — index, bounds, flags, timestamped text — into one file, so writing the
  prose summary is a single read rather than forty, and the digest survives as a
  grep-able index of the recording.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from .audio import AudioError
from .config import AUDIO_EXTENSIONS, DIGEST_NAME, SUMMARY_NAME, Config, format_time
from .pipeline import PipelineError, _now, _read_json, _write_json, process_file
from .segments import OverrideError, default_override_path
from .transcribe import Transcriber, TranscriberUnavailable
from .vad import SpeechDetector, VadUnavailable

log = logging.getLogger(__name__)

PROCESSED_DIRNAME = "processed"
FAILED_DIRNAME = "failed"
BATCHES_DIRNAME = "_batches"
LATEST_REPORT_NAME = "latest.json"

# Problems worth naming individually before a report becomes a wall of text.
_MAX_LISTED_PROBLEMS = 5


class BatchError(RuntimeError):
    """Raised when the inbox itself is unusable."""


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def discover_inbox(raw_dir: Path, *, min_age: float = 0.0) -> tuple[list[Path], list[dict]]:
    """Audio sitting *directly* in the inbox, plus whatever was skipped and why.

    Flat by design — see the module docstring. ``min_age`` guards against
    grabbing a file that is still being copied in: anything modified more
    recently than that is left for the next run.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise BatchError(f"No such inbox directory: {raw_dir}")

    sources: list[Path] = []
    skipped: list[dict] = []
    now = time.time()

    for path in sorted(raw_dir.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            skipped.append({"source": path.name, "reason": "not a recognised audio extension"})
            continue
        if min_age > 0:
            age = now - path.stat().st_mtime
            if age < min_age:
                skipped.append({
                    "source": path.name,
                    "reason": f"modified {age:.0f}s ago, under --min-age {min_age:.0f}s; "
                              f"it may still be copying",
                })
                continue
        sources.append(path)

    return sources, skipped


# --------------------------------------------------------------------------
# filing
# --------------------------------------------------------------------------

def _unique_destination(dest_dir: Path, name: str) -> Path:
    """``foo.wav`` -> ``foo-2.wav`` when the archive already holds a ``foo.wav``."""
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 2
    while (candidate := dest_dir / f"{stem}-{counter}{suffix}").exists():
        counter += 1
    return candidate


def file_source(source: Path, dest_dir: Path) -> list[Path]:
    """Move the audio, and any override sidecar beside it, into ``dest_dir``.

    Returns the new paths, audio first. ``shutil.move`` rather than
    ``Path.replace`` so that an inbox on a different volume from its archive
    still works on Windows.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dest = _unique_destination(dest_dir, source.name)
    shutil.move(str(source), str(audio_dest))
    moved = [audio_dest]

    sidecar = default_override_path(source)
    if sidecar.is_file():
        # Named off the *destination* stem, so a renamed collision keeps its pair.
        sidecar_dest = audio_dest.with_name(f"{audio_dest.stem}.overrides.json")
        shutil.move(str(sidecar), str(sidecar_dest))
        moved.append(sidecar_dest)

    return moved


# --------------------------------------------------------------------------
# outcome
# --------------------------------------------------------------------------

def classify_outcome(manifest: dict) -> list[str]:
    """Why this run should be filed as failed. An empty list means it succeeded.

    Quality flags (``sparse``, ``low_confidence``, ``repetitive``, ...) are
    deliberately absent: they mark clips the pipeline was unsure about, not a
    run that went wrong, and filtering on them is a downstream decision made
    against the full record.
    """
    counts = manifest.get("counts")
    if not counts:
        # process_file writes counts last; without it, the run never finished.
        return ["the run did not complete (no counts block in the manifest)"]

    reasons: list[str] = []

    if not (manifest.get("stats") or {}).get("clip_count"):
        reasons.append("no clips were produced — the VAD heard no speech anywhere")

    if counts.get("failed"):
        reasons.append(
            f"{counts['failed']} of {counts.get('total', '?')} clip(s) failed to "
            f"extract or transcribe"
        )

    pairing = manifest.get("pairing") or {}
    if not pairing.get("ok", True):
        problems = pairing.get("problems") or []
        listed = "; ".join(problems[:_MAX_LISTED_PROBLEMS])
        if len(problems) > _MAX_LISTED_PROBLEMS:
            listed += f"; (+{len(problems) - _MAX_LISTED_PROBLEMS} more)"
        reasons.append(f"clip/transcript pairing check failed: {listed}")

    return reasons


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------

def _trim_ms(hms: str) -> str:
    """``00:03:32.336`` -> ``00:03:32``. The digest is read by people."""
    return hms.split(".")[0]


def _short_hms(seconds: float) -> str:
    return _trim_ms(format_time(seconds))


def _clip_text_lines(out_dir: Path, record: dict) -> list[str]:
    """Transcript text for one clip, timestamped against the *original* recording.

    Prefers the sidecar's segments, because those carry ``global_start`` and so
    point at a moment in the source rather than an offset inside a clip. Falls
    back to the flat ``.txt`` when segments are unavailable.
    """
    sidecar = _read_json(out_dir / record["sidecar_file"]) if record.get("sidecar_file") else None
    segments = ((sidecar or {}).get("transcript") or {}).get("segments") or []

    lines = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        stamp = segment.get("global_start_hms") or format_time(segment.get("global_start", 0.0))
        lines.append(f"- **[{_trim_ms(stamp)}]** {text}")
    if lines:
        return lines

    text_name = record.get("text_file")
    if text_name and (out_dir / text_name).is_file():
        flat = (out_dir / text_name).read_text(encoding="utf-8").strip()
        if flat:
            return [f"- **[{_short_hms(record.get('start', 0.0))}]** {flat}"]

    return ["_(no speech transcribed — the clip and an empty transcript are both kept)_"]


def build_digest(out_dir: Path, manifest: dict) -> str:
    """Render every clip in a finished run as one markdown document."""
    source = manifest.get("source") or {}
    stats = manifest.get("stats") or {}
    config = manifest.get("config") or {}
    records = manifest.get("clips") or []

    name = Path(source.get("name") or out_dir.name).stem
    flag_counts = stats.get("flag_counts") or {}
    flags_seen = ", ".join(f"{flag} ({n})" for flag, n in sorted(flag_counts.items())) or "none"
    retained = 100 * stats.get("retained_fraction", 0.0)

    out = [
        f"# {name} — transcript digest",
        "",
        f"- **Source audio:** `{source.get('name', '?')}` — {source.get('duration_hms', '?')}, "
        f"{source.get('codec', '?')}, {source.get('sample_rate', '?')} Hz, "
        f"{source.get('channels', '?')} ch",
        f"- **Clips:** {stats.get('clip_count', 0)} · "
        f"{_short_hms(stats.get('clip_seconds', 0.0))} of speech retained "
        f"({retained:.1f}% of the recording) · {stats.get('flagged_clips', 0)} flagged",
        f"- **Flags seen:** {flags_seen}",
        f"- **Transcribed with:** faster-whisper `{config.get('model', '?')}`, "
        f"language {config.get('language') or 'autodetected per clip'}",
        f"- **Generated:** {_now()} by `audua batch`",
        "",
        f"Every clip below is a citable source. In `{SUMMARY_NAME}`, cite clip N as `[^N]`; "
        "the footnote carries the timestamp, the transcript, and the audio.",
        "",
    ]

    for record in records:
        out += [
            f"## [{record.get('index', '?')}] {record.get('stem', '?')} — "
            f"{_short_hms(record.get('start', 0.0))} → {_short_hms(record.get('end', 0.0))} "
            f"({record.get('duration', 0.0):.1f}s)",
            "",
        ]

        if record.get("status") == "failed":
            out += [f"_(failed: {record.get('error', 'unknown error')})_", ""]
            continue

        files = " · ".join(
            f"`{record[key]}`"
            for key in ("audio_file", "text_file", "sidecar_file")
            if record.get(key)
        )
        meta = [
            files,
            f"origin {record.get('origin', '?')}",
            f"speech {100 * record.get('speech_ratio', 0.0):.0f}%",
            f"flags: {', '.join(record.get('flags') or []) or 'none'}",
        ]
        if record.get("labels"):
            meta.append(f"labels: {', '.join(record['labels'])}")

        out += [" · ".join(meta), ""]
        out += _clip_text_lines(out_dir, record)
        out += [""]

    return "\n".join(out).rstrip() + "\n"


def write_digest(out_dir: Path, manifest: dict) -> Path:
    """Write ``transcript_digest.md`` into a finished run's output folder."""
    path = out_dir / DIGEST_NAME
    path.write_text(build_digest(out_dir, manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# summary citations
# --------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[\^(\d+)\](?!:)")
_DEFINITION_RE = re.compile(r"^\[\^(\d+)\]:(.*)$", re.MULTILINE)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def check_summary(out_dir: Path, manifest: dict) -> dict:
    """Check that every citation in ``summary.md`` resolves to a real clip.

    A citation is a promise that a specific clip says a specific thing. The
    numbers are written by hand, so they are checked by machine: each ``[^N]``
    must have a definition, N must be a clip index that exists in this run, and
    every file the definition links to must be on disk. Clips nobody cited are
    reported too — not a problem, but the point of the summary is findability,
    and an uncited clip is one you cannot find through it.
    """
    path = out_dir / SUMMARY_NAME
    if not path.is_file():
        return {"ok": True, "present": False, "problems": [], "cited": [], "uncited": []}

    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    used = {int(n) for n in _CITATION_RE.findall(text)}
    defined = {int(match.group(1)): match.group(2) for match in _DEFINITION_RE.finditer(text)}
    indexes = {record.get("index") for record in (manifest.get("clips") or [])}

    for number in sorted(used - set(defined)):
        problems.append(f"[^{number}] is cited but never defined")
    for number in sorted(set(defined) - used):
        problems.append(f"[^{number}] is defined but never cited")
    for number in sorted((used | set(defined)) - indexes):
        problems.append(f"[^{number}] does not match any clip in this run")

    for number, body in sorted(defined.items()):
        for target in _LINK_RE.findall(body):
            target = target.split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (out_dir / target).is_file():
                problems.append(f"[^{number}] links to {target}, which is not in this folder")

    return {
        "ok": not problems,
        "present": True,
        "problems": problems,
        "cited": sorted(used),
        "uncited": sorted(index for index in indexes if index not in used),
    }


# --------------------------------------------------------------------------
# the batch
# --------------------------------------------------------------------------

def _process_one(
    source: Path,
    config: Config,
    *,
    processed_dir: Path,
    failed_dir: Path,
    move: bool,
    transcriber: Transcriber | None,
    detector: SpeechDetector | None,
) -> dict:
    """Run one source through the pipeline, judge the result, and file it."""
    started = time.monotonic()
    out_dir = Path(config.output_root) / source.stem
    result: dict = {
        "source": source.name,
        "stem": source.stem,
        "source_path": str(source.resolve()),
        "output_dir": str(out_dir),
        "status": "processed",
        "reasons": [],
        "digest": None,
        "summary": str(out_dir / SUMMARY_NAME),
        "summary_needed": False,
        "moved_to": None,
    }

    manifest: dict | None = None
    try:
        manifest = process_file(source, config, transcriber=transcriber, detector=detector)
    except (VadUnavailable, TranscriberUnavailable):
        # A missing dependency is not this recording's fault, and every
        # remaining file would fail identically. Let the caller stop the batch.
        raise
    except (AudioError, OverrideError, PipelineError, OSError) as exc:
        result["reasons"].append(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # one bad file must not take the batch down
        log.exception("%s: unexpected failure", source.name)
        result["reasons"].append(f"unexpected {type(exc).__name__}: {exc}")
    else:
        result["reasons"] = classify_outcome(manifest)
        result["duration_hms"] = (manifest.get("source") or {}).get("duration_hms")
        result["counts"] = manifest.get("counts")
        stats = manifest.get("stats") or {}
        result["clip_count"] = stats.get("clip_count", 0)
        result["flagged_clips"] = stats.get("flagged_clips", 0)
        result["flag_counts"] = stats.get("flag_counts") or {}

    if manifest and manifest.get("clips"):
        # Written even for a failed run — the digest is how you find out what
        # went wrong without re-running an hour of audio.
        try:
            result["digest"] = str(write_digest(out_dir, manifest))
        except OSError as exc:
            result["reasons"].append(f"could not write {DIGEST_NAME}: {exc}")

    result["status"] = "failed" if result["reasons"] else "processed"

    if move:
        destination = failed_dir if result["status"] == "failed" else processed_dir
        try:
            result["moved_to"] = str(file_source(source, destination)[0])
        except OSError as exc:
            # Filing failed, the processing did not. Say so, and leave the
            # source where it is rather than losing track of it.
            result["reasons"].append(f"could not move source into {destination}: {exc}")
            result["status"] = "failed"

    # Decided last: filing can still turn a finished run into a failed one.
    result["summary_needed"] = (
        result["status"] == "processed" and not (out_dir / SUMMARY_NAME).is_file()
    )
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    log.info(
        "  -> %s%s",
        result["status"],
        f" ({'; '.join(result['reasons'])})" if result["reasons"] else "",
    )
    return result


def run_inbox(
    config: Config,
    *,
    raw_dir: Path,
    processed_dir: Path | None = None,
    failed_dir: Path | None = None,
    move: bool = True,
    min_age: float = 0.0,
    report_path: Path | None = None,
    transcriber: Transcriber | None = None,
    detector: SpeechDetector | None = None,
) -> dict:
    """Process everything in the inbox. Returns (and writes) the batch report."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir) if processed_dir else raw_dir / PROCESSED_DIRNAME
    failed_dir = Path(failed_dir) if failed_dir else raw_dir / FAILED_DIRNAME
    output_root = Path(config.output_root)

    sources, skipped = discover_inbox(raw_dir, min_age=min_age)
    started = time.monotonic()

    report: dict = {
        "schema": "audua/batch/1",
        "started": _now(),
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "processed_dir": str(processed_dir),
        "failed_dir": str(failed_dir),
        "moved": move,
        "config": config.to_dict(),
        "results": [],
        "skipped": skipped,
        "aborted": None,
    }

    log.info("Inbox %s: %d file(s) to process, %d skipped.", raw_dir, len(sources), len(skipped))

    # One Transcriber for the whole batch. The model loads lazily on the first
    # clip and is then reused, instead of being reloaded once per source.
    if transcriber is None and not config.segment_only and not config.dry_run:
        transcriber = Transcriber(config)

    for position, source in enumerate(sources, start=1):
        log.info("[%d/%d] %s", position, len(sources), source.name)
        try:
            report["results"].append(_process_one(
                source, config,
                processed_dir=processed_dir, failed_dir=failed_dir, move=move,
                transcriber=transcriber, detector=detector,
            ))
        except (VadUnavailable, TranscriberUnavailable) as exc:
            report["aborted"] = str(exc)
            log.error("Stopping the batch: %s", exc)
            break

    results = report["results"]
    report["completed"] = _now()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["counts"] = {
        "total": len(sources),
        "processed": sum(1 for r in results if r["status"] == "processed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": len(skipped),
        "not_reached": len(sources) - len(results),
        "summaries_pending": sum(1 for r in results if r.get("summary_needed")),
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    batches_dir = output_root / BATCHES_DIRNAME
    path = Path(report_path) if report_path else batches_dir / f"batch_{stamp}.json"
    report["report_path"] = str(path)
    _write_json(path, report)
    # A stable name, so the next step never has to guess which run was the last.
    _write_json(batches_dir / LATEST_REPORT_NAME, report)

    return report
