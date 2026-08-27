"""Orchestration: source file -> per-source folder of clips and transcripts.

Design notes that matter more than the code:

* **Resumable.** Every expensive step caches its result and is skipped on a
  re-run when its inputs are unchanged. Interrupt a four-hour file halfway and
  the next run picks up at the clip it was on.
* **Idempotent.** Running twice on the same input produces the same tree. It
  does not append, duplicate, or renumber.
* **Never destructive.** When settings change and old outputs no longer match,
  they are moved into a ``_superseded_*`` folder, not deleted.
* **Checkable without re-running.** ``manifest.json`` records what happened to
  every clip, so progress and failures can be inspected from disk alone.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from .audio import AudioError, AudioInfo, extract_clip, probe
from .config import AUDIO_EXTENSIONS, DIGEST_NAME, SUMMARY_NAME, Config, format_time
from .segments import (
    Clip,
    default_override_path,
    build_clips,
    load_overrides,
    summarise,
)
from .transcribe import Transcriber
from .vad import SpeechDetector, detect_speech_regions

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
VAD_CACHE_NAME = "vad_cache.json"


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _vad_params(config: Config) -> dict:
    return {
        "engine": "silero",
        "threshold": config.vad_threshold,
        "min_speech_ms": config.min_speech_ms,
        "min_silence_ms": config.min_silence_ms,
        "speech_pad_ms": config.speech_pad_ms,
        "window_seconds": config.vad_window_seconds,
        "highpass_hz": config.vad_highpass_hz,
        "denoise": config.vad_denoise,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON via a temp file so an interrupted write cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Ignoring unreadable %s: %s", path.name, exc)
        return None


def discover_sources(target: Path, recursive: bool = True) -> list[Path]:
    """Expand a file or directory into the list of audio files to process."""
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise PipelineError(f"No such file or directory: {target}")

    pattern = "**/*" if recursive else "*"
    found = sorted(
        p for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not found:
        raise PipelineError(f"No audio files found under {target}")
    return found


def _supersede(out_dir: Path, reason: str) -> None:
    """Move stale outputs aside instead of deleting them."""
    existing = [
        p for p in out_dir.iterdir()
        if p.is_file() and p.name not in {MANIFEST_NAME, VAD_CACHE_NAME}
    ]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    attic = out_dir / f"_superseded_{stamp}"
    attic.mkdir(parents=True, exist_ok=True)
    for item in existing:
        item.replace(attic / item.name)
    (attic / "REASON.txt").write_text(reason + "\n", encoding="utf-8")
    log.warning("Moved %d stale file(s) into %s (%s)", len(existing), attic.name, reason)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _speech_regions(
    info: AudioInfo,
    out_dir: Path,
    config: Config,
    detector: SpeechDetector | None,
) -> list[tuple[float, float]]:
    """Load cached VAD output, or run the detector and cache it.

    The cache is keyed on the source fingerprint and the VAD parameters only —
    changing the merge gap or the Whisper model does not force a re-listen.
    """
    cache_path = out_dir / VAD_CACHE_NAME
    cached = None if config.force else _read_json(cache_path)

    if (
        cached
        and cached.get("source_fingerprint") == info.fingerprint()
        and cached.get("params") == _vad_params(config)
    ):
        regions = [(float(s), float(e)) for s, e in cached.get("regions", [])]
        log.info("Reusing cached VAD result (%d regions).", len(regions))
        return regions

    started = time.monotonic()
    regions = detect_speech_regions(info, config, detector=detector)
    _write_json(
        cache_path,
        {
            "schema": "audua/vad_cache/1",
            "source": info.path.name,
            "source_fingerprint": info.fingerprint(),
            "params": _vad_params(config),
            "created": _now(),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "regions": [[round(s, 3), round(e, 3)] for s, e in regions],
        },
    )
    return regions


def _clip_is_done(
    clip: Clip,
    audio_path: Path,
    text_path: Path,
    sidecar_path: Path,
    *,
    transcribing: bool,
) -> bool:
    """Has this exact clip already been produced?

    The sidecar is the completion marker and it records the clip's bounds, so a
    boundary change is detected even if the filenames happen to line up.
    """
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        return False
    if transcribing and not text_path.is_file():
        return False

    sidecar = _read_json(sidecar_path)
    if not sidecar:
        return False

    recorded = sidecar.get("clip") or {}
    if abs(float(recorded.get("start", -1)) - clip.start) > 1e-3:
        return False
    if abs(float(recorded.get("end", -1)) - clip.end) > 1e-3:
        return False
    if transcribing and "transcript" not in sidecar:
        return False
    return True


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def process_file(
    source: Path,
    config: Config,
    *,
    transcriber: Transcriber | None = None,
    detector: SpeechDetector | None = None,
) -> dict:
    """Run the full pipeline for one audio file. Returns the manifest dict."""
    source = Path(source)
    info = probe(source)
    out_dir = Path(config.output_root) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== %s (%s) ===", source.name, format_time(info.duration))

    regions = _speech_regions(info, out_dir, config, detector)

    override_path = config.overrides_path or default_override_path(source)
    overrides = load_overrides(override_path, info.duration)
    if overrides:
        log.info("Loaded %d override window(s) from %s", len(overrides), Path(override_path).name)

    clips = build_clips(regions, overrides, config, info.duration)
    stats = summarise(clips, info.duration)
    log.info(
        "%d clip(s), %s of %s retained (%.1f%%), %d flagged",
        stats["clip_count"],
        format_time(stats["clip_seconds"]),
        format_time(info.duration),
        100 * stats["retained_fraction"],
        stats["flagged_clips"],
    )

    seg_fp = config.segmentation_fingerprint()
    manifest_path = out_dir / MANIFEST_NAME
    previous = _read_json(manifest_path)

    if config.dry_run:
        plan = _base_manifest(source, info, config, seg_fp, stats)
        plan["dry_run"] = True
        plan["clips"] = [c.to_dict() for c in clips]
        _write_json(out_dir / "plan.json", plan)
        log.info("Dry run: wrote %s", (out_dir / "plan.json"))
        return plan

    # Settings that move clip boundaries invalidate every existing output.
    if previous and not config.force:
        changed = (
            previous.get("segmentation_fingerprint") != seg_fp
            or previous.get("source_fingerprint") != info.fingerprint()
        )
        if changed:
            _supersede(out_dir, "segmentation settings or source file changed")
    elif config.force and previous:
        _supersede(out_dir, "--force was requested")

    transcribing = not config.segment_only
    if transcribing and transcriber is None:
        transcriber = Transcriber(config)

    manifest = _base_manifest(source, info, config, seg_fp, stats)
    manifest["clips"] = []
    records: list[dict] = []
    reused = 0
    failed = 0

    for clip in clips:
        audio_path = out_dir / f"{clip.stem}{source.suffix}"
        text_path = out_dir / f"{clip.stem}.txt"
        sidecar_path = out_dir / f"{clip.stem}.json"

        if not config.force and _clip_is_done(
            clip, audio_path, text_path, sidecar_path, transcribing=transcribing
        ):
            reused += 1
            existing = _read_json(sidecar_path) or {}
            records.append(
                {
                    **clip.to_dict(),
                    "status": "reused",
                    "flags": existing.get("flags", clip.flags),
                    "audio_file": audio_path.name,
                    "text_file": text_path.name if transcribing else None,
                    "sidecar_file": sidecar_path.name,
                }
            )
            continue

        record = {**clip.to_dict(), "audio_file": audio_path.name,
                  "sidecar_file": sidecar_path.name}
        try:
            extraction = extract_clip(source, audio_path, clip.start, clip.end)

            if transcribing:
                from .transcribe import transcribe_clip

                sidecar = transcribe_clip(
                    transcriber, clip, audio_path, text_path, sidecar_path,
                    config, extraction=extraction,
                )
                record["flags"] = sidecar["flags"]
                record["text_file"] = text_path.name
                record["chars"] = len(sidecar["transcript"]["text"])
            else:
                _write_json(
                    sidecar_path,
                    {
                        "schema": "audua/clip/1",
                        "clip": clip.to_dict(),
                        "audio_file": audio_path.name,
                        "extraction": extraction,
                        "flags": clip.flags,
                    },
                )
                record["text_file"] = None

            record["status"] = "ok"
            record["extraction_mode"] = extraction["mode"]
            log.info(
                "  %s  %s–%s  (%.1fs)%s",
                clip.stem, format_time(clip.start), format_time(clip.end),
                clip.duration,
                f"  [{', '.join(record.get('flags') or [])}]" if record.get("flags") else "",
            )
        except (AudioError, OSError, RuntimeError) as exc:
            failed += 1
            record["status"] = "failed"
            record["error"] = str(exc)
            log.error("  %s FAILED: %s", clip.stem, exc)

        records.append(record)
        manifest["clips"] = records
        _write_json(manifest_path, manifest)   # incremental: survives a crash

    manifest["clips"] = records
    manifest["completed"] = _now()
    manifest["counts"] = {
        "total": len(records),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "reused": reused,
        "failed": failed,
    }
    manifest["pairing"] = verify_pairing(out_dir, records, transcribing=transcribing)
    _write_json(manifest_path, manifest)

    log.info(
        "Done: %d ok, %d reused, %d failed -> %s",
        manifest["counts"]["ok"], reused, failed, out_dir,
    )
    return manifest


def _base_manifest(
    source: Path, info: AudioInfo, config: Config, seg_fp: str, stats: dict
) -> dict:
    return {
        "schema": "audua/manifest/1",
        "source": {
            "name": source.name,
            "path": str(source.resolve()),
            "duration": round(info.duration, 3),
            "duration_hms": format_time(info.duration),
            "codec": info.codec,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
        },
        "source_fingerprint": info.fingerprint(),
        "segmentation_fingerprint": seg_fp,
        "transcription_fingerprint": config.transcription_fingerprint(),
        "started": _now(),
        "config": config.to_dict(),
        "stats": stats,
    }


def verify_pairing(out_dir: Path, records: list[dict], *, transcribing: bool) -> dict:
    """Assert the 1:1 clip/transcript invariant and report any violation.

    Checks both directions: every successful clip has its pair on disk, and
    every file on disk belongs to a clip in the manifest.
    """
    problems: list[str] = []
    expected_names: set[str] = set()

    for record in records:
        if record.get("status") == "failed":
            continue
        audio = out_dir / record["audio_file"]
        sidecar = out_dir / record["sidecar_file"]
        expected_names.update({audio.name, sidecar.name})

        if not audio.is_file():
            problems.append(f"missing audio: {audio.name}")
        if not sidecar.is_file():
            problems.append(f"missing sidecar: {sidecar.name}")

        if transcribing:
            text_name = record.get("text_file")
            if not text_name:
                problems.append(f"{record['stem']}: no transcript recorded")
                continue
            expected_names.add(text_name)
            if not (out_dir / text_name).is_file():
                problems.append(f"missing transcript: {text_name}")

    known = expected_names | {MANIFEST_NAME, VAD_CACHE_NAME, "plan.json",
                             DIGEST_NAME, SUMMARY_NAME}
    for path in out_dir.iterdir():
        if path.is_dir() or path.name in known or path.name.startswith("_"):
            continue
        if path.name.endswith(".tmp") or ".partial" in path.name:
            problems.append(f"leftover temp file: {path.name}")
        else:
            problems.append(f"orphan file not in manifest: {path.name}")

    return {"ok": not problems, "problems": problems}
