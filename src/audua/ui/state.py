"""The read-only view model: an audua filetree, read as it stands on disk.

Everything here is a pure function of the tree. Nothing writes, nothing
launches a pipeline, and nothing caches -- a request reads the files, so the
answer is never staler than the last save. That is cheap because the pipeline
already keeps its state in files designed to be read this way:
``manifest.json`` is rewritten after every clip, and the batch report says what
happened to each source.

Judgements are borrowed rather than reimplemented. Whether a run counts as
failed is :func:`audua.batch.classify_outcome`, whether its citations resolve is
:func:`audua.batch.check_summary`, and whether its files pair up is
:func:`audua.pipeline.verify_pairing` -- so the UI cannot drift from what
``audua verify`` would tell you at the terminal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..batch import (
    BATCHES_DIRNAME,
    FAILED_DIRNAME,
    PROCESSED_DIRNAME,
    check_summary,
    classify_outcome,
)
from ..config import AUDIO_EXTENSIONS, DIGEST_NAME, SUMMARY_NAME
from ..pipeline import MANIFEST_NAME, verify_pairing
from ..segments import default_override_path
from . import markdown

log = logging.getLogger(__name__)

DEFAULT_RAW = Path("processing/raw")
DEFAULT_OUTPUT = Path("processing/output")

# Documents the reading pane will open. Clip audio is served separately.
READABLE_SUFFIXES = frozenset({".md", ".txt", ".json"})

# One line, in a list, on a screen. Long enough for a real sentence.
_SUMMARY_LINE_MAX = 220
_CLIP_PREVIEW_MAX = 260


class StateError(LookupError):
    """Raised when a requested run or file is not in the tree."""


@dataclass(frozen=True)
class Roots:
    """Where the inbox and the outputs live. Mirrors ``audua batch``."""

    raw: Path = DEFAULT_RAW
    output: Path = DEFAULT_OUTPUT

    @classmethod
    def resolved(cls, raw: Path | str, output: Path | str) -> Roots:
        return cls(raw=Path(raw).resolve(), output=Path(output).resolve())

    @property
    def processed(self) -> Path:
        return self.raw / PROCESSED_DIRNAME

    @property
    def failed(self) -> Path:
        return self.raw / FAILED_DIRNAME

    @property
    def batches(self) -> Path:
        return self.output / BATCHES_DIRNAME


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    """Read JSON, tolerating a file being rewritten underneath us."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log.debug("unreadable json %s: %s", path, exc)
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _modified(path: Path) -> str | None:
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(stamp, tz=UTC).isoformat(timespec="seconds")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _hms(seconds: float) -> str:
    """``3727.8`` -> ``01:02:07``. Whole seconds; the UI is not a scrub bar."""
    seconds = max(0, int(seconds or 0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _trim(value: str) -> str:
    """``00:03:32.336`` -> ``00:03:32``."""
    return (value or "").split(".")[0]


def _within(root: Path, *parts: str) -> Path:
    """Resolve ``parts`` under ``root``, refusing anything that escapes it.

    The server binds to localhost and only ever reads, but a path from a query
    string is still a path from outside, so it is checked rather than trusted.
    """
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise StateError(f"invalid path component: {part!r}")
    candidate = (root / Path(*parts)).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise StateError("path escapes its root")
    return candidate


def summary_line(text: str) -> str:
    """The first real sentence of a summary, as one plain-text line.

    Skips the title, the bold statistics line the summary format opens with,
    and any heading, to land on the first sentence of actual prose -- which is
    what tells you whether this is the recording you were looking for.
    """
    for block in text.replace("\r\n", "\n").split("\n\n"):
        block = block.strip()
        if not block or block.startswith(("#", ">", "-", "*", "|", "[^", "```")):
            continue
        plain = markdown.strip_inline(block)
        # The metadata line is entirely bold; stripping it leaves only stats.
        if not plain or (block.startswith("**") and block.endswith("**")):
            continue
        stop = plain.find(". ")
        if 60 <= stop <= _SUMMARY_LINE_MAX:
            return plain[: stop + 1]
        if len(plain) <= _SUMMARY_LINE_MAX:
            return plain
        return plain[:_SUMMARY_LINE_MAX].rsplit(" ", 1)[0] + "…"
    return ""


# --------------------------------------------------------------------------
# batch reports
# --------------------------------------------------------------------------

def read_reports(roots: Roots) -> list[dict]:
    """Every batch report in the tree, oldest first."""
    if not roots.batches.is_dir():
        return []
    reports = []
    for path in roots.batches.glob("*.json"):
        report = _read_json(path)
        if report and report.get("schema", "").startswith("audua/batch/"):
            report["_path"] = str(path)
            reports.append(report)
    return sorted(reports, key=lambda r: (r.get("started") or "", r.get("_path") or ""))


def latest_report(roots: Roots) -> dict | None:
    reports = read_reports(roots)
    return reports[-1] if reports else None


def _result_index(roots: Roots) -> dict[str, dict]:
    """Map every name a source has been known by to its most recent result.

    Indexed twice on purpose: by the name it had in the inbox, and by the name
    it was filed under -- filing renames on collision, so ``foo.wav`` in the
    report can be ``foo-2.wav`` on disk.
    """
    index: dict[str, dict] = {}
    for report in read_reports(roots):
        for result in report.get("results") or []:
            entry = dict(result)
            entry["batch_started"] = report.get("started")
            entry["report_path"] = report.get("report_path") or report.get("_path")
            index[result.get("source", "")] = entry
            moved = result.get("moved_to")
            if moved:
                index[Path(moved).name] = entry
    index.pop("", None)
    return index


# --------------------------------------------------------------------------
# the inbox
# --------------------------------------------------------------------------

def _run_name_for(stem: str, roots: Roots, result: dict | None) -> str | None:
    """Which output folder belongs to this source, if any."""
    if result and result.get("output_dir"):
        candidate = Path(result["output_dir"]).name
        if (roots.output / candidate / MANIFEST_NAME).is_file():
            return candidate
    if (roots.output / stem / MANIFEST_NAME).is_file():
        return stem
    # Filed as `name-2.wav` after a collision; the run kept the original stem.
    base = stem.rsplit("-", 1)[0]
    if base != stem and (roots.output / base / MANIFEST_NAME).is_file():
        return base
    return None


def _source_record(path: Path, status: str, roots: Roots, index: dict[str, dict]) -> dict:
    result = index.get(path.name) or {}
    run = _run_name_for(path.stem, roots, result)

    # The batch report is the better source — it says what happened *this* time
    # — but it only covers sources processed by `audua batch`. Anything run
    # through `audua run`, or whose report has since been pruned, still has its
    # manifest, so fall back to that rather than showing a row of dashes.
    manifest = _read_json(roots.output / run / MANIFEST_NAME) if run else None
    stats = (manifest or {}).get("stats") or {}
    source = (manifest or {}).get("source") or {}

    # `process_file` rewrites manifest.json after every clip and only adds
    # `completed` once the whole source is done -- so a source still sitting in
    # the inbox with a manifest that has clips but no `completed` is the one
    # `audua batch` is working on right now. `clips_done` undercounts slightly
    # while the leading clips were all reused (see pipeline.process_file), but
    # it never overcounts, and it always catches up on the next real clip.
    clips_written = (manifest or {}).get("clips")
    processing = (
        status == "ready" and manifest is not None and "completed" not in manifest
    )

    return {
        "name": path.name,
        "stem": path.stem,
        "status": status,
        "size": _size(path),
        "modified": _modified(path),
        "has_overrides": default_override_path(path).is_file(),
        "run": run,
        "duration_hms": _trim(result.get("duration_hms") or source.get("duration_hms") or ""),
        "clip_count": result.get("clip_count", stats.get("clip_count")),
        "flagged_clips": result.get("flagged_clips", stats.get("flagged_clips")),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "reasons": result.get("reasons") or [],
        "last_run": result.get("batch_started") or (manifest or {}).get("completed"),
        "summary_needed": bool(result.get("summary_needed")),
        "processing": processing,
        "clips_done": len(clips_written) if processing and clips_written is not None else None,
    }


def list_sources(roots: Roots) -> list[dict]:
    """Every raw recording in the tree: waiting, processed, or failed.

    ``ready`` is simply "still sitting in the inbox". The inbox is scanned top
    level only, the same way ``audua batch`` scans it, so the archive folders
    below it are not mistaken for work waiting to be done.
    """
    index = _result_index(roots)
    records: list[dict] = []

    for directory, status in (
        (roots.raw, "ready"),
        (roots.processed, "processed"),
        (roots.failed, "failed"),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_dir() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            records.append(_source_record(path, status, roots, index))

    order = {"ready": 0, "failed": 1, "processed": 2}
    records.sort(key=lambda r: (order[r["status"]], r["modified"] or ""), reverse=False)
    return records


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def _run_dirs(roots: Roots) -> list[Path]:
    if not roots.output.is_dir():
        return []
    return sorted(
        path for path in roots.output.iterdir()
        if path.is_dir() and not path.name.startswith("_") and not path.name.startswith(".")
    )


def _run_summary(run_dir: Path, manifest: dict | None) -> dict:
    """The card-level facts about one run: what it is, and whether it is sound."""
    source = (manifest or {}).get("source") or {}
    stats = (manifest or {}).get("stats") or {}
    config = (manifest or {}).get("config") or {}
    counts = (manifest or {}).get("counts") or {}

    summary_path = run_dir / SUMMARY_NAME
    summary_text = _read_text(summary_path) if summary_path.is_file() else None

    if manifest is None:
        status, reasons = "incomplete", ["no manifest.json — this run never started, or is mid-flight"]
    else:
        reasons = classify_outcome(manifest)
        status = "failed" if reasons else "ok"

    return {
        "name": run_dir.name,
        "source_name": source.get("name"),
        "duration": source.get("duration"),
        "duration_hms": _trim(source.get("duration_hms") or ""),
        "codec": source.get("codec"),
        "status": status,
        "reasons": reasons,
        "clip_count": stats.get("clip_count", 0),
        "flagged_clips": stats.get("flagged_clips", 0),
        # A flag nothing carries is not news; only report the ones that fired.
        "flag_counts": {
            flag: n for flag, n in (stats.get("flag_counts") or {}).items() if n
        },
        "clip_seconds": stats.get("clip_seconds", 0.0),
        "speech_hms": _hms(stats.get("clip_seconds", 0.0)),
        "retained_fraction": stats.get("retained_fraction", 0.0),
        "model": config.get("model"),
        "language": config.get("language"),
        "started": (manifest or {}).get("started"),
        "completed": (manifest or {}).get("completed"),
        "counts": counts,
        "has_summary": summary_text is not None,
        "has_digest": (run_dir / DIGEST_NAME).is_file(),
        "has_manifest": manifest is not None,
        "summary_line": summary_line(summary_text) if summary_text else "",
        "modified": _modified(summary_path if summary_text else run_dir / MANIFEST_NAME),
    }


def list_outputs(roots: Roots) -> list[dict]:
    """Every finished (or half-finished) run, newest first."""
    runs = [_run_summary(d, _read_json(d / MANIFEST_NAME)) for d in _run_dirs(roots)]
    runs.sort(key=lambda r: (r.get("completed") or r.get("started") or "", r["name"]), reverse=True)
    return runs


def _clip_records(run_dir: Path, manifest: dict, cited: set[int]) -> list[dict]:
    """Every clip, with enough text to recognise it and links to open it."""
    records = []
    for clip in manifest.get("clips") or []:
        text_name = clip.get("text_file")
        text = _read_text(run_dir / text_name) if text_name else None
        preview = " ".join((text or "").split())
        truncated = len(preview) > _CLIP_PREVIEW_MAX
        if truncated:
            preview = preview[:_CLIP_PREVIEW_MAX].rsplit(" ", 1)[0] + "…"

        records.append({
            "index": clip.get("index"),
            "stem": clip.get("stem"),
            "start": clip.get("start"),
            "end": clip.get("end"),
            "start_hms": _trim(clip.get("start_hms") or ""),
            "end_hms": _trim(clip.get("end_hms") or ""),
            "duration": clip.get("duration"),
            "origin": clip.get("origin"),
            "labels": clip.get("labels") or [],
            "speech_ratio": clip.get("speech_ratio"),
            "flags": clip.get("flags") or [],
            "status": clip.get("status"),
            "error": clip.get("error"),
            "audio_file": clip.get("audio_file"),
            "text_file": text_name,
            "sidecar_file": clip.get("sidecar_file"),
            "preview": preview,
            "empty": not preview,
            "cited": clip.get("index") in cited,
        })
    return records


def output_detail(roots: Roots, name: str) -> dict:
    """One run in full: its stats, its clips, and every health check we have."""
    run_dir = _within(roots.output, name)
    if not run_dir.is_dir():
        raise StateError(f"no such output: {name}")

    manifest = _read_json(run_dir / MANIFEST_NAME)
    detail = _run_summary(run_dir, manifest)

    if manifest is None:
        detail |= {"clips": [], "citations": None, "pairing": None, "documents": []}
        return detail

    citations = check_summary(run_dir, manifest)
    transcribing = not (manifest.get("config") or {}).get("segment_only", False)
    pairing = verify_pairing(run_dir, manifest.get("clips") or [], transcribing=transcribing)

    detail |= {
        "clips": _clip_records(run_dir, manifest, set(citations.get("cited") or [])),
        "citations": citations,
        "pairing": pairing,
        "manifest_pairing": manifest.get("pairing") or {},
        "documents": [
            {"file": path.name, "size": _size(path), "modified": _modified(path)}
            for path in sorted(run_dir.iterdir())
            if path.is_file() and path.name in {SUMMARY_NAME, DIGEST_NAME, MANIFEST_NAME}
        ],
    }
    return detail


# --------------------------------------------------------------------------
# documents and media
# --------------------------------------------------------------------------

def document(roots: Roots, name: str, filename: str) -> dict:
    """Render one file from a run for the reading pane."""
    path = _within(roots.output, name, filename)
    if not path.is_file():
        raise StateError(f"no such file: {name}/{filename}")
    if path.suffix.lower() not in READABLE_SUFFIXES:
        raise StateError(f"not a readable document: {filename}")

    text = _read_text(path)
    if text is None:
        raise StateError(f"could not read {filename}")

    if path.suffix.lower() == ".md":
        kind, html = "markdown", markdown.render(text)
    elif path.suffix.lower() == ".json":
        parsed = _read_json(path)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else text
        kind, html = "json", f"<pre class=\"doc-plain\">{markdown.escape(pretty)}</pre>"
    else:
        kind = "text"
        html = (
            f'<pre class="doc-plain">{markdown.escape(text.strip())}</pre>'
            if text.strip()
            else '<p class="muted">Empty transcript — the clip is kept, but Whisper '
                 "returned no text for it.</p>"
        )

    return {
        "run": name,
        "file": path.name,
        "kind": kind,
        "html": html,
        "size": _size(path),
        "modified": _modified(path),
    }


def media_path(roots: Roots, name: str, filename: str) -> Path:
    """Resolve a clip's audio file for streaming, or refuse."""
    path = _within(roots.output, name, filename)
    if not path.is_file():
        raise StateError(f"no such file: {name}/{filename}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise StateError(f"not an audio file: {filename}")
    return path


# --------------------------------------------------------------------------
# the dashboard
# --------------------------------------------------------------------------

def _processing_status(waiting: list[dict]) -> dict:
    """What `audua batch` is doing to the inbox right now, if anything.

    `audua batch` works the inbox one source at a time, so at most one waiting
    source can be mid-run; every other waiting source is simply next in line.
    The overall fraction weighs the active source by its own clip progress and
    every other waiting source as not-yet-started -- the fairest read available
    without the batch itself publishing a plan, and it can only undercount
    (never overcount) if a source is dropped into the inbox mid-run.
    """
    active = next((s for s in waiting if s["processing"]), None)
    total = len(waiting)

    fraction = None
    if total:
        done = 0.0
        if active and active.get("clip_count"):
            done = min(1.0, (active.get("clips_done") or 0) / active["clip_count"])
        fraction = done / total

    return {
        "in_progress": active is not None,
        "active_source": active["name"] if active else None,
        "clips_done": active.get("clips_done") if active else None,
        "clips_total": active.get("clip_count") if active else None,
        "queue_total": total,
        "fraction": fraction,
        "percent": round(fraction * 100) if fraction is not None else None,
    }


def overview(roots: Roots) -> dict:
    """Everything the front page shows, in one read of the tree."""
    sources = list_sources(roots)
    runs = list_outputs(roots)
    report = latest_report(roots)

    by_status: dict[str, int] = {"ready": 0, "processed": 0, "failed": 0}
    for source in sources:
        by_status[source["status"]] += 1

    waiting = [s for s in sources if s["status"] == "ready"]

    return {
        "roots": {
            "raw": str(roots.raw),
            "output": str(roots.output),
            "processed": str(roots.processed),
            "failed": str(roots.failed),
        },
        "inbox": {
            "ready": by_status["ready"],
            "processed": by_status["processed"],
            "failed": by_status["failed"],
            "ready_bytes": sum(s["size"] for s in waiting),
            "oldest_waiting": min((s["modified"] or "" for s in waiting), default=None),
        },
        "processing": _processing_status(waiting),
        "outputs": {
            "total": len(runs),
            "ok": sum(1 for r in runs if r["status"] == "ok"),
            "failed": sum(1 for r in runs if r["status"] == "failed"),
            "incomplete": sum(1 for r in runs if r["status"] == "incomplete"),
            "summarized": sum(1 for r in runs if r["has_summary"]),
            "awaiting_summary": sum(1 for r in runs if r["status"] == "ok" and not r["has_summary"]),
        },
        "clips": {
            "total": sum(r["clip_count"] for r in runs),
            "flagged": sum(r["flagged_clips"] for r in runs),
            "speech_hms": _hms(sum(r["clip_seconds"] for r in runs)),
            "audio_hms": _hms(sum(r["duration"] or 0 for r in runs)),
        },
        "latest_batch": None if not report else {
            "started": report.get("started"),
            "completed": report.get("completed"),
            "elapsed_seconds": report.get("elapsed_seconds"),
            "counts": report.get("counts") or {},
            "aborted": report.get("aborted"),
            "report_path": report.get("report_path") or report.get("_path"),
            "results": [
                {
                    "source": r.get("source"),
                    "status": r.get("status"),
                    "clip_count": r.get("clip_count"),
                    "reasons": r.get("reasons") or [],
                }
                for r in (report.get("results") or [])
            ],
            "skipped": report.get("skipped") or [],
        },
        "recent_runs": runs[:5],
        "waiting": waiting[:5],
    }
