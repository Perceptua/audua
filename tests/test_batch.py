"""Inbox tests: discovery, filing, the digest, and what counts as a failure.

The fakes below duplicate the ones in ``test_pipeline.py`` deliberately. The
offline shim in ``tools/run_tests.py`` resolves fixtures per module and never
loads a ``conftest.py``, and pytest runs with ``--import-mode=importlib``, so
neither runner can share fixtures across test modules. Duplication here keeps
both of them working.

The recording is short (15s, two bursts) because these tests are about the
conveyor belt, not about segmentation — ``test_pipeline.py`` covers that.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from audua.audio import probe
from audua.batch import (
    build_digest,
    check_summary,
    classify_outcome,
    discover_inbox,
    file_source,
    run_inbox,
)
from audua.config import DIGEST_NAME, SUMMARY_NAME, Config, find_ffmpeg
from audua.pipeline import MANIFEST_NAME, verify_pairing
from audua.transcribe import Transcriber

pytestmark = pytest.mark.usefixtures("ffmpeg_required")


@pytest.fixture(scope="session")
def ffmpeg_required():
    try:
        find_ffmpeg()
    except Exception:  # pragma: no cover
        pytest.skip("ffmpeg not available")


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """A 15s WAV with tone bursts at 2-4s and 8-10s, near-silence elsewhere."""
    sample_rate = 16_000
    duration = 15
    audio = np.zeros(sample_rate * duration, dtype=np.float32)
    t = np.arange(sample_rate * duration) / sample_rate

    for start, end in [(2, 4), (8, 10)]:
        window = slice(start * sample_rate, end * sample_rate)
        audio[window] = 0.4 * np.sin(2 * np.pi * 220 * t[window])

    audio += 0.002 * np.random.default_rng(0).standard_normal(audio.size).astype(np.float32)

    path = tmp_path / "source" / "session_one.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    subprocess.run(
        [find_ffmpeg(), "-v", "error", "-y", "-f", "s16le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "-", str(path)],
        input=pcm16, check=True, capture_output=True,
    )
    return path


@pytest.fixture
def inbox(tmp_path: Path, recording: Path) -> Path:
    """An inbox holding one recording, with the archive folders already present."""
    raw = tmp_path / "raw"
    (raw / "processed").mkdir(parents=True, exist_ok=True)
    (raw / "failed").mkdir(parents=True, exist_ok=True)
    shutil.copy2(recording, raw / recording.name)
    return raw


class FakeDetector:
    """Stands in for Silero: reports the bursts we synthesised, per window."""

    BURSTS = [(2.0, 4.0), (8.0, 10.0)]

    def detect(self, samples):
        length = len(samples) / 16_000
        return [(s, e) for s, e in self.BURSTS if e <= length + 0.01]


class FakeTranscriber(Transcriber):
    """Returns deterministic text without loading a model."""

    def __init__(self, config, text="a five mile run and some Melville"):
        super().__init__(config)
        self.text = text

    def transcribe(self, audio_path):
        info = probe(audio_path)
        return {
            "segments": [{
                "id": 0, "start": 0.0, "end": round(info.duration, 3),
                "text": self.text, "avg_logprob": -0.2,
                "no_speech_prob": 0.05, "compression_ratio": 1.4,
            }] if self.text else [],
            "language": "en",
            "language_probability": 0.99,
            "duration": info.duration,
        }


def base_config(tmp_path: Path, **kwargs) -> Config:
    config = Config(output_root=tmp_path / "output", pad=0.0, **kwargs)
    config.validate()
    return config


def batch(tmp_path: Path, raw: Path, *, move=True, text="a five mile run and some Melville",
          **config_kwargs) -> dict:
    config = base_config(tmp_path, **config_kwargs)
    return run_inbox(
        config, raw_dir=raw, move=move,
        transcriber=FakeTranscriber(config, text=text), detector=FakeDetector(),
    )


# ------------------------------------------------------------------- discovery

def test_discover_inbox_is_flat_and_ignores_the_archive(tmp_path):
    raw = tmp_path / "raw"
    (raw / "processed").mkdir(parents=True)
    (raw / "failed").mkdir(parents=True)
    (raw / "a.wav").write_bytes(b"x")
    (raw / "b.m4a").write_bytes(b"x")
    (raw / "processed" / "old.wav").write_bytes(b"x")
    (raw / "failed" / "bad.wav").write_bytes(b"x")

    sources, _ = discover_inbox(raw)

    assert [p.name for p in sources] == ["a.wav", "b.m4a"]


def test_discover_inbox_skips_non_audio_and_says_why(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.wav").write_bytes(b"x")
    (raw / "notes.txt").write_text("not audio", encoding="utf-8")

    sources, skipped = discover_inbox(raw)

    assert [p.name for p in sources] == ["a.wav"]
    assert len(skipped) == 1
    assert skipped[0]["source"] == "notes.txt"
    assert "audio extension" in skipped[0]["reason"]


def test_discover_inbox_leaves_a_file_that_may_still_be_copying(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.wav").write_bytes(b"x")

    sources, skipped = discover_inbox(raw, min_age=3600)

    assert sources == []
    assert "min-age" in skipped[0]["reason"]


# --------------------------------------------------------------------- filing

def test_a_finished_source_is_moved_into_processed(tmp_path, inbox):
    report = batch(tmp_path, inbox)

    assert report["counts"]["processed"] == 1
    assert report["counts"]["failed"] == 0
    assert not (inbox / "session_one.wav").exists()
    assert (inbox / "processed" / "session_one.wav").is_file()
    assert report["results"][0]["moved_to"].endswith("session_one.wav")


def test_a_source_that_cannot_be_read_is_moved_into_failed(tmp_path, inbox):
    (inbox / "broken.wav").write_bytes(b"this is not a wav file")

    report = batch(tmp_path, inbox)

    result = next(r for r in report["results"] if r["source"] == "broken.wav")
    assert result["status"] == "failed"
    assert result["reasons"]
    assert (inbox / "failed" / "broken.wav").is_file()
    assert not (inbox / "broken.wav").exists()
    # The healthy file in the same batch is unaffected.
    assert (inbox / "processed" / "session_one.wav").is_file()
    assert report["counts"] == {
        "total": 2, "processed": 1, "failed": 1, "skipped": 0,
        "not_reached": 0, "summaries_pending": 1,
    }


def test_an_override_sidecar_travels_with_its_audio(tmp_path, inbox):
    (inbox / "session_one.overrides.json").write_text(
        json.dumps({"windows": [{"start": 5.0, "end": 6.0, "label": "quiet"}]}),
        encoding="utf-8",
    )

    batch(tmp_path, inbox)

    assert (inbox / "processed" / "session_one.wav").is_file()
    assert (inbox / "processed" / "session_one.overrides.json").is_file()


def test_a_name_collision_gets_a_suffix_instead_of_an_overwrite(tmp_path, inbox):
    archived = inbox / "processed" / "session_one.wav"
    archived.write_bytes(b"an older recording of the same name")

    batch(tmp_path, inbox)

    assert archived.read_bytes() == b"an older recording of the same name"
    assert (inbox / "processed" / "session_one-2.wav").is_file()


def test_file_source_keeps_a_renamed_pair_together(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "clip.wav").write_bytes(b"audio")
    (raw / "clip.overrides.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "clip.wav").write_bytes(b"older")

    moved = file_source(raw / "clip.wav", archive)

    assert [p.name for p in moved] == ["clip-2.wav", "clip-2.overrides.json"]


def test_no_move_leaves_the_inbox_untouched(tmp_path, inbox):
    report = batch(tmp_path, inbox, move=False)

    assert (inbox / "session_one.wav").is_file()
    assert not (inbox / "processed" / "session_one.wav").exists()
    assert report["results"][0]["moved_to"] is None
    assert report["results"][0]["status"] == "processed"


def test_an_empty_inbox_is_not_an_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()

    report = batch(tmp_path, raw)

    assert report["counts"]["total"] == 0
    assert report["results"] == []


# ------------------------------------------------------------------- outcomes

def test_quality_flags_alone_are_not_a_failure():
    manifest = {
        "counts": {"total": 3, "ok": 3, "reused": 0, "failed": 0},
        "stats": {"clip_count": 3, "flagged_clips": 3,
                  "flag_counts": {"sparse": 2, "low_confidence": 1}},
        "pairing": {"ok": True, "problems": []},
    }

    assert classify_outcome(manifest) == []


def test_a_failed_clip_or_broken_pairing_is_a_failure():
    manifest = {
        "counts": {"total": 3, "ok": 2, "reused": 0, "failed": 1},
        "stats": {"clip_count": 3},
        "pairing": {"ok": False, "problems": ["missing transcript: clip_0002.txt"]},
    }

    reasons = classify_outcome(manifest)

    assert any("1 of 3 clip(s) failed" in r for r in reasons)
    assert any("pairing check failed" in r for r in reasons)


def test_a_run_with_no_clips_is_a_failure():
    manifest = {"counts": {"total": 0, "ok": 0, "reused": 0, "failed": 0},
                "stats": {"clip_count": 0}, "pairing": {"ok": True, "problems": []}}

    assert any("no clips" in r for r in classify_outcome(manifest))


def test_an_unfinished_run_is_a_failure():
    assert classify_outcome({"stats": {"clip_count": 4}}) == [
        "the run did not complete (no counts block in the manifest)"
    ]


# --------------------------------------------------------------------- digest

def test_the_digest_lists_every_clip_with_its_files_and_text(tmp_path, inbox):
    report = batch(tmp_path, inbox)
    out_dir = Path(report["results"][0]["output_dir"])
    digest = (out_dir / DIGEST_NAME).read_text(encoding="utf-8")
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert report["results"][0]["digest"] == str(out_dir / DIGEST_NAME)
    for record in manifest["clips"]:
        assert f"## [{record['index']}] {record['stem']}" in digest
        assert f"`{record['audio_file']}`" in digest
        assert f"`{record['text_file']}`" in digest
    assert digest.count("a five mile run and some Melville") == len(manifest["clips"])
    assert "flags: none" in digest


def test_the_digest_timestamps_text_against_the_original_recording(tmp_path, inbox):
    # A gap shorter than --merge-gap would fold both bursts into one clip; 2s
    # keeps them separate so the second clip has a non-zero offset to get wrong.
    report = batch(tmp_path, inbox, merge_gap=2.0)
    digest = (Path(report["results"][0]["output_dir"]) / DIGEST_NAME).read_text(encoding="utf-8")

    assert "## [2] clip_0002" in digest
    # The second burst starts at 8s in the source, not at 0s inside its own clip.
    assert "**[00:00:08]**" in digest


def test_an_empty_transcript_is_reported_not_hidden(tmp_path, inbox):
    report = batch(tmp_path, inbox, text="")
    digest = (Path(report["results"][0]["output_dir"]) / DIGEST_NAME).read_text(encoding="utf-8")

    assert "no speech transcribed" in digest
    assert "empty_transcript" in digest


def test_a_failed_clip_is_named_in_the_digest(tmp_path):
    out_dir = tmp_path / "output" / "session_one"
    out_dir.mkdir(parents=True)
    manifest = {
        "source": {"name": "session_one.wav", "duration_hms": "00:00:15.000"},
        "stats": {"clip_count": 1},
        "clips": [{"index": 1, "stem": "clip_0001", "start": 2.0, "end": 4.0,
                   "duration": 2.0, "status": "failed", "error": "ffmpeg exploded"}],
    }

    digest = build_digest(out_dir, manifest)

    assert "## [1] clip_0001" in digest
    assert "ffmpeg exploded" in digest


# ------------------------------------------------------------- derived artifacts

def test_the_digest_and_summary_are_not_treated_as_orphans(tmp_path, inbox):
    """Regression guard: verify_pairing must know these belong in the folder."""
    report = batch(tmp_path, inbox)
    out_dir = Path(report["results"][0]["output_dir"])
    (out_dir / SUMMARY_NAME).write_text("# summary\n", encoding="utf-8")
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert verify_pairing(out_dir, manifest["clips"], transcribing=True)["ok"]

    # And a second pass over the same source, with both files now on disk, is
    # still a clean run rather than a pairing failure.
    shutil.copy2(inbox / "processed" / "session_one.wav", inbox / "session_one.wav")
    again = batch(tmp_path, inbox)
    assert again["counts"]["failed"] == 0


def test_a_summary_is_flagged_as_pending_until_it_exists(tmp_path, inbox):
    report = batch(tmp_path, inbox)
    out_dir = Path(report["results"][0]["output_dir"])

    assert report["results"][0]["summary_needed"] is True
    assert report["counts"]["summaries_pending"] == 1
    assert report["results"][0]["summary"] == str(out_dir / SUMMARY_NAME)

    (out_dir / SUMMARY_NAME).write_text("# summary\n", encoding="utf-8")
    shutil.copy2(inbox / "processed" / "session_one.wav", inbox / "session_one.wav")

    again = batch(tmp_path, inbox)
    assert again["results"][0]["summary_needed"] is False
    assert again["counts"]["summaries_pending"] == 0


# --------------------------------------------------------------------- report

def test_the_report_is_written_and_latest_points_at_the_same_run(tmp_path, inbox):
    report = batch(tmp_path, inbox)

    written = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    latest = json.loads(
        (tmp_path / "output" / "_batches" / "latest.json").read_text(encoding="utf-8")
    )

    assert written["started"] == report["started"] == latest["started"]
    assert latest["report_path"] == report["report_path"]
    assert written["counts"]["processed"] == 1


# ---------------------------------------------------------- summary citations

def two_clip_run(tmp_path: Path, inbox: Path) -> tuple[Path, dict]:
    """A finished run with two separate clips, plus its manifest."""
    report = batch(tmp_path, inbox, merge_gap=2.0)
    out_dir = Path(report["results"][0]["output_dir"])
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(manifest["clips"]) == 2
    return out_dir, manifest


GOOD_SUMMARY = """# session_one

A run[^1], then some Melville[^2].

[^1]: **00:00:02** - [transcript](clip_0001.txt) - [audio](clip_0001.wav) - "a five mile run"
[^2]: **00:00:08** - [transcript](clip_0002.txt) - [audio](clip_0002.wav) - "some Melville"
"""


def test_a_summary_whose_citations_all_resolve_passes(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)
    (out_dir / SUMMARY_NAME).write_text(GOOD_SUMMARY, encoding="utf-8")

    result = check_summary(out_dir, manifest)

    assert result["ok"]
    assert result["present"]
    assert result["cited"] == [1, 2]
    assert result["uncited"] == []


def test_a_citation_without_a_definition_is_caught(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)
    (out_dir / SUMMARY_NAME).write_text(
        GOOD_SUMMARY.replace("[^2]: ", "[^9]: "), encoding="utf-8"
    )

    result = check_summary(out_dir, manifest)

    assert not result["ok"]
    assert "[^2] is cited but never defined" in result["problems"]
    assert "[^9] is defined but never cited" in result["problems"]


def test_a_citation_that_names_no_clip_in_this_run_is_caught(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)
    (out_dir / SUMMARY_NAME).write_text(
        GOOD_SUMMARY.replace("[^2]", "[^7]"), encoding="utf-8"
    )

    result = check_summary(out_dir, manifest)

    assert not result["ok"]
    assert any("does not match any clip" in problem for problem in result["problems"])


def test_a_definition_pointing_at_a_missing_file_is_caught(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)
    (out_dir / SUMMARY_NAME).write_text(
        GOOD_SUMMARY.replace("clip_0002.wav", "clip_0002.m4a"), encoding="utf-8"
    )

    result = check_summary(out_dir, manifest)

    assert not result["ok"]
    assert any("clip_0002.m4a" in problem for problem in result["problems"])


def test_uncited_clips_are_reported_without_failing(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)
    # Drop every mention of the second clip: citation, definition and all.
    only_clip_one = GOOD_SUMMARY.replace(", then some Melville[^2]", "")
    (out_dir / SUMMARY_NAME).write_text(only_clip_one.split("[^2]:")[0], encoding="utf-8")

    result = check_summary(out_dir, manifest)

    assert result["ok"]
    assert result["uncited"] == [2]


def test_a_run_with_no_summary_yet_is_not_a_problem(tmp_path, inbox):
    out_dir, manifest = two_clip_run(tmp_path, inbox)

    result = check_summary(out_dir, manifest)

    assert result["ok"]
    assert result["present"] is False
