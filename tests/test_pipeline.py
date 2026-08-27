"""End-to-end tests over real audio, with the two heavy dependencies faked.

The VAD model and Whisper are stubbed so the suite runs anywhere in seconds.
Everything else is real: ffmpeg decodes and cuts actual files, the manifest is
written to a real directory, and resume/idempotency are exercised by running
the pipeline twice.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from audua.audio import extract_clip, probe
from audua.config import Config, find_ffmpeg
from audua.pipeline import MANIFEST_NAME, process_file, verify_pairing
from audua.transcribe import Transcriber, confidence_flags

pytestmark = pytest.mark.usefixtures("ffmpeg_required")


@pytest.fixture(scope="session")
def ffmpeg_required():
    try:
        find_ffmpeg()
    except Exception:  # pragma: no cover
        pytest.skip("ffmpeg not available")


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """A 60s WAV: tone bursts at 5-7s, 10-12s (3s gap) and 40-41s, silence elsewhere."""
    sample_rate = 16_000
    duration = 60
    audio = np.zeros(sample_rate * duration, dtype=np.float32)
    t = np.arange(sample_rate * duration) / sample_rate

    for start, end in [(5, 7), (10, 12), (40, 41)]:
        window = slice(start * sample_rate, end * sample_rate)
        audio[window] = 0.4 * np.sin(2 * np.pi * 220 * t[window])

    audio += 0.002 * np.random.default_rng(0).standard_normal(audio.size).astype(np.float32)

    path = tmp_path / "session_one.wav"
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2").tobytes()
    subprocess.run(
        [find_ffmpeg(), "-v", "error", "-y", "-f", "s16le", "-ar", str(sample_rate),
         "-ac", "1", "-i", "-", str(path)],
        input=pcm16, check=True, capture_output=True,
    )
    return path


class FakeDetector:
    """Stands in for Silero: reports the bursts we synthesised, per window."""

    BURSTS = [(5.0, 7.0), (10.0, 12.0), (40.0, 41.0)]

    def __init__(self):
        self.calls = 0

    def detect(self, samples):
        self.calls += 1
        # iter_windows hands us the whole file in one window at default settings
        length = len(samples) / 16_000
        return [(s, e) for s, e in self.BURSTS if e <= length + 0.01]


class FakeTranscriber(Transcriber):
    """Returns deterministic text without loading a model."""

    def __init__(self, config, text="hello there"):
        super().__init__(config)
        self.text = text
        self.calls = 0

    def transcribe(self, audio_path):
        self.calls += 1
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


def run(recording, config, text="hello there"):
    detector = FakeDetector()
    transcriber = FakeTranscriber(config, text=text)
    manifest = process_file(recording, config, transcriber=transcriber, detector=detector)
    return manifest, detector, transcriber


# ------------------------------------------------------------------- extraction

def test_extract_clip_produces_the_requested_span(recording, tmp_path):
    dest = tmp_path / "cut.wav"
    result = extract_clip(recording, dest, 10.0, 12.0)
    assert dest.is_file()
    assert result["mode"] == "copy"
    assert probe(dest).duration == pytest.approx(2.0, abs=0.3)


def test_stream_copy_rounds_outward_never_inward(recording, tmp_path):
    """Cut points snap to codec frame boundaries; they must never shave audio."""
    for start, end in [(10.0, 12.0), (5.3, 7.7), (40.0, 41.0)]:
        dest = tmp_path / f"cut_{start}.wav"
        extract_clip(recording, dest, start, end)
        actual = probe(dest).duration
        requested = end - start
        assert actual >= requested - 1e-6, f"clip was shortened: {actual} < {requested}"
        assert actual <= requested + 0.25, f"clip overshot badly: {actual} vs {requested}"


def test_extract_clip_preserves_the_source_codec(recording, tmp_path):
    dest = tmp_path / "cut.wav"
    extract_clip(recording, dest, 0.0, 1.0)
    assert probe(dest).codec == probe(recording).codec


def test_extract_clip_rejects_inverted_bounds(recording, tmp_path):
    with pytest.raises(ValueError):
        extract_clip(recording, tmp_path / "x.wav", 10.0, 10.0)


def test_extract_clip_leaves_no_partial_file(recording, tmp_path):
    dest = tmp_path / "cut.wav"
    extract_clip(recording, dest, 1.0, 2.0)
    assert not list(tmp_path.glob("*.partial"))


# ---------------------------------------------------------------------- the run

def test_pipeline_produces_greedy_clips(recording, tmp_path):
    manifest, _, _ = run(recording, base_config(tmp_path))
    # 5-7 and 10-12 are 3s apart -> one clip. 40-41 stands alone.
    assert manifest["stats"]["clip_count"] == 2
    spans = [(c["start"], c["end"]) for c in manifest["clips"]]
    assert spans == [(5.0, 12.0), (40.0, 41.0)]


def test_pipeline_skips_the_silence(recording, tmp_path):
    manifest, _, _ = run(recording, base_config(tmp_path))
    # 8s of clips out of a 60s file
    assert manifest["stats"]["clip_seconds"] == pytest.approx(8.0)
    assert manifest["stats"]["retained_fraction"] < 0.2


def test_every_clip_has_exactly_one_transcript(recording, tmp_path):
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)
    out_dir = config.output_root / recording.stem

    audio_files = sorted(p.stem for p in out_dir.glob("clip_*.wav"))
    text_files = sorted(p.stem for p in out_dir.glob("clip_*.txt"))
    sidecars = sorted(p.stem for p in out_dir.glob("clip_*.json"))

    assert audio_files == text_files == sidecars
    assert len(audio_files) == manifest["stats"]["clip_count"]
    assert manifest["pairing"]["ok"], manifest["pairing"]["problems"]


def test_transcript_text_is_written(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)
    text = (config.output_root / recording.stem / "clip_0001.txt").read_text(encoding="utf-8")
    assert text.strip() == "hello there"


def test_sidecar_carries_global_timestamps(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)
    sidecar = json.loads(
        (config.output_root / recording.stem / "clip_0001.json").read_text(encoding="utf-8")
    )
    segment = sidecar["transcript"]["segments"][0]
    # clip starts at 5.0s in the source, so a clip-local 0.0 maps to global 5.0
    assert segment["global_start"] == pytest.approx(5.0, abs=0.01)
    assert segment["global_start_hms"].startswith("00:00:05")


def test_empty_transcript_is_kept_and_flagged(recording, tmp_path):
    """A clip the model finds nothing in is still a result worth keeping."""
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config, text="")
    out_dir = config.output_root / recording.stem

    assert (out_dir / "clip_0001.wav").is_file()
    assert (out_dir / "clip_0001.txt").is_file()
    assert (out_dir / "clip_0001.txt").read_text(encoding="utf-8") == ""
    assert "empty_transcript" in manifest["clips"][0]["flags"]
    assert manifest["pairing"]["ok"]


# ------------------------------------------------------------------- overrides

def test_override_window_is_always_saved(recording, tmp_path):
    """A window over pure silence, which the VAD would never pick up."""
    (recording.parent / f"{recording.stem}.overrides.json").write_text(
        json.dumps({"windows": [{"start": "00:00:20", "end": "00:00:25", "label": "manual"}]})
    )
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)

    forced = [c for c in manifest["clips"] if c["origin"] == "override"]
    assert len(forced) == 1
    assert (forced[0]["start"], forced[0]["end"]) == (20.0, 25.0)
    assert forced[0]["labels"] == ["manual"]
    assert "no_vad_speech" in forced[0]["flags"]

    clip_path = config.output_root / recording.stem / f"{forced[0]['stem']}.wav"
    # Stream copy rounds outward, so the window is fully covered and never cut short.
    assert probe(clip_path).duration >= 5.0


def test_override_clip_still_gets_a_transcript(recording, tmp_path):
    (recording.parent / f"{recording.stem}.overrides.json").write_text(
        json.dumps([{"start": 20, "end": 25}])
    )
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)
    out_dir = config.output_root / recording.stem
    forced = next(c for c in manifest["clips"] if c["origin"] == "override")
    assert (out_dir / f"{forced['stem']}.txt").is_file()
    assert manifest["pairing"]["ok"]


def test_override_overlapping_speech_does_not_duplicate_audio(recording, tmp_path):
    (recording.parent / f"{recording.stem}.overrides.json").write_text(
        json.dumps([{"start": 6, "end": 8}])
    )
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)
    spans = sorted((c["start"], c["end"]) for c in manifest["clips"])
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] <= later[0] + 1e-6, f"clips overlap: {earlier} and {later}"


# ------------------------------------------------------ resume and idempotency

def test_second_run_reuses_everything(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)

    detector = FakeDetector()
    transcriber = FakeTranscriber(config)
    manifest = process_file(recording, config, transcriber=transcriber, detector=detector)

    assert manifest["counts"]["reused"] == manifest["counts"]["total"]
    assert manifest["counts"]["ok"] == 0
    assert transcriber.calls == 0, "should not have re-transcribed"
    assert detector.calls == 0, "should not have re-run VAD"


def test_resume_after_interruption_redoes_only_the_missing_clip(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)
    out_dir = config.output_root / recording.stem

    # Simulate a crash partway: the second clip's sidecar never got written.
    (out_dir / "clip_0002.json").unlink()

    transcriber = FakeTranscriber(config)
    manifest = process_file(recording, config, transcriber=transcriber, detector=FakeDetector())

    assert transcriber.calls == 1
    assert manifest["counts"]["reused"] == 1
    assert manifest["pairing"]["ok"]


def test_changing_merge_gap_supersedes_old_output_without_deleting_it(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)
    out_dir = config.output_root / recording.stem

    wider = base_config(tmp_path, merge_gap=40.0)
    run(recording, wider)

    attic = list(out_dir.glob("_superseded_*"))
    assert attic, "old clips should be preserved, not deleted"
    assert list(attic[0].glob("clip_*.wav"))
    # 5-7, 10-12 and 40-41 all merge at a 40s gap
    assert json.loads((out_dir / MANIFEST_NAME).read_text())["stats"]["clip_count"] == 1


def test_vad_cache_survives_a_merge_gap_change(recording, tmp_path):
    """Re-cutting should not mean re-listening."""
    config = base_config(tmp_path)
    run(recording, config)

    detector = FakeDetector()
    wider = base_config(tmp_path, merge_gap=40.0)
    process_file(recording, wider, transcriber=FakeTranscriber(wider), detector=detector)
    assert detector.calls == 0


def test_force_redoes_the_work(recording, tmp_path):
    config = base_config(tmp_path)
    run(recording, config)

    forced = base_config(tmp_path)
    forced.force = True
    transcriber = FakeTranscriber(forced)
    manifest = process_file(recording, forced, transcriber=transcriber, detector=FakeDetector())
    assert transcriber.calls == manifest["counts"]["total"]


# ---------------------------------------------------------------- verification

def test_verify_pairing_catches_a_missing_transcript(recording, tmp_path):
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)
    out_dir = config.output_root / recording.stem

    (out_dir / "clip_0001.txt").unlink()
    result = verify_pairing(out_dir, manifest["clips"], transcribing=True)
    assert not result["ok"]
    assert any("clip_0001.txt" in p for p in result["problems"])


def test_verify_pairing_catches_an_orphan_file(recording, tmp_path):
    config = base_config(tmp_path)
    manifest, _, _ = run(recording, config)
    out_dir = config.output_root / recording.stem

    (out_dir / "clip_0099.wav").write_bytes(b"stray")
    result = verify_pairing(out_dir, manifest["clips"], transcribing=True)
    assert not result["ok"]
    assert any("orphan" in p for p in result["problems"])


# ----------------------------------------------------------------------- modes

def test_segment_only_stops_before_transcription(recording, tmp_path):
    config = base_config(tmp_path)
    config.segment_only = True
    manifest = process_file(recording, config, detector=FakeDetector())
    out_dir = config.output_root / recording.stem
    assert list(out_dir.glob("clip_*.wav"))
    assert not list(out_dir.glob("clip_*.txt"))
    assert manifest["pairing"]["ok"]


def test_dry_run_writes_only_a_plan(recording, tmp_path):
    config = base_config(tmp_path)
    config.dry_run = True
    plan = process_file(recording, config, detector=FakeDetector())
    out_dir = config.output_root / recording.stem
    assert plan["dry_run"] is True
    assert (out_dir / "plan.json").is_file()
    assert not list(out_dir.glob("clip_*"))


# -------------------------------------------------------------------- flagging

def test_confidence_flags_detect_low_quality():
    config = Config()
    result = {"segments": [
        {"text": "mm", "avg_logprob": -2.0, "no_speech_prob": 0.9, "compression_ratio": 3.0}
    ]}
    flags = confidence_flags(result, config)
    assert set(flags) == {"low_confidence", "high_no_speech", "repetitive"}


def test_confidence_flags_stay_quiet_on_good_audio():
    config = Config()
    result = {"segments": [
        {"text": "a clear sentence", "avg_logprob": -0.15,
         "no_speech_prob": 0.01, "compression_ratio": 1.3}
    ]}
    assert confidence_flags(result, config) == []
