"""UI tests: the view model, the markdown renderer, and the HTTP surface.

Deliberately dependency-free — no ffmpeg, no numpy, no model. The UI only ever
reads the tree, so a tree built out of plain files exercises every path through
it, and the whole module runs in well under a second on both runners.

The ``.wav`` files below are bytes, not audio. Nothing in the UI decodes them:
the server streams ranges of them and the browser is what plays them back, so
their content is irrelevant and their length is what the Range tests need.
"""

from __future__ import annotations

import contextlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from audua.ui import Roots, StateError, make_server, state
from audua.ui import markdown as md

# --------------------------------------------------------------------------
# a tree to read
# --------------------------------------------------------------------------

SUMMARY = """# done

**00:10:00 · 3 clips · 00:01:00 of speech · transcribed with large-v3**

## What's in this recording

A recording made on a bright morning to exercise the reading pane end to end.
It holds three clips and cites two of them.

## The frame problem (00:00-00:01)

Something about the cart frame[^1] and then a second thought[^2].

[^1]: **00:00:10 – 00:00:20** · [transcript](clip_0001.txt) · [audio](clip_0001.wav) — "first"
[^2]: **00:00:30 – 00:00:40** · [transcript](clip_0002.txt) · [audio](clip_0002.wav) — "second"
"""


def _clip(index: int, start: float, flags: list[str] | None = None) -> dict:
    return {
        "index": index,
        "stem": f"clip_{index:04d}",
        "start": start,
        "end": start + 10,
        "duration": 10.0,
        "start_hms": f"00:00:{int(start):02d}.000",
        "end_hms": f"00:00:{int(start) + 10:02d}.000",
        "origin": "vad",
        "labels": [],
        "speech_seconds": 6.0,
        "speech_ratio": 0.6,
        "flags": flags or [],
        "audio_file": f"clip_{index:04d}.wav",
        "sidecar_file": f"clip_{index:04d}.json",
        "text_file": f"clip_{index:04d}.txt",
        "status": "ok",
    }


def _manifest(name: str, clips: list[dict], *, failed: int = 0) -> dict:
    return {
        "schema": "audua/manifest/1",
        "source": {
            "name": f"{name}.wav", "duration": 600.0, "duration_hms": "00:10:00.000",
            "codec": "pcm_s16le", "sample_rate": 48000, "channels": 1,
        },
        "started": "2026-08-20T10:00:00+00:00",
        "completed": "2026-08-20T10:05:00+00:00",
        "config": {"model": "large-v3", "language": None, "segment_only": False},
        "stats": {
            "clip_count": len(clips), "source_duration": 600.0, "clip_seconds": 30.0,
            "retained_fraction": 0.05, "override_clips": 0,
            "flagged_clips": sum(1 for c in clips if c["flags"]),
            "flag_counts": {"sparse": sum(1 for c in clips if "sparse" in c["flags"])},
        },
        "clips": clips,
        "counts": {"total": len(clips), "ok": len(clips) - failed, "reused": 0, "failed": failed},
        "pairing": {"ok": True, "problems": []},
    }


def _write_run(root: Path, name: str, clips: list[dict], *, summary: str | None,
               failed: int = 0) -> Path:
    run = root / name
    run.mkdir(parents=True)
    for clip in clips:
        (run / clip["audio_file"]).write_bytes(b"RIFF" + bytes(4000))
        (run / clip["text_file"]).write_text(f"text of {clip['stem']}", encoding="utf-8")
        (run / clip["sidecar_file"]).write_text("{}", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(_manifest(name, clips, failed=failed)), encoding="utf-8"
    )
    (run / "transcript_digest.md").write_text(f"# {name} — transcript digest\n", encoding="utf-8")
    if summary:
        (run / "summary.md").write_text(summary, encoding="utf-8")
    return run


@pytest.fixture
def tree(tmp_path: Path) -> Roots:
    """An inbox with one of everything, and two runs — one sound, one failed."""
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    (raw / "processed").mkdir(parents=True)
    (raw / "failed").mkdir(parents=True)

    (raw / "waiting.wav").write_bytes(b"RIFF" + bytes(1000))
    (raw / "waiting.overrides.json").write_text('{"windows": []}', encoding="utf-8")
    (raw / "readme.txt").write_text("not audio", encoding="utf-8")
    (raw / "processed" / "done.wav").write_bytes(b"RIFF" + bytes(1000))
    (raw / "failed" / "broken.wav").write_bytes(b"RIFF" + bytes(1000))

    _write_run(output, "done", [_clip(1, 10), _clip(2, 30), _clip(3, 50, ["sparse"])],
               summary=SUMMARY)
    _write_run(output, "broken", [_clip(1, 10)], summary=None, failed=1)

    batches = output / "_batches"
    batches.mkdir()
    report = {
        "schema": "audua/batch/1",
        "started": "2026-08-20T10:00:00+00:00",
        "completed": "2026-08-20T10:06:00+00:00",
        "elapsed_seconds": 360.0,
        "counts": {"total": 2, "processed": 1, "failed": 1, "skipped": 0,
                   "not_reached": 0, "summaries_pending": 0},
        "results": [
            {"source": "done.wav", "status": "processed", "clip_count": 3, "flagged_clips": 1,
             "reasons": [], "output_dir": str(output / "done"),
             "moved_to": str(raw / "processed" / "done.wav"), "duration_hms": "00:10:00.000"},
            {"source": "broken.wav", "status": "failed", "clip_count": 1, "flagged_clips": 0,
             "reasons": ["1 of 1 clip(s) failed to extract or transcribe"],
             "output_dir": str(output / "broken"),
             "moved_to": str(raw / "failed" / "broken.wav")},
        ],
        "skipped": [],
    }
    (batches / "batch_20260820-100000.json").write_text(json.dumps(report), encoding="utf-8")
    (batches / "latest.json").write_text(json.dumps(report), encoding="utf-8")

    return Roots.resolved(raw, output)


@contextlib.contextmanager
def running(roots: Roots):
    """A live server on an ephemeral port, torn down on the way out."""
    server = make_server(roots, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def fetch(url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def fetch_json(url: str) -> dict:
    return json.loads(fetch(url)[2])


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------

def test_markdown_renders_the_blocks_the_documents_use():
    html = md.render(
        "# Title\n\n"
        "Prose with **bold**, _italic_, `code` and a [link](https://example.com).\n\n"
        "- one\n- two\n\n"
        "> quoted\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    )
    assert '<h1 id="title" class="doc-h1">Title</h1>' in html
    assert "<strong>bold</strong>" in html and "<em>italic</em>" in html
    assert "<code>code</code>" in html
    assert 'href="https://example.com"' in html and 'target="_blank"' in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<blockquote>" in html
    assert "<th>a</th>" in html and "<td>2</td>" in html


def test_markdown_escapes_html_in_the_source():
    html = md.render("A <script>alert(1)</script> in the transcript & more")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp;" in html


def test_citations_become_superscripts_and_a_numbered_reference_list():
    html = md.render(SUMMARY)
    assert '<sup class="fn-ref" id="fnref-1">' in html
    assert '<a href="#fn-1" data-fn="1">1</a>' in html
    # The number is the clip index, so the list item carries it explicitly
    # rather than being renumbered by the browser.
    assert '<li id="fn-2" value="2">' in html
    assert html.index('class="footnotes"') > html.index("The frame problem")


def test_clip_links_become_in_app_actions():
    html = md.render(SUMMARY)
    assert 'class="ln-audio" data-audio="clip_0001.wav"' in html
    assert 'class="ln-doc" data-doc="clip_0001.txt"' in html
    # ...and nothing is left pointing at a file the browser would download.
    assert 'href="clip_0001.wav"' not in html


def test_a_link_to_something_unplayable_is_not_offered_as_a_link():
    html = md.render("see [the source](recording.zip)")
    assert '<span class="ln-dead"' in html
    assert "<a" not in html


@pytest.mark.parametrize("source,expected", [
    ("**00:10:00 · 3 clips**\n\nThe real first line.", "The real first line."),
    ("# Title\n\n## Heading\n\nProse follows.", "Prose follows."),
    ("- a list is not a summary\n\nProse follows.", "Prose follows."),
    ("", ""),
])
def test_summary_line_finds_the_first_prose(source, expected):
    assert state.summary_line(source) == expected


def test_summary_line_cuts_at_a_sentence_and_truncates_a_long_one():
    long_sentence = (
        "An easy run on the evening of 19 August, recorded on a TASCAM with a collar mic, "
        "made as a proof of concept[^1]. And then a second sentence nobody needs."
    )
    assert state.summary_line(long_sentence).endswith("proof of concept.")

    runaway = "word " * 200
    line = state.summary_line(runaway)
    assert len(line) <= 221 and line.endswith("…")


# --------------------------------------------------------------------------
# the inbox
# --------------------------------------------------------------------------

def test_sources_are_classified_by_where_they_are_filed(tree):
    by_name = {s["name"]: s for s in state.list_sources(tree)}

    assert by_name["waiting.wav"]["status"] == "ready"
    assert by_name["done.wav"]["status"] == "processed"
    assert by_name["broken.wav"]["status"] == "failed"
    # The inbox scan is flat, like the batch's: the archive folders below it
    # are not work waiting to be done.
    assert len(by_name) == 3
    assert "readme.txt" not in by_name


def test_a_source_carries_its_overrides_run_and_failure_reasons(tree):
    by_name = {s["name"]: s for s in state.list_sources(tree)}

    assert by_name["waiting.wav"]["has_overrides"] is True
    assert by_name["done.wav"]["has_overrides"] is False
    assert by_name["done.wav"]["run"] == "done"
    assert by_name["done.wav"]["clip_count"] == 3
    assert by_name["broken.wav"]["reasons"] == [
        "1 of 1 clip(s) failed to extract or transcribe"
    ]


def test_facts_fall_back_to_the_manifest_when_no_report_covers_a_source(tree):
    """A source processed by `audua run`, or whose report was pruned."""
    for name in ("batch_20260820-100000.json", "latest.json"):
        (tree.output / "_batches" / name).unlink()

    done = next(s for s in state.list_sources(tree) if s["name"] == "done.wav")
    assert done["reasons"] == []          # nothing claims it failed
    assert done["clip_count"] == 3        # but the manifest still knows
    assert done["duration_hms"] == "00:10:00"


def _start_run(tree: Roots, name: str, clips_planned: int, clips_written: list[dict]) -> None:
    """Simulate `process_file` mid-run: a manifest with clips but no `completed`."""
    run = tree.output / name
    run.mkdir()
    manifest = _manifest(name, clips_written)
    manifest["stats"]["clip_count"] = clips_planned
    del manifest["completed"]
    del manifest["counts"]
    del manifest["pairing"]
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_a_source_mid_run_is_reported_as_processing(tree):
    _start_run(tree, "waiting", clips_planned=10, clips_written=[_clip(1, 10), _clip(2, 30)])

    waiting = next(s for s in state.list_sources(tree) if s["name"] == "waiting.wav")
    assert waiting["status"] == "ready"
    assert waiting["processing"] is True
    assert waiting["clips_done"] == 2
    assert waiting["clip_count"] == 10

    # A finished run, or one that never started, is not "processing".
    done = next(s for s in state.list_sources(tree) if s["name"] == "done.wav")
    broken = next(s for s in state.list_sources(tree) if s["name"] == "broken.wav")
    assert done["processing"] is False
    assert broken["processing"] is False


def test_overview_reports_the_active_run_and_the_queue_fraction(tree):
    _start_run(tree, "waiting", clips_planned=10, clips_written=[_clip(1, 10), _clip(2, 30)])

    processing = state.overview(tree)["processing"]
    assert processing == {
        "in_progress": True,
        "active_source": "waiting.wav",
        "clips_done": 2,
        "clips_total": 10,
        "queue_total": 1,
        "fraction": 0.2,
        "percent": 20,
    }


def test_overview_processing_is_quiet_when_nothing_is_running(tree):
    # `waiting.wav` sits in the inbox with no manifest yet -- queued, not started.
    processing = state.overview(tree)["processing"]
    assert processing["in_progress"] is False
    assert processing["active_source"] is None
    assert processing["fraction"] == 0.0

    # With nothing at all waiting, there is no queue to report a fraction for.
    (tree.raw / "waiting.wav").unlink()
    (tree.raw / "waiting.overrides.json").unlink()
    processing = state.overview(tree)["processing"]
    assert processing["fraction"] is None
    assert processing["percent"] is None


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def test_output_status_matches_what_the_batch_would_have_decided(tree):
    by_name = {r["name"]: r for r in state.list_outputs(tree)}

    assert by_name["done"]["status"] == "ok"
    assert by_name["broken"]["status"] == "failed"
    assert by_name["broken"]["reasons"]
    assert by_name["done"]["has_summary"] is True
    assert by_name["broken"]["has_summary"] is False


def test_an_output_carries_a_readable_one_line_summary(tree):
    done = next(r for r in state.list_outputs(tree) if r["name"] == "done")
    assert done["summary_line"].startswith("A recording made on a bright morning")
    assert "[^" not in done["summary_line"]


def test_a_run_with_no_manifest_is_incomplete_rather_than_missing(tree):
    (tree.output / "half").mkdir()
    half = next(r for r in state.list_outputs(tree) if r["name"] == "half")
    assert half["status"] == "incomplete"
    assert half["clip_count"] == 0
    # So the UI does not offer a document that is not there to open.
    assert half["has_manifest"] is False
    assert half["has_summary"] is False and half["has_digest"] is False


def test_only_flags_that_actually_fired_are_reported(tree):
    by_name = {r["name"]: r for r in state.list_outputs(tree)}
    assert by_name["done"]["flag_counts"] == {"sparse": 1}
    # `broken`'s manifest carries a sparse count of zero. A flag nothing
    # carries is not news, and showing "sparse 0" reads as if it were.
    assert by_name["broken"]["flag_counts"] == {}


def test_output_detail_reports_clips_pairing_and_citations(tree):
    detail = state.output_detail(tree, "done")

    assert [c["index"] for c in detail["clips"]] == [1, 2, 3]
    assert detail["clips"][0]["preview"] == "text of clip_0001"
    assert detail["clips"][2]["flags"] == ["sparse"]

    assert detail["pairing"]["ok"] is True
    assert detail["citations"]["ok"] is True
    assert detail["citations"]["cited"] == [1, 2]
    # The useful signal: clip 3 exists and nothing in the summary points at it.
    assert detail["citations"]["uncited"] == [3]
    assert detail["clips"][2]["cited"] is False


def test_output_detail_notices_a_broken_pairing_on_disk(tree):
    (tree.output / "done" / "clip_0002.txt").unlink()
    detail = state.output_detail(tree, "done")

    assert detail["pairing"]["ok"] is False
    assert any("clip_0002.txt" in p for p in detail["pairing"]["problems"])


def test_overview_counts_the_whole_tree(tree):
    data = state.overview(tree)

    assert data["inbox"] == {
        "ready": 1, "processed": 1, "failed": 1,
        "ready_bytes": 1004, "oldest_waiting": data["inbox"]["oldest_waiting"],
    }
    assert data["outputs"]["total"] == 2
    assert data["outputs"]["ok"] == 1 and data["outputs"]["failed"] == 1
    assert data["outputs"]["summarized"] == 1
    assert data["clips"]["total"] == 4
    assert data["latest_batch"]["counts"]["processed"] == 1


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

def test_a_transcript_opens_as_text_and_a_summary_as_markdown(tree):
    assert state.document(tree, "done", "summary.md")["kind"] == "markdown"

    transcript = state.document(tree, "done", "clip_0001.txt")
    assert transcript["kind"] == "text"
    assert "text of clip_0001" in transcript["html"]


def test_an_empty_transcript_is_explained_rather_than_shown_blank(tree):
    (tree.output / "done" / "clip_0001.txt").write_text("", encoding="utf-8")
    assert "Empty transcript" in state.document(tree, "done", "clip_0001.txt")["html"]


@pytest.mark.parametrize("filename", [
    "../../secret.txt", "..", "/etc/passwd", "sub/clip_0001.txt",
    "clip_0001.wav",  # audio is streamed, not rendered into the page
    "nope.md",
])
def test_the_reading_pane_refuses_anything_outside_the_run(tree, filename):
    (tree.output.parent / "secret.txt").write_text("private", encoding="utf-8")
    with pytest.raises(StateError):
        state.document(tree, "done", filename)


def test_media_refuses_a_non_audio_file(tree):
    assert state.media_path(tree, "done", "clip_0001.wav").is_file()
    with pytest.raises(StateError):
        state.media_path(tree, "done", "summary.md")
    with pytest.raises(StateError):
        state.media_path(tree, "..", "pyproject.toml")


# --------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------

def test_the_pages_and_the_api_are_served(tree):
    with running(tree) as base:
        status, headers, body = fetch(f"{base}/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"<title>audua</title>" in body

        assert fetch(f"{base}/static/app.js")[0] == 200
        assert fetch_json(f"{base}/api/overview")["outputs"]["total"] == 2
        assert len(fetch_json(f"{base}/api/sources")["sources"]) == 3
        assert len(fetch_json(f"{base}/api/outputs")["outputs"]) == 2
        assert fetch_json(f"{base}/api/outputs/done")["citations"]["uncited"] == [3]

        doc = fetch_json(f"{base}/api/outputs/done/doc?file=summary.md")
        assert 'data-audio="clip_0001.wav"' in doc["html"]


def test_clip_audio_is_streamed_and_seekable(tree):
    with running(tree) as base:
        url = f"{base}/media/done/clip_0001.wav"
        total = 4004

        status, headers, body = fetch(url)
        assert status == 200
        assert headers["Accept-Ranges"] == "bytes"
        assert headers["Content-Type"] == "audio/wav"
        assert len(body) == total

        # Seeking is a range request; answering it properly is what makes the
        # scrub bar work instead of re-downloading the clip on every drag.
        status, headers, body = fetch(url, {"Range": "bytes=100-199"})
        assert status == 206
        assert headers["Content-Range"] == f"bytes 100-199/{total}"
        assert len(body) == 100

        status, headers, _ = fetch(url, {"Range": "bytes=4000-"})
        assert status == 206 and headers["Content-Range"] == f"bytes 4000-{total - 1}/{total}"

        status, headers, _ = fetch(url, {"Range": "bytes=-50"})
        assert status == 206 and headers["Content-Range"] == f"bytes {total - 50}-{total - 1}/{total}"

        status, headers, _ = fetch(url, {"Range": "bytes=99999-"})
        assert status == 416 and headers["Content-Range"] == f"bytes */{total}"


@pytest.mark.parametrize("path", [
    "/api/outputs/nope",
    "/api/outputs/done/doc?file=../../pyproject.toml",
    "/api/outputs/done/doc",
    "/api/nonsense",
    "/media/done/nope.wav",
    "/nowhere",
])
def test_bad_requests_are_refused_without_leaking_the_filesystem(tree, path):
    with running(tree) as base:
        status, _, body = fetch(base + path)
        assert status in {400, 404}
        assert b"Traceback" not in body


def test_the_server_answers_only_reads(tree):
    """Read-only by construction: there is no handler for a write."""
    with running(tree) as base:
        request = urllib.request.Request(f"{base}/api/overview", method="DELETE")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 501  # Unsupported method
