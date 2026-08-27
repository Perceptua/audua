"""Unit tests for clip-formation logic. No audio, no models, no filesystem."""

from __future__ import annotations

import json

import pytest

from audua.config import Config, parse_time
from audua.segments import (
    Override,
    OverrideError,
    build_clips,
    greedy_merge,
    load_overrides,
    overlap_seconds,
    pad_intervals,
    split_long,
    subtract,
    summarise,
)


def cfg(**kwargs) -> Config:
    base = dict(pad=0.0, merge_gap=5.0, min_clip_duration=0.5, sparse_ratio=0.35)
    base.update(kwargs)
    return Config(**base)


# ---------------------------------------------------------------- time parsing

@pytest.mark.parametrize(
    "value,expected",
    [
        (12, 12.0),
        (12.5, 12.5),
        ("12.5", 12.5),
        ("12.5s", 12.5),
        ("01:30", 90.0),
        ("00:01:30", 90.0),
        ("01:02:03.250", 3723.25),
        ("1:02:03", 3723.0),
    ],
)
def test_parse_time_accepts_common_formats(value, expected):
    assert parse_time(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "abc", "1:99", "-5", None, True, "00:60:00"])
def test_parse_time_rejects_nonsense(value):
    with pytest.raises(ValueError):
        parse_time(value)


# ---------------------------------------------------------------- greedy merge

def test_three_second_gap_becomes_one_clip():
    """The requirement's own example: speech -> 3s silence -> speech."""
    regions = [(10.0, 12.0), (15.0, 17.0)]  # 3s gap
    assert greedy_merge(regions, 5.0) == [(10.0, 17.0)]


def test_gap_larger_than_threshold_stays_split():
    regions = [(10.0, 12.0), (25.0, 27.0)]  # 13s gap
    assert greedy_merge(regions, 5.0) == [(10.0, 12.0), (25.0, 27.0)]


def test_gap_exactly_at_threshold_merges():
    """<= is intentional: the boundary case should be greedy, not split."""
    assert greedy_merge([(0.0, 1.0), (6.0, 7.0)], 5.0) == [(0.0, 7.0)]


def test_merge_chains_across_many_short_gaps():
    regions = [(0.0, 1.0), (3.0, 4.0), (6.0, 7.0), (9.0, 10.0)]
    assert greedy_merge(regions, 3.0) == [(0.0, 10.0)]


def test_merge_handles_unsorted_and_nested_input():
    regions = [(20.0, 22.0), (0.0, 10.0), (2.0, 5.0)]
    assert greedy_merge(regions, 1.0) == [(0.0, 10.0), (20.0, 22.0)]


def test_merge_of_empty_is_empty():
    assert greedy_merge([], 5.0) == []


# ---------------------------------------------------------------------- padding

def test_padding_clamps_to_file_bounds():
    assert pad_intervals([(0.1, 9.9)], 0.5, 10.0) == [(0.0, 10.0)]


def test_padding_merges_neighbours_it_pushes_together():
    # 0.6s apart, padded by 0.5 each side -> they touch and must not overlap.
    assert pad_intervals([(1.0, 2.0), (2.6, 3.0)], 0.5, 10.0) == [(0.5, 3.5)]


def test_zero_padding_is_identity():
    regions = [(1.0, 2.0), (5.0, 6.0)]
    assert pad_intervals(regions, 0.0, 10.0) == regions


# ------------------------------------------------------------ interval algebra

def test_subtract_punches_a_hole_in_the_middle():
    assert subtract((0.0, 10.0), [(4.0, 6.0)]) == [(0.0, 4.0), (6.0, 10.0)]


def test_subtract_consuming_whole_interval_returns_nothing():
    assert subtract((4.0, 6.0), [(0.0, 10.0)]) == []


def test_subtract_ignores_non_overlapping_holes():
    assert subtract((0.0, 5.0), [(8.0, 9.0)]) == [(0.0, 5.0)]


def test_overlap_seconds_sums_only_the_covered_part():
    assert overlap_seconds((0.0, 10.0), [(1.0, 2.0), (5.0, 9.0), (20.0, 30.0)]) == pytest.approx(5.0)


def test_split_long_prefers_cutting_in_a_silence():
    speech = [(0.0, 5.0), (8.0, 12.0), (30.0, 35.0)]
    pieces = split_long((0.0, 35.0), 20.0, speech)
    assert len(pieces) == 2
    cut = pieces[0][1]
    # The silence between 12 and 30 is the only candidate at or below 20s.
    assert 12.0 < cut <= 20.0


def test_split_long_falls_back_to_hard_cut_without_silence():
    pieces = split_long((0.0, 30.0), 10.0, [(0.0, 30.0)])
    assert pieces == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_split_long_is_a_noop_when_under_the_limit():
    assert split_long((0.0, 5.0), 20.0, []) == [(0.0, 5.0)]


# -------------------------------------------------------------- override files

def test_load_overrides_parses_object_form(tmp_path):
    path = tmp_path / "a.overrides.json"
    path.write_text(json.dumps({
        "windows": [{"start": "00:12:30", "end": "00:13:45", "label": "intro"}]
    }))
    overrides = load_overrides(path, duration=3600)
    assert overrides == [Override(750.0, 825.0, "intro")]


def test_load_overrides_parses_bare_list(tmp_path):
    path = tmp_path / "a.overrides.json"
    path.write_text(json.dumps([{"start": 5, "end": 10}]))
    assert load_overrides(path, duration=60)[0].as_interval() == (5.0, 10.0)


def test_missing_override_file_is_not_an_error(tmp_path):
    assert load_overrides(tmp_path / "nope.json", duration=60) == []
    assert load_overrides(None, duration=60) == []


def test_override_past_end_is_clamped(tmp_path):
    path = tmp_path / "a.overrides.json"
    path.write_text(json.dumps([{"start": 50, "end": 900}]))
    assert load_overrides(path, duration=60)[0].end == 60.0


def test_override_starting_past_end_is_skipped(tmp_path):
    path = tmp_path / "a.overrides.json"
    path.write_text(json.dumps([{"start": 500, "end": 900}]))
    assert load_overrides(path, duration=60) == []


@pytest.mark.parametrize("payload", [
    [{"start": 10, "end": 5}],          # inverted
    [{"start": 10}],                    # missing end
    ["not-an-object"],
    {"windows": "not-a-list"},
])
def test_malformed_overrides_raise(tmp_path, payload):
    path = tmp_path / "a.overrides.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(OverrideError):
        load_overrides(path, duration=60)


def test_invalid_json_raises(tmp_path):
    path = tmp_path / "a.overrides.json"
    path.write_text("{ not json")
    with pytest.raises(OverrideError):
        load_overrides(path, duration=60)


# ------------------------------------------------------------------ build_clips

def test_build_clips_merges_the_three_second_gap():
    clips = build_clips([(10.0, 12.0), (15.0, 17.0)], [], cfg(), duration=60)
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end) == (10.0, 17.0)
    assert clips[0].origin == "vad"


def test_build_clips_skips_pure_silence():
    """Only detected speech becomes a clip; ambient noise is not emitted."""
    clips = build_clips([(10.0, 12.0)], [], cfg(), duration=600)
    assert len(clips) == 1
    assert clips[0].duration == pytest.approx(2.0)


def test_clips_are_indexed_in_time_order():
    clips = build_clips([(30.0, 31.0), (5.0, 6.0)], [], cfg(), duration=60)
    assert [c.index for c in clips] == [1, 2]
    assert clips[0].start < clips[1].start
    assert clips[0].stem == "clip_0001"


def test_override_is_emitted_verbatim_in_isolate_mode():
    overrides = [Override(100.0, 110.0, "keep-me")]
    clips = build_clips([(10.0, 12.0)], overrides, cfg(), duration=600)
    forced = [c for c in clips if c.origin == "override"]
    assert len(forced) == 1
    assert (forced[0].start, forced[0].end) == (100.0, 110.0)
    assert forced[0].labels == ["keep-me"]


def test_override_with_no_speech_still_produces_a_clip():
    """A window over pure silence must still be saved — and flagged, not dropped."""
    clips = build_clips([], [Override(100.0, 110.0)], cfg(), duration=600)
    assert len(clips) == 1
    assert clips[0].origin == "override"
    assert "no_vad_speech" in clips[0].flags


def test_override_is_carved_out_of_an_overlapping_vad_clip():
    """No audio should appear in two clips at once."""
    clips = build_clips([(0.0, 60.0)], [Override(20.0, 30.0)], cfg(), duration=60)
    spans = [(round(c.start, 3), round(c.end, 3), c.origin) for c in clips]
    assert spans == [(0.0, 20.0, "vad"), (20.0, 30.0, "override"), (30.0, 60.0, "vad")]
    total = sum(c.duration for c in clips)
    assert total == pytest.approx(60.0)


def test_override_swallowing_a_vad_clip_leaves_only_the_override():
    clips = build_clips([(21.0, 25.0)], [Override(20.0, 30.0)], cfg(), duration=60)
    assert len(clips) == 1
    assert clips[0].origin == "override"


def test_merge_mode_keeps_override_inside_a_larger_clip():
    clips = build_clips(
        [(10.0, 12.0)], [Override(14.0, 16.0, "x")], cfg(override_mode="merge"), duration=60
    )
    assert len(clips) == 1
    assert clips[0].start <= 14.0 and clips[0].end >= 16.0
    assert clips[0].origin == "vad+override"
    assert clips[0].labels == ["x"]


def test_padding_never_eats_into_an_override_window():
    clips = build_clips(
        [(19.0, 19.5)], [Override(20.0, 30.0)], cfg(pad=2.0), duration=60
    )
    forced = [c for c in clips if c.origin == "override"]
    assert (forced[0].start, forced[0].end) == (20.0, 30.0)


def test_short_clip_is_flagged_not_dropped():
    clips = build_clips([(10.0, 10.2)], [], cfg(), duration=60)
    assert len(clips) == 1
    assert "short" in clips[0].flags


def test_sparse_clip_is_flagged():
    """Greedy merging can bridge a lot of silence; say so, but keep the clip."""
    regions = [(0.0, 0.5), (5.0, 5.5), (10.0, 10.5)]
    clips = build_clips(regions, [], cfg(merge_gap=5.0), duration=60)
    assert len(clips) == 1
    assert clips[0].speech_seconds == pytest.approx(1.5)
    assert "sparse" in clips[0].flags


def test_dense_clip_is_not_flagged():
    clips = build_clips([(0.0, 9.0), (9.5, 10.0)], [], cfg(), duration=60)
    assert clips[0].flags == []


def test_nothing_is_ever_dropped_for_being_dubious():
    regions = [(1.0, 1.05), (30.0, 30.05), (50.0, 50.05)]
    clips = build_clips(regions, [], cfg(), duration=60)
    assert len(clips) == 3
    assert all("short" in c.flags for c in clips)


def test_summarise_reports_expected_totals():
    clips = build_clips([(0.0, 10.0), (30.0, 40.0)], [Override(50.0, 55.0)], cfg(), duration=60)
    stats = summarise(clips, 60.0)
    assert stats["clip_count"] == 3
    assert stats["override_clips"] == 1
    assert stats["clip_seconds"] == pytest.approx(25.0)
    assert stats["retained_fraction"] == pytest.approx(25.0 / 60.0, abs=1e-4)


def test_clip_dict_round_trips_the_fields_downstream_needs():
    clip = build_clips([(65.5, 70.0)], [], cfg(), duration=600)[0]
    data = clip.to_dict()
    assert data["start_hms"] == "00:01:05.500"
    assert data["stem"] == "clip_0001"
    assert set(data) >= {"index", "start", "end", "duration", "origin", "flags", "speech_ratio"}
