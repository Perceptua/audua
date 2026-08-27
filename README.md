# audua — Stage 1 pre-processing pipeline

Turns long recordings that are mostly silence into clips of speech, each paired
with its transcript. Fully local: no audio leaves the machine, no per-minute
cost, no LLM tokens spent on raw audio.

```bash
uv sync                          # once
uv run audua run recording.m4a
```

```
output/recording/
  manifest.json
  vad_cache.json
  clip_0001.m4a   clip_0001.txt   clip_0001.json
  clip_0002.m4a   clip_0002.txt   clip_0002.json
```

---

## How it decides what to keep

**1. Detect.** Silero VAD runs over the file in 10-minute windows, so a
four-hour recording uses the same memory as a four-minute one. It reports the
spans where someone is speaking.

**2. Merge, greedily.** Speech separated by a short silence is *one* clip, not
two. The default absorbs gaps up to 10 seconds, so `speech → 3s silence →
speech` produces a single clip. Tune with `--merge-gap`.

**3. Pad.** Each clip gets 0.5s of lead-in and tail-out, because VAD
boundaries tend to shave word onsets. `--pad 0` disables it. Override windows
are never padded — you asked for exact bounds.

**4. Cut.** Clips are stream-copied with ffmpeg in the source codec, so there
is no re-encode and no generational loss. Cut points snap to the nearest codec
frame boundary, and they always snap *outward* — a clip may be ~50ms longer
than requested, never shorter.

**5. Transcribe.** Each clip goes through faster-whisper individually, writing
a `.txt` and a `.json` beside it.

Long silences between bursts are never written out. What is written is
everything with a plausible claim to containing speech.

---

## Nothing is ever deleted

Clips that look dubious are **flagged in the sidecar, not dropped**. The flags:

| Flag | Meaning |
|---|---|
| `short` | Below `--min-clip-duration` (default 0.5s). A single word is still speech. |
| `sparse` | Greedy merging bridged a lot of silence; speech is under 35% of the clip. |
| `no_vad_speech` | The VAD heard nothing here. Only expected on override windows. |
| `empty_transcript` | Whisper returned no text. The clip and an empty `.txt` are both kept. |
| `low_confidence` | Mean `avg_logprob` below `-1.0`. |
| `high_no_speech` | Mean `no_speech_prob` above `0.6`. |
| `repetitive` | Compression ratio over 2.4 — the classic Whisper looping failure. |

Filtering on these is a Stage 2 decision, made against a complete record.

To find every clip the pipeline was unsure about:

```bash
python - <<'EOF'
import json, pathlib
for m in pathlib.Path("output").glob("*/manifest.json"):
    for clip in json.loads(m.read_text())["clips"]:
        if clip.get("flags"):
            print(m.parent.name, clip["stem"], clip["start_hms"], clip["flags"])
EOF
```

---

## Time-window overrides

A window listed in the override sidecar is **always** saved as a clip, whether
or not the VAD heard anything there. Name the file after the audio and put it
alongside:

```
recordings/session_one.m4a
recordings/session_one.overrides.json
```

```json
{
  "windows": [
    { "start": "00:12:30", "end": "00:13:45", "label": "quiet-passage" },
    { "start": 3600, "end": 3665.5 }
  ]
}
```

Times accept seconds (`12.5`), `MM:SS`, or `HH:MM:SS.mmm`. Point at a different
file with `--overrides path/to/file.json`. A full example is in
`examples/session_one.overrides.json`.

**Two modes.** Default `isolate` emits the window verbatim as its own clip and
carves it out of any overlapping VAD clip, so no audio appears twice and you
get exactly the bounds you asked for. `--override-mode merge` instead folds the
window into the surrounding speech — the window is still guaranteed to be
inside a clip, but the clip may extend past it.

The invariant is asserted in code: if a requested window is ever not fully
contained in an emitted clip, segmentation fails loudly rather than silently
losing it.

---

## Re-running is cheap and safe

Every stage caches. Re-running an unchanged file does no work at all.

- **Interrupted partway?** Run the same command again. Finished clips are
  reused; it resumes at the one it was on.
- **Changed `--merge-gap`?** Clip boundaries move, so old outputs no longer
  match. They are moved into `output/<name>/_superseded_<timestamp>/`, never
  deleted. The VAD cache survives — re-cutting does not mean re-listening.
- **Changed `--model`?** Only transcription settings changed, so the clips
  stand. (Use `--force` to redo everything.)
- **Want to check progress without re-running?** Read `manifest.json`. It is
  rewritten after every clip and records the status of each one.

The `manifest.json` `pairing` block reports the 1:1 invariant, checked in both
directions: every clip has its `.txt` and `.json`, and every file in the folder
belongs to a clip in the manifest. Re-check an old run any time:

```bash
uv run audua verify output/recording
```

---

## Commands

```bash
# One file, or a whole folder (recurses by default)
uv run audua run recording.m4a
uv run audua run ./recordings --output ./output

# Preview the cut without writing clips - useful for tuning
uv run audua plan recording.m4a
uv run audua plan recording.m4a --merge-gap 10 --pad 0.5

# Cut clips but skip transcription
uv run audua run recording.m4a --segment-only

# Re-check pairing on a finished run, and any summary's citations
uv run audua verify output/recording

# Work an inbox: run everything in it, then file each source
uv run audua batch

# Browse the whole tree in a browser: status, clips, summaries
uv run audua ui
```

Inside an already-activated venv, `audua ...` and `python -m audua ...` both
work identically.

### Options worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--merge-gap SEC` | `10` | Silence absorbed into one clip. The greediness dial. |
| `--pad SEC` | `0.5` | Lead-in/tail-out. `0` disables. |
| `--vad-threshold` | `0.35` | Lower keeps more. Raise if noise is being picked up as speech. |
| `--vad-highpass HZ` | `100` | High-pass filter before VAD detection only — saved clips are untouched. Cuts wind/handling rumble. `0` disables. |
| `--vad-denoise` | on | FFT noise reduction before VAD detection only. `--no-vad-denoise` disables. |
| `--min-clip-duration` | `0.5` | Flag threshold only; never drops anything. |
| `--max-clip-duration` | `0` (off) | Splits over-long clips, preferring a silence. |
| `--override-mode` | `isolate` | `isolate` or `merge`. |
| `--model` | `large-v3` | Any faster-whisper model. `medium` or `small` on CPU-only boxes. |
| `--device` | `auto` | `auto` picks cuda when available, else cpu. |
| `--language` | autodetect | Set it if you know it — faster and more accurate. |
| `--force` | off | Redo everything. Old output is superseded, not deleted. |

### Tuning

Use `uv run audua plan` — it is fast and writes nothing but `plan.json`.

- **Clips are splitting mid-thought** → raise `--merge-gap`.
- **One clip swallowed the whole recording** → lower `--merge-gap`, or set
  `--max-clip-duration`.
- **Background noise is becoming clips** → raise `--vad-threshold` toward 0.5.
  Check what you lose; the defaults deliberately lean permissive.
- **Word onsets are clipped** → raise `--pad`. If it's still not enough, also
  try raising `--speech-pad-ms` (Silero's own padding, applied before merge —
  affects where a region is considered to start/end in the first place,
  rather than padding the clip after the fact).
- **Real speech is missing entirely, and the recording has wind/handling
  noise** → this is not a `--vad-threshold` problem. Wind can push Silero's
  speech probability to zero regardless of how loud or clear the speech
  underneath actually is, so lowering the threshold does nothing. By default,
  VAD's detection pass (never the saved clip audio) runs through a highpass
  filter and FFT denoise first specifically to counter this; if speech is
  still going undetected in heavy wind, try raising `--vad-highpass` (e.g.
  150-200) rather than touching `--vad-threshold`.

---

## Working an inbox

`audua run` processes what you point it at. `audua batch` works a folder as a
queue: everything in `processing/raw/` goes through the pipeline, and each
source is then filed under `processed/` or `failed/` so the inbox drains.

```bash
uv run audua batch
```

```
processing/
  raw/                       <- drop recordings here
    processed/260819_0002.wav
    failed/
  output/
    260819_0002/
      clip_0001.wav  clip_0001.txt  clip_0001.json   ...
      manifest.json
      transcript_digest.md       <- every clip, timestamped, in one file
      summary.md                 <- written from the digest, with citations
    _batches/
      batch_20260821-135536.json
      latest.json                <- always the most recent run
```

The inbox is scanned **top level only**. `audua run processing/raw` would
recurse and drag the whole archive back through the pipeline; `batch` will not.

**What sends a source to `failed/`:** a crash, unreadable audio, a clip that
failed to extract or transcribe, a broken clip/transcript pairing, or no clips
at all. Quality flags do not — a `sparse` or `low_confidence` clip is a result,
not a failure, and filtering on flags stays a downstream decision. Failed
sources keep their output folder and digest, so you can see what went wrong
without re-running an hour of audio.

Nothing is overwritten. A source whose name already exists in the archive is
filed as `name-2.wav`, and an `.overrides.json` sidecar moves with its audio.

| Flag | Default | Notes |
|---|---|---|
| `--raw DIR` | `processing/raw` | The inbox. |
| `-o, --output DIR` | `processing/output` | Output root. |
| `--no-move` | off | Process everything, leave the inbox untouched. |
| `--min-age SEC` | `0` | Skip files modified more recently than this — they may still be copying in. |
| `--report PATH` | `<output>/_batches/batch_<stamp>.json` | Where the run report goes. |

Every `run` flag works here too (`--merge-gap`, `--pad`, `--model`, `--force`, ...).

### The digest, and the summary

`transcript_digest.md` is written by the batch: every clip, with its index,
bounds, flags, files, and transcript text timestamped against the *original*
recording rather than against the clip. One file per source, so finding a moment
is one read and one search rather than forty.

`summary.md` is written from that digest by Claude, via the two skills in
`.claude/skills/`:

- **audua-pipeline** — the whole flow: check the inbox, run the batch, read the
  report, then summarize each source that finished.
- **audua-summarize** — write or refresh one `summary.md`, with the citation
  format spelled out.

Citations are markdown footnotes numbered by clip index, so they render as
superscripts with a reference list at the foot of the page:

```markdown
An easy run recorded as a proof of concept[^1], with the cart frame
cost problem worked through mid-run[^14][^40].

[^1]: **00:00:10 – 00:01:06** · [transcript](clip_0001.txt) · [audio](clip_0001.wav) — "getting ready for my first easy run wearing the TASCAM"
```

The number *is* the clip index — `[^14]` means `clip_0014` — so a citation
is an address, not a footnote counter. `audua verify` checks them: every
citation has a definition, every number matches a clip in that run, and every
linked file is really in the folder. It also lists clips nothing cited, which is
usually the more useful signal.

```bash
uv run audua verify processing/output/260819_0002
```

```
OK — 44 clip(s), pairing intact.
summary.md OK — 43 clip(s) cited.
  uncited clip(s): 24
```

---

## Browsing the tree

`audua ui` serves the filetree on localhost and opens a browser. Nothing extra
to install — it is standard-library only, and the browser is the one media
player and markdown viewer already present on both Windows and Linux.

```bash
uv run audua ui
```

**Dashboard** — what is waiting in the inbox, what has been processed or
failed, how many runs still owe a summary, and what the last batch did.

**Processing** — every raw recording in the tree with its status: `ready`
while it is still sitting in the inbox, then `processed` or `failed` once the
batch has filed it. A failure carries the reason the batch recorded, so the
answer to "why is this in `failed/`" is on the screen rather than in a JSON
report.

**Outputs** — one line per run, taken from the first sentence of its
`summary.md`, so the recording you meant is findable without opening five that
you didn't.

**Output detail** — the summary rendered beside the clip list, with the same
checks `audua verify` runs, run live against the folder: clip/transcript
pairing, whether every citation resolves, and which clips nothing cites.

Clips play in the page and transcripts render in it. That includes from inside
a citation — `[audio]` starts the clip, `[transcript]` opens its text beside
the prose — because a finding aid you have to leave in order to follow is not
much of one.

The split is yours to set: drag the divider (arrow keys nudge it, double-click
resets it), or collapse the reading pane to a rail down the right edge and
click it to bring the document back. Both the width and the collapsed state are
remembered. Opening a citation expands the pane, because asking to see
something should show it; arriving at another run does not, because that would
undo a collapse you meant.

The UI is **read-only**. Only GET is routed, so nothing in it can alter a
recording, and it binds to `127.0.0.1` because clip audio is personal data.

| Flag | Default | Notes |
|---|---|---|
| `--raw DIR` | `processing/raw` | The inbox. |
| `-o, --output DIR` | `processing/output` | Output root. |
| `--host` | `127.0.0.1` | This machine only. Widen deliberately. |
| `--port` | `8765` | `0` picks a free one. |
| `--no-browser` | off | Do not open a browser window on start. |
| `-b, --background` | off | Start detached and return immediately. Logs to `<output>/_ui.log`. |

---

## Output format

`clip_0007.json` alongside `clip_0007.m4a` and `clip_0007.txt`:

```jsonc
{
  "schema": "audua/clip/1",
  "clip": {
    "index": 7, "stem": "clip_0007",
    "start": 754.25, "end": 771.5, "duration": 17.25,
    "start_hms": "00:12:34.250",
    "origin": "vad",              // "vad" | "override" | "vad+override"
    "labels": [],
    "speech_seconds": 12.1, "speech_ratio": 0.7014,
    "flags": []
  },
  "audio_file": "clip_0007.m4a",
  "text_file": "clip_0007.txt",
  "transcript": {
    "text": "...",
    "language": "en",
    "segments": [
      {
        "start": 0.0, "end": 4.2,        // within the clip
        "global_start": 754.25,           // within the original recording
        "global_start_hms": "00:12:34.250",
        "text": "...",
        "avg_logprob": -0.21, "no_speech_prob": 0.01
      }
    ]
  },
  "extraction": { "mode": "copy", "actual_duration": 17.28 },
  "flags": []
}
```

Every segment carries `global_start` / `global_end`, so a Stage 2 workflow can
locate a quote in the original audio without consulting the manifest.

---

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). Two things are
needed: uv itself, and ffmpeg (a system binary, not a Python package).

- **Linux:** [INSTALL-linux.md](INSTALL-linux.md)
- **Windows:** [INSTALL-windows.md](INSTALL-windows.md)

Then, from the project root:

```bash
uv sync
```

That creates `.venv`, installs the pipeline and its dev tools, and pins
everything in `uv.lock`. There is no separate "activate the venv" step — `uv
run` handles it:

```bash
uv run audua run recording.m4a
uv run audua --help
```

### torch is pinned to CPU wheels

`silero-vad` depends on torch, and torch's default PyPI wheels bundle CUDA —
several GB. This pipeline only uses torch to run Silero VAD, a few megabytes of
model, so `pyproject.toml` points torch at PyTorch's CPU index. Install drops to
a few hundred MB.

On a machine with an NVIDIA GPU you want the CUDA build instead, since
faster-whisper is the part that benefits:

```bash
uv sync --extra-index-url https://download.pytorch.org/whl/cu124
```

Or delete the `[tool.uv.sources]` block to always take whatever PyPI serves.

Note that torch is declared as a direct dependency even though it arrives
transitively. `[tool.uv.sources]` is only honoured for direct dependencies, so
without that line the CPU pin would be silently ignored.

The core pipeline is identical on both platforms — `pathlib` throughout, no
shell invocations, no hardcoded separators. Only the install steps differ.

---

## Development

```bash
uv sync                  # install, including dev tools
uv run pytest            # 145 tests, a couple of minutes
uv run ruff check .      # lint
uv run ruff format .     # format
uv add <package>         # add a dependency (updates pyproject + lock)
uv lock --upgrade        # refresh the lock
```

`src/` layout: the package lives at `src/audua/`, so tests import the
*installed* package rather than a stray copy on `sys.path`. Packaging mistakes
surface in the test run instead of after someone installs it elsewhere.

The VAD model and Whisper are stubbed in the suite, so it runs in seconds
anywhere; everything else is real — ffmpeg genuinely decodes and cuts audio,
and resume/idempotency are exercised by running the pipeline twice over a
synthetic recording built to match the target shape (bursts at 5-7s, 10-12s
and 40-41s inside 60s of near-silence).

If you are somewhere PyPI is unreachable and cannot install pytest at all, a
bundled shim runs the same test files with no dependencies:

```bash
python tools/run_tests.py
```

---

## Design notes

**Why VAD before Whisper, when Whisper has its own VAD?** Whisper's filter
decides what to *transcribe*; it does not decide what to *keep on disk*. Doing
detection separately means the clip boundaries are inspectable, tunable, and
cached independently of the model. Whisper's internal VAD is explicitly
disabled at transcribe time so it cannot discard audio the pipeline chose to
keep.

**Why stream copy instead of decoding to WAV?** No re-encode, no generational
loss, and the clip is byte-identical to the source audio. faster-whisper
decodes it at transcribe time anyway, so nothing is gained by normalising the
format up front.

**Why is the sidecar the completion marker?** It is written last. A crash
between the audio and the transcript leaves a clip with no sidecar, which the
next run detects as unfinished and redoes — instead of a half-finished clip
that looks done.

**Why flag instead of filter?** Because the pipeline cannot know what a
downstream workflow considers signal. Discarding is irreversible; flagging is
not. Stage 2 can filter on `flags` with the full record in front of it.
