---
name: audua-pipeline
description: Process the audua inbox end to end — segment and transcribe every recording in processing/raw/, file each source under processed/ or failed/, then write a cited markdown summary for each one. Use when asked to process new recordings, work the inbox, run the pipeline, or transcribe what is waiting.
---

# audua: inbox to summaries

Five steps, in order. Step 4 is not optional — a batch without summaries is a
folder of clips nobody can find anything in, which is the whole problem this
pipeline exists to solve.

## 1. Look before you run

```bash
uv run python -c "from audua.batch import discover_inbox; from pathlib import Path; s, k = discover_inbox(Path('processing/raw')); print('to process:', [p.name for p in s]); print('skipped:', k)"
```

Tell the user what is queued. Transcription is slow — roughly real time on CPU
with `large-v3`, much faster on CUDA — so an inbox holding three hours of audio
is a commitment, not a quick command. If the inbox is empty, say so and stop.

## 2. Run the batch

Launch it fully detached, not with a plain `run_in_background: true` Bash call.
Transcription can run for hours, and a process left attached to this session's
shell gets SIGHUP'd and dies silently — no exception, no `aborted` field in the
report, it just vanishes — the moment this session ends or gets torn down.
This has happened repeatedly. Detach it so it survives independently:

```bash
touch /tmp/audua_batch_started
nohup uv run audua batch > processing/output/_batch.log 2>&1 < /dev/null & disown
```

Tell the user it is running and where the log is. To be notified once it
finishes without a manual polling loop, use `Bash` with `run_in_background:
true` on a short wait command *for the notification only* — the `audua batch`
process above is already independent of this wait command's fate, so it is
fine if this wait command itself gets cut off by a session ending:

```bash
until [ -f processing/output/_batches/latest.json ] && \
      [ processing/output/_batches/latest.json -nt /tmp/audua_batch_started ]; do
  sleep 5
done
```

The marker file, touched right before launch, is what makes "just finished"
distinguishable from "a report already sat there from a prior run". If this
session ends before the wait command fires, re-check
`processing/output/_batches/latest.json`'s `completed` timestamp or
`processing/output/_batch.log` in the next session to see whether it finished.

That one command does all of this:

- scans `processing/raw/` — **top level only**, so the archive folders are never
  reprocessed
- runs the full segment-and-transcribe pipeline on each file
- writes `transcript_digest.md` into each output folder
- moves each source to `processing/raw/processed/` or `processing/raw/failed/`
- writes a report to `processing/output/_batches/`

Never substitute `audua run processing/raw`. It recurses by design, so it would
drag every already-archived recording back through the pipeline.

Flags worth reaching for, all optional:

| Flag | When |
|---|---|
| `--pad 0.25` | Tighter lead-in/tail-out than the 0.5 default. |
| `--merge-gap SEC` | Clips splitting mid-thought (raise) or swallowing everything (lower). |
| `--model medium` | CPU-only machine where `large-v3` is too slow. |
| `--language en` | Known language: faster and more accurate than autodetect. |
| `--min-age 60` | Files may still be copying in from a recorder. |
| `--no-move` | Test run: process everything, leave the inbox untouched. |
| `--force` | Redo a source from scratch. Old output is superseded, not deleted. |

Pass through whatever the user asked for; otherwise take the defaults.

## 3. Read the report

```bash
cat processing/output/_batches/latest.json
```

Act on it in this order:

- **`aborted` is set** → ffmpeg or faster-whisper is missing. The batch stopped
  and every source is still in the inbox, untouched. Report the message, point
  at `INSTALL-windows.md` / `INSTALL-linux.md`, and stop. Do not retry.
- **`results[].status == "failed"`** → report `reasons` verbatim. The source is
  in `processing/raw/failed/`; the output folder and digest are still there to
  diagnose from. Do not re-run it without the user asking — the same failure
  will repeat.
- **`skipped`** → files that were not audio, or too fresh under `--min-age`.
  Mention them; they are still in the inbox.
- **`results[].summary_needed == true`** → step 4.

## 4. Write a summary for each processed source

For every result with `"summary_needed": true`, follow the **audua-summarize**
skill, working from that result's `digest` path. Do them one at a time.

## 5. Report back

Give the user, in a few lines:

- how many processed, failed, skipped
- for each processed source: the output folder, clip count, and a one-line sense
  of what is in it
- for each failure: the reason, and where the source now sits
- any clip flags worth knowing about (`sparse`, `low_confidence`, `repetitive`)
  — these are not failures, but they mark transcripts to trust less

Link the paths so they are clickable. Do not paste the report JSON.

## Re-running is safe

Every stage caches on fingerprints, so a re-run of an unchanged source does no
work. A source already in `processed/` is out of the inbox and will not be
picked up again; to redo one, move it back into `processing/raw/` and run the
batch with `--force`.
