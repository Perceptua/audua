---
name: audua-summarize
description: Write or refresh summary.md for a finished audua run — a detailed, densely cited markdown summary built from that run's transcript_digest.md, with numbered footnote citations that link back to each clip's transcript and audio. Use after audua batch, or when asked to summarize a recording, re-summarize a source, or fix a summary's citations.
---

# audua: write the summary

The summary is a **finding aid**, not an executive summary. Its reader already
made the recording; what they need is to locate the four minutes where they
talked about a particular thing, out of an hour of clips named `clip_0031.wav`.
Optimise for that, and err on the side of too much detail.

## 1. Read the digest

```bash
cat processing/output/<source>/transcript_digest.md
```

One file holds every clip: index, timestamps, files, flags, and the transcript
timestamped against the original recording. Read it whole. Do not read the
clips one at a time — the digest exists so you do not have to.

If `summary.md` already exists, do not overwrite it without being asked. Say it
exists and offer to refresh it.

## 2. Write `processing/output/<source>/summary.md`

Same folder as the clips, so every citation is a plain relative filename.

### Shape

```markdown
# 260819_0002

**01:02:07 · 44 clips · 00:21:09 of speech · transcribed with large-v3**

## What's in this recording

Two or three sentences of orientation. Where it seems to have been recorded,
who is talking, what it mostly is — a run, a drive, a reading session, notes
after a meeting.

## Morning run and the Melville thread (00:00–00:14)

Detailed prose. Every claim carries the clip it came from[^1]. Group by topic
rather than by clock time when the recording circles back to something[^4][^9].

## Metaphysics in Moby Dick (00:14–00:41)

Keep the speaker's own words for anything searchable — names, book titles,
technical terms, place names, the phrasing of a recurring idea[^7]. Those are
what they will type into a search box later.

## Threads left open

- Questions asked and not answered, ideas marked to come back to[^12].

## Clips not cited above

- **[38]** 00:52:10 — 40s of wind noise, nothing intelligible recovered.
- **[41]** 00:57:03 — repeats [37] almost word for word (`repetitive`).

[^1]: **00:00:10 – 00:01:06** · [transcript](clip_0001.txt) · [audio](clip_0001.wav) — "all right we're starting the watch now wait for it to acquire gps"
[^4]: **00:03:32 – 00:04:12** · [transcript](clip_0004.txt) · [audio](clip_0004.wav) — "the thing about the whale is that it refuses to be a symbol"
```

Rendered, `[^1]` is a superscript number and the definitions become a numbered
reference list at the foot of the page — the scientific-paper look, and it works
in GitHub, Obsidian, and VS Code preview alike.

### Citation rules

1. **The number is the clip index.** `[^7]` means clip 7 — `clip_0007` in the
   digest. Never renumber to make the citations read in order; the number is an
   address, not a footnote counter.
2. **Cite as you write, mid-sentence**, immediately after the claim it supports.
   Adjacent citations for one claim go side by side: `[^4][^5]`.
3. **Every definition, exactly once, at the bottom**, in ascending order:
   bold time range · transcript link · audio link · a short verbatim quote from
   that clip. The quote is what makes the reference list scannable — pick the
   most identifying line, not the first one.
4. **Use the filenames from the digest.** The audio extension follows the source
   (`.wav` here, `.m4a` elsewhere).
5. **Mark untrustworthy clips** in their definition: append
   `⚠ low_confidence`, `⚠ repetitive`, or `⚠ sparse` when the digest shows that
   flag. The reader needs to know the text may be wrong before they go looking.
6. **Cover everything.** Every clip carrying content gets cited at least once.
   Anything left — silence, noise, duplicates — goes under *Clips not cited
   above* with a one-line note. Nothing disappears without a trace.

### Writing rules

- **Never invent.** Everything traces to a transcript. Whisper garbles names and
  jargon: when a clip is plainly mangled, say what it seems to be and cite it
  anyway (`sounds like "Pequod", garbled[^22]`) rather than guessing confidently
  or dropping it.
- **Follow the speaker.** Solo voice memos read naturally in the speaker's own
  framing; keep their vocabulary verbatim.
- **Describe, don't interpret.** No psychoanalysing, no drawing conclusions the
  speaker did not draw.
- **Length: err long.** A dense hour deserves several hundred words and a dozen
  sections. Detail costs nothing here and missing detail costs a search.
- **Section headings carry their time range** — half of finding something is
  knowing roughly when it happened.

## 3. Verify before you report

```bash
uv run audua verify processing/output/<source>
```

This checks the clip/transcript pairing *and* every citation: that each `[^N]`
has a definition, that N is a clip in this run, and that every linked file is
really in the folder. Fix anything it reports, then run it again. Uncited clips
are listed too — if there are more than a couple, you probably owe them a line
under *Clips not cited above*.

Report to the user: the path to the summary, how many clips it cites, and
anything the recording contains that they may want to act on.
