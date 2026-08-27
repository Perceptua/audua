"""Command-line entry point.

    python -m audua run recording.m4a
    python -m audua run ./inbox --output ./output --merge-gap 8
    python -m audua plan recording.m4a          # segment preview, writes nothing
    python -m audua batch                       # process the processing/raw inbox
    python -m audua verify ./output/recording   # re-check 1:1 pairing on disk
    python -m audua ui                          # browse the filetree in a browser
    python -m audua ui --background             # same, detached; logs to <output>/_ui.log
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from .batch import BatchError, check_summary, run_inbox
from .config import Config, ConfigError
from .pipeline import PipelineError, discover_sources, process_file, verify_pairing
from .segments import OverrideError
from .transcribe import TranscriberUnavailable
from .ui import Roots, serve
from .vad import VadUnavailable


def _add_segmentation_args(sp: argparse.ArgumentParser) -> None:
    """Everything that decides *where clips are cut*. Shared by run, plan and batch."""
    group = sp.add_argument_group("clip formation")
    group.add_argument("--merge-gap", type=float, default=10.0, metavar="SEC",
                       help="Silences up to this long stay inside one clip. Default: 10")
    group.add_argument("--pad", type=float, default=0.5, metavar="SEC",
                       help="Lead-in/tail-out added to each clip. 0 disables. Default: 0.5")
    group.add_argument("--min-clip-duration", type=float, default=0.5, metavar="SEC",
                       help="Clips shorter than this are flagged, never dropped. Default: 0.5")
    group.add_argument("--max-clip-duration", type=float, default=0.0, metavar="SEC",
                       help="Split clips longer than this at an internal silence. "
                            "0 = no limit. Default: 0")
    group.add_argument("--sparse-ratio", type=float, default=0.35,
                       help="Speech fraction below which a clip is flagged 'sparse'. "
                            "Default: 0.35")
    group.add_argument("--override-mode", choices=("isolate", "merge"), default="isolate",
                       help="isolate: emit override windows verbatim as their own clips. "
                            "merge: fold them into surrounding speech. Default: isolate")

    group = sp.add_argument_group("voice activity detection")
    group.add_argument("--vad-threshold", type=float, default=0.35,
                       help="Silero speech probability. Lower keeps more. Default: 0.35")
    group.add_argument("--min-speech-ms", type=int, default=150)
    group.add_argument("--min-silence-ms", type=int, default=400)
    group.add_argument("--speech-pad-ms", type=int, default=150)
    group.add_argument("--vad-window", type=float, default=600.0, metavar="SEC",
                       help="Decode window size; bounds memory on long files. Default: 600")
    group.add_argument("--vad-highpass", type=float, default=100.0, metavar="HZ",
                       help="High-pass filter applied before VAD detection only (saved "
                            "clips are untouched). Cuts wind/handling rumble that can "
                            "otherwise suppress speech probability to zero. 0 disables. "
                            "Default: 100")
    group.add_argument("--vad-denoise", action=argparse.BooleanOptionalAction, default=True,
                       help="FFT noise reduction before VAD detection only. Default: on")


def _add_transcription_args(sp: argparse.ArgumentParser, *, segment_only: bool) -> None:
    """Whisper settings. ``batch`` omits --segment-only: transcripts are the point."""
    group = sp.add_argument_group("transcription")
    group.add_argument("--model", default="large-v3", help="faster-whisper model. Default: large-v3")
    group.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    group.add_argument("--compute-type", default="auto",
                       help="e.g. int8, float16, int8_float16. Default: auto")
    group.add_argument("--language", default=None,
                       help="Force a language code instead of autodetecting per clip.")
    group.add_argument("--beam-size", type=int, default=5)
    if segment_only:
        group.add_argument("--segment-only", action="store_true",
                           help="Cut clips but skip transcription.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audua",
        description="Segment sporadic speech out of long recordings and transcribe it.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Segment and transcribe.")
    plan = sub.add_parser("plan", help="Show the clips that would be cut. Writes only plan.json.")
    batch = sub.add_parser(
        "batch",
        help="Work an inbox end to end: run every file in it, write a transcript digest, "
             "and file each source under processed/ or failed/.",
    )

    for sp in (run, plan):
        sp.add_argument("input", type=Path, help="Audio file, or a directory of them.")
        sp.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="Output root. Default: ./output")
        sp.add_argument("--overrides", type=Path, default=None,
                        help="Override sidecar. Default: <source_stem>.overrides.json "
                             "next to the audio.")
        sp.add_argument("--no-recursive", action="store_true",
                        help="When given a directory, do not descend into subfolders.")

    group = batch.add_argument_group("inbox")
    group.add_argument("--raw", type=Path, default=Path("processing/raw"),
                       help="Inbox folder, scanned top level only. Default: processing/raw")
    group.add_argument("-o", "--output", type=Path, default=Path("processing/output"),
                       help="Output root. Default: processing/output")
    group.add_argument("--processed-dir", type=Path, default=None,
                       help="Where finished sources are filed. Default: <raw>/processed")
    group.add_argument("--failed-dir", type=Path, default=None,
                       help="Where problem sources are filed. Default: <raw>/failed")
    group.add_argument("--no-move", action="store_true",
                       help="Process everything but leave the sources in the inbox.")
    group.add_argument("--min-age", type=float, default=0.0, metavar="SEC",
                       help="Skip files modified more recently than this — they may still "
                            "be copying in. Default: 0 (take everything)")
    group.add_argument("--report", type=Path, default=None,
                       help="Where to write the batch report. "
                            "Default: <output>/_batches/batch_<timestamp>.json")

    for sp in (run, plan, batch):
        _add_segmentation_args(sp)
    _add_transcription_args(run, segment_only=True)
    _add_transcription_args(batch, segment_only=False)

    for sp in (run, batch):
        sp.add_argument("--force", action="store_true",
                        help="Redo everything, ignoring cached work. Old outputs are moved "
                             "to a _superseded_* folder, not deleted.")

    verify = sub.add_parser("verify", help="Re-check clip/transcript pairing for a finished run.")
    verify.add_argument("directory", type=Path, help="A per-source output folder.")

    ui = sub.add_parser(
        "ui",
        help="Browse the filetree in a browser: what is waiting, what ran, and what it said.",
    )
    ui.add_argument("--raw", type=Path, default=Path("processing/raw"),
                    help="Inbox folder. Default: processing/raw")
    ui.add_argument("-o", "--output", type=Path, default=Path("processing/output"),
                    help="Output root. Default: processing/output")
    ui.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind. Default: 127.0.0.1 (this machine only). "
                         "Clip audio is personal data; widen this deliberately.")
    ui.add_argument("--port", type=int, default=8765,
                    help="Port to bind. 0 picks a free one. Default: 8765")
    ui.add_argument("--no-browser", action="store_true",
                    help="Do not open a browser window on start.")
    ui.add_argument("-b", "--background", action="store_true",
                    help="Start detached and return immediately. Logs to "
                         "<output>/_ui.log.")

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config(
        output_root=args.output,
        overrides_path=getattr(args, "overrides", None),
        vad_threshold=args.vad_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        vad_window_seconds=args.vad_window,
        vad_highpass_hz=args.vad_highpass,
        vad_denoise=args.vad_denoise,
        merge_gap=args.merge_gap,
        pad=args.pad,
        min_clip_duration=args.min_clip_duration,
        max_clip_duration=args.max_clip_duration,
        sparse_ratio=args.sparse_ratio,
        override_mode=args.override_mode,
        verbose=args.verbose,
    )
    if args.command in {"run", "batch"}:
        config.model = args.model
        config.device = args.device
        config.compute_type = args.compute_type
        config.language = args.language
        config.beam_size = args.beam_size
        config.segment_only = getattr(args, "segment_only", False)
        config.force = args.force
    else:
        config.dry_run = True
    config.validate()
    return config


def _verify_command(directory: Path) -> int:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest.json in {directory}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transcribing = not manifest.get("config", {}).get("segment_only", False)
    result = verify_pairing(directory, manifest.get("clips", []), transcribing=transcribing)
    summary = check_summary(directory, manifest)
    exit_code = 0

    if result["ok"]:
        print(f"OK — {len(manifest.get('clips', []))} clip(s), pairing intact.")
    else:
        print("Pairing problems found:", file=sys.stderr)
        for problem in result["problems"]:
            print(f"  — {problem}", file=sys.stderr)
        exit_code = 1

    # A citation is a promise that a named clip says a particular thing. The
    # numbers are written by hand, so they get checked rather than trusted.
    if summary["present"]:
        if summary["ok"]:
            print(f"summary.md OK — {len(summary['cited'])} clip(s) cited.")
        else:
            print("Summary citation problems found:", file=sys.stderr)
            for problem in summary["problems"]:
                print(f"  — {problem}", file=sys.stderr)
            exit_code = 1
        if summary["uncited"]:
            print(
                "  uncited clip(s): "
                + ", ".join(str(index) for index in summary["uncited"])
            )

    return exit_code


def _ui_command(args: argparse.Namespace) -> int:
    """Serve the read-only UI. Needs no Config — it runs nothing, it only reads."""
    roots = Roots.resolved(args.raw, args.output)
    if not roots.raw.is_dir() and not roots.output.is_dir():
        print(
            f"error: neither {args.raw} nor {args.output} exists. Point --raw and "
            f"--output at an audua tree, or run `audua batch` to create one.",
            file=sys.stderr,
        )
        return 2
    if args.background:
        return _ui_background(args, roots)
    return serve(roots, host=args.host, port=args.port, open_browser=not args.no_browser)


def _ui_background(args: argparse.Namespace, roots: Roots) -> int:
    """Relaunch ``ui`` as a detached child and return immediately.

    Detaching needs different flags per platform, and neither Popen argument
    is accepted on the other OS, so the two are built separately rather than
    unified.
    """
    roots.output.mkdir(parents=True, exist_ok=True)
    log_path = roots.output / "_ui.log"

    command = [
        sys.executable, "-m", "audua", "ui",
        "--raw", str(args.raw),
        "--output", str(args.output),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.no_browser:
        command.append("--no-browser")

    detach: dict = {}
    if sys.platform == "win32":
        detach["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        detach["start_new_session"] = True

    with open(log_path, "ab") as log_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **detach,
        )

    shown = "localhost" if args.host in {"127.0.0.1", "0.0.0.0", "::1"} else args.host
    print(f"audua UI starting in background (pid {process.pid}).")
    if args.port:
        print(f"  url:  http://{shown}:{args.port}/")
    else:
        print("  url:  port 0 auto-assigns — check the log for the bound port.")
    print(f"  logs: {log_path}")
    return 0


def _batch_command(args: argparse.Namespace, config: Config) -> int:
    try:
        report = run_inbox(
            config,
            raw_dir=args.raw,
            processed_dir=args.processed_dir,
            failed_dir=args.failed_dir,
            move=not args.no_move,
            min_age=args.min_age,
            report_path=args.report,
        )
    except BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    counts = report["counts"]
    print()
    if not counts["total"]:
        print(f"Inbox {args.raw} is empty — nothing to do.")
    else:
        print(
            f"Batch: {counts['processed']} processed, {counts['failed']} failed, "
            f"{counts['skipped']} skipped."
        )
    for result in report["results"]:
        detail = (
            "; ".join(result["reasons"]) if result["reasons"]
            else f"{result.get('clip_count', 0)} clips, {result.get('flagged_clips', 0)} flagged"
        )
        print(f"  {result['status']:<10} {result['source']}  ({detail})")
    for skipped in report["skipped"]:
        print(f"  {'skipped':<10} {skipped['source']}  ({skipped['reason']})")

    print(f"\nReport: {report['report_path']}")
    if counts["summaries_pending"]:
        print(
            f"{counts['summaries_pending']} summary/summaries pending — "
            "run the audua-summarize skill, or ask Claude to write summary.md "
            "from each transcript_digest.md."
        )

    if report.get("aborted"):
        print(f"\nerror: {report['aborted']}", file=sys.stderr)
        return 2
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "verify":
        return _verify_command(args.directory)

    if args.command == "ui":
        return _ui_command(args)

    try:
        config = _config_from_args(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "batch":
        return _batch_command(args, config)

    try:
        sources = discover_sources(args.input, recursive=not args.no_recursive)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    logging.info("Processing %d file(s).", len(sources))

    exit_code = 0
    transcriber = None
    for source in sources:
        try:
            manifest = process_file(source, config, transcriber=transcriber)
            # Reuse the loaded model across files instead of paying the load cost
            # once per recording.
            if transcriber is None and not config.segment_only and not config.dry_run:
                from .transcribe import Transcriber

                transcriber = transcriber or Transcriber(config)
            if manifest.get("counts", {}).get("failed"):
                exit_code = 1
            if not manifest.get("pairing", {}).get("ok", True):
                exit_code = 1
        except (VadUnavailable, TranscriberUnavailable) as exc:
            # A missing dependency will fail identically on every remaining
            # file, so stop rather than emit the same traceback N times.
            print(f"\nerror: {exc}", file=sys.stderr)
            return 2
        except (OverrideError, PipelineError) as exc:
            logging.error("%s: %s", source.name, exc)
            exit_code = 1
        except Exception as exc:  # keep going through a batch
            logging.exception("%s: unexpected failure: %s", source.name, exc)
            exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
