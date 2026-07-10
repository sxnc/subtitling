#!/usr/bin/env python3
"""
optimize_subs.py — Deterministic parse/write helper for subtitle optimization.

This script does NOT call any API and needs no API key. It is a structural tool
used by a Claude Code instance: the model reads the events, optimizes the text
itself, and writes the result back. The script's only jobs are to (1) parse an
.srt/.vtt file into indexed events and (2) re-attach the ORIGINAL timestamps to
optimized text, so timing can never be corrupted.

Three subcommands:

    # 1. Dump events as JSON (for the model to read)
    python optimize_subs.py extract "video.srt"

    # 2. Apply optimized text (JSON list of {index, text}) and write the output.
    #    Edits come from a file path, or "-" for stdin.
    #    Optionally close small gaps between events in the same pass, and/or
    #    enforce Swiss Standard German orthography (ß -> ss).
    python optimize_subs.py apply "video.srt" edits.json
    python optimize_subs.py apply "video.srt" -   < edits.json
    python optimize_subs.py apply "video.srt" edits.json --close-gaps
    python optimize_subs.py apply "video.srt" edits.json --swiss

    # 3. Only close small gaps (no text changes).
    python optimize_subs.py close-gaps "video.srt" --max-gap 1.5

`apply` and `close-gaps` write <name>_optimized.<ext> beside the input and print
its path. The input file is never modified. Event count and indices are always
preserved. Timestamps are preserved too, EXCEPT that gap-closing (opt-in) may
extend an event's END time forward to touch the next event's start — it never
moves a start, never shortens an event, and never creates an overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Gaps at or below this many seconds are treated as incidental (STT flicker,
# short breaths) and closed; anything larger is kept as a deliberate pause —
# a title card, scene change, or dramatic beat. Aligns with the "~1.5s pause"
# threshold in the optimization rules.
MAX_GAP_DEFAULT = 1.5

# Swiss Standard German (Schweizer Hochdeutsch) has no eszett: every ß is
# written ss, and the capital ẞ likewise becomes SS. This holds in every
# position, including proper nouns and loanwords, so it is safe to do as a
# blind character substitution.
SWISS_SUBSTITUTIONS = {"ß": "ss", "ẞ": "SS"}


def to_swiss(text: str) -> str:
    for src, dst in SWISS_SUBSTITUTIONS.items():
        text = text.replace(src, dst)
    return text

try:
    import srt as srt_lib
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency 'srt'. Run: pip install -r requirements.txt")

try:
    import webvtt
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency 'webvtt-py'. Run: pip install -r requirements.txt")


@dataclass
class Event:
    index: int            # 0-based global index
    start: str            # timestamp string, as in the source format
    end: str              # timestamp string, as in the source format
    start_seconds: float  # start time in seconds (for pause reasoning)
    gap_before: float     # seconds between the previous event's end and this start
    text: str


# --- timestamp helpers ------------------------------------------------------
def _fmt_td(td) -> str:
    total = td.total_seconds()
    hours, rem = divmod(int(total), 3600)
    minutes, seconds = divmod(rem, 60)
    millis = int(round((total - int(total)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            return 0.0
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        return 0.0


def _with_gaps(rows) -> list[Event]:
    """rows: list of (start_str, end_str, text). Compute gap_before for each."""
    events: list[Event] = []
    prev_end = None
    for i, (start, end, text) in enumerate(rows):
        s = _ts_to_seconds(start)
        gap = 0.0 if prev_end is None else max(0.0, s - prev_end)
        events.append(
            Event(index=i, start=start, end=end, start_seconds=s,
                  gap_before=round(gap, 3), text=text)
        )
        prev_end = _ts_to_seconds(end)
    return events


# --- format-specific load / save --------------------------------------------
def load_srt(path: Path):
    subs = list(srt_lib.parse(path.read_text(encoding="utf-8-sig")))
    if not subs:
        raise ValueError("No subtitle events were found in the file.")
    rows = [(_fmt_td(s.start), _fmt_td(s.end), s.content) for s in subs]
    return _with_gaps(rows), subs


def save_srt(subs, events, out_path: Path):
    for sub, ev in zip(subs, events):
        sub.content = ev.text
    out_path.write_text(srt_lib.compose(subs), encoding="utf-8")


def load_vtt(path: Path):
    vtt = webvtt.read(str(path))
    caps = list(vtt.iter_slice())
    if not caps:
        raise ValueError("No subtitle events were found in the file.")
    rows = [(c.start, c.end, c.text) for c in caps]
    return _with_gaps(rows), vtt


def save_vtt(vtt, events, out_path: Path):
    for cap, ev in zip(vtt.iter_slice(), events):
        cap.text = ev.text
    vtt.save(str(out_path))


# --- gap closing ------------------------------------------------------------
def _close_gaps_srt(subs, max_gap: float) -> int:
    """Extend each event's end to the next event's start when the gap between
    them is >0 and <= max_gap. subs[i].start/.end are datetime.timedelta."""
    closed = 0
    for cur, nxt in zip(subs, subs[1:]):
        gap = (nxt.start - cur.end).total_seconds()
        if 0 < gap <= max_gap:
            cur.end = nxt.start
            closed += 1
    return closed


def _close_gaps_vtt(caps, max_gap: float) -> int:
    """Same as SRT, but webvtt captions carry timestamps as strings; use the
    computed *_in_seconds for the gap and copy the next start string across."""
    caps = list(caps)
    closed = 0
    for cur, nxt in zip(caps, caps[1:]):
        gap = nxt.start_in_seconds - cur.end_in_seconds
        if 0 < gap <= max_gap:
            cur.end = nxt.start
            closed += 1
    return closed


def close_gaps(fmt: str, native, max_gap: float) -> int:
    """Close small gaps in-place on the native container. Returns the count."""
    if fmt == "srt":
        return _close_gaps_srt(native, max_gap)
    return _close_gaps_vtt(native.iter_slice(), max_gap)


def load(path: Path):
    ext = path.suffix.lower()
    if ext == ".srt":
        return "srt", *load_srt(path)
    if ext == ".vtt":
        return "vtt", *load_vtt(path)
    raise ValueError(f"unsupported file type '{ext}'. Expected .srt or .vtt.")


def save(fmt: str, native, events, out_path: Path):
    (save_srt if fmt == "srt" else save_vtt)(native, events, out_path)


# --- subcommands ------------------------------------------------------------
def cmd_extract(path: Path) -> int:
    fmt, events, _ = load(path)
    json.dump(
        {"format": fmt, "count": len(events),
         "events": [asdict(e) for e in events]},
        sys.stdout, ensure_ascii=False, indent=2,
    )
    sys.stdout.write("\n")
    return 0


def cmd_apply(path: Path, edits_src: str, close: bool, max_gap: float,
              swiss: bool) -> int:
    fmt, events, native = load(path)

    raw = sys.stdin.read() if edits_src == "-" else Path(edits_src).read_text("utf-8")
    edits = json.loads(raw)
    # Accept either a bare list or {"events": [...]}
    if isinstance(edits, dict):
        edits = edits.get("events", [])

    by_index = {e.index: e for e in events}
    applied = 0
    for item in edits:
        idx = item.get("index")
        if idx in by_index and "text" in item:
            by_index[idx].text = item["text"]
            applied += 1

    # Run over every event, not just the edited ones: an untouched event can
    # still carry an ß from the source file.
    eszett = 0
    if swiss:
        for ev in events:
            swissified = to_swiss(ev.text)
            if swissified != ev.text:
                eszett += 1
                ev.text = swissified

    closed = close_gaps(fmt, native, max_gap) if close else 0

    out_path = path.with_name(f"{path.stem}_optimized{path.suffix}")
    save(fmt, native, events, out_path)
    gap_note = (f", {closed} gap(s) closed (<= {max_gap}s)" if close
                else ", timestamps preserved")
    swiss_note = f", ß→ss in {eszett} event(s)" if swiss else ""
    print(f"Wrote {out_path}  ({applied}/{len(events)} events updated"
          f"{gap_note}{swiss_note})")
    return 0


def cmd_close_gaps(path: Path, max_gap: float) -> int:
    fmt, events, native = load(path)
    closed = close_gaps(fmt, native, max_gap)
    out_path = path.with_name(f"{path.stem}_optimized{path.suffix}")
    save(fmt, native, events, out_path)
    print(f"Wrote {out_path}  ({closed}/{max(len(events) - 1, 0)} gaps closed "
          f"(<= {max_gap}s), text unchanged)")
    return 0


# --- CLI --------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ex = sub.add_parser("extract", help="Print events as JSON.")
    p_ex.add_argument("file")

    p_ap = sub.add_parser("apply", help="Apply optimized text and write output.")
    p_ap.add_argument("file")
    p_ap.add_argument("edits", help="Path to JSON edits, or '-' for stdin.")
    p_ap.add_argument("--close-gaps", action="store_true",
                      help="Also close small gaps between events.")
    p_ap.add_argument("--max-gap", type=float, default=MAX_GAP_DEFAULT,
                      help=f"Max gap (seconds) to close. Default {MAX_GAP_DEFAULT}.")
    p_ap.add_argument("--swiss", action="store_true",
                      help="Swiss Standard German orthography: rewrite every "
                           "ß as ss (and ẞ as SS) across all events.")

    p_cg = sub.add_parser("close-gaps",
                          help="Close small gaps between events (no text change).")
    p_cg.add_argument("file")
    p_cg.add_argument("--max-gap", type=float, default=MAX_GAP_DEFAULT,
                      help=f"Max gap (seconds) to close. Default {MAX_GAP_DEFAULT}.")

    args = parser.parse_args()
    path = Path(args.file).expanduser()

    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        if args.command == "extract":
            return cmd_extract(path)
        if args.command == "close-gaps":
            return cmd_close_gaps(path, args.max_gap)
        return cmd_apply(path, args.edits, args.close_gaps, args.max_gap,
                         args.swiss)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # malformed subtitle / IO
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
