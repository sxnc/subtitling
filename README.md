# subtitling

Proofread and re-flow subtitle files (`.srt` / `.vtt`) **with the Claude Code
instance you already have open** — no API key, no separate API billing, no
network calls. Claude reads the subtitles, applies the optimization rules, and
writes the result. Your timestamps are never touched.

```
Sie haben zum Beispiel nicht gleich gut          Sie haben zum Beispiel
geschmeckt oder nachhaltige T-Shirts     ─►      nicht gleich gut geschmeckt,
haben auch abgefährt.                            oder nachhaltige T-Shirts
                                                 haben auch abgefärbt.
```

## Use it

1. Open this folder in a terminal.
2. Run `claude`.
3. **Drag a `.srt` or `.vtt` file into the terminal window and press enter.**

That's it. Dropping the file pastes its path, and the included `CLAUDE.md` tells
Claude Code to treat a bare subtitle path as "optimize this file." On the first
run it sets up a local `.venv` for the helper. The result is written beside your
file as `<name>_optimized.<ext>`; **the original is never modified.**

You can also just type a request like `optimize subtitles.srt`, or pass several
paths at once.

## Why it can't corrupt your timing

The model never writes a subtitle file. It only ever emits text keyed by event
index, and `optimize_subs.py` re-attaches the **original timestamps** by that
index:

1. `optimize_subs.py extract file.srt` → JSON of every event
   (`index`, `start`, `end`, `start_seconds`, `gap_before`, `text`).
2. Claude optimizes the **text only**, returning `{"index": …, "text": …}` pairs.
3. `optimize_subs.py apply file.srt edits.json` → writes `file_optimized.srt`.

Because timestamps come from the source and the event count is fixed, timing
survives whatever the model does to the text. Events are never added, removed,
or reordered. The single sanctioned timing change is opt-in gap closing (below),
which only ever *extends* an end time.

`optimize_subs.py` is a deterministic, offline parse/write helper. It makes no
network calls and needs no credentials.

## What gets optimized

- **Proofreading** — auto-detects the source language and edits only in that
  language. Fixes spelling, grammar, typos, and punctuation while preserving
  meaning, tone, register, and slang. It never translates, formalizes, censors,
  or summarizes. In practice the big win is speech-to-text mishearings:
  `abgefährt` → `abgefärbt`, `CO2-Fußprint` → `CO2-Footprint`.
- **Swiss Standard German** — German-language subtitles are written in
  *Schweizer Hochdeutsch*: no `ß` (always `ss` — `heisst`, `gross`, `Strasse`),
  «guillemets» rather than „German quotes“, and `1'000` for thousands. This is a
  spelling rule only; the speaker's wording is left exactly as spoken, and Swiss
  German dialect is never flattened into standard German.
- **Semantic line breaking** — keeps words that belong together on one line:
  article+noun, adjective+noun, auxiliary+main verb, preposition+object,
  multi-word names. Breaks fall at syntactic boundaries, never mid-phrase.
- **Orphan & punctuation control** — no lone word stranded at an event's start or
  end, and events end on natural punctuation where timing allows. Text is
  redistributed across the events that already exist. The exception: if the
  silence around a word exceeds ~1.5s it's a deliberate dramatic pause, and the
  lone word is left to stand.
- **Gap closing** *(opt-in, `--close-gaps`)* — STT output is full of tiny gaps
  between consecutive events that make subtitles flicker. Each event's end time
  is extended forward to touch the next event's start, but only when the gap is
  ≤ 1.5s (`--max-gap`). Larger gaps — title cards, scene changes, dramatic beats
  — are left intact. A start is never moved, an event is never shortened, and an
  overlap is never created.

## Manual use

You can drive the helper by hand:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python optimize_subs.py extract video.srt              # dump events as JSON
.venv/bin/python optimize_subs.py apply   video.srt edits.json   # write output
.venv/bin/python optimize_subs.py apply   video.srt edits.json --close-gaps --swiss
.venv/bin/python optimize_subs.py close-gaps video.srt --max-gap 1.5   # timing only
```

`edits.json` is a list of `{"index": <int>, "text": "<optimized>"}` — use `\n`
for an in-event line break, and omit events you didn't change. Pass `-` instead
of a path to read the edits from stdin.

| Flag | Effect |
| --- | --- |
| `--close-gaps` | Extend end times to close gaps ≤ `--max-gap` |
| `--max-gap <s>` | Gap-closing threshold in seconds (default `1.5`) |
| `--swiss` | Rewrite every `ß` as `ss` (and `ẞ` as `SS`) across all events, edited or not |

## Files

| File | Purpose |
| --- | --- |
| `CLAUDE.md` | The instructions and rules the Claude Code instance follows |
| `optimize_subs.py` | Offline parse/write helper (`extract` / `apply` / `close-gaps`) |
| `requirements.txt` | `srt`, `webvtt-py` — no `anthropic`, no API used |

## Requirements

Python 3.9+ and [Claude Code](https://claude.com/claude-code).
