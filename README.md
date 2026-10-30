# Social-first subtitler

Turns a voice-over or dialogue audio file into short, punchy social-media
subtitles: 1–4 words per event, bold italic, timed to the word, with
Swiss Standard German orthography and carefully checked punctuation. Swiss
German audio comes out as Schweizer Hochdeutsch.

## How to use

Open this folder in Claude Code and drop in an audio file:

```
@"~/Downloads/20261012_CEMBRA_Präsentation - V2.wav"
```

That's all. Claude follows [CLAUDE.md](CLAUDE.md): it transcribes with two
Whisper models, re-checks every doubtful passage in isolation, writes and
punctuates the text, chunks it into events, and builds the SRTs. You get:

- `output/<name>.srt` — timeline start 01:00:00:00 (drops straight onto a Resolve timeline)
- `output/<name> (00h start).srt` — the same, starting at 00:00:00
- the full text in chat for proofreading, plus a list of anything uncertain

Add instructions in the same message if a job differs ("English audio",
"no bold", "keep the dialect").

## How it works

| Step | Who | What |
| --- | --- | --- |
| Transcribe | `subtitle.py transcribe` | Whisper large-v3 + large-v3-turbo (MLX), word timestamps |
| Verify | Claude + `subtitle.py clip` | Compare models, re-transcribe doubtful ranges, spot hallucinations over music/silence |
| Write | Claude | Final text, Swiss orthography, punctuation, brand spellings |
| Chunk | Claude → `chunks.json` | Which words form each event |
| Build | `subtitle.py build` | Timing from the word table, SRT output, QA warnings |

Timing is never hand-typed: each event inherits the timestamps of the words it
covers. Events within a sentence run back-to-back; at a pause the last one holds
for 0.4 s and clears.

## Manual use

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python subtitle.py transcribe ~/Downloads/clip.wav    # → work/clip/
.venv/bin/python subtitle.py table clip                          # re-print word table
.venv/bin/python subtitle.py clip clip 0 3.6 22.8 25.6           # re-check ranges
# write work/clip/chunks.json, then:
.venv/bin/python subtitle.py build clip --swiss --name "Clip V1"
```

`chunks.json` is a list of `[token_count, "event text"]`; `[n, null]` skips n
tokens (e.g. words Whisper invented over an intro jingle).

### Tight timing (e.g. max 2 words per event)

Ask for "tight timing" or "max 2 words". Instead of borrowing Whisper's word
timings, the final text is force-aligned to the audio with two engines (Whisper
DTW and Meta's MMS phoneme aligner), checked against the speech profile, merged,
and every event boundary is snapped to the nearest pause between words:

```bash
.venv/bin/python subtitle.py score   clip 25.3 28.5 "Variante A" "Variante B"   # which wording fits the audio
.venv/bin/python subtitle.py align   clip --engine mms      # script.txt → words_mms.json
.venv/bin/python subtitle.py align   clip --engine whisper  # script.txt → words_whisper.json
.venv/bin/python subtitle.py compare clip 52.9 55.8          # speech profile + both engines
.venv/bin/python subtitle.py merge   clip --mms "31.3-32.5" --avg "54.0-55.7"
.venv/bin/python subtitle.py build   clip --words aligned --tight --swiss
```

## Requirements

- Apple Silicon Mac, Python 3.10+, `ffmpeg` (`brew install ffmpeg`)
- `mlx-whisper`, plus `torch`/`torchaudio` for tight timing (see `requirements.txt`). The models download to
  `~/.cache/huggingface` on first use (~3 GB each).

## Files

| Path | Purpose |
| --- | --- |
| `CLAUDE.md` | Procedure, style and language rules Claude follows (incl. tight timing) |
| `subtitle.py` | Transcribe / clip / build helper |
| `reference/Uncovered.srt` | Approved style reference (gitignored) |
| `work/`, `output/` | Per-job files and delivered SRTs (gitignored) |
