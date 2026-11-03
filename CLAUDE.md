# Social-first subtitler — agent instructions

**You (this Claude Code instance) are the subtitler.** `subtitle.py` does the
mechanical work (transcription, clip checks, SRT timing). You do the editorial
work: deciding what was actually said, writing clean Swiss Standard German,
punctuating it correctly, and choosing where each subtitle event breaks.

## Trigger

When the user's message is essentially an audio file path (`.wav`, `.mp3`,
`.m4a`, `.aif`, …) — pasted or `@`-attached, with or without "same as before" —
run the full procedure below without asking first. Clean the path: strip
quotes, un-escape `\ `, expand `~`. Several paths → one job each.

Defaults unless the user says otherwise: German output in Swiss orthography,
the style in "Style" below, both timecode variants.

## Procedure

1. **Set up once.** If `.venv/bin/python -c "import mlx_whisper"` fails:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
   ```
   Needs Apple Silicon and `ffmpeg` (Homebrew). Models load from
   `~/.cache/huggingface` (downloaded on first use, ~3 GB each); once cached,
   `subtitle.py` runs offline so a slow network can't stall it.
   **This folder lives in iCloud Drive:** the environment must be
   `.venv.nosync` (iCloud never syncs `*.nosync`) with `.venv` as a symlink to
   it. Otherwise iCloud syncs ~1 GB of packages and imports wait on iCloud
   (`import mlx_whisper` took 106 s instead of 1 s). When creating it fresh:
   `python3 -m venv .venv.nosync && ln -s .venv.nosync .venv`.

2. **Transcribe with both models.**
   ```bash
   .venv/bin/python subtitle.py transcribe "<audio path>"
   ```
   Creates `work/<stem>/` and prints the large-v3 transcript, the turbo
   transcript, and an indexed word table (`words_large.json` — this is the
   timing source). `~` marks zero-length words, `▼` a gap of more than 1 s.
   Swiss German audio comes out as Standard German; that is intended.

3. **Find the doubtful spots.** Compare the two transcripts. Suspect:
   - anywhere the models disagree on a word or name;
   - zero-length words (`~`), especially at the very start — Whisper invents
     speech over intros and music ("Das ist ein bisschen zu schnell",
     "Vielen Dank", "Untertitel von …" are classic hallucinations);
   - words that make no sense in context ("Nächstes Like" was "Nächste Slide");
   - silent stretches (`▼`) — check whether anything is said there.

4. **Re-check them in isolation.**
   ```bash
   .venv/bin/python subtitle.py clip <stem> 0 3.6 22.8 25.6
   ```
   (start/end pairs in seconds, from `in.wav`). Prints mean loudness and both
   models' reading of each clip. Interpreting it:
   - both models produce *different* nonsense → music/SFX, no subtitle;
   - "Vielen Dank" / empty output on a quiet range → silence, no subtitle;
   - both agree in isolation → trust it, even over the full-file pass;
   - still disagree → pick the best reading and **flag it to the user**.
   Look up brand and product spellings on the web when a brand is involved
   (e.g. Cembra's card is "Certo!" with an exclamation mark).

5. **Write the final text** for the whole piece, with correct punctuation (see
   "Language"), then **write `work/<stem>/chunks.json`**: one item per event,
   in order, against the word table:
   ```json
   [[12, null], [2, "Findet raus:"], [4, "Wie viele Gramm Protein"], [1, "täglich"]]
   ```
   `[n, "text"]` = one event spanning the next *n* tokens of the word table,
   displayed as `text` (you may correct words/punctuation; timing still comes
   from those n tokens). `[n, null]` = skip *n* tokens (hallucinations).
   Note that numbers often split into several tokens (`0`, `.9g` → 2 tokens,
   shown as `0,9 g`; `16`, `-Jähriger` → 2 tokens). The counts must sum to the
   table length exactly — `build` refuses otherwise.

6. **Build.**
   ```bash
   .venv/bin/python subtitle.py build <stem> --swiss --name "<nice name>"
   ```
   Writes `output/<name>.srt` (starts 01:00:00,000) and
   `output/<name> (00h start).srt`, and prints a QA table: each event's timing,
   the displayed text, what the model heard, and warnings (`SHORT` < 0.45 s,
   `WORDS` > 4, `LONG` > 22 characters). Fix every `SHORT` by merging with a
   neighbour, and fix `WORDS` unless splitting would break a phrase. **Always
   fix `LONG`**, even if that splits a phrase (see "Style": safe zones).
   Edit `chunks.json` and rebuild until it is clean.

### Tight timing / few words per event

Use this when the user asks for tight timing or ≤ 2 words per event, and for
messy audio (dialogue, overlapping speakers) where Whisper's own words are too
wrong to borrow timing from. Instead of `chunks.json` against Whisper's word
table, the **final text** is aligned to the audio:

a. **Settle the text** (steps 2–4). Pass the topic vocabulary to `clip` with
   `--prompt "Gespräch über Brot, Brötli, Brotkorb …"`. It helps a lot with
   dialect. For passages that stay unclear, rank candidate wordings:
   ```bash
   .venv/bin/python subtitle.py score <stem> 25.3 28.5 "Aber er hat einen Brotkorb." "Aber wir haben einen Brotkorb."
   ```
   (higher = better supported by the audio). The score favours fluent and
   repetitive phrasing, so treat it as evidence alongside duration and context,
   not as proof.
b. **Write `work/<stem>/script.txt`**: `start end | text`, one segment per
   phrase, **split at every real pause** (≤ 30 s each). The text is exactly what
   will be displayed. Leave unclear stretches out (comment them with `#`).
c. **Align with both engines.**
   ```bash
   .venv/bin/python subtitle.py align <stem> --engine mms
   .venv/bin/python subtitle.py align <stem> --engine whisper
   ```
   - *whisper* (attention DTW) copes with dialect, since the text is Standard
     German and the audio is Swiss German, but it can stretch the first word of a
     segment or squash several words together.
   - *mms* (phoneme-level CTC) is precise to ~20 ms where the words sound like
     the text, but can be badly off where dialect differs ("zu Hause"/"dihei").
   - `⏸` marks a word that spans a pause: move the segment boundary there and
     re-run. Segment boundaries must sit in real pauses; a boundary inside speech
     puts that speech on the wrong word.
d. **Adjudicate** every place where the engines disagree by > 0.25 s:
   ```bash
   .venv/bin/python subtitle.py compare <stem> 52.9 55.8
   ```
   draws the speech profile (20 ms/char) with both engines' word starts. The
   right engine is the one whose starts sit on onsets after dips (`▁·`).
   Combine them:
   ```bash
   .venv/bin/python subtitle.py merge <stem> --mms "8.3-9.2,31.3-32.5" --avg "54.0-55.7"
   ```
   (ranges match the whisper start of each word.)
   **When the final text is close to what Whisper transcribed** (clean
   audio), use `merge --median` instead: per word, it takes the median of
   whisper-align, mms and the transcription timestamps, so one engine's outlier
   is outvoted.
   Then **check every phrase start after a pause** against the voice onset
   with `.venv/bin/python subtitle.py onsets <stem>`. That shows 0.4 s before
   and 0.6 s after each start: the word should begin at the first sustained
   `▅`/`█` after the quiet (a music bed shows as `▃`; laughter or a scream
   before a phrase is loud too, so judge by the shape). Then
   record verified corrections in the same command, never by hand-editing:
   `merge … --set "5=8.08,16=15.56"` (word index = start in seconds).
e. **Chunks** are plain strings against `words_aligned.json`, e.g.
   `["Du hast", "zwei Brote", "zu Hause?"]`. Respect the word limit. Keep
   article + noun together where two words allow; otherwise break the way the
   reference does ("wieder ein / nicht ganz so").
f. **Build tight:**
   ```bash
   .venv/bin/python subtitle.py build <stem> --words aligned --tight --swiss --name "<name>"
   ```
   Tight mode snaps boundaries between back-to-back events to the nearest
   loudness dip (±0.2 s), and clears the screen 0.1 s after speech. It joins
   events across a gap ≤ 0.6 s that still contains speech, and gives
   flash-short events ~0.25 s by shifting boundaries ≤ 0.12 s. `--max-words`
   defaults to 2. A few `SHORT` warnings around 0.22–0.25 s in very fast speech
   are acceptable.
   **Accurate timing in the normal style** (1–4 words, as in the reference):
   the same pipeline with `--tight --max-words 4 --hold 0.4`. Events then
   keep the usual 0.4 s hold, and balancing aims for ≥ 0.4 s per event.

7. **Deliver.** Send both SRTs with SendUserFile (`display: attach`). In the
   reply:
   - the **full text** as a blockquote, so punctuation can be proofread in one
     read;
   - **conventions/choices** made (only the non-obvious ones);
   - a numbered **"please confirm"** list: every uncertain word, name, or
     spelling, with the time and the alternative reading;
   - which ranges were **deliberately left blank** (music, silence) and why.

## Style (from the approved reference, `reference/Uncovered.srt`)

- Every event is `<i><b>…</b></i>` (bold italic).
- **1–4 words per event**, ideally ≤ 20 characters; short punchy events.
- Break at phrase boundaries: keep article + noun, preposition + object,
  auxiliary + verb together. Good: "Wie viele Gramm Protein / solltet ihr mit 16 /
  täglich / zu euch nehmen?" A long word may stand alone ("Lieblingshändlern",
  "Präsentation").
- **Hard limit: 22 characters per event.** Longer events don't fit the
  short-form safe zones. That limit beats keeping a phrase together: put the
  long word in its own event ("Und mit dieser / Einstellung",
  "sondern um / Rangordnung,"). The user had to split such events by hand in
  "Uncovered Top Tier Mann - V3".
- Events of one sentence are back-to-back (no flicker); at a pause the last
  event holds ~0.4 s, then the screen clears. `build` does this — never
  hand-edit timestamps.
- Minimum ~0.45 s on screen; merge anything shorter.
- Leave music, jingles, and silence unsubtitled.

## Language — Swiss Standard German

- Audio may be Swiss German; subtitles are **Schweizer Hochdeutsch**. Convert
  dialect into Standard German grammar ("vo de" → "von der", "Nächschti Slide"
  → "Nächste Slide"), but keep the speaker's wording, tone and English
  phrases ("Let me explain.", "What a deal, oder?", "Cashback").
- **No ß, ever** → `ss` (`weisst`, `Ausser`, `heisst`). `--swiss` enforces it.
- Quotes are «guillemets», nested ‹single›.
- Thousands separator is an apostrophe: `1'000`, `25'000`. Decimal comma for
  spoken quantities (`0,9 g`, `1,2 g`). Percent sign directly after the number,
  no space: `70%`, `1%` (German and English; only French subtitles use
  `70 %`).
- Numbers as digits, as in the reference ("bis zu 5 Tage", "mit 16").

**Punctuation must be spot on** — the user checks it closely:
- Comma before every subordinate clause (`weisst nicht, was es ist?`,
  `wird gerechnet, ob du …`, `bekommst, wird …`) and around relative clauses.
- Infinitive groups: comma where required (with *um, ohne, statt, als*, or a
  noun/correlate); otherwise optional — set it when the speaker pauses.
- A colon after a label or announcement; lower case after it unless a full
  sentence follows (`Grund Nummer 1: der Name.`, `Kurz gesagt: sehr robust`).
- Split run-on spoken questions into separate questions
  (`Also, was verdienst du? Was geht schon weg …? Und was bleibt übrig?`).
- Abbreviation point + question mark both stay (`Steuern etc.?`).
- Greeting interjections take `!` (`Ciao!`).
- Tag questions: `…, oder?`
- Trailing-off speech: `…` (`Diesen Punkt hast du definitiv …`).

## Notes

- `work/` (audio copies, transcripts, clips) and `output/` are gitignored;
  never commit client subtitles or audio.
- Do not invent words to complete a sentence; flag it instead.
- For a non-German job pass `--language <code>` to `transcribe` and `clip`,
  and drop `--swiss`.
