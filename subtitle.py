#!/usr/bin/env python3
"""
subtitle.py — Audio → social-first SRT helper.

The model (Claude Code, following CLAUDE.md) does the editorial work: deciding
what was actually said, punctuation, and where to break events. This script does
the mechanical parts deterministically so timing can never drift:

    # 1. Transcribe with two Whisper models (word-level timestamps).
    #    Writes work/<stem>/{in.wav, words_large.json, words_turbo.json}
    #    and prints both transcripts plus an indexed word table.
    python subtitle.py transcribe "~/Downloads/clip.wav"

    # 2. Re-transcribe doubtful ranges in isolation (both models + loudness).
    python subtitle.py clip work/<stem> 0 3.6 22.8 25.6

    # 3. Build the SRT from the word table + a chunks file you wrote.
    python subtitle.py build work/<stem> --swiss

chunks.json (written by the model into work/<stem>/) is a JSON list, one item
per subtitle event, consumed in order against words_large.json:

    [[2, "Findet raus:"], [4, "Wie viele Gramm Protein"], [3, null], ...]

    [n, "text"]  -> one event spanning the next n word tokens, shown as "text"
    [n, null]    -> skip the next n tokens (hallucinations over music/silence)

The token counts must add up to the word table length exactly, or build fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUTPUT = ROOT / "output"

MODELS = {
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

# Models already cached → don't let a slow network stall start-up on a version check.
_HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if all((_HF / f"models--{r.replace('/', '--')}").is_dir() for r in MODELS.values()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Timing rules (tuned on the approved reference files).
JOIN_GAP = 0.3        # gap to next event <= this -> extend end to next start (no flicker)
HOLD = 0.4            # otherwise hold this long after the last word, then clear
MIN_DURATION = 0.4    # stretch shorter events to this, if the next start allows
TC_OFFSET = 3600.0    # 01:00:00:00 timeline start, matching the reference SRTs

# QA thresholds — warnings, not errors.
WARN_SHORT = 0.45
WARN_WORDS = 4
WARN_CHARS = 22

SWISS = {"ß": "ss", "ẞ": "SS"}

# --tight: subtitles hug the speech. Boundaries next to a pause are snapped to
# the measured speech onset/offset; inside continuous speech Whisper's word
# timings are kept and cross-checked against the second model.
TIGHT_HOLD = 0.10          # clear this soon after speech stops
TIGHT_MIN_DURATION = 0.20
TIGHT_WARN_SHORT = 0.25
TIGHT_LEAD = 0.04
DIP_WINDOW = 0.20          # tight: search this far around an event boundary for a dip
DIP_DISTANCE_COST = 20.0   # dB per second: prefer the nearest dip over a slightly deeper far one
TALK_JOIN_MAX = 0.60       # tight: keep a sentence on screen across a gap this short if speech continues
BALANCE_MIN = 0.25         # tight: aim for at least this on screen (~6 frames)...
BALANCE_MIN_WIDE = 0.40    # ...or this with more than 2 words per event
BALANCE_SHIFT = 0.12       # ...moving any boundary at most this far (~3 frames)          # show ~1 frame before the onset (reads as in sync)
PAUSE = 0.15               # a gap at least this long counts as a pause
SNAP_WINDOW = (0.30, 0.25) # search this far before / after Whisper's boundary
DRIFT = 0.12               # flag words whose start differs between models by more
FRAME = 0.01               # loudness envelope resolution
MIN_PAUSE_INSIDE = 0.20    # align: a word containing this much silence is misplaced


# ── helpers ──────────────────────────────────────────────────────────────────

def need(binary: str) -> None:
    if not shutil.which(binary):
        sys.exit(f"'{binary}' not found on PATH (brew install ffmpeg)")


def to_16k_mono(src: Path, dst: Path, start: float | None = None, end: float | None = None) -> None:
    need("ffmpeg")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", str(start), "-to", str(end)]
    cmd += ["-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    subprocess.run(cmd, check=True)


def mean_volume(wav: Path) -> str:
    out = subprocess.run(
        ["ffmpeg", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    m = re.search(r"mean_volume: (-?[\d.]+) dB", out)
    return f"{m.group(1)} dB" if m else "?"


def whisper(wav: Path, model: str, language: str, words: bool, prompt: str | None = None):
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("Missing mlx-whisper. Run: .venv/bin/pip install -r requirements.txt")
    return mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=MODELS[model], language=language,
        word_timestamps=words, condition_on_previous_text=False, initial_prompt=prompt,
    )


def tc(t: float, offset: float) -> str:
    ms = round((t + offset) * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def resolve_job(arg: str) -> Path:
    p = Path(arg).expanduser()
    if not p.is_dir():
        p = WORK / arg
    if not p.is_dir():
        sys.exit(f"No job folder at {arg} (run 'transcribe' first)")
    return p


def envelope(wav: Path):
    """Loudness in dBFS per FRAME of a 16 kHz mono 16-bit wav, plus a speech threshold."""
    import wave
    import numpy as np
    with wave.open(str(wav)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        hop = int(w.getframerate() * FRAME)
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor, speech = np.percentile(db, 10), np.percentile(db, 90)
    return db, floor + 0.35 * (speech - floor)


def snap_to_audio(events: list[dict], wav: Path) -> None:
    """Move event starts after a pause to the measured speech onset, and event
    ends before a pause to the measured offset. Never crosses a neighbour."""
    db, thr = envelope(wav)
    loud = db >= thr
    at = lambda t: max(0, min(len(db) - 1, int(round(t / FRAME))))

    for k, ev in enumerate(events):
        prev_end = events[k - 1]["e"] if k else 0.0
        nxt_start = events[k + 1]["s"] if k + 1 < len(events) else len(db) * FRAME
        ev["snap"] = {}

        if k == 0 or ev["s"] - prev_end >= PAUSE:
            lo = max(prev_end + 0.05, ev["s"] - SNAP_WINDOW[0])
            hi = ev["s"] + SNAP_WINDOW[1]
            for f in range(at(lo), at(hi)):
                if loud[f:f + 3].all():                 # sustained onset, not a click
                    new = max(lo, f * FRAME - TIGHT_LEAD)
                    if abs(new - ev["s"]) >= 0.02:
                        ev["snap"]["s"] = new - ev["s"]
                        ev["s"] = round(new, 3)
                    break

        if k + 1 == len(events) or nxt_start - ev["e"] >= PAUSE:
            lo = max(ev["s"] + 0.1, ev["e"] - SNAP_WINDOW[1])
            hi = min(nxt_start - 0.05, ev["e"] + SNAP_WINDOW[0])
            for f in range(at(hi), at(lo), -1):
                if loud[max(0, f - 2):f + 1].all():     # sustained offset
                    new = (f + 1) * FRAME
                    if abs(new - ev["e"]) >= 0.02:
                        ev["snap"]["e"] = new - ev["e"]
                        ev["e"] = round(new, 3)
                    break


def snap_to_dips(events: list[dict], wav: Path) -> None:
    """Move each boundary between back-to-back events to the deepest nearby
    loudness dip (word gaps show up as dips even in fast speech)."""
    import numpy as np
    db, thr = envelope(wav)
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    for k in range(len(events) - 1):
        a, b = events[k], events[k + 1]
        if abs(a["e"] - b["s"]) > 1e-6:
            continue
        t = b["s"]
        lo = max(t - DIP_WINDOW, a["s"] + 0.08)
        hi = min(t + DIP_WINDOW, b["e"] - 0.08)
        if hi <= lo:
            continue
        fr = np.arange(int(lo / FRAME), int(hi / FRAME))
        cost = sm[fr] + DIP_DISTANCE_COST * np.abs(fr * FRAME - t)
        best = fr[int(np.argmin(cost))]
        if sm[best] < thr + 3 and sm[best] < sm[int(t / FRAME)] - 3:
            new = round((best + 1) * FRAME, 3)
            a["snap"] = a.get("snap", {}) | {"e": new - a["e"]}
            a["e"] = b["s"] = new


def balance(events: list[dict], target: float = None) -> None:
    """Give flash-short events a readable minimum by moving shared boundaries
    between back-to-back events (or into an adjacent pause) by at most
    BALANCE_SHIFT, so nothing drifts visibly off its words."""
    target = target or BALANCE_MIN
    orig = [(ev["s"], ev["e"]) for ev in events]
    dur = lambda k: events[k]["e"] - events[k]["s"]
    for _ in range(4):
        for k, ev in enumerate(events):
            need = target - dur(k)
            if need <= 0:
                continue
            # borrow earlier
            if k:
                prev = events[k - 1]
                joined = abs(prev["e"] - ev["s"]) < 1e-6
                room = (dur(k - 1) - target) if joined else (ev["s"] - prev["e"] - 0.04)
                room = min(room, BALANCE_SHIFT - (orig[k][0] - ev["s"]))
                take = max(0.0, min(need / 2 if k + 1 < len(events) else need, room))
                ev["s"] -= take
                if joined:
                    prev["e"] = ev["s"]
                need -= take
            # borrow later
            if need > 0 and k + 1 < len(events):
                nxt = events[k + 1]
                joined = abs(nxt["s"] - ev["e"]) < 1e-6
                room = (dur(k + 1) - target) if joined else (nxt["s"] - ev["e"] - 0.04)
                room = min(room, BALANCE_SHIFT - (ev["e"] - orig[k][1]))
                take = max(0.0, min(need, room))
                ev["e"] += take
                if joined:
                    nxt["s"] = ev["e"]
    for ev in events:
        ev["s"], ev["e"] = round(ev["s"], 3), round(ev["e"], 3)


def model_drift(job: Path, base: str) -> list[tuple[float, float]]:
    """(time, delta) for words that the timing source and an independent model
    (turbo) both have, but whose start differs by more than DRIFT."""
    import difflib
    other = "large" if base == "turbo" else "turbo"
    a = json.loads((job / f"words_{base}.json").read_text(encoding="utf-8"))
    b = json.loads((job / f"words_{other}.json").read_text(encoding="utf-8"))
    norm = lambda w: re.sub(r"\W", "", w["w"].lower())
    sm = difflib.SequenceMatcher(None, [norm(w) for w in a], [norm(w) for w in b], autojunk=False)
    out = []
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            wa, wb = a[blk.a + j], b[blk.b + j]
            if abs(wb["s"] - wa["s"]) > DRIFT:
                out.append((wa["s"], wb["s"] - wa["s"]))
    return out


# ── subcommands ──────────────────────────────────────────────────────────────

def cmd_transcribe(a) -> None:
    src = Path(a.audio).expanduser()
    if not src.is_file():
        sys.exit(f"Audio not found: {src}")
    job = WORK / src.stem
    job.mkdir(parents=True, exist_ok=True)
    wav = job / "in.wav"
    to_16k_mono(src, wav)
    (job / "source.txt").write_text(str(src), encoding="utf-8")

    texts = {}
    for name in ("large", "turbo"):
        r = whisper(wav, name, a.language, words=True)
        words = [{"w": w["word"].strip(), "s": round(w["start"], 3), "e": round(w["end"], 3)}
                 for seg in r["segments"] for w in seg["words"]]
        (job / f"words_{name}.json").write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
        texts[name] = r["text"].strip()

    print(f"JOB {job}\n")
    for name, text in texts.items():
        print(f"── {name} ──\n{text}\n")
    print_table(json.loads((job / "words_large.json").read_text(encoding="utf-8")))


def print_table(words: list[dict]) -> None:
    print("── word table (words_large.json) ──")
    print("  i   start    end  word        flags: ~ zero-length (suspect)  ▼ gap >1s before")
    prev_end = 0.0
    for i, w in enumerate(words):
        flags = ("~" if w["e"] - w["s"] < 0.01 else " ") + ("▼" if w["s"] - prev_end > 1.0 else " ")
        print(f"{i:3} {w['s']:6.2f} {w['e']:6.2f} {flags} {w['w']}")
        prev_end = w["e"]


def cmd_table(a) -> None:
    job = resolve_job(a.job)
    print_table(json.loads((job / "words_large.json").read_text(encoding="utf-8")))


def cmd_clip(a) -> None:
    job = resolve_job(a.job)
    if len(a.ranges) % 2:
        sys.exit("Give ranges as start/end pairs: 0 3.6 22.8 25.6")
    pairs = list(zip(a.ranges[0::2], a.ranges[1::2]))
    clips = job / "clips"
    clips.mkdir(exist_ok=True)
    for s, e in pairs:
        wav = clips / f"{s:.2f}-{e:.2f}.wav"
        to_16k_mono(job / "in.wav", wav, s, e)
        print(f"{s:6.2f}–{e:6.2f}  mean {mean_volume(wav)}")
        for name in ("large", "turbo"):
            print(f"   {name:5} | {whisper(wav, name, a.language, words=False, prompt=a.prompt)['text'].strip()}")


class Aligner:
    """Forced alignment of known text to audio with Whisper large-v3's alignment heads."""

    def __init__(self, wav: Path, language: str):
        import mlx.core as mx
        from mlx_whisper.audio import load_audio
        from mlx_whisper.tokenizer import get_tokenizer
        from mlx_whisper.transcribe import ModelHolder
        self.mx = mx
        self.model = ModelHolder.get_model(MODELS["large"], mx.float16)
        self.tok = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages,
                                 language=language, task="transcribe")
        self.audio = load_audio(str(wav))
        self.db, self.thr = envelope(wav)

    def trim(self, s: float, e: float) -> tuple[float, float]:
        """Shrink [s, e] to the speech inside it (keeps a 0.08 s margin)."""
        loud = self.db >= self.thr
        f0, f1 = int(s / FRAME), min(len(loud), int(e / FRAME))
        idx = [f for f in range(f0, f1 - 2) if loud[f:f + 3].all()]
        if not idx:
            return s, e
        return max(s, idx[0] * FRAME - 0.08), min(e, (idx[-1] + 3) * FRAME + 0.08)

    def align(self, s: float, e: float, text: str):
        """Return (timings, mean token log-prob) for text spoken within [s, e]."""
        import numpy as np
        from mlx_whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE, \
            log_mel_spectrogram, pad_or_trim
        from mlx_whisper.timing import find_alignment
        mx = self.mx
        clip = self.audio[int(s * SAMPLE_RATE): int(e * SAMPLE_RATE)]
        mel = log_mel_spectrogram(clip, n_mels=self.model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        tokens = self.tok.encode(" " + text)
        timing = find_alignment(self.model, self.tok, tokens, mel, len(clip) // HOP_LENGTH)
        probs = np.concatenate([[t.probability] * len(t.tokens) for t in timing]) if timing else np.array([1e-9])
        return timing, float(np.mean(np.log(np.maximum(probs, 1e-9))))


def mms_align(al: "Aligner", s: float, e: float, words: list[str]) -> list[dict]:
    """Phoneme-level CTC forced alignment (Meta MMS via torchaudio): ~20 ms precise.
    Text is romanised for the aligner (ä→a, ß→ss, punctuation dropped); words with
    nothing alignable (pure digits, dashes) are placed between their neighbours."""
    import numpy as np
    import torch
    import torchaudio.functional as F
    from torchaudio.pipelines import MMS_FA
    if not hasattr(al, "mms"):
        al.mms = MMS_FA.get_model(with_star=False).eval()
        al.mms_dict = MMS_FA.get_dict(star=None)
    table = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "é": "e", "è": "e", "à": "a", "ç": "c", "ß": "ss"})
    norm = lambda w: [c for c in w.lower().translate(table) if c in al.mms_dict and c != "-"]
    toks = [[al.mms_dict[c] for c in norm(w)] for w in words]

    clip = torch.from_numpy(np.array(al.audio[int(s * 16000): int(e * 16000)], dtype=np.float32))[None]
    with torch.inference_mode():
        emission, _ = al.mms(clip)
    emission = torch.log_softmax(emission, dim=-1)
    flat = [t for w in toks for t in w]
    path, scores = F.forced_align(emission, torch.tensor([flat], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(path[0], scores[0].exp())
    ratio = clip.size(1) / emission.size(1) / 16000

    out, i = [], 0
    for w, t in zip(words, toks):
        sp = spans[i:i + len(t)]
        i += len(t)
        if sp:
            out.append({"w": w, "s": round(s + sp[0].start * ratio, 3), "e": round(s + sp[-1].end * ratio, 3),
                        "p": float(np.mean([x.score for x in sp]))})
        else:
            out.append({"w": w, "s": None, "e": None, "p": 1.0})
    for k, w in enumerate(out):                      # place unalignable words between neighbours
        if w["s"] is None:
            prev = out[k - 1]["e"] if k else s
            nxt = next((x["s"] for x in out[k + 1:] if x["s"] is not None), e)
            w["s"], w["e"] = prev, max(prev, nxt)
    return out


def cmd_score(a) -> None:
    """Rank alternative wordings of one passage by how well the audio supports them."""
    job = resolve_job(a.job)
    al = Aligner(job / "in.wav", a.language)
    s, e = al.trim(a.start, a.end)
    print(f"{a.start:.2f}–{a.end:.2f} (speech {s:.2f}–{e:.2f})")
    ranked = sorted(((al.align(s, e, c)[1], c) for c in a.candidates), reverse=True)
    for score, c in ranked:
        print(f"  {score:7.2f}  {c}")


def cmd_align(a) -> None:
    """Force-align the final text (script.txt) to the audio, segment by segment."""
    import string

    job = resolve_job(a.job)
    segments = []
    for line in (job / a.script).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        span, text = line.split("|", 1)
        s, e = map(float, span.split())
        segments.append((s, e, text.strip()))

    al = Aligner(job / "in.wav", a.language)

    out = []
    for s, e, text in segments:
        if e - s > 30:
            sys.exit(f"segment {s}–{e} is longer than 30 s; split it at a pause")
        s, e = al.trim(s, e)          # leading/trailing silence would smear the edge words
        # standalone punctuation ("immer …", "– ja") belongs to the word before it
        expected = []
        for tok in text.split():
            if expected and all(c in string.punctuation + "«»‹›…–" for c in tok):
                expected[-1] += " " + tok
            else:
                expected.append(tok)
        if a.engine == "mms":
            seg_words = mms_align(al, s, e, expected)
            for x in seg_words:
                x["_seg"] = s
            out += seg_words
            continue
        timing, _ = al.align(s, e, text)

        merged = []                                   # glue punctuation onto its word
        for t in timing:
            w = t.word.strip()
            if merged and (not t.word.startswith(" ") or all(c in string.punctuation + "«»‹›…–" for c in w)):
                merged[-1]["w"] += w
                merged[-1]["p"] = min(merged[-1]["p"], float(t.probability))
            else:
                merged.append({"w": w, "s": round(s + float(t.start), 3),
                               "e": round(s + float(t.end), 3), "p": float(t.probability)})
        nospace = lambda ws: [x.replace(" ", "") for x in ws]
        if nospace(m["w"] for m in merged) != nospace(expected):
            sys.exit(f"alignment split the words of segment {s}–{e} differently:\n"
                     f"  {[m['w'] for m in merged]}\n  {expected}")
        for m_, w_ in zip(merged, expected):
            m_["w"] = w_
            m_["_seg"] = s
        out += merged

    quiet = al.db < al.thr
    # A word stretched over a pause into the next word's speech: end it at the
    # pause and start the next word at the onset after it.
    for k in range(len(out) - 1):
        w, nx = out[k], out[k + 1]
        if abs(nx["s"] - w["e"]) > 0.06:
            continue
        f0, f1 = int(w["s"] / FRAME), int(w["e"] / FRAME)
        runs, f = [], f0
        while f < f1:
            if quiet[f]:
                g = f
                while g < f1 and quiet[g]:
                    g += 1
                if (g - f) * FRAME >= 0.10 and g < f1 and any(not q for q in quiet[f0:f]):
                    runs.append((f, g))
                f = g
            else:
                f += 1
        if runs:
            q0, q1 = runs[-1]
            w["e"], nx["s"] = round(q0 * FRAME, 3), round(q1 * FRAME - 0.02, 3)

    # Alignments can bleed into neighbouring silence; shrink each word to its audible part.
    for w in out:
        f0, f1 = int(w["s"] / FRAME), int(w["e"] / FRAME)
        loud = [f for f in range(f0, f1) if not quiet[f]]
        if loud:
            w["s"] = round(max(w["s"], loud[0] * FRAME - 0.02), 3)
            w["e"] = round(min(w["e"], (loud[-1] + 1) * FRAME + 0.02), 3)

    # ...and if the aligner started a word late after a pause, grow it back to the onset.
    for k, w in enumerate(out):
        prev_e = max(out[k - 1]["e"] if k else 0.0, w["_seg"])   # never grow outside the segment
        f = int(w["s"] / FRAME) - 1
        while f >= 0 and not quiet[f] and (f * FRAME) > prev_e + 0.02:
            f -= 1
        g = f                                         # the quiet run before that speech
        while g >= 0 and quiet[g]:
            g -= 1
        if (f - g) * FRAME >= 0.08 and (f + 1) * FRAME < w["s"]:
            w["s"] = round(max((f + 1) * FRAME - 0.02, prev_e), 3)

    for w in out:
        w.pop("_seg", None)
    for name in (f"words_{a.engine}.json", "words_aligned.json"):
        (job / name).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")

    # A word that still spans a real pause was misplaced by DTW: split the segment there.
    def pause_inside(w) -> bool:
        f0, f1 = int((w["s"] + 0.05) / FRAME), int((w["e"] - 0.05) / FRAME)
        run = 0
        for f in range(f0, max(f0, f1)):
            run = run + 1 if quiet[f] else 0
            if run * FRAME >= MIN_PAUSE_INSIDE:
                return True
        return False

    print("  i   start    end  prob  word        (! low probability: re-check text · ⏸ spans a pause: split segment)")
    bad = 0
    for i, w in enumerate(out):
        p = pause_inside(w)
        bad += p
        print(f"{i:3} {w['s']:6.2f} {w['e']:6.2f}  {w['p']:.2f} {'!' if w['p'] < 0.2 else ' '}{'⏸' if p else ' '} {w['w']}")
    print(f"\n{len(out)} words aligned → {job / 'words_aligned.json'}")
    if bad:
        print(f"⏸ {bad} word(s) span a pause — split script.txt at those pauses and re-run align")


def cmd_compare(a) -> None:
    """Draw the speech profile (20 ms/char) with both engines' word starts marked."""
    job = resolve_job(a.job)
    db, thr = envelope(job / "in.wav")
    w = json.loads((job / "words_whisper.json").read_text(encoding="utf-8"))
    m = json.loads((job / "words_mms.json").read_text(encoding="utf-8"))
    s, e = a.start, a.end
    n = int((e - s) / 0.02)
    def ch(t):
        d = db[min(len(db) - 1, int(t / FRAME))] - thr
        return "█" if d > 12 else "▅" if d > 6 else "▃" if d > 0 else "▁" if d > -6 else "·"
    def marks(ws):
        row = [" "] * n
        for x in ws:
            k = int(round((x["s"] - s) / 0.02))
            if 0 <= k < n:
                row[k] = "|"
        return "".join(row)
    print("  W " + marks(w))
    print("    " + "".join(ch(s + k * 0.02) for k in range(n)))
    print("  M " + marks(m))
    print("    " + "".join(f"{s + k * 0.2:<10.1f}" for k in range(n // 10 + 1)))
    print("  W: " + "  ".join(f"{x['w']}@{x['s']:.2f}" for x in w if s <= x["s"] < e))
    print("  M: " + "  ".join(f"{x['w']}@{x['s']:.2f}" for x in m if s <= x["s"] < e))


def apply_overrides(out: list[dict], spec: str | None) -> None:
    """--set "5=8.08,16=15.56": verified word starts (index from the align table),
    then keep the sequence monotonic with no overlaps."""
    for item in (spec.split(",") if spec else []):
        i, t = item.split("=")
        out[int(i)]["s"] = float(t)
        out[int(i)]["src"] = "set"
    for k in range(1, len(out)):
        out[k]["s"] = max(out[k]["s"], out[k - 1]["s"] + 0.02)
        out[k - 1]["e"] = min(out[k - 1]["e"], out[k]["s"])
        out[k]["e"] = max(out[k]["e"], out[k]["s"] + 0.02)


def cmd_onsets(a) -> None:
    """Every phrase start after a pause, with 0.4 s of speech profile before and
    0.6 s after it (20 ms/char): the word should start where the voice gets loud."""
    job = resolve_job(a.job)
    db, thr = envelope(job / "in.wav")
    w = json.loads((job / "words_aligned.json").read_text(encoding="utf-8"))
    def ch(t):
        d = db[min(len(db) - 1, int(t / FRAME))] - thr
        return "█" if d > 12 else "▅" if d > 6 else "▃" if d > 0 else "▁" if d > -6 else "·"
    for k, x in enumerate(w):
        if k and x["s"] - w[k - 1]["e"] < 0.2:
            continue
        s0 = max(0.0, x["s"] - 0.4)
        row = "".join(ch(s0 + i * 0.02) for i in range(50))
        cut = int(round((x["s"] - s0) / 0.02))
        print(f"{k:3} {x['w']:16} {x['s']:6.2f}  {row[:cut]}|{row[cut:]}")


def cmd_merge(a) -> None:
    """words_aligned.json = whisper timings, with MMS timings (or the average of
    both) for words whose whisper start falls in the given ranges."""
    job = resolve_job(a.job)
    w = json.loads((job / "words_whisper.json").read_text(encoding="utf-8"))
    m = json.loads((job / "words_mms.json").read_text(encoding="utf-8"))
    if [x["w"] for x in w] != [x["w"] for x in m]:
        sys.exit("whisper and mms word lists differ — re-run both aligns on the same script.txt")
    if a.median:
        # Third opinion: Whisper's own transcription timestamps, matched by word.
        import difflib
        import statistics
        t = json.loads((job / "words_large.json").read_text(encoding="utf-8"))
        norm = lambda x: re.sub(r"\W", "", x["w"].lower())
        sm = difflib.SequenceMatcher(None, [norm(x) for x in w], [norm(x) for x in t], autojunk=False)
        third = {}
        for blk in sm.get_matching_blocks():
            for j in range(blk.size):
                third[blk.a + j] = t[blk.b + j]
        out = []
        for i, (x, y) in enumerate(zip(w, m)):
            z = dict(x)
            starts, ends = [x["s"], y["s"]], [x["e"], y["e"]]
            if i in third and third[i]["e"] - third[i]["s"] > 0.01:
                starts.append(third[i]["s"]); ends.append(third[i]["e"])
            z.update(s=round(statistics.median(starts), 3), e=round(statistics.median(ends), 3),
                     src=f"median{len(starts)}")
            out.append(z)
        apply_overrides(out, a.set)
        (job / "words_aligned.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"merged by median: {sum(z['src'] == 'median3' for z in out)} words from 3 sources, "
              f"{sum(z['src'] == 'median2' for z in out)} from 2 → {job / 'words_aligned.json'}")
        return
    rng = lambda spec: [tuple(map(float, r.split("-"))) for r in spec.split(",")] if spec else []
    use_mms, use_avg = rng(a.mms), rng(a.avg)
    inside = lambda t, rs: any(lo <= t < hi for lo, hi in rs)
    out = []
    for x, y in zip(w, m):
        z = dict(x)
        if inside(x["s"], use_mms):
            z.update(s=y["s"], e=y["e"], src="mms")
        elif inside(x["s"], use_avg):
            z.update(s=round((x["s"] + y["s"]) / 2, 3), e=round((x["e"] + y["e"]) / 2, 3), src="avg")
        out.append(z)
    apply_overrides(out, a.set)
    (job / "words_aligned.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"merged: {sum(z.get('src') == 'mms' for z in out)} mms, {sum(z.get('src') == 'avg' for z in out)} averaged, "
          f"{sum('src' not in z for z in out)} whisper → {job / 'words_aligned.json'}")


def cmd_build(a) -> None:
    job = resolve_job(a.job)
    words = json.loads((job / f"words_{a.words}.json").read_text(encoding="utf-8"))
    chunks = json.loads((job / a.chunks).read_text(encoding="utf-8"))
    # a bare string is shorthand for [number of words in it, string]
    # (standalone punctuation like the "…" in "immer …" is not a word of its own)
    import string as _string
    words_in = lambda c: sum(not all(ch in _string.punctuation + "«»‹›…–" for ch in t) for t in c.split())
    chunks = [[words_in(c), c] if isinstance(c, str) else c for c in chunks]

    total = sum(n for n, _ in chunks)
    if total != len(words):
        sys.exit(f"chunks cover {total} tokens but the word table has {len(words)} — fix chunks.json")

    events, i = [], 0
    for n, text in chunks:
        ws = words[i:i + n]
        i += n
        if text is None:
            continue
        if a.swiss:
            for k, v in SWISS.items():
                text = text.replace(k, v)
        events.append({"s": ws[0]["s"], "e": ws[-1]["e"], "text": text,
                       "heard": " ".join(w["w"] for w in ws)})

    hold, min_dur, warn_short = (TIGHT_HOLD, TIGHT_MIN_DURATION, TIGHT_WARN_SHORT) if a.tight \
        else (HOLD, MIN_DURATION, WARN_SHORT)
    if a.hold is not None:
        hold = a.hold
    warn_words = a.max_words or (2 if a.tight else WARN_WORDS)

    if a.tight and a.words != "aligned":
        snap_to_audio(events, job / "in.wav")
        drift = model_drift(job, a.words)
        for ev in events:
            ev["drift"] = [d for t, d in drift if ev["s"] - 0.05 <= t < ev["e"]]

    talking = None
    if a.tight:                                       # join across gaps that still contain speech
        db, thr = envelope(job / "in.wav")
        talking = lambda t0, t1: (db[int(t0 / FRAME):int(t1 / FRAME)] >= thr).mean() > 0.5 if t1 > t0 else True

    for k, ev in enumerate(events):
        nxt = events[k + 1]["s"] if k + 1 < len(events) else ev["e"] + 2 * hold
        join = nxt - ev["e"] <= JOIN_GAP or (talking and k + 1 < len(events)
                                             and nxt - ev["e"] <= TALK_JOIN_MAX and talking(ev["e"], nxt))
        ev["e"] = nxt if join else min(nxt, ev["e"] + hold)
        ev["e"] = max(ev["e"], min(nxt, ev["s"] + min_dur))

    if a.tight:
        if a.words == "aligned":
            snap_to_dips(events, job / "in.wav")
        balance(events, BALANCE_MIN if warn_words <= 2 else BALANCE_MIN_WIDE)

    name = a.name or job.name
    OUTPUT.mkdir(exist_ok=True)
    paths = []
    for suffix, offset in (("", TC_OFFSET), (" (00h start)", 0.0)):
        path = OUTPUT / f"{name}{suffix}.srt"
        with path.open("w", encoding="utf-8") as f:
            for n, ev in enumerate(events, 1):
                body = f"<i><b>{ev['text']}</b></i>" if a.style else ev["text"]
                f.write(f"{n}\n{tc(ev['s'], offset)} --> {tc(ev['e'], offset)}\n{body}\n\n")
        paths.append(path)

    print("  start    end   dur  text                         | heard")
    warnings = 0
    for ev in events:
        dur = ev["e"] - ev["s"]
        flags = []
        if dur < warn_short:
            flags.append("SHORT")
        if len(ev["text"].split()) > warn_words:
            flags.append("WORDS")
        if len(ev["text"]) > WARN_CHARS:
            flags.append("LONG")
        if ev.get("drift"):
            flags.append("DRIFT " + "/".join(f"{d:+.2f}" for d in ev["drift"]))
        warnings += bool(flags)
        snap = "".join(f" {k}{v:+.2f}" for k, v in ev.get("snap", {}).items())
        print(f"{ev['s']:6.2f} {ev['e']:6.2f} {dur:5.2f}  {ev['text']:28} | {ev['heard']}"
              + (f"   ⇥{snap}" if snap else "")
              + (f"   ⚠ {', '.join(flags)}" if flags else ""))
    print(f"\n{len(events)} events, {warnings} with warnings")
    for p in paths:
        print(f"WROTE {p}")


# ── cli ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe audio with both models")
    t.add_argument("audio")
    t.add_argument("--language", default="de")
    t.set_defaults(fn=cmd_transcribe)

    tb = sub.add_parser("table", help="re-print the indexed word table of a job")
    tb.add_argument("job")
    tb.set_defaults(fn=cmd_table)

    c = sub.add_parser("clip", help="re-transcribe time ranges in isolation")
    c.add_argument("job")
    c.add_argument("ranges", nargs="+", type=float)
    c.add_argument("--language", default="de")
    c.add_argument("--prompt", help="context hint for Whisper: topic words, names, spellings")
    c.set_defaults(fn=cmd_clip)

    al = sub.add_parser("align", help="force-align the final text in script.txt to the audio")
    al.add_argument("job")
    al.add_argument("--script", default="script.txt")
    al.add_argument("--engine", default="whisper", choices=["whisper", "mms"],
                    help="whisper: attention DTW, robust to dialect vs. standard text; "
                         "mms: phoneme-level CTC, precise when the words match the sounds (needs torch)")
    al.add_argument("--language", default="de")
    al.set_defaults(fn=cmd_align)

    sc = sub.add_parser("score", help="rank alternative wordings of a passage against the audio")
    sc.add_argument("job")
    sc.add_argument("start", type=float)
    sc.add_argument("end", type=float)
    sc.add_argument("candidates", nargs="+")
    sc.add_argument("--language", default="de")
    sc.set_defaults(fn=cmd_score)

    cp = sub.add_parser("compare", help="speech profile with whisper vs mms word starts")
    cp.add_argument("job")
    cp.add_argument("start", type=float)
    cp.add_argument("end", type=float)
    cp.set_defaults(fn=cmd_compare)

    on = sub.add_parser("onsets", help="check every phrase start against the voice onset")
    on.add_argument("job")
    on.set_defaults(fn=cmd_onsets)

    mg = sub.add_parser("merge", help="combine whisper and mms alignments into words_aligned.json")
    mg.add_argument("job")
    mg.add_argument("--mms", help="ranges (s) where mms timing wins, e.g. 31.3-32.5,38.8-39.5")
    mg.add_argument("--avg", help="ranges (s) where both are averaged")
    mg.add_argument("--set", help='verified word starts as index=seconds, e.g. "5=8.08,16=15.56"')
    mg.add_argument("--median", action="store_true",
                    help="per word, median of whisper-align, mms and the transcription timestamps "
                         "(use when the final text is close to what Whisper transcribed)")
    mg.set_defaults(fn=cmd_merge)

    b = sub.add_parser("build", help="build SRTs from chunks.json")
    b.add_argument("job")
    b.add_argument("--chunks", default="chunks.json")
    b.add_argument("--words", default="large", choices=["large", "turbo", "aligned"],
                   help="word timing source (aligned = output of 'align')")
    b.add_argument("--name", help="output file name (default: audio file stem)")
    b.add_argument("--swiss", action="store_true", help="enforce ß→ss")
    b.add_argument("--no-style", dest="style", action="store_false", help="omit <i><b> tags")
    b.add_argument("--tight", action="store_true",
                   help="snap pause boundaries to the audio, short hold, flag model timing drift")
    b.add_argument("--hold", type=float, help="seconds an event stays after speech before a pause "
                   "(default 0.4, tight 0.1)")
    b.add_argument("--max-words", type=int, help="warn above this many words per event (default 4, tight 2)")
    b.set_defaults(fn=cmd_build)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
