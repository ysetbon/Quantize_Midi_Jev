"""Bach chorales as sequences of exact four-voice chords.

Each event is one change of sounding chord on a beat: the exact pitches of
bass, tenor, alto and soprano (MIDI numbers), its Roman numeral in the
chorale's key, and whether the soprano carries a fermata (a phrase ending).
Only beats whose four notes form a plain triad or seventh chord are kept as
events; beats with passing tones are skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

from music21 import chord, corpus, expressions, key, roman

CACHE = "data/chorales.json"


def _sounding(part, offset):
    """The note sounding in ``part`` at ``offset`` (quarter notes), or None."""
    for n in part.recurse().notes:
        start = n.getOffsetInHierarchy(part)
        if start <= offset < start + n.quarterLength:
            return n if n.isNote else None
        if start > offset:
            break
    return None


def chorale_events(path: str) -> dict | None:
    s = corpus.parse(path)
    if len(s.parts) != 4:
        return None
    parts = list(s.parts)  # soprano, alto, tenor, bass
    k = s.analyze("key")
    end = s.highestTime
    events, last = [], None
    beat = 0.0
    while beat < end:
        notes = [_sounding(p, beat) for p in parts]
        if all(n is not None for n in notes):
            midi = [n.pitch.midi for n in reversed(notes)]  # bass, tenor, alto, soprano
            c = chord.Chord([n.pitch for n in notes])
            if (c.isTriad() or c.isSeventh()) and midi != last:
                fig = roman.romanNumeralFromChord(c, k).figure
                ferm = any(isinstance(e, expressions.Fermata) for e in notes[0].expressions)
                events.append({"offset": beat, "notes": midi, "figure": fig, "fermata": ferm})
                last = midi
        beat += 1.0
    if len(events) < 8:
        return None
    name = os.path.basename(str(path)).rsplit(".", 1)[0]
    return {"id": name, "tonic": k.tonic.name, "mode": k.mode, "events": events}


def build(out: str = CACHE, workers: int = 4) -> list[dict]:
    if os.path.exists(out):
        return json.load(open(out))
    paths = [str(p) for p in corpus.getComposer("bach") if "bwv" in str(p)]
    with ProcessPoolExecutor(workers) as ex:
        res = [r for r in ex.map(chorale_events, paths, chunksize=8) if r]
    seen, uniq = set(), []
    for r in res:  # some chorales appear twice in the corpus
        sig = tuple(tuple(e["notes"]) for e in r["events"][:12])
        if sig not in seen:
            seen.add(sig)
            uniq.append(r)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(uniq, open(out, "w"))
    return uniq


def split_of(chorale_id: str) -> str:
    h = int(hashlib.md5(chorale_id.encode()).hexdigest()[:8], 16) % 10
    return "test" if h < 2 else "dev" if h < 3 else "train"


def key_of(ch: dict) -> key.Key:
    return key.Key(ch["tonic"], ch["mode"])
