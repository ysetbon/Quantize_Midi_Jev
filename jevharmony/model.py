"""Suggest the next harmony and its exact piano notes from the recent chords.

Code proposes, Jev picks:

1. ``label_of`` turns exact notes into a chord label in the key: root scale
   degree, quality, inversion (e.g. V65 = dominant seventh, first inversion).
2. ``NGram`` (trained on chorales) ranks every candidate next chord.
3. ``voicings`` enumerates every exact four-voice voicing of a candidate that
   fits the voice ranges and spacing, has no parallel fifths/octaves against
   the previous chord, and ranks them by voice-leading cost.
4. One Jev request asks, in words and note names, which chord comes next
   and, for the code's top chords, which exact voicing (speculative fan-out).
5. Code combines: code log-score + a * log P(Jev).
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from itertools import product
from math import exp, log

from music21 import chord as m21chord
from music21 import pitch as m21pitch

from jevquant.jev import ask

TEMPLATES = {  # quality -> intervals above the root (fifth optional for sevenths)
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "dom7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "hdim7": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
}
DEGREE = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
DEGREE_WORD = ["tonic", "lowered supertonic", "supertonic", "lowered mediant", "mediant", "subdominant",
               "raised subdominant", "dominant", "lowered submediant", "submediant", "subtonic", "leading tone"]
QUALITY_WORD = {"maj": "major triad", "min": "minor triad", "dim": "diminished triad", "aug": "augmented triad",
                "dom7": "dominant seventh chord", "maj7": "major seventh chord", "min7": "minor seventh chord",
                "hdim7": "half-diminished seventh chord", "dim7": "diminished seventh chord"}
INV_WORD = ["root position", "first inversion", "second inversion", "third inversion"]
RANGES = [(40, 62), (48, 69), (53, 74), (60, 81)]  # bass, tenor, alto, soprano
VOICE = ["bass", "tenor", "alto", "soprano"]
TONIC_PC = {n: i for i, n in enumerate(["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"])}


def tonic_pc(name: str) -> int:
    return m21pitch.Pitch(name).pitchClass


def label_of(notes: list[int], tonic: int) -> tuple | None:
    """(degree, quality, inversion) of exact notes in a key with tonic pitch class ``tonic``."""
    pcs = {n % 12 for n in notes}
    bass = min(notes) % 12
    roots = [m21chord.Chord(notes).root().pitchClass] + sorted(pcs)
    for r in roots:
        rel = {(p - r) % 12 for p in pcs}
        for q, t in TEMPLATES.items():
            need = set(t) - ({7} if len(t) == 4 else set())
            if need <= rel <= set(t):
                inv = t.index((bass - r) % 12) if (bass - r) % 12 in t else None
                if inv is not None:
                    return ((r - tonic) % 12, q, inv)
    return None


def name_of(lab: tuple) -> str:
    deg, q, inv = lab
    s = DEGREE[deg]
    if q in ("min", "dim", "min7", "hdim7", "dim7"):
        s = s.lower()
    s += {"dim": "o", "aug": "+", "hdim7": "ø", "dim7": "o"}.get(q, "")
    if len(TEMPLATES[q]) == 4:
        s += ["7", "65", "43", "42"][inv] if q != "maj7" else ["M7", "M65", "M43", "M42"][inv]
    else:
        s += ["", "6", "64"][inv]
    return s


def words_of(lab: tuple, tonic: int) -> str:
    deg, q, inv = lab
    root = m21pitch.Pitch((tonic + deg) % 12).name
    bass = m21pitch.Pitch((tonic + deg + TEMPLATES[q][inv]) % 12).name
    return f"{QUALITY_WORD[q]} on the {DEGREE_WORD[deg]} ({root}), {INV_WORD[inv]}, {bass} in the bass"


def note_name(m: int) -> str:
    return m21pitch.Pitch(midi=m).nameWithOctave


# ------------------------------------------------------------------ chord model

class NGram:
    """Interpolated trigram over chord labels, per mode."""

    def __init__(self, seqs: list[tuple[str, list[tuple]]], l3=0.5, l2=0.35, k=0.1):
        self.c3, self.c2, self.c1 = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.vocab: dict[str, Counter] = defaultdict(Counter)
        for mode, seq in seqs:
            for i, lab in enumerate(seq):
                self.c1[mode][lab] += 1
                self.vocab[mode][lab] += 1
                if i >= 1:
                    self.c2[(mode, seq[i - 1])][lab] += 1
                if i >= 2:
                    self.c3[(mode, seq[i - 2], seq[i - 1])][lab] += 1
        self.l3, self.l2, self.k = l3, l2, k

    def candidates(self, mode: str, min_count: int = 5) -> list[tuple]:
        return [lab for lab, n in self.vocab[mode].items() if n >= min_count]

    def probs(self, mode: str, hist: list[tuple], cands: list[tuple]) -> dict[tuple, float]:
        def dist(c: Counter):
            tot = sum(c.values()) + self.k * len(cands)
            return {x: (c[x] + self.k) / tot for x in cands}
        p1 = dist(self.c1[mode])
        p2 = dist(self.c2[(mode, hist[-1])]) if hist else p1
        p3 = dist(self.c3[(mode, hist[-2], hist[-1])]) if len(hist) >= 2 else p2
        w3 = self.l3 if len(hist) >= 2 and self.c3[(mode, hist[-2], hist[-1])] else 0
        w2 = self.l2 if hist and self.c2[(mode, hist[-1])] else 0
        return {x: w3 * p3[x] + w2 * p2[x] + (1 - w3 - w2) * p1[x] for x in cands}


# ------------------------------------------------------------------ voicings

def voicings(lab: tuple, tonic: int, prev: list[int], top: int = 8) -> list[tuple[tuple, float]]:
    """Exact (bass, tenor, alto, soprano) voicings of ``lab`` with their voice-leading cost."""
    deg, q, inv = lab
    root = (tonic + deg) % 12
    pcs = {(root + i) % 12 for i in TEMPLATES[q]}
    must = {(root + i) % 12 for i in TEMPLATES[q] if not (len(TEMPLATES[q]) == 4 and i == 7)}
    bass_pc = (root + TEMPLATES[q][inv]) % 12
    leading = (tonic + 11) % 12
    seventh = (root + TEMPLATES[q][3]) % 12 if len(TEMPLATES[q]) == 4 else None
    opts = [[m for m in range(lo, hi + 1) if m % 12 in pcs] for lo, hi in RANGES]
    opts[0] = [m for m in opts[0] if m % 12 == bass_pc]
    out = []
    for v in product(*opts):
        b, t, a, s = v
        if not (b < t < a < s) or s - a > 12 or a - t > 12 or t - b > 19:
            continue
        if not must <= {m % 12 for m in v}:
            continue
        if _parallels(prev, v):
            continue
        cost = sum(abs(x - y) for x, y in zip(v[1:], prev[1:])) + 0.5 * abs(b - prev[0])
        cost += sum(2.0 for x, y in zip(v[1:3], prev[1:3]) if abs(x - y) > 4)  # inner-voice leaps
        cost += 1.5 * (abs(s - prev[3]) > 7)
        doubled = Counter(m % 12 for m in v)
        cost += 4.0 * (doubled[leading] > 1) + (4.0 * (doubled[seventh] > 1) if seventh is not None else 0)
        cost += 1.0 * (len(TEMPLATES[q]) == 3 and inv == 0 and doubled[root] < 2)  # prefer doubled root
        out.append((v, cost))
    out.sort(key=lambda x: x[1])
    return out[:top]


def _parallels(prev, cur) -> bool:
    for i in range(4):
        for j in range(i + 1, 4):
            a0, a1 = (prev[j] - prev[i]) % 12, (cur[j] - cur[i]) % 12
            if a0 == a1 and a0 in (0, 7) and prev[i] != cur[i] and prev[j] != cur[j]:
                return True
    return False


def soft(costs: dict, tau: float) -> dict:
    m = min(costs.values())
    w = {k: exp(-(c - m) / tau) for k, c in costs.items()}
    z = sum(w.values())
    return {k: x / z for k, x in w.items()}


# ------------------------------------------------------------------ Jev

def _motion(prev: list[int], v: tuple) -> str:
    parts = []
    for name, a, b in zip(VOICE, prev, v):
        d = b - a
        how = "holds" if d == 0 else ("rises" if d > 0 else "falls") + (
            " a step" if abs(d) <= 2 else " a third" if abs(d) <= 4 else " a fourth" if abs(d) == 5 else " a leap")
        parts.append(f"{name} {note_name(a)}→{note_name(b)} {how}")
    return "; ".join(parts)


def voicing_words(prev: list[int], v: tuple, lab: tuple, tonic: int) -> str:
    deg, q, _ = lab
    root = (tonic + deg) % 12
    dbl = [m21pitch.Pitch(p).name for p, n in Counter(m % 12 for m in v).items() if n > 1]
    fifth_missing = (root + 7) % 12 not in {m % 12 for m in v} and len(TEMPLATES[q]) == 4
    spacing = "close position" if v[3] - v[1] <= 12 else "open position"
    return (f"{' '.join(note_name(m) for m in v)} ({spacing}; doubled: {', '.join(dbl) or 'none'}"
            f"{'; fifth omitted' if fifth_missing else ''}). Motion: {_motion(prev, v)}")


CHORD_Q = ("Based on the recent chords in `recent_chords`, which chord is most likely to come next in this "
           "Bach chorale (four-part harmony, Baroque common practice)? Consider the phrase position, cadences "
           "(a fermata marks a phrase ending) and typical progressions.")
VOICING_Q = ("If the next chord is `{name}`, which exact voicing would Bach write after the last chord? Consider "
             "smooth voice leading, resolution of tendency tones (leading tone up, sevenths down), doubling and "
             "the melodic line of the soprano.")


def jev_suggest(ctx: dict, chord_opts: list[tuple], voicing_opts: dict, model: str = "jev-latest",
                seed: str = "") -> tuple[dict, dict, dict]:
    """One request: P(next chord) over ``chord_opts`` and P(voicing | chord) for each chord in ``voicing_opts``."""
    tonic = ctx["tonic_pc"]
    rng = random.Random(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    order = list(chord_opts)
    rng.shuffle(order)
    crit = {name_of(l): words_of(l, tonic) for l in order}
    qs = {"next_chord": {"type": "choice", "instructions": CHORD_Q, "criteria": crit},
          "next_chord_reversed": {"type": "choice", "instructions": CHORD_Q, "criteria": dict(reversed(list(crit.items())))}}
    for j, (lab, vs) in enumerate(voicing_opts.items()):
        qs[f"voicing_{j}"] = {"type": "choice", "instructions": VOICING_Q.format(name=name_of(lab)),
                              "criteria": {f"v{i}": voicing_words(ctx["prev_notes"], v, lab, tonic) for i, (v, _) in enumerate(vs)}}
    r = ask(ctx["state"], qs, model)
    a = r["answers"]
    by_name = {name_of(l): l for l in order}
    pc = {by_name[n]: (a["next_chord"]["probabilities"].get(n, 0) + a["next_chord_reversed"]["probabilities"].get(n, 0)) / 2
          for n in by_name}
    pv = {}
    for j, (lab, vs) in enumerate(voicing_opts.items()):
        pv[lab] = {vs[int(k[1:])][0]: p for k, p in a[f"voicing_{j}"]["probabilities"].items()}
    return pc, pv, r


def state_of(ch: dict, i: int, hist_len: int = 6) -> dict:
    """What Jev sees: the key and the recent chords (names, words and exact notes)."""
    tonic = tonic_pc(ch["tonic"])
    recent = []
    for e in ch["events"][max(0, i - hist_len):i]:
        lab = label_of(e["notes"], tonic)
        recent.append({"chord": name_of(lab) if lab else "?", "meaning": words_of(lab, tonic) if lab else "unclear",
                       "notes_bass_to_soprano": " ".join(note_name(m) for m in e["notes"]),
                       **({"fermata": "phrase ends here"} if e["fermata"] else {})})
    return {"key": f"{ch['tonic'].replace('-', 'b')} {ch['mode']}", "style": "Bach chorale, four voices",
            "recent_chords": recent}
