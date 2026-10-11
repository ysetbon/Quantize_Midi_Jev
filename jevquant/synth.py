"""Synthetic piano chunks: a random notated score plus a humanized performance.

There is no public, downloadable aligned performance/score dataset reachable
from this environment, so the benchmark generates its own.  Because each
performed note is generated *from* a score note, the ground truth alignment is
exact, which is what we need to score a quantizer.

Every chunk starts with a note on the downbeat (position 0).
"""
from __future__ import annotations

import random
from fractions import Fraction as F

import numpy as np

from .model import Chunk, PerfNote, ScoreNote

# Rhythm cells, in quarter notes.  A negative value is a rest of that length.
CELLS: dict[str, list[list[F]]] = {
    "quarters": [[F(1)], [F(1)], [F(2)], [F(1, 2), F(1, 2)]],
    "eighths": [[F(1)], [F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)], [F(3, 2), F(1, 2)]],
    "sixteenths": [
        [F(1, 4)] * 4,
        [F(1, 2), F(1, 4), F(1, 4)],
        [F(1, 4), F(1, 4), F(1, 2)],
        [F(1, 2), F(1, 2)],
        [F(3, 4), F(1, 4)],
    ],
    "dotted": [[F(3, 4), F(1, 4)], [F(3, 4), F(1, 4)], [F(1, 2), F(1, 2)], [F(1)], [F(3, 2), F(1, 2)]],
    "triplets": [[F(1, 3)] * 3, [F(1, 3)] * 3, [F(2, 3), F(1, 3)], [F(1)]],
    "mixed": [[F(1, 3)] * 3, [F(1, 2), F(1, 2)], [F(1, 4)] * 4, [F(1)], [F(1, 2), F(1, 4), F(1, 4)]],
    "syncopated": [[F(1, 2), F(1), F(1, 2)], [F(1, 2), F(1, 2), F(1)], [F(3, 4), F(3, 4), F(1, 2)], [F(1)]],
    "rests": [[F(-1, 2), F(1, 2)], [F(1, 2), F(-1, 2)], [F(1)], [F(1, 2), F(1, 2)], [F(-1)]],
}
# 6/8 cells: one dotted-quarter beat (3 eighths = 3/2 quarter notes) each.
CELLS_68 = [
    [F(1, 2)] * 3,
    [F(1, 2)] * 3,
    [F(1), F(1, 2)],
    [F(3, 2)],
    [F(1, 4), F(1, 4), F(1, 2), F(1, 2)],
    [F(3, 4), F(1, 4), F(1, 2)],
]
# Tempo range (quarter-note bpm) where each style is idiomatic.
TEMPO = {
    "quarters": (70, 150),
    "eighths": (60, 140),
    "sixteenths": (50, 100),
    "dotted": (55, 120),
    "triplets": (55, 120),
    "mixed": (55, 100),
    "syncopated": (70, 140),
    "rests": (60, 130),
}
METER_BEATS = {"4/4": F(4), "3/4": F(3), "6/8": F(3)}

DIFFICULTY = {
    #          jitter ms, tempo drift, chord spread ms, rit at end
    "easy": dict(jitter=0.008, drift=0.02, spread=0.006, rit=0.0),
    "medium": dict(jitter=0.016, drift=0.05, spread=0.012, rit=0.05),
    "hard": dict(jitter=0.026, drift=0.08, spread=0.018, rit=0.12),
}

C_MAJOR = [0, 2, 4, 5, 7, 9, 11]


def _scale_walk(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    degrees = [o * 12 + d for o in range(0, 11) for d in C_MAJOR]
    degrees = [p for p in degrees if lo <= p <= hi]
    i = rng.randrange(len(degrees))
    out = []
    for _ in range(n):
        out.append(degrees[i])
        i = min(max(i + rng.choice([-2, -1, -1, 1, 1, 2, 0, 3, -3]), 0), len(degrees) - 1)
    return out


def _fill(rng: random.Random, cells: list[list[F]], length: F, cell_len) -> list[F]:
    """Concatenate random cells until ``length`` quarter notes are filled."""
    out: list[F] = []
    total = F(0)
    while total < length:
        options = [c for c in cells if total + sum(abs(x) for x in c) <= length]
        if cell_len is not None:
            options = [c for c in options if sum(abs(x) for x in c) == cell_len] or options
        c = rng.choice(options)
        out.extend(c)
        total += sum(abs(x) for x in c)
    return out


def make_score(rng: random.Random, meter: str, style: str, bars: int, accomp: str) -> list[ScoreNote]:
    bar_len = METER_BEATS[meter]
    length = bar_len * bars
    if meter == "6/8":
        rhythm = _fill(rng, CELLS_68, length, F(3, 2))
    else:
        rhythm = _fill(rng, CELLS[style], length, None)
    if rhythm[0] < 0:  # chunks always start with a note on the downbeat
        rhythm[0] = -rhythm[0]
    pitches = _scale_walk(rng, len(rhythm), 64, 84)
    notes: list[ScoreNote] = []
    pos = F(0)
    for dur, p in zip(rhythm, pitches):
        if dur > 0:
            notes.append(ScoreNote(p, pos, dur, hand="R"))
        pos += abs(dur)

    # Left hand.  Bass pitches follow a simple I-IV-V-I-ish root motion.
    roots = [48, 53, 55, 48, 45, 50, 43, 48]
    beat = F(3, 2) if meter == "6/8" else F(1)
    for b in range(bars):
        root = roots[(b + rng.randrange(2)) % len(roots)]
        start = bar_len * b
        if accomp == "bass":
            if meter == "4/4":
                notes += [ScoreNote(root, start, F(2), hand="L"), ScoreNote(root + 7, start + 2, F(2), hand="L")]
            elif meter == "6/8":
                notes += [ScoreNote(root, start, F(3, 2), hand="L"), ScoreNote(root + 7, start + F(3, 2), F(3, 2), hand="L")]
            else:
                notes.append(ScoreNote(root, start, F(3), hand="L"))
        elif accomp == "chords":
            # waltz: bass on 1, chord on the other beats; others: chord every beat
            n_beats = int(bar_len / beat)
            for k in range(n_beats):
                t = start + beat * k
                if k == 0 or meter != "3/4":
                    notes.append(ScoreNote(root - 12 if k == 0 else root, t, beat, hand="L"))
                if k > 0 or meter == "4/4":
                    for iv in (4, 7):
                        notes.append(ScoreNote(root + iv, t, beat, hand="L"))
        elif accomp == "broken":
            step = F(1, 2)
            pattern = [0, 7, 4, 7] if meter == "4/4" else [0, 7, 4, 12, 4, 7]
            for k in range(int(bar_len / step)):
                notes.append(ScoreNote(root + pattern[k % len(pattern)], start + step * k, step, hand="L"))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def accent(meter: str, pos: F) -> int:
    """Metrical velocity accent, so the meter is audible like in a real performance."""
    bar_len = METER_BEATS[meter]
    p = pos % bar_len
    if p == 0:
        return 14
    if meter == "4/4" and p == 2:
        return 7
    if meter == "6/8" and p == F(3, 2):
        return 8
    if p.denominator == 1 and meter != "6/8":
        return 3
    return 0


def humanize(rng: random.Random, notes: list[ScoreNote], meter: str, bpm: float, level: str) -> list[PerfNote]:
    prm = DIFFICULTY[level]
    nprng = np.random.default_rng(rng.randrange(2**32))
    length = float(max(n.onset + n.duration for n in notes))
    # Beat period as a function of score position: linear drift + slow random walk
    # + ritardando over the last beat.
    res = 48
    grid = np.arange(0, length + 1, 1 / res)
    p0 = 60.0 / bpm
    drift = rng.uniform(-prm["drift"], prm["drift"])
    walk = np.cumsum(nprng.normal(0, prm["drift"] / 6, len(grid))) / np.sqrt(res)
    rit = np.clip((grid - (length - 1)) / 1.0, 0, 1) * prm["rit"]
    period = p0 * (1 + drift * (grid / length - 0.5) + walk + rit)
    times = np.concatenate([[0.0], np.cumsum(period[:-1] / res)])

    def t_of(pos: F) -> float:
        return float(np.interp(float(pos), grid, times))

    artic = rng.uniform(0.55, 1.02)
    base_vel = rng.randint(52, 72)
    chord_shift: dict[F, float] = {}
    out = []
    for n in notes:
        if n.onset not in chord_shift:  # shared timing error of a chord, plus per-note spread
            chord_shift[n.onset] = nprng.normal(0, prm["jitter"])
        lead = -0.008 if n.hand == "R" else 0.0  # melody lead
        on = t_of(n.onset) + chord_shift[n.onset] + nprng.normal(0, prm["spread"]) + lead
        dur = (t_of(n.onset + n.duration) - t_of(n.onset)) * max(0.3, artic + nprng.normal(0, 0.06))
        vel = base_vel + accent(meter, n.onset) + int(nprng.normal(0, 5)) + (6 if n.hand == "R" else 0)
        out.append(PerfNote(n.pitch, on, on + max(0.05, dur), int(np.clip(vel, 20, 120)), n.hand))
    t0 = min(p.onset for p in out)
    for p in out:
        p.onset -= t0
        p.offset -= t0
    return out


def make_chunk(seed: int, level: str = "medium", meter: str | None = None, style: str | None = None,
               bars: int = 2, accomp: str | None = None) -> Chunk:
    rng = random.Random(seed)
    meter = meter or rng.choices(["4/4", "3/4", "6/8"], [0.6, 0.25, 0.15])[0]
    style = style or rng.choice(list(CELLS))
    if meter == "6/8":
        style = "compound"
    accomp = accomp or rng.choices(["none", "bass", "chords", "broken"], [0.3, 0.35, 0.2, 0.15])[0]
    if meter == "6/8":
        bpm = rng.uniform(40, 80) * 1.5  # dotted-quarter tempo -> quarter-note bpm
    else:
        bpm = rng.uniform(*TEMPO[style])
    truth = make_score(rng, meter, style, bars, accomp)
    perf = humanize(rng, truth, meter, bpm, level)
    order = sorted(range(len(perf)), key=lambda i: (perf[i].onset, perf[i].pitch))
    return Chunk(
        perf=[perf[i] for i in order],
        truth=[truth[i] for i in order],
        meta=dict(seed=seed, level=level, meter=meter, style=style, accomp=accomp, bpm=bpm, bars=bars),
    )
