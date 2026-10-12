"""Round 3, stage 2: the written length of every note.

Given a chosen hypothesis (bar lines + onset positions), code lists the
plausible written lengths of each note:

* until the next onset in the same hand (legato, the usual case),
* shorter, followed by a rest (detached / staccato),
* longer, sounding under the next notes (a held voice),

and a timing model ranks them from how long the key was held (converted to
written beats with the passage's tempo map).  Jev is asked only about the notes
where that model is unsure (confidence-gated routing): one Choice per note,
options described in words, and its probabilities multiply the model's.
"""
from __future__ import annotations

from fractions import Fraction as F
from math import exp, log

import numpy as np

from .jev import ask
from .notevalues import note_name

SPLIT = 60  # same staff split as notation.py
STD = [F(1, 8), F(1, 6), F(1, 4), F(1, 3), F(3, 8), F(1, 2), F(2, 3), F(3, 4), F(1), F(3, 2), F(2), F(3), F(4), F(6)]
NAMES = {F(1, 8): "thirty-second note", F(1, 6): "sextuplet sixteenth", F(1, 4): "sixteenth note",
         F(1, 3): "triplet eighth", F(3, 8): "dotted sixteenth", F(1, 2): "eighth note", F(2, 3): "triplet quarter",
         F(3, 4): "dotted eighth", F(1): "quarter note", F(3, 2): "dotted quarter", F(2): "half note",
         F(3): "dotted half", F(4): "whole note", F(6): "dotted whole"}


def value_words(v: F) -> str:
    if v in NAMES:
        return NAMES[v]
    return f"{float(v):g} beats (tied notes)"


def note_context(chunk, h, cl) -> list[dict]:
    """Per performed note: written onset, hand, gap to the next onset in that hand, held beats."""
    pos = h.positions()
    tm = h.cand.timemap
    end = (max(pos) // h.L + 1) * h.L  # passage ends at the next bar line
    staff_on: dict[bool, list[F]] = {True: [], False: []}
    for i, n in enumerate(chunk.perf):
        staff_on[n.pitch >= SPLIT].append(pos[cl.note_cluster[i]])
    for k in staff_on:
        staff_on[k] = sorted(set(staff_on[k]))
    out = []
    for i, n in enumerate(chunk.perf):
        s = pos[cl.note_cluster[i]]
        later = [x for x in staff_on[n.pitch >= SPLIT] if x > s]
        g = (later[0] if later else end) - s
        b_on, b_off = tm.beat(np.array([n.onset, n.offset]))
        held = max(float(b_off - b_on), 0.02)
        out.append({"i": i, "onset": s, "rh": n.pitch >= SPLIT, "gap": g, "held": held, "pitch": n.pitch,
                    "last": not later})
    return out


def options(ctx: dict, k: int = 6) -> list[F]:
    g, held = ctx["gap"], ctx["held"]
    vals = {g} | {v for v in STD if v < g} | {v for v in STD if g < v <= 4 * g and v <= 2.5 * held}
    ranked = sorted(vals, key=lambda v: abs(log(held / float(v))))
    keep = ranked[:k]
    if g not in keep:
        keep[-1] = g
    return sorted(keep)


def code_probs(ctx: dict, opts: list[F], sigma: float = 0.45, ratio: float = 0.8, legato: float = 1.5,
               cx: float = 0.3) -> dict[F, float]:
    """Timing model: log(held / (ratio * value)) ~ N(0, sigma), plus a bonus for
    'until the next note' and a small penalty for tuplet / dotted-16th values."""
    w = {}
    for v in opts:
        r = ratio if v <= ctx["gap"] else 1.0  # held voices are held about their full length
        s = -0.5 * (log(ctx["held"] / (r * float(v))) / sigma) ** 2
        s += legato * (v == ctx["gap"]) - cx * (v.denominator % 3 == 0 or v == F(3, 8)) - 2.0 * (v > ctx["gap"])
        w[v] = s
    m = max(w.values())
    z = sum(exp(x - m) for x in w.values())
    return {v: exp(x - m) / z for v, x in w.items()}


def legato_rule(ctx: dict) -> F:
    """The plain engraving rule used by notation.py: legato unless clearly released early."""
    if ctx["held"] >= 0.5 * float(ctx["gap"]):
        return ctx["gap"]
    return min(STD, key=lambda v: abs(log(ctx["held"] / float(v))))


def _held_words(ctx: dict) -> str:
    r = ctx["held"] / float(ctx["gap"])
    feel = ("held about as long as the time until the next note in this hand" if 0.75 <= r <= 1.15 else
            "released early, well before the next note in this hand" if r < 0.5 else
            "released a little before the next note in this hand" if r < 0.75 else
            "held past the next note in this hand")
    near = min(STD, key=lambda v: abs(log(ctx["held"] / float(v))))
    return f"{feel}; the key was held for about a {value_words(near)} ({round(100 * r)}% of the time until the next note)"


def _bar_beat(x: F, L: F, u: F) -> str:
    bar = int(x // L)
    within = x - bar * L
    beat = int(within // u) + 1
    off = within - (beat - 1) * u
    where = f"beat {beat}" if off == 0 else f"beat {beat} plus {value_words(off)}"
    return ("pickup, " if bar < 0 else f"bar {bar + 1}, ") + where


def option_text(v: F, ctx: dict) -> str:
    name = value_words(v)
    if v == ctx["gap"]:
        return f"{name}: lasts exactly until the next note in this hand starts (legato)"
    if v < ctx["gap"]:
        return f"{name}, then a rest of {value_words(ctx['gap'] - v)} before the next note in this hand"
    return f"{name}: keeps sounding while this hand plays the next notes (a separate held voice)"


INSTRUCTIONS = ("Which written note length should this note have in the sheet music? Pianists usually release "
                "keys a little early (about 70-95% of the written length) when playing legato, much earlier for "
                "staccato, and they hold bass notes and inner voices for their full written length. Choose what a "
                "skilled engraver would write for this piece.")


def jev_durations(chunk, h, ctxs: list[dict], ask_idx: list[int], model: str = "jev-latest", max_q: int = 48):
    """Jev's probabilities over the options of the notes in ``ask_idx``."""
    state = {"passage": {"time_signature": h.ts, "tempo": f"{round(h.cand.bpm)} quarter notes per minute",
                         "music": "classical piano"}}
    qs = {}
    for i in ask_idx:
        c = ctxs[i]
        opts = options(c)
        qs[f"note_{i:03d}"] = {
            "type": "choice",
            "instructions": {"question": INSTRUCTIONS, "note": {
                "pitch": note_name(c["pitch"]), "hand": "right hand (treble staff)" if c["rh"] else "left hand (bass staff)",
                "starts": _bar_beat(c["onset"], h.L, h.unit),
                "next_note_in_this_hand": ("comes " + value_words(c["gap"]) + " later") if not c["last"] else
                "none: the passage ends " + value_words(c["gap"]) + " later",
                "played": _held_words(c)}},
            "criteria": {f"opt_{j}": option_text(v, c) for j, v in enumerate(opts)},
        }
    probs, usage = {}, {"input_tokens": 0, "requests": 0, "latency_s": 0.0}
    names = list(qs)
    for a in range(0, len(names), max_q):
        part = {n: qs[n] for n in names[a:a + max_q]}
        r = ask(state, part, model)
        for n, ans in r["answers"].items():
            i = int(n.split("_")[1])
            opts = options(ctxs[i])
            probs[i] = {opts[int(k.split("_")[1])]: p for k, p in ans["probabilities"].items()}
        usage["input_tokens"] += r["usage"]["input_tokens"]
        usage["requests"] += 1
        usage["latency_s"] = max(usage["latency_s"], r.get("latency_s") or 0)
    return probs, usage
