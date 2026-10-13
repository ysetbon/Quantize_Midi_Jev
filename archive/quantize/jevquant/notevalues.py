"""Round 2: Jev names each note's written value (quarter, eighth, triplet...).

Instead of choosing between whole-bar notations prepared by code, Jev gets the
played notes of one bar and answers one ``choice`` question per onset group:
which written note value fills the time until the next onset.  The answers are
turned into positions either by summing Jev's top choices (``decode_argmax``)
or by the best-scoring sequence that exactly fills the bar (``decode_fit``).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import log

import numpy as np

from .candidates import cluster_onsets
from .jev import ask

# name -> (length in quarter notes, description for Jev)
VALUES: dict[str, tuple[F, str]] = {
    "whole": (F(4), "whole note: 4 quarter-note beats"),
    "dotted_half": (F(3), "dotted half note: 3 quarter-note beats"),
    "half": (F(2), "half note: 2 quarter-note beats"),
    "dotted_quarter": (F(3, 2), "dotted quarter note: 1.5 beats"),
    "quarter": (F(1), "quarter note: 1 beat"),
    "dotted_eighth": (F(3, 4), "dotted eighth note: 3/4 of a beat"),
    "triplet_quarter": (F(2, 3), "quarter-note triplet: 2/3 of a beat (1/6 of a whole note)"),
    "eighth": (F(1, 2), "eighth note: half a beat"),
    "triplet_eighth": (F(1, 3), "eighth-note triplet: 1/3 of a beat (1/12 of a whole note)"),
    "sixteenth": (F(1, 4), "sixteenth note: a quarter of a beat"),
    "sextuplet": (F(1, 6), "sixteenth-note sextuplet: 1/6 of a beat"),
    "thirty_second": (F(1, 8), "thirty-second note: 1/8 of a beat"),
    "grace": (F(0), "grace note or part of a rolled chord: an ornament played just before the main note, with no rhythmic value of its own"),
}
OFFSETS = {"on_downbeat": (F(0), "the first note starts on the downbeat")} | {
    k: (v, f"after a rest of one {d.split(':')[0]}") for k, (v, d) in VALUES.items() if 0 < v < 4}
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def note_name(p: int) -> str:
    return f"{NAMES[p % 12]}{p // 12 - 1}"


@dataclass
class Groups:
    """Onset groups of one bar (chords count once)."""

    t: np.ndarray            # group onset times, s from the downbeat
    members: list[list[int]]  # note indices per group
    bar_s: float
    bar_q: F

    @property
    def beat_s(self) -> float:
        return self.bar_s / float(self.bar_q)

    def gap_s(self, k: int) -> float:
        nxt = self.t[k + 1] if k + 1 < len(self.t) else self.bar_s
        return float(nxt - self.t[k])


def groups_of(chunk) -> Groups:
    t, idx = cluster_onsets([n.onset for n in chunk.perf])
    members = [[] for _ in t]
    for i, g in enumerate(idx):
        members[g].append(i)
    return Groups(t=t, members=members, bar_s=chunk.meta["bar_seconds"], bar_q=F(chunk.meta["bar_quarters"]))


def fitting(bar_q: F, table=VALUES) -> dict:
    return {k: v for k, v in table.items() if v[0] <= bar_q}


def build_state(chunk, g: Groups, style: str) -> dict:
    rows = []
    for k, mem in enumerate(g.members):
        held = max(chunk.perf[i].offset - chunk.perf[i].onset for i in mem)
        row = {"onset": k + 1, "notes": " ".join(note_name(chunk.perf[i].pitch) for i in sorted(mem, key=lambda i: chunk.perf[i].pitch)),
               "starts_ms_after_downbeat": round(max(g.t[k], 0) * 1000), "ms_until_next_onset": round(g.gap_s(k) * 1000),
               "held_ms": round(held * 1000)}
        if style == "beats":
            row["beats_until_next_onset"] = round(g.gap_s(k) / g.beat_s, 2)
        rows.append(row)
    return {
        "task": "One bar of a human classical piano performance (Mozart). Name the written note value of every onset.",
        "time_signature": chunk.meta["timesig"],
        "bar_length_ms": round(g.bar_s * 1000),
        "quarter_note_beat_ms": round(g.beat_s * 1000),
        "onsets": rows,
        "last_onset_note": "the last onset lasts until the next bar's downbeat",
    }


def questions(chunk, g: Groups, style: str) -> dict:
    vals = fitting(g.bar_q)
    crit = {k: d for k, (_, d) in vals.items()}
    names = [r["notes"] for r in build_state(chunk, g, style)["onsets"]]
    qs = {}
    first_ms = round(max(g.t[0], 0) * 1000)
    qs["first_start"] = {"type": "choice", "criteria": {k: d for k, (v, d) in OFFSETS.items() if v < g.bar_q},
                         "instructions": f"The first note starts {first_ms} ms after the downbeat "
                                         f"(a beat is {round(g.beat_s * 1000)} ms). Is it on the downbeat, or after a rest of which written value? "
                                         "Pianists play up to about 40 ms early or late."}
    for k in range(len(g.t)):
        gap = round(g.gap_s(k) * 1000)
        extra = f" ({round(g.gap_s(k) / g.beat_s, 2)} beats)" if style == "beats" else ""
        until = "the next onset" if k + 1 < len(g.t) else "the end of the bar"
        qs[f"onset_{k + 1:02d}"] = {
            "type": "choice", "criteria": crit,
            "instructions": f"Onset {k + 1} ({names[k]}) lasts {gap} ms{extra} until {until}; "
                            f"a quarter-note beat is {round(g.beat_s * 1000)} ms. Which written note value is it? "
                            "Pianists stretch and rush by 5-15%, so pick the value an engraver would write.",
        }
    return qs


def ask_values(chunk, style: str = "beats", model: str = "jev-latest", max_q: int = 64):
    """Jev's probability distributions: ({question: {value: p}}, merged response info)."""
    g = groups_of(chunk)
    state = build_state(chunk, g, style)
    qs = questions(chunk, g, style)
    names = list(qs)
    probs, info = {}, {"latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0, "cached": True}
    for i in range(0, len(names), max_q):
        part = {n: qs[n] for n in names[i:i + max_q]}
        r = ask(state, part, model)
        for n, a in r["answers"].items():
            probs[n] = a["probabilities"]
        info["latency_s"] = max(info["latency_s"], r.get("latency_s") or 0)
        info["input_tokens"] += r["usage"]["input_tokens"]
        info["output_tokens"] += r["usage"]["output_tokens"]
        info["requests"] += 1
        info["cached"] &= bool(r.get("cached"))
        info["model"] = r.get("model")
    return g, probs, info


# ------------------------------------------------------------------ decoding

def decode_argmax(g: Groups, probs: dict) -> list[F]:
    """Sum Jev's top value per onset (no bar-length constraint)."""
    pos = OFFSETS[max(probs["first_start"], key=probs["first_start"].get)][0]
    out = []
    for k in range(len(g.t)):
        out.append(pos)
        p = probs[f"onset_{k + 1:02d}"]
        pos += VALUES[max(p, key=p.get)][0]
    return out


def decode_fit(g: Groups, probs: dict, floor: float = 1e-4) -> list[F]:
    """Highest total log-probability value sequence that exactly fills the bar."""
    K = len(g.t)
    best: dict[F, tuple[float, list[F]]] = {}
    for name, (v, _) in OFFSETS.items():
        if v < g.bar_q and name in probs["first_start"]:
            best[v] = (log(max(probs["first_start"][name], floor)), [])
    for k in range(K):
        p = probs[f"onset_{k + 1:02d}"]
        nxt: dict[F, tuple[float, list[F]]] = {}
        for pos, (s, path) in best.items():
            for name, (v, _) in VALUES.items():
                if name not in p:
                    continue
                end = pos + v
                if end > g.bar_q or (k + 1 < K and end >= g.bar_q):
                    continue
                cand = (s + log(max(p[name], floor)), path + [pos])
                if end not in nxt or cand[0] > nxt[end][0]:
                    nxt[end] = cand
        best = nxt
    if g.bar_q in best:
        return best[g.bar_q][1]
    return decode_argmax(g, probs)  # no exact fill possible: fall back


# ------------------------------------------------------------------ code-only baselines

def timing_probs(g: Groups, sigma: float = 0.18, simplicity: float = 0.3) -> dict:
    """A timing model in the same shape as Jev's answers: Gaussian on the log ratio
    of played to written length, plus a small prior for simpler values."""
    cx = {F(4): 0, F(3): 1, F(2): 0, F(3, 2): 1, F(1): 0, F(3, 4): 1.5, F(2, 3): 2, F(1, 2): 0.5,
          F(1, 3): 1.5, F(1, 4): 1, F(1, 6): 2.5, F(1, 8): 2, F(0): 3}

    def dist(x_beats: float, table) -> dict:
        w = {}
        for name, (v, _) in table.items():
            if v == 0:
                w[name] = np.exp(-0.5 * (x_beats / 0.08) ** 2)
                continue
            w[name] = np.exp(-0.5 * (np.log(max(x_beats, 1e-3) / float(v)) / sigma) ** 2 - simplicity * cx.get(v, 2))
        z = sum(w.values()) or 1.0
        return {k: float(x / z) for k, x in w.items()}

    out = {"first_start": dist(max(g.t[0], 0) / g.beat_s, {k: v for k, v in OFFSETS.items() if v[0] < g.bar_q})}
    vals = fitting(g.bar_q)
    for k in range(len(g.t)):
        out[f"onset_{k + 1:02d}"] = dist(g.gap_s(k) / g.beat_s, vals)
    return out


def decode_nearest(g: Groups) -> list[F]:
    """Round every gap to the nearest value (log scale); no prior, no bar constraint."""
    return decode_argmax(g, timing_probs(g, sigma=1.0, simplicity=0.0))


# ------------------------------------------------------------------ truth helpers

def truth_positions(chunk, g: Groups) -> list[F | None]:
    """Written position of each onset group (earliest matched note in it)."""
    out = []
    for mem in g.members:
        ts = [chunk.truth[i].onset for i in mem if chunk.truth[i] is not None]
        out.append(min(ts) if ts else None)
    return out


def value_name(v: F) -> str | None:
    for k, (x, _) in VALUES.items():
        if x == v:
            return k
    return None
