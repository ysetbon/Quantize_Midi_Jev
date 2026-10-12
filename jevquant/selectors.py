"""Pick one candidate quantization per chunk.

Every selector sees the same shortlist (``shortlist``) so the comparison is
fair: Jev, Sonnet and the plain-code baselines all choose from the same
labelled options A-F.  The order of the options is shuffled deterministically
per chunk so that the label/position never leaks which one is best.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from math import log2

from .candidates import Candidate
from .jev import ask

LABELS = "ABCDEFGH"
K = 6

INSTRUCTIONS = (
    "You are transcribing one bar of a human classical piano performance into sheet music. "
    "Each candidate is a different way to notate the same played notes: a number of quarter-note beats "
    "in the bar (the time signature) and a rhythmic grid. 'counting' shows where notes start in each beat "
    "('1 e + a' = sixteenths, '1 trip let' = triplets, '-' = no new note in that beat). "
    "Timing error is how far the played notes are from the notated positions; skilled pianists usually "
    "deviate 10-40 ms. Choose the notation a skilled music engraver would write for this bar of a "
    "Classical-era sonata: it should match the performance within normal human timing deviation while "
    "using the simplest, most conventional rhythm. For this repertoire the quarter-note tempo is usually "
    "between 60 and 160 bpm. Avoid unnecessary fine subdivisions or tuplets, and "
    "avoid oversimplifying in a way that leaves large timing errors or merges distinct notes."
)


def prerank_cost(c: Candidate) -> float:
    """Lenient cost used only to build the shortlist (keeps it diverse)."""
    return c.mean_err_ms / 8 + 0.6 * c.feats["complexity"] + 2.0 * c.collisions


def shortlist(cands: list[Candidate], seed_text: str, quarters_per_bar=None, k: int = 6, per_meter: int = 2) -> dict[str, Candidate]:
    """The options every selector chooses from, shuffled deterministically and
    labelled A...

    Meter unknown: the ``per_meter`` cheapest candidates of every beats-per-bar
    hypothesis (2 x 4 = 8 options), so the meter decision stays open.
    Meter known (e.g. read from the score's time signature): the ``k`` cheapest
    candidates with that many quarter notes per bar.
    """
    if quarters_per_bar is None:
        top = []
        for n in sorted({c.feats["quarters_per_bar"] for c in cands}):
            top += sorted([c for c in cands if c.feats["quarters_per_bar"] == n], key=prerank_cost)[:per_meter]
    else:
        top = sorted([c for c in cands if c.feats["quarters_per_bar"] == quarters_per_bar], key=prerank_cost)[:k]
    rng = random.Random(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16))
    rng.shuffle(top)
    return {LABELS[i]: c for i, c in enumerate(top)}


def candidate_view(c: Candidate) -> dict:
    f = c.feats
    return {
        "quarter_beats_in_bar": f.get("quarters_per_bar"),
        "tempo_bpm": f["bpm"],
        "grid": f["grid"],
        "counting": f["counting"],
        "mean_timing_error_ms": f["mean_timing_error_ms"],
        "max_timing_error_ms": f["max_timing_error_ms"],
        "beats_with_tuplets": f["beats_with_tuplets"],
        "distinct_played_onsets_merged": f["merged_onsets"],
    }


def payload(chunk, options: dict[str, Candidate], meter_known: bool = False) -> dict:
    """The exact state sent to Jev (and to Sonnet for the head-to-head)."""
    bar = {"duration_s": round(chunk.meta.get("bar_seconds", 0) or 0, 3), "notes_played": len(chunk.perf)}
    if meter_known:
        bar["time_signature"] = chunk.meta["timesig"]
    return {"bar": bar, "candidates": {lab: candidate_view(c) for lab, c in options.items()}}


# ------------------------------------------------------------------ baselines

def sel_min_error(options: dict[str, Candidate], **_) -> str:
    return min(options, key=lambda l: (round(options[l].mean_err_ms), options[l].feats["complexity"]))


@dataclass
class Heuristic:
    """error + complexity + tuplets + merges + tempo prior; weights tuned on dev bars."""

    w_cx: float = 1.0
    w_tup: float = 0.5
    w_merge: float = 3.0
    w_tempo: float = 1.0

    def cost(self, c: Candidate) -> float:
        return (c.mean_err_ms / 10 + self.w_cx * c.feats["complexity"] + self.w_tup * c.feats["beats_with_tuplets"]
                + self.w_merge * c.collisions + self.w_tempo * abs(log2(c.bpm / 110)))

    def __call__(self, options: dict[str, Candidate], **_) -> str:
        return min(options, key=lambda l: self.cost(options[l]))


# ------------------------------------------------------------------ Jev

def sel_jev_choice(options, chunk, model="jev-latest", meter_known=False, **_) -> tuple[str, dict]:
    q = {"best": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {lab: None for lab in options}}}
    r = ask(payload(chunk, options, meter_known), q, model)
    return r["answers"]["best"]["choice"], r


def sel_jev_rate(options, chunk, model="jev-latest", meter_known=False, **_) -> tuple[str, dict]:
    levels = [
        "wrong: does not match the performance or an engraver would never write it",
        "possible but awkward or inaccurate",
        "good notation",
        "exactly what an engraver would write",
    ]
    q = {f"rate_{lab}": {"type": "score", "instructions": f"Rate candidate {lab}. " + INSTRUCTIONS, "criteria": levels}
         for lab in options}
    r = ask(payload(chunk, options, meter_known), q, model)
    scores = {lab: r["answers"][f"rate_{lab}"]["score"] for lab in options}
    return max(scores, key=scores.get), r


def sel_jev_blind(options, chunk, model="jev-latest", meter_known=False, **_) -> tuple[str, dict]:
    """Ablation: raw played onsets + bare candidate names, no arithmetic done for Jev."""
    state = {
        "bar_duration_s": round(chunk.meta.get("bar_seconds", 0) or 0, 3),
        "played_onsets_s": [round(n.onset, 3) for n in chunk.perf],
        "candidates": {lab: f"{c.feats.get('quarters_per_bar')} quarter beats per bar, {c.feats['grid']}"
                       for lab, c in options.items()},
    }
    if meter_known:
        state["time_signature"] = chunk.meta["timesig"]
    q = {"best": {"type": "choice", "criteria": {lab: None for lab in options},
                  "instructions": "Which candidate (beats per bar and rhythmic grid) best notates these played "
                                  "note onsets as one bar of sheet music?"}}
    r = ask(state, q, model)
    return r["answers"]["best"]["choice"], r
