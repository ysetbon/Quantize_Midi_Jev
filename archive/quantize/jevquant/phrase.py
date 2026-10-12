"""Round 3: quantize a whole ~4-bar passage with no bar lines given.

Code enumerates every reasonable reading of the passage and Jev re-ranks a
shortlist.  A reading ("hypothesis") is three nested decisions:

1. onset grid: tempo, tempo octave and rhythmic grid, fitted across the whole
   passage with rubato (``candidates.generate``), so positions are known
   relative to the first note;
2. bar structure: time signature (2/4 ... 12/8) and where the first bar line
   falls, i.e. the pickup length;
3. note lengths: the written duration of every note (``durations.py``).

For 1+2 code computes metrical evidence for each hypothesis (bass notes,
accents, long notes and onsets on strong beats, bars repeating the same
rhythm, timing error, tempo plausibility) and a conditional-logit model trained
on dev passages turns it into a code score.  Jev sees the top hypotheses as
musician-readable options (counting lines per bar plus the evidence in words,
never raw arithmetic) and answers one Choice, asked twice in reversed option
order to cancel position bias.  Final pick = code score + weight * log P(Jev).
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from fractions import Fraction as F
from math import floor, log, log2

import numpy as np

from .candidates import Candidate, cluster_onsets
from .jev import ask
from .passage import beat_unit
from .tracker import passage_candidates

METERS = ("2/4", "3/4", "4/4", "2/2", "3/8", "6/8", "9/8", "12/8")
# typical tempo of the counted beat, in beats per minute
BEAT_BPM = {F(1): 100.0, F(2): 66.0, F(3, 2): 66.0, F(1, 2): 180.0}
LABELS = "ABCDEFGHIJKL"
FEATURES = ("err", "cx", "tup", "merge", "tempo", "db_miss", "beat_miss", "db_bass", "db_acc", "db_long",
            "db_dens", "beat_bass", "beat_acc", "beat_long", "repeat", "sync", "pickup", "no_pickup",
            "dp_cost")


def bar_len(ts: str) -> F:
    num, den = (int(x) for x in ts.split("/"))
    return F(4 * num, den)


@dataclass
class Clusters:
    """Onset groups of a passage plus the per-group evidence that marks strong beats."""

    t: np.ndarray
    note_cluster: list[int]
    new_bass: np.ndarray  # 1 where a new bass note (lowest note below middle C, new pitch) starts
    vel_z: np.ndarray     # loudest velocity, z-scored over the passage
    long_z: np.ndarray    # log time to the next onset, z-scored
    dens_z: np.ndarray    # notes per onset, z-scored


def _z(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def clusters_of(chunk) -> Clusters:
    t, idx = cluster_onsets([n.onset for n in chunk.perf])
    K = len(t)
    low = np.full(K, 200)
    vel, n = np.zeros(K), np.zeros(K)
    held = np.zeros(K)
    for i, k in enumerate(idx):
        p = chunk.perf[i]
        low[k] = min(low[k], p.pitch)
        vel[k] = max(vel[k], p.velocity)
        n[k] += 1
        held[k] = max(held[k], p.offset - p.onset)
    ioi = np.append(np.diff(t), held[-1])
    nb, prev = np.zeros(K), None
    for k in range(K):
        if low[k] < 60:
            nb[k] = float(prev is None or low[k] != prev)
            prev = low[k]
    return Clusters(t, idx, nb, _z(vel), _z(np.log(np.maximum(ioi, 0.03))), _z(n))


@dataclass
class Hyp:
    cand: Candidate
    ts: str
    d: F                     # first bar line, on the candidate's position scale
    feats: dict = field(default_factory=dict)
    code: float = 0.0        # code score (log-linear)
    key: str = ""

    @property
    def L(self) -> F:
        return bar_len(self.ts)

    @property
    def unit(self) -> F:
        return beat_unit(self.ts)

    @property
    def pickup(self) -> F:
        """Written time from the first note to bar 1's downbeat."""
        return self.d - self.cand.q[0]

    def positions(self) -> list[F]:
        """Written position of each onset group, quarter notes after bar 1's downbeat."""
        return [q - self.d for q in self.cand.q]


def _contrast(x: np.ndarray, m: np.ndarray) -> float:
    if m.all() or not m.any():
        return 0.0
    return float(x[m].mean() - x[~m].mean())


def _features(h: Hyp, cl: Clusters) -> dict:
    c, L, u = h.cand, h.L, h.unit
    q = np.array([float(v) for v in c.q])
    rel = q - float(h.d)
    pos = np.mod(rel, float(L))
    on_db = np.isclose(pos, 0) | np.isclose(pos, float(L))
    on_beat = np.isclose(np.mod(rel, float(u)), 0) | np.isclose(np.mod(rel, float(u)), float(u))
    span = (rel.min(), rel.max())
    db_slots = np.arange(np.ceil(span[0] / float(L) - 1e-9), np.floor(span[1] / float(L) + 1e-9) + 1) * float(L)
    beat_slots = np.arange(np.ceil(span[0] / float(u) - 1e-9), np.floor(span[1] / float(u) + 1e-9) + 1) * float(u)
    have = set(np.round(rel, 6))
    db_cov = np.mean([round(s, 6) in have for s in db_slots]) if len(db_slots) else 0.0
    beat_cov = np.mean([round(s, 6) in have for s in beat_slots]) if len(beat_slots) else 0.0
    bar_idx = np.floor(rel / float(L) + 1e-9).astype(int)
    bars = [frozenset(np.round(pos[bar_idx == b], 6)) for b in range(0, bar_idx.max())]  # full bars only
    rep = 0.0
    if len(bars) >= 2:
        rep = float(np.mean([max(len(a & b) / max(1, len(a | b)) for j, b in enumerate(bars) if j != i)
                             for i, a in enumerate(bars)]))
    nxt = np.append(np.diff(q), float(u))
    sync = float(np.mean(~on_beat & (nxt > float(u) + 1e-9)))
    bpm = c.bpm / float(u)
    tup = c.feats["beats_with_tuplets"] if u == 1 or u == 2 else sum(
        (v - h.d) % F(1, 2) != 0 and ((v - h.d) % F(1, 2)).denominator % 3 == 0 for v in set(c.q))
    return {
        "err": c.mean_err_ms / 10, "cx": c.feats["complexity"], "tup": tup / max(1, len(db_slots)),
        "merge": c.collisions, "tempo": abs(log2(bpm / BEAT_BPM.get(u, 100.0))),
        "db_miss": 1 - db_cov, "beat_miss": 1 - beat_cov,
        "db_bass": _contrast(cl.new_bass, on_db), "db_acc": _contrast(cl.vel_z, on_db),
        "db_long": _contrast(cl.long_z, on_db), "db_dens": _contrast(cl.dens_z, on_db),
        "beat_bass": _contrast(cl.new_bass, on_beat), "beat_acc": _contrast(cl.vel_z, on_beat),
        "beat_long": _contrast(cl.long_z, on_beat), "repeat": rep, "sync": sync,
        "pickup": float(h.pickup / L), "no_pickup": float(h.pickup == 0),
    }


def hypotheses(chunk, meters=METERS, qbpm: float | None = None) -> tuple[list[Hyp], Clusters]:
    """Every reasonable reading of the passage: the metrical tracker decodes the
    rhythm once per time signature x position of the first note in its bar (the
    pickup) x grid; each decoded path is one hypothesis with its own bar lines."""
    cl = clusters_of(chunk)
    out, seen = [], set()
    for ts, x0, c, cost in passage_candidates(chunk, meters, qbpm):
        h = Hyp(c, ts, F(0) if x0 == 0 else bar_len(ts))
        h.key = f"{ts}|{h.pickup}|{c.key}"
        if h.key in seen:
            continue
        seen.add(h.key)
        h.feats = _features(h, cl)
        h.feats["dp_cost"] = cost
        out.append(h)
    return out, cl


# ------------------------------------------------------------------ code scorer

def feature_matrix(hs: list[Hyp]) -> np.ndarray:
    return np.array([[h.feats[f] for f in FEATURES] for h in hs])


def fit_scorer(groups: list[tuple[np.ndarray, np.ndarray]], l2: float = 0.05, steps: int = 3000, lr: float = 0.05):
    """Conditional logit: P(h) = softmax(X w) within each passage; maximize the
    probability of the correct hypotheses (``y`` mask) with an L2 penalty."""
    groups = [(X, y) for X, y in groups if y.any() and len(X)]
    mu = np.concatenate([X for X, _ in groups]).mean(0)
    sd = np.concatenate([X for X, _ in groups]).std(0) + 1e-9
    w = np.zeros(len(FEATURES))
    for _ in range(steps):
        g = -l2 * w
        for X, y in groups:
            Z = (X - mu) / sd
            s = Z @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            py = p * y
            g += Z.T @ (py / py.sum() - p)
        w += lr * g / len(groups)
    return {"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "features": list(FEATURES)}


def apply_scorer(hs: list[Hyp], model: dict) -> None:
    if not hs:
        return
    Z = (feature_matrix(hs) - np.array(model["mu"])) / np.array(model["sd"])
    for h, s in zip(hs, Z @ np.array(model["w"])):
        h.code = float(s)


def shortlist(hs: list[Hyp], k: int = 8) -> list[Hyp]:
    """Best hypothesis per (time signature, pickup, grid, tempo octave), top ``k`` by code score."""
    best: dict = {}
    for h in sorted(hs, key=lambda h: -h.code):
        best.setdefault((h.ts, h.pickup, h.cand.grid, round(log2(h.cand.bpm))), h)
    return sorted(best.values(), key=lambda h: -h.code)[:k]


# ------------------------------------------------------------------ Jev view

def _syllable(frac: F, u: F) -> str:
    x = frac / u
    if u == F(3, 2):  # compound beat: eighths are "la li", sixteenths "ta"
        return {F(1, 3): "la", F(2, 3): "li"}.get(x, "ta" if (x * 6).denominator == 1 else ".")
    return {F(1, 2): "+", F(1, 4): "e", F(3, 4): "a", F(1, 3): "trip", F(2, 3): "let"}.get(x, ".")


def _ordinal_list(xs: list[int]) -> str:
    return ", ".join(str(x) for x in xs) if xs else "none"


def view(h: Hyp, cl: Clusters) -> dict:
    """Musician-readable description of one hypothesis (numbers already turned into words)."""
    L, u = h.L, h.unit
    pos = h.positions()
    bars: dict[int, list] = {}
    for k, p in enumerate(pos):
        b = floor(p / L)
        bars.setdefault(b, []).append((p - b * L, k))
    lines, bass_db, db_bars = [], 0, 0
    for b in sorted(bars):
        words = []
        for x, k in bars[b]:
            beat = int(x // u) + 1
            w = str(beat) if x % u == 0 else _syllable(x % u, u)
            if cl.new_bass[k]:
                w += "(bass)"
            if cl.vel_z[k] > 1.0:
                w += ">"
            words.append(w)
        name = "pickup" if b < 0 else f"bar {b + 1}"
        lines.append(f"{name}: {' '.join(words)}")
        if b >= 0:
            db_bars += 1
            bass_db += any(x == 0 and cl.new_bass[k] for x, k in bars[b])
    f = h.feats
    unit_name = {F(1): "quarter note", F(2): "half note", F(3, 2): "dotted quarter note", F(1, 2): "eighth note"}[u]
    bpm = h.cand.bpm / float(u)
    pace = "slow" if bpm < 70 else "moderate" if bpm < 110 else "fast" if bpm < 160 else "very fast"
    err = h.cand.mean_err_ms
    fit = "tight" if err < 20 else "normal" if err < 35 else "loose" if err < 60 else "very loose"
    pick = "none: the first note is beat 1 of bar 1" if h.pickup == 0 else \
        f"{_value_words(h.pickup)} before bar 1 (the passage starts on {_beat_words(L - h.pickup, u)} of a bar)"
    acc = f["db_acc"]
    return {
        "time_signature": h.ts,
        "counted_beat": f"{unit_name}, about {round(bpm)} per minute ({pace})",
        "pickup": pick,
        "counting": lines,
        "beat_1_has_a_note": f"{round((1 - f['db_miss']) * 100)}% of bars",
        "new_bass_note_on_beat_1": f"{bass_db} of {db_bars} bars",
        "beat_1_louder": "clearly" if acc > 0.6 else "slightly" if acc > 0.2 else "no",
        "long_notes_start_on": "beat 1" if f["db_long"] > 0.3 else "beats" if f["beat_long"] > 0.3 else "anywhere",
        "bars_repeat_same_rhythm": "mostly" if f["repeat"] > 0.7 else "partly" if f["repeat"] > 0.4 else "no",
        "timing_fit": f"{fit} (average {round(err)} ms off the written rhythm)",
        "tuplets": "none" if not h.cand.feats["beats_with_tuplets"] else f"in {h.cand.feats['beats_with_tuplets']} beats",
    }


def _value_words(q: F) -> str:
    names = {F(4): "a whole note", F(3): "a dotted half note", F(2): "a half note", F(3, 2): "a dotted quarter",
             F(1): "one quarter note", F(3, 4): "a dotted eighth", F(1, 2): "one eighth note", F(1, 4): "one sixteenth",
             F(1, 3): "one triplet eighth", F(2, 3): "two triplet eighths"}
    if q in names:
        return names[q]
    if (q * 2).denominator == 1:
        return f"{int(q * 2)} eighth notes"
    if (q * 4).denominator == 1:
        return f"{int(q * 4)} sixteenths"
    return f"{q} quarter notes"


def _beat_words(x: F, u: F) -> str:
    beat = int(x // u) + 1
    return f"beat {beat}" if x % u == 0 else f"the '{_syllable(x % u, u)}' after beat {beat}"


def short_label(h: Hyp) -> str:
    pick = "no pickup" if h.pickup == 0 else f"pickup of {_value_words(h.pickup)}"
    return f"{h.ts}, {pick}, rhythm on a grid of {h.cand.feats['grid']}"


INSTRUCTIONS = (
    "Each option in `options` is a different way to write the same short passage of a human classical piano "
    "performance as sheet music: a time signature, where the first bar line goes (a passage may start with a "
    "pickup before bar 1), and the rhythm, shown as counting per bar ('1 + 2 +' are eighths, '1 e + a' "
    "sixteenths, 'trip let' triplets, '1 la li' eighths in a compound beat; '(bass)' marks a new bass note, "
    "'>' a louder note). Which option would a skilled music engraver write? Beat 1 of a bar is usually the "
    "strongest beat: new bass notes, chord changes, louder and longer notes fall there; bars of one phrase often "
    "repeat a rhythm; tempos and rhythms should be conventional and the timing should fit within normal human "
    "deviation."
)


def jev_rank(chunk, options: list[Hyp], cl: Clusters, qbpm: float | None = None,
             model: str = "jev-latest") -> tuple[dict[str, float], dict]:
    """Jev's probability for each option (average of two Choices in opposite option order)."""
    seed = int(hashlib.md5(chunk.meta["id"].encode()).hexdigest()[:8], 16)
    order = list(options)
    random.Random(seed).shuffle(order)
    labs = {LABELS[i]: h for i, h in enumerate(order)}
    passage = {"duration_s": round(float(cl.t[-1]), 2), "notes_played": len(chunk.perf)}
    if qbpm:
        passage["tempo"] = f"given: {round(qbpm)} quarter notes per minute"
    state = {"passage": passage,
             "options": {lab: view(h, cl) for lab, h in labs.items()}}
    crit = {lab: short_label(h) for lab, h in labs.items()}
    qs = {"notation": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit},
          "notation_reversed": {"type": "choice", "instructions": INSTRUCTIONS,
                                "criteria": dict(reversed(list(crit.items())))}}
    r = ask(state, qs, model)
    p1, p2 = r["answers"]["notation"]["probabilities"], r["answers"]["notation_reversed"]["probabilities"]
    probs = {h.key: (p1.get(lab, 0) + p2.get(lab, 0)) / 2 for lab, h in labs.items()}
    return probs, r


def combine(options: list[Hyp], probs: dict[str, float], a: float) -> Hyp:
    return max(options, key=lambda h: h.code + a * log(max(probs.get(h.key, 0.0), 1e-4)))


# ------------------------------------------------------------------ scoring against ground truth

def evaluate(chunk, h: Hyp, cl: Clusters) -> dict:
    """Accuracy of one hypothesis: time signature, bar lines, every onset."""
    pos = h.positions()
    L_true = bar_len(chunk.meta["timesig"])
    pairs = [(pos[cl.note_cluster[i]], t.onset) for i, t in enumerate(chunk.truth) if t is not None]
    on = sum(p == t for p, t in pairs) / len(pairs)
    # bar lines in time: every note lands in the same bar (pickup = -1) as written, at any note-value scale
    scale = h.L / L_true
    bars_ok = all(floor(p / h.L) == floor(t / L_true) for p, t in pairs)
    scaled = sum(p / scale == t for p, t in pairs) / len(pairs)
    ts_ok = h.ts == chunk.meta["timesig"]
    # local rhythm: written gap between consecutive onset groups (robust to one early slip)
    tq: dict[int, F] = {}
    for i, t in enumerate(chunk.truth):
        if t is not None:
            k = cl.note_cluster[i]
            tq[k] = min(tq.get(k, t.onset), t.onset)
    gaps = [h.cand.q[k] - h.cand.q[k - 1] == tq[k] - tq[k - 1] for k in range(1, len(pos)) if k in tq and k - 1 in tq]
    # the upbeat: length of the pickup (written time from the first note to bar 1) and bar 1 itself
    first = min(tq) if tq else None
    upbeat_ok = ts_ok and first == 0 and h.pickup == -tq[0]
    bar1 = [k for k in tq if 0 <= tq[k] < L_true]
    bar1_ok = upbeat_ok and all(pos[k] == tq[k] for k in bar1 + [k for k in tq if tq[k] < 0])
    return {"ts_ok": ts_ok, "bars_ok": bars_ok, "pickup_ok": ts_ok and bars_ok, "upbeat_ok": upbeat_ok,
            "bar1_ok": bar1_ok, "onset_acc": on, "onset_acc_any_unit": scaled,
            "gap_acc": sum(gaps) / max(1, len(gaps)), "exact": ts_ok and on == 1.0}
