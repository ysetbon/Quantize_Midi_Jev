"""Turn a performed chunk into a small set of candidate quantizations.

This is the deterministic half of the pipeline: it does all of the arithmetic
(tempo search, grid snapping, least-squares time fit, error measurement) and
describes each candidate in musician-readable terms ("1 e + a", "1 trip let").
Choosing *between* candidates is the judgement call handed to Jev (or to a
baseline) in ``selectors.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F
from math import floor, log2

import numpy as np

from .model import Chunk

CLUSTER_S = 0.045  # notes starting within this window are one chord / onset

# Grid name -> alternative per-beat position sets.  With several alternatives
# each beat independently uses the set that fits it best (binary OR ternary,
# never mixed inside one beat).
GRIDS: dict[str, list[list[F]]] = {
    "4": [[F(0)]],
    "8": [[F(0), F(1, 2)]],
    "16": [[F(k, 4) for k in range(4)]],
    "8t": [[F(k, 3) for k in range(3)]],
    "8|8t": [[F(0), F(1, 2)], [F(k, 3) for k in range(3)]],
    "16|8t": [[F(k, 4) for k in range(4)], [F(k, 3) for k in range(3)]],
    "16+8t": [sorted({F(k, 4) for k in range(4)} | {F(k, 3) for k in range(3)})],
    "32": [[F(k, 8) for k in range(8)]],
    "16t": [[F(k, 6) for k in range(6)]],
}
GRID_LABEL = {
    "4": "quarter notes only",
    "8": "eighth notes",
    "16": "sixteenth notes",
    "8t": "eighth-note triplets",
    "8|8t": "eighths or triplets (per beat)",
    "16|8t": "sixteenths or triplets (per beat)",
    "16+8t": "sixteenths and triplets together (cross-rhythm)",
    "32": "thirty-second notes",
    "16t": "sixteenth-note sextuplets",
}
SYLLABLE = {F(0): "1", F(1, 4): "e", F(1, 2): "+", F(3, 4): "a", F(1, 3): "trip", F(2, 3): "let"}


@dataclass
class Candidate:
    bpm: float
    grid: str
    a: float  # time (s) of beat 0
    p: float  # seconds per quarter note
    q: list[F]  # quantized position of each onset cluster (quarter notes)
    resid_ms: np.ndarray
    collisions: int
    feats: dict = field(default_factory=dict)
    key: str = ""
    timemap: "TimeMap | None" = None

    @property
    def mean_err_ms(self) -> float:
        return float(np.mean(np.abs(self.resid_ms)))

    @property
    def max_err_ms(self) -> float:
        return float(np.max(np.abs(self.resid_ms)))


def cluster_onsets(onsets: list[float]) -> tuple[np.ndarray, list[int]]:
    """Group near-simultaneous onsets (chords). Returns cluster times and, per
    note (in the given order, which must be sorted), its cluster index."""
    times: list[list[float]] = []
    idx = []
    for t in onsets:
        if times and t - times[-1][0] < CLUSTER_S:
            times[-1].append(t)
        else:
            times.append([t])
        idx.append(len(times) - 1)
    return np.array([float(np.mean(c)) for c in times]), idx


def snap(x: np.ndarray, grid: str) -> list[F]:
    """Snap beat positions to a grid, choosing binary/ternary per beat."""
    alts = GRIDS[grid]
    beats = np.floor(x).astype(int)
    out: list[F | None] = [None] * len(x)
    for b in np.unique(beats):
        members = np.where(beats == b)[0]
        best = None
        for ai, alt in enumerate(alts):
            pts = [float(v) for v in alt] + [1.0]
            vals = [F(int(b)) + (alt + [F(1)])[int(np.argmin([abs(x[i] - b - v) for v in pts]))] for i in members]
            err = sum(abs(x[i] - float(v)) for i, v in zip(members, vals)) + 1e-3 * ai
            if best is None or err < best[0]:
                best = (err, vals)
        for i, v in zip(members, best[1]):
            out[i] = v
    return out  # type: ignore[return-value]


class TimeMap:
    """Score position (quarter notes) <-> seconds: constant tempo ``a + p*q`` plus
    a smooth warp ``w(q)`` (kernel-smoothed residuals) that follows rubato."""

    def __init__(self, a: float, p: float, knots_q=None, knots_w=None):
        self.a, self.p = a, p
        self.kq = np.array([0.0, 1.0]) if knots_q is None else knots_q
        self.kw = np.zeros(2) if knots_w is None else knots_w

    def time(self, q):
        return self.a + self.p * np.asarray(q, float) + np.interp(q, self.kq, self.kw)

    def beat(self, t):
        lo, hi = self.kq[0] - 8, self.kq[-1] + 8
        qs = np.linspace(lo, hi, int((hi - lo) * 96) + 1)
        ts = self.time(qs)
        ts = np.maximum.accumulate(ts)  # keep monotonic for the inverse
        return np.interp(t, ts, qs)

    def local_tempo_range(self) -> float:
        """Max relative deviation of the local beat period from the mean, in %."""
        if len(self.kq) < 3:
            return 0.0
        slope = np.diff(self.kw) / np.diff(self.kq)
        return float(np.max(np.abs(slope)) / self.p * 100)


def _warp(qf: np.ndarray, resid: np.ndarray, p: float, bw: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    kq = np.arange(np.floor(qf.min()), np.ceil(qf.max()) + 1, 0.5)
    w = np.exp(-0.5 * ((kq[:, None] - qf[None, :]) / bw) ** 2)
    kw = (w @ resid) / (w.sum(1) + 1.0)  # +1 shrinks toward 0 where data is sparse
    # never let the warp reverse time: local period must stay > 0.6 * p
    for i in range(1, len(kw)):
        kw[i] = max(kw[i], kw[i - 1] - 0.4 * p * (kq[i] - kq[i - 1]))
    return kq, kw


def fit(t: np.ndarray, p: float, grid: str, b0: F = F(0), iters: int = 4, warp: bool = True,
        anchors: list[tuple[float, float]] | None = None) -> Candidate | None:
    """Snap onsets to ``grid``; refit tempo + phase; optionally follow rubato.

    Without ``anchors`` the first onset is beat ``b0``.  With anchors (known
    (time, beat) pairs such as annotated downbeats) the fit is pinned to them
    instead and the first onset may fall anywhere.
    """
    p_init = p
    if anchors:
        at = np.array([a[0] for a in anchors])
        aq = np.array([a[1] for a in anchors], float)
        tm = TimeMap(at[0] - aq[0] * p, p)
    else:
        at = aq = np.zeros(0)
        tm = TimeMap(t[0] - float(b0) * p, p)
    wa = 3.0  # anchor weight in the least-squares fit
    q = snap(tm.beat(t), grid)
    for it in range(iters + (2 if warp else 0)):
        qf = np.array([float(v) for v in q])
        if len(set(q)) + len(aq) < 2:
            break
        X = np.concatenate([qf, aq])
        Y = np.concatenate([t, at])
        W = np.concatenate([np.ones_like(qf), np.full(len(aq), wa)])
        A = np.vstack([np.ones_like(X), X]).T * W[:, None]
        (a2, p2), *_ = np.linalg.lstsq(A, Y * W, rcond=None)
        if not (0.8 * p_init < p2 < 1.25 * p_init):
            break
        tm = TimeMap(float(a2), float(p2))
        if warp and it >= 1 and len(set(q)) >= 4:
            tm = TimeMap(float(a2), float(p2), *_warp(X, Y - (a2 + p2 * X), p2))
        q_new = snap(tm.beat(t), grid)
        if q_new == q and it >= (2 if warp else 0):
            break
        q = q_new
    if not anchors and q[0] != b0:
        return None
    qf = np.array([float(v) for v in q])
    resid = (t - tm.time(qf)) * 1000
    collisions = len(q) - len(set(q))
    c = Candidate(bpm=60 / tm.p, grid=grid, a=tm.a, p=tm.p, q=q, resid_ms=resid, collisions=collisions)
    c.timemap = tm
    return c


def tempo_candidates(t: np.ndarray, b0: F = F(0), top: int = 3, hints: list[float] = ()) -> list[float]:
    """Scan one tempo octave (60-120 quarter bpm) and return the best local minima,
    folded back to that octave.  Octave choice is left to the selector."""
    if len(t) < 3:
        return [90.0]
    bpms = 60 * 2 ** (np.arange(0, 1, 0.005))
    errs = []
    for bpm in bpms:
        c = fit(t, 60 / bpm, "16|8t", b0, iters=2, warp=False)
        errs.append(np.inf if c is None else float(np.mean(np.abs(c.resid_ms))) / 1000 / c.p + 0.2 * c.collisions / len(t))
    errs = np.array(errs)
    order = [i for i in np.argsort(errs) if np.isfinite(errs[i])]
    picked: list[float] = []
    for i in order:
        if all(abs(log2(bpms[i] / b)) > 0.04 for b in picked):
            picked.append(float(bpms[i]))
        if len(picked) >= top:
            break
    for h in hints:
        fold = h * 2 ** -floor(log2(h / 60))
        if all(abs(log2(fold / b)) > 0.04 for b in picked):
            picked.append(fold)
    return picked


# ---------------------------------------------------------------- description

def _count(beat_positions: list[F]) -> str:
    """'1 e + a' style counting for the onsets inside one beat."""
    if not beat_positions:
        return "-"
    words = []
    for v in beat_positions:
        words.append(SYLLABLE.get(v, f"({v})"))
    return " ".join(words)


def counting(q: list[F], n_beats: int | None = None) -> str:
    """Beat-by-beat counting string for a quantized onset list."""
    uniq = sorted(set(q))
    n_beats = max(n_beats or 0, int(floor(uniq[-1])) + 1)
    per_beat: list[list[F]] = [[] for _ in range(n_beats)]
    for v in uniq:
        per_beat[int(floor(v))].append(v - floor(v))
    return " | ".join(_count(b) for b in per_beat)


def complexity(v: F) -> float:
    d = (v - floor(v)).denominator
    return {1: 0.0, 2: 1.0, 4: 2.0, 3: 2.5, 8: 4.0, 6: 4.0}.get(d, 5.0)


def describe(c: Candidate) -> None:
    uniq = sorted(set(c.q))
    tuplet_beats = len({floor(v) for v in uniq if (v - floor(v)).denominator % 3 == 0})
    c.feats = {
        "bpm": round(c.bpm),
        "grid": GRID_LABEL[c.grid],
        "counting": counting(c.q),
        "mean_timing_error_ms": round(c.mean_err_ms),
        "max_timing_error_ms": round(c.max_err_ms),
        "beats_with_tuplets": tuplet_beats,
        "merged_onsets": c.collisions,
        "tempo_variation_pct": round(c.timemap.local_tempo_range()),
        "complexity": round(float(np.mean([complexity(v) for v in uniq])), 2),
    }
    c.key = f"{round(c.bpm)}|{','.join(str(v) for v in c.q)}"


def generate(chunk: Chunk, b0: F = F(0), tempo_hints: list[float] = (), grids=None) -> tuple[list[Candidate], np.ndarray, list[int]]:
    """All distinct candidate quantizations of a chunk."""
    onsets = [n.onset for n in chunk.perf]
    t, note_cluster = cluster_onsets(onsets)
    grids = grids or list(GRIDS)
    out: dict[str, Candidate] = {}
    for base in tempo_candidates(t, b0, hints=tempo_hints):
        for mult in (0.5, 1.0, 2.0):
            for g in grids:
                c = fit(t, 60 / (base * mult), g, b0)
                if c is None or not (30 <= c.bpm <= 260):
                    continue
                describe(c)
                if c.key not in out or c.mean_err_ms < out[c.key].mean_err_ms:
                    out[c.key] = c
    return list(out.values()), t, note_cluster


BAR_BEATS = (2, 3, 4, 6)


def generate_bar(chunk: Chunk, t_start: float, t_end: float, beats=BAR_BEATS, grids=None) -> tuple[list[Candidate], np.ndarray, list[int]]:
    """Candidates for one bar whose downbeat times are known (e.g. annotated or
    from a beat tracker).  Hypotheses: the bar holds ``n`` quarter notes, for each
    n in ``beats``, times every grid."""
    t, note_cluster = cluster_onsets([n.onset for n in chunk.perf])
    grids = grids or list(GRIDS)
    out: dict[str, Candidate] = {}
    for n in beats:
        p = (t_end - t_start) / n
        for g in grids:
            c = fit(t, p, g, anchors=[(t_start, 0.0), (t_end, float(n))])
            if c is None:
                continue
            describe(c)
            c.feats["counting"] = counting(c.q, n)
            c.feats["quarters_per_bar"] = n
            c.key = f"{n}|{c.key}"
            if c.key not in out or c.mean_err_ms < out[c.key].mean_err_ms:
                out[c.key] = c
    return list(out.values()), t, note_cluster
