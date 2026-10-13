"""Metrical tempo-tracking quantizer for whole passages (round 3).

A Viterbi search over onset groups, run once per (time signature, tempo band,
grid).  The state is (position in the bar, local tempo), so the first state is
the pickup and every path carries its own bar lines.  Costs:

* timing: played gap vs written gap x local tempo, with an absolute (ms) and a
  relative (%) noise term;
* rubato: change of log tempo, scaled by the written time that passed;
* notation: landing on weaker metrical positions (off-beat eighths, sixteenths,
  tuplets) costs more, and binary and ternary subdivisions may not mix inside
  one beat;
* accents: onsets that carry a new bass note, a louder or a longer note cost
  extra when they land off the beat or off the downbeat.

This replaces the single-tempo grid fit of ``candidates.fit`` for passages:
over 4 bars rubato accumulates, and timing alone cannot tell triplet eighths
at one tempo from sixteenths at another; the bar position can.
"""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np

from .candidates import GRIDS, Candidate, TimeMap, _warp, cluster_onsets, describe, tempo_candidates


def _fracs(grid: str) -> list[F]:
    return sorted({v for alt in GRIDS[grid] for v in alt})


def level_cost(x: F, L: F, u: F) -> float:
    """How weak metrical position ``x`` (quarters into the bar) is: 0 = downbeat."""
    if x == 0:
        return 0.0
    if x % u == 0:
        return 0.3
    sub = [u / 3, u / 6, u / 12] if u == F(3, 2) else [u / 2, u / 4, u / 8]
    for c, s in zip((0.8, 1.6, 2.8), sub):
        if x % s == 0:
            return c
    tern = [u / 3, u / 6] if u != F(3, 2) else [u / 2, u / 4]  # tuplets against the beat
    for c, s in zip((1.8, 2.8), tern):
        if x % s == 0:
            return c
    return 4.0


def _ternary(x: F, u: F) -> bool:
    f = (x % u) / u
    return f != 0 and (f.denominator % 3 == 0) != (u == F(3, 2))


class Tracker:
    def __init__(self, L: F, u: F, grid: str, s_abs=0.025, s_rel=0.07, s_tempo=0.06, w_cx=1.0, w_sal=0.3,
                 grace_s=0.09, max_gap: F = F(4)):
        self.L, self.u = L, u
        fr = _fracs(grid)
        pos = sorted({F(b) + f for b in range(int(np.ceil(L))) for f in fr if F(b) + f < L})
        self.pos = pos
        S = len(pos)
        self.lvl = np.array([level_cost(x, L, u) for x in pos]) * w_cx
        self.on_db = np.array([x == 0 for x in pos])
        self.on_beat = np.array([x % u == 0 for x in pos])
        fr_, to_, v_, extra = [], [], [], []
        gap = max(max_gap, L)
        for i, a in enumerate(pos):
            for j, b in enumerate(pos):
                v = b - a
                while v <= gap:
                    if v >= 0:
                        same_beat = v < u and (a // u) == (b // u)
                        mix = same_beat and a % u != 0 and _ternary(a, u) != _ternary(b, u)
                        fr_.append(i), to_.append(j), v_.append(float(v)), extra.append(4.0 if mix else 0.0)
                    v += L
        self.m_from, self.m_to = np.array(fr_), np.array(to_)
        self.m_v, self.m_extra = np.array(v_), np.array(extra)
        self.S = S
        self.s_abs, self.s_rel, self.s_tempo, self.w_sal, self.grace_s = s_abs, s_rel, s_tempo, w_sal, grace_s

    def run(self, t: np.ndarray, sal: np.ndarray, p_center: float, n_tempi: int = 15, band: float = 1.35,
            start: F | None = None):
        """Best path: (positions in quarters from bar start of the first onset, cost per onset).
        ``start`` forces the first onset's position in the bar (i.e. the pickup)."""
        K, S = len(t), self.S
        logp = np.log(p_center) + np.linspace(-np.log(band), np.log(band), n_tempi)
        P = np.exp(logp)
        dlp = (logp[None, :] - logp[:, None]) ** 2  # [j_from, j_to]
        emit = lambda k: self.lvl + sal[k] * self.w_sal * ((~self.on_db) + (~self.on_beat))  # noqa: E731
        cost = np.repeat((emit(0) + 0.3 * (np.arange(S) > 0))[:, None], n_tempi, 1)
        if start is not None:
            cost[[x != start for x in self.pos]] = np.inf
        back = []
        v, fr, to = self.m_v, self.m_from, self.m_to
        for k in range(1, K):
            ioi = t[k] - t[k - 1]
            ok = (v > 0) | (ioi < self.grace_s)
            mv, mf, mt, mx = v[ok], fr[ok], to[ok], self.m_extra[ok]
            pred = mv[:, None] * P[None, :]                                   # [M, j_to]
            obs = (ioi - pred) ** 2 / (2 * (self.s_abs ** 2 + (self.s_rel * pred) ** 2))
            obs[mv == 0] = (ioi / self.s_abs) ** 2 / 2 + 2.0                   # grace note / spread chord
            tr = cost[mf][:, :, None] + dlp[None] / (2 * self.s_tempo ** 2 * np.maximum(mv, 0.25))[:, None, None]
            tr[mv == 0] = np.where(np.eye(n_tempi, dtype=bool), cost[mf][mv == 0][:, :, None], np.inf)
            jb = np.argmin(tr, axis=1)                                        # [M, j_to]
            tot = np.take_along_axis(tr, jb[:, None, :], 1)[:, 0, :] + obs + mx[:, None] + emit(k)[mt][:, None]
            flat = np.argsort(tot, axis=None)
            mi, ji = np.unravel_index(flat, tot.shape)
            _, first = np.unique(mt[mi] * n_tempi + ji, return_index=True)  # cheapest move into each state
            mi, ji = mi[first], ji[first]
            new = np.full((S, n_tempi), np.inf)
            best_m = np.full((S, n_tempi), -1)
            new[mt[mi], ji] = tot[mi, ji]
            best_m[mt[mi], ji] = mi
            idx_m = np.nonzero(ok)[0]
            back.append((best_m, jb, idx_m))
            cost = new
        s, j = np.unravel_index(np.argmin(cost), cost.shape)
        total = float(cost[s, j])
        gaps = []
        for best_m, jb, idx_m in reversed(back):
            m = best_m[s, j]
            gaps.append(_true_gap(self, idx_m[m]))
            s, j = self.m_from[idx_m[m]], jb[m, j]
        q = [self.pos[s]]
        for g in reversed(gaps):
            q.append(q[-1] + g)
        return q, total / K


def _true_gap(tr: Tracker, m: int) -> F:
    a, b = tr.pos[tr.m_from[m]], tr.pos[tr.m_to[m]]
    v = b - a
    while abs(float(v) - tr.m_v[m]) > 1e-9:
        v += tr.L
    return v


def salience(cl) -> np.ndarray:
    """0..~3 per onset group: new bass note, louder than usual, longer gap after it."""
    return cl.new_bass + np.clip(cl.vel_z, 0, 2) * 0.5 + np.clip(cl.long_z, 0, 2) * 0.5


def as_candidate(t: np.ndarray, q: list[F], grid: str) -> Candidate:
    """Wrap tracked positions as a Candidate: global tempo fit + smooth rubato warp."""
    qf = np.array([float(v) for v in q])
    if len(set(q)) >= 2:
        A = np.vstack([np.ones_like(qf), qf]).T
        (a, p), *_ = np.linalg.lstsq(A, t, rcond=None)
    else:
        a, p = float(t[0]), 0.5
    tm = TimeMap(float(a), float(p))
    if len(set(q)) >= 4:
        tm = TimeMap(float(a), float(p), *_warp(qf, t - (a + p * qf), float(p), bw=1.0))
    resid = (t - tm.time(qf)) * 1000
    c = Candidate(bpm=60 / tm.p, grid=grid, a=tm.a, p=tm.p, q=list(q), resid_ms=resid,
                  collisions=len(q) - len(set(q)))
    c.timemap = tm
    describe(c)
    return c


_TRACKERS: dict = {}


def starts(L: F, u: F) -> list[F]:
    """Where the first note may sit in its bar: every half beat (every eighth in compound time)."""
    step = u / 3 if u == F(3, 2) else u / 2
    return [step * k for k in range(int(L / step))]


def passage_candidates(chunk, meters, qbpm: float | None = None, grids=("16", "16|8t", "8t", "32", "16t"),
                       octaves=(0.5, 1.0, 2.0), hints: int = 2, beat_bpm=(35, 260)) -> list[tuple]:
    """(time signature, start, onset candidate, cost per onset) for every meter x first-note
    position x grid, and x tempo band when no tempo is given.  With ``qbpm`` (quarter notes per
    minute, as a DAW or MIDI file states it) only rubato around that tempo is searched.  The
    candidate's positions are bar positions (first bar line at 0 or L)."""
    from .passage import beat_unit
    from .phrase import bar_len, clusters_of  # local import: phrase depends on this module
    t, _ = cluster_onsets([n.onset for n in chunk.perf])
    sal = salience(clusters_of(chunk))
    tempi = [qbpm] if qbpm else [b * m for b in tempo_candidates(t, top=hints) for m in octaves]
    out = []
    for ts in meters:
        L, u = bar_len(ts), beat_unit(ts)
        for g in grids:
            key = (ts, g)
            if key not in _TRACKERS:
                _TRACKERS[key] = Tracker(L, u, g)
            tr = _TRACKERS[key]
            for bpm in tempi:
                if not qbpm and not beat_bpm[0] <= bpm / float(u) <= beat_bpm[1]:
                    continue
                for x0 in starts(L, u):
                    if x0 not in tr.pos:
                        continue
                    q, cost = tr.run(t, sal, 60 / bpm, n_tempi=9 if qbpm else 15, band=1.12 if qbpm else 1.35, start=x0)
                    out.append((ts, x0, as_candidate(t, q, g), cost))
    return out
