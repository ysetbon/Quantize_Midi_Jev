"""Multi-bar passages (about 4 bars) of real performances, with no bar lines given.

Round 3 data.  A window starts on any beat of a bar, so it may open with a
pickup, and its time signature is not given to the quantizer.  Ground truth
for every performed note is its written position, measured in quarter notes
from the first full bar's downbeat (pickup notes are negative), and its written
duration.

Performed and written notes are paired through ASAP's row-aligned beat
annotations: a performed onset is mapped to score time by interpolating
between annotated beats, then matched to the nearest written note of the same
pitch.  Score times come from the score MIDI's ticks, so compound meters (6/8,
3/8) and 2/2 work too.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

from .asap import PASSAGE_PIECES, _collapse_ornaments, read_annotations
from .midi_io import read_notes
from .model import Chunk, PerfNote, ScoreNote


def _q(x: float) -> F:
    return F(round(x * 48), 48)


@dataclass
class Bar:
    q0: F
    q1: F
    timesig: str


class Performance:
    """One performance of a piece: notes, written notes and bar lines in both time bases."""

    def __init__(self, root: str, piece: str, performer: str):
        d = os.path.join(root, PASSAGE_PIECES[piece])
        self.piece, self.performer = piece, performer
        self.perf, _, _ = read_notes(os.path.join(d, performer + ".mid"))
        score, ticks, tpq = read_notes(os.path.join(d, "midi_score.mid"))
        self.score = [ScoreNote(n.pitch, F(t0, tpq), F(round((t1 - t0) * 48 / tpq), 48), n.velocity)
                      for n, (t0, t1) in zip(score, ticks)]
        # score seconds -> quarter notes through the score MIDI's own notes
        sec = np.array([n.onset for n in score])
        qs = np.array([t0 / tpq for t0, _ in ticks])
        pa = read_annotations(os.path.join(d, performer + "_annotations.txt"))
        sa = read_annotations(os.path.join(d, "midi_score_annotations.txt"))
        assert len(pa) == len(sa), "annotation files must be row-aligned"
        self.beat_t = np.array([r[0] for r in pa])
        self.beat_q = np.interp([r[0] for r in sa], sec, qs)
        ts, self.bars = None, []
        db = [i for i, r in enumerate(sa) if r[1]]
        for k, i in enumerate(db):
            ts = sa[i][2] or ts
            if k + 1 < len(db):
                self.bars.append(Bar(_q(self.beat_q[i]), _q(self.beat_q[db[k + 1]]), ts))

    def score_q(self, t) -> np.ndarray:
        """Performance seconds -> score quarter notes (linear between annotated beats)."""
        return np.interp(t, self.beat_t, self.beat_q)

    def window(self, k: int, pickup: F = F(0), n_bars: int = 4) -> Chunk | None:
        """Bars k .. k+n_bars-1, preceded by the last ``pickup`` quarter notes of bar k-1."""
        if k < 1 or k + n_bars > len(self.bars):
            return None
        bars = self.bars[k - 1:k + n_bars]
        ts = bars[1].timesig
        L = bars[1].q1 - bars[1].q0
        if any(b.timesig != ts or b.q1 - b.q0 != L for b in bars[1:]) or pickup >= L:
            return None
        if pickup and bars[0].q1 - bars[0].q0 != L:
            return None
        Q0, Qd, Q1 = bars[1].q0 - pickup, bars[1].q0, bars[-1].q1
        beat = float(np.median(np.diff(self.beat_q)))
        sn = [s for s in self.score if Q0 <= s.onset < Q1]
        cand = [i for i, n in enumerate(self.perf) if Q0 - beat <= self.score_q(n.onset) < Q1 + beat]
        mq = {i: float(self.score_q(self.perf[i].onset)) for i in cand}
        pairs = sorted((abs(mq[i] - float(s.onset)), i, j) for i in cand for j, s in enumerate(sn)
                       if s.pitch == self.perf[i].pitch and abs(mq[i] - float(s.onset)) < 0.35 * beat)
        used_i, used_j, match = set(), set(), {}
        for _, i, j in pairs:
            if i not in used_i and j not in used_j:
                used_i.add(i)
                used_j.add(j)
                match[i] = sn[j]
        lo, hi = float(Q0) - 0.1 * beat, float(Q1) - 0.1 * beat
        idx = sorted(i for i in cand if i in match or lo <= mq[i] < hi)
        if not idx:
            return None
        t0 = self.perf[idx[0]].onset
        perf = [PerfNote(self.perf[i].pitch, self.perf[i].onset - t0, self.perf[i].offset - t0, self.perf[i].velocity)
                for i in idx]
        truth = [ScoreNote(m.pitch, m.onset - Qd, m.duration, m.velocity) if (m := match.get(i)) else None for i in idx]
        truth, n_orn = _collapse_ornaments(perf, truth)
        pts = [(float(tr.onset), p.onset) for p, tr in zip(perf, truth) if tr is not None]
        qbpm = 60 / np.polyfit([x for x, _ in pts], [y for _, y in pts], 1)[0]  # the given tempo (DAW-style BPM)
        return Chunk(perf=perf, truth=truth, meta=dict(
            id=f"{self.piece}/{self.performer}/{k}/{pickup}", piece=self.piece, performer=self.performer, bar=k,
            timesig=ts, bar_quarters=str(L), pickup=str(pickup), n_bars=n_bars, qbpm=round(float(qbpm), 1),
            match_rate=round(len(match) / max(1, len(sn)), 3), extra_perf_notes=len(idx) - len(match),
            ornament_notes=n_orn, abs_start=t0))


def beat_unit(timesig: str) -> F:
    """The counted beat in quarter notes (dotted quarter in 6/8, half in 2/2)."""
    num, den = (int(x) for x in timesig.split("/"))
    unit = F(4, den)
    return unit * 3 if den == 8 and num % 3 == 0 and num > 3 else unit


def sample_windows(root: str, pieces: dict, per_piece: int, split: str, seed: int = 0,
                   min_match: float = 0.85, max_bar: dict | None = None) -> list[str]:
    """Window ids for one split.  Bars are split into blocks of 8; even blocks are
    'eval', odd blocks 'dev', so the two splits never share written bars."""
    from .asap import performers
    rng = random.Random(f"{seed}/{split}")
    out = []
    for pc in pieces:
        pfs = performers(root, pc)
        perfs = {pf: Performance(root, pc, pf) for pf in pfs}
        tries, got = 0, 0
        while got < per_piece and tries < 400:
            tries += 1
            pf = rng.choice(pfs)
            P = perfs[pf]
            top = min(len(P.bars) - 4, (max_bar or {}).get(pc, 10 ** 6))
            if top < 2:
                break
            k = rng.randrange(1, top)
            if (k // 8) % 2 != (0 if split == "eval" else 1):
                continue
            L = P.bars[k].q1 - P.bars[k].q0
            u = beat_unit(P.bars[k].timesig or "4/4")
            n_beats = int(L / u)
            pickup = F(0) if rng.random() < 0.4 or n_beats < 2 else u * rng.randrange(1, n_beats)
            if n_beats < 2 and rng.random() < 0.6:
                pickup = L / 3 * rng.randrange(1, 3)  # 3/8: pickup of one or two eighths
            ch = P.window(k, pickup)
            wid = f"{pc}/{pf}/{k}/{pickup}"
            if ch is None or wid in out or ch.meta["match_rate"] < min_match or len(ch.perf) < 8:
                continue
            if sum(t is None for t in ch.truth) > 0.15 * len(ch.perf):
                continue
            out.append(wid)
            got += 1
    return out


def load_windows(ids: list[str], root: str = "data/asap") -> list[Chunk]:
    cache: dict = {}
    out = []
    for wid in ids:
        pc, pf, k, pk = wid.split("/", 3)
        if (pc, pf) not in cache:
            cache[(pc, pf)] = Performance(root, pc, pf)
        ch = cache[(pc, pf)].window(int(k), F(pk))
        ch.meta["id"] = wid
        out.append(ch)
    return out
