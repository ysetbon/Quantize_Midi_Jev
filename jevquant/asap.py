"""One-bar chunks of real human performances from the ASAP dataset.

ASAP (github.com/fosfrancesco/asap-dataset) pairs performance MIDI (MAESTRO /
e-competition pianists) with the unfolded score MIDI and beat/downbeat
annotations for both.  The two annotation files are row-aligned: row i of the
performance file is the same beat as row i of the score file.  That gives us
bar boundaries in the performance and the ground-truth notated rhythm.
"""
from __future__ import annotations

import os
import subprocess
from fractions import Fraction as F

from .midi_io import read_notes
from .model import Chunk, PerfNote, ScoreNote

REPO = "https://github.com/fosfrancesco/asap-dataset"
PIECES = {
    "K331-3": "Mozart/Piano_Sonatas/11-3",  # Alla Turca, 2/4
    "K332-1": "Mozart/Piano_Sonatas/12-1",  # 3/4
    "K310-1": "Mozart/Piano_Sonatas/8-1",   # 4/4
}
EARLY = 0.07  # performed notes may anticipate the downbeat by this much (s)


def fetch(root: str = "data/asap") -> str:
    """Sparse-clone only the Mozart folders we use."""
    if not os.path.isdir(root):
        os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, root], check=True)
        subprocess.run(["git", "-C", root, "sparse-checkout", "set", *PIECES.values()], check=True)
    return root


def read_annotations(path: str) -> list[tuple[float, bool, str | None]]:
    """Rows of (time, is_downbeat, time_signature_if_given)."""
    rows = []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        tags = parts[2].split(",")
        ts = tags[1] if len(tags) > 1 and "/" in tags[1] else None
        rows.append((float(parts[0]), tags[0] == "db", ts))
    return rows


def performers(root: str, piece: str) -> list[str]:
    d = os.path.join(root, PIECES[piece])
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".mid") and f != "midi_score.mid")


def _match(perf_idx, score_notes, perf, u_of, v_of, tol=0.2):
    """Pair performed and score notes of the same pitch by closest bar-relative
    position (both mapped to 0..1 across the bar).  Returns {perf_i: score_note}."""
    pairs = []
    for i in perf_idx:
        for j, s in enumerate(score_notes):
            if s.pitch == perf[i].pitch:
                d = abs(u_of(perf[i].onset) - v_of(s.onset))
                if d < tol:
                    pairs.append((d, i, j))
    pairs.sort()
    used_i, used_j, out = set(), set(), {}
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out[i] = score_notes[j]
    return out


def _collapse_ornaments(perf: list[PerfNote], truth: list, window: float = 0.045, span: F = F(1, 4)):
    """The score MIDI spells out rolled chords and grace notes as tiny rhythmic
    offsets (e.g. 13/160 of a quarter) that the sheet shows as a single chord
    with an ornament sign.  When the pianist played such notes together (within
    ``window`` s) and their written positions differ by at most ``span``, score
    them against the chord's written position (the earliest one)."""
    out = list(truth)
    n = 0
    i = 0
    while i < len(perf):
        j = i
        while j + 1 < len(perf) and perf[j + 1].onset - perf[i].onset < window:
            j += 1
        group = [k for k in range(i, j + 1) if out[k] is not None]
        if group:
            pos = [out[k].onset for k in group]
            lo = min(pos)
            if max(pos) != lo and max(pos) - lo <= span:
                for k in group:
                    if out[k].onset != lo:
                        s = out[k]
                        out[k] = ScoreNote(s.pitch, lo, s.duration + s.onset - lo, s.velocity)
                        n += 1
        i = j + 1
    return out, n


class Piece:
    """A performance of a piece with access to its bars."""

    def __init__(self, root: str, piece: str, performer: str):
        d = os.path.join(root, PIECES[piece])
        self.piece, self.performer = piece, performer
        self.perf, _, _ = read_notes(os.path.join(d, performer + ".mid"))
        score, ticks, tpq = read_notes(os.path.join(d, "midi_score.mid"))
        # score positions in quarter notes, exact (ticks / ticks_per_quarter)
        self.score = [ScoreNote(n.pitch, F(t0, tpq), F(round((t1 - t0) * 48 / tpq), 48), n.velocity) for n, (t0, t1) in zip(score, ticks)]
        pa = read_annotations(os.path.join(d, performer + "_annotations.txt"))
        sa = read_annotations(os.path.join(d, "midi_score_annotations.txt"))
        assert len(pa) == len(sa), "annotation files must be row-aligned"
        # score annotation times are seconds of a constant-tempo score MIDI; convert to quarters
        sec_per_q = (sa[1][0] - sa[0][0])  # first beat interval = one quarter (2/4, 3/4, 4/4 pieces)
        ts = None
        self.bars = []  # (perf_start, perf_end, score_start_q, score_end_q, timesig)
        db = [(i, r) for i, r in enumerate(pa) if r[1]]
        for k, (i, r) in enumerate(db):
            ts = sa[i][2] or ts
            if k + 1 < len(db):
                j = db[k + 1][0]
                self.bars.append((r[0], pa[j][0], F(sa[i][0] / sec_per_q).limit_denominator(48),
                                  F(sa[j][0] / sec_per_q).limit_denominator(48), ts))

    def bar(self, k: int) -> Chunk:
        t0, t1, q0, q1, ts = self.bars[k]
        idx = [i for i, n in enumerate(self.perf) if t0 - EARLY <= n.onset < t1 - EARLY]
        sn = [ScoreNote(s.pitch, s.onset - q0, s.duration, s.velocity) for s in self.score if q0 <= s.onset < q1]
        bar_q = q1 - q0
        m = _match(idx, sn, self.perf, lambda t: (t - t0) / (t1 - t0), lambda q: float(q / bar_q))
        perf = [self.perf[i] for i in idx]
        # times relative to the annotated downbeat
        rel = [PerfNote(p.pitch, p.onset - t0, p.offset - t0, p.velocity) for p in perf]
        truth = [m.get(i) for i in idx]
        truth, n_orn = _collapse_ornaments(rel, truth)
        return Chunk(
            perf=rel,
            truth=truth,
            meta=dict(piece=self.piece, performer=self.performer, bar=k, timesig=ts,
                      bar_quarters=str(bar_q), bar_seconds=t1 - t0, t_start=0.0, t_end=t1 - t0,
                      match_rate=round(len(m) / max(1, len(sn)), 3) if sn else 1.0,
                      extra_perf_notes=len(idx) - len(m), ornament_notes=n_orn, score_notes=sn, abs_start=t0),
        )
