"""Sheet-music output (MusicXML via music21).

Engraving is deliberately simple: two staves split at middle C, and within a
staff each onset becomes one chord lasting until the next onset of that staff
(or until its own release, if the gap is long enough to need a rest).  The same
rules are used for the ground truth and for the quantizers, so differences in
the rendered bars are differences in rhythm only.
"""
from __future__ import annotations

from fractions import Fraction as F

import numpy as np
from music21 import chord, clef, layout, meter, metadata, note, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

from .candidates import Candidate, snap
from .model import ScoreNote

SPLIT = 60  # pitches below go to the bass staff


def quantized_notes(chunk, c: Candidate, idx: list[int]) -> list[ScoreNote]:
    """The chosen candidate as score notes (onsets from the candidate, durations
    from the snapped release times)."""
    offs = c.timemap.beat(np.array([n.offset for n in chunk.perf]))
    off_q = snap(offs, c.grid)
    out = []
    for i, n in enumerate(chunk.perf):
        on = c.q[idx[i]]
        out.append(ScoreNote(n.pitch, on, max(off_q[i] - on, F(1, 4)), n.velocity))
    return out


def _staff_events(notes: list[ScoreNote], bar_len: F):
    """Monophonic chord sequence for one staff: [(onset, duration, [pitches])]."""
    groups: dict[F, list[ScoreNote]] = {}
    for n in notes:
        if 0 <= n.onset < bar_len:
            groups.setdefault(n.onset, []).append(n)
    onsets = sorted(groups)
    ev = []
    for k, on in enumerate(onsets):
        nxt = onsets[k + 1] if k + 1 < len(onsets) else bar_len
        own = max(n.duration for n in groups[on])
        dur = nxt - on if own >= (nxt - on) / 2 else own  # legato unless clearly released early
        ev.append((on, dur, sorted({n.pitch for n in groups[on]})))
    return ev


def _clean(d: F) -> F:
    """Durations music21 can always spell: multiples of 1/12 or 1/8."""
    for den in (8, 12):
        if (d * den).denominator == 1:
            return d
    return F(round(d * 24), 24) or F(1, 24)


def _measure(notes: list[ScoreNote], bar_len: F, number: int, sel, first: bool, cl, timesig: str) -> stream.Measure:
    m = stream.Measure(number=number)
    if first:
        m.append(cl)
        m.append(meter.TimeSignature(timesig))
    pos = F(0)
    for on, dur, pitches in _staff_events([n for n in notes if sel(n.pitch)], bar_len):
        if on > pos:
            m.append(note.Rest(quarterLength=_clean(on - pos)))
        dur = _clean(dur)
        el = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
        el.quarterLength = dur
        m.append(el)
        pos = on + dur
    if pos < bar_len:
        m.append(note.Rest(quarterLength=_clean(bar_len - pos)))
    return m


def piece_score(bars: list[list[ScoreNote]], timesig: str, title: str = "") -> stream.Score:
    """A two-staff score with one measure per entry of ``bars`` (bar-relative positions)."""
    num, den = (int(x) for x in timesig.split("/"))
    bar_len = F(num * 4, den)
    sc = stream.Score()
    if title:
        sc.metadata = metadata.Metadata(title=title)
    parts = []
    for name, sel, mk in (("RH", lambda p: p >= SPLIT, clef.TrebleClef), ("LH", lambda p: p < SPLIT, clef.BassClef)):
        part = stream.PartStaff(id=name)
        for k, notes in enumerate(bars):
            part.append(_measure(notes, bar_len, k + 1, sel, k == 0, mk(), timesig))
        parts.append(part)
    for p in parts:
        sc.insert(0, p)
    sc.insert(0, layout.StaffGroup(parts, symbol="brace"))
    return sc


def bar_score(notes: list[ScoreNote], timesig: str, title: str = "") -> stream.Score:
    return piece_score([notes], timesig, title)


def to_musicxml(sc: stream.Score) -> str:
    sc = sc.makeNotation()
    return GeneralObjectExporter(sc).parse().decode("utf-8")
