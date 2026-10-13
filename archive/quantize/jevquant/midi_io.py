"""Minimal MIDI reading with mido."""
from __future__ import annotations

from fractions import Fraction

import mido

from .model import PerfNote


def read_notes(path: str) -> tuple[list[PerfNote], list[tuple[int, int]], int]:
    """All notes of a MIDI file (all tracks/channels merged), sorted by onset.

    Returns (notes in seconds, the same notes as (on_tick, off_tick), ticks_per_beat).
    """
    mid = mido.MidiFile(path)
    # Build a merged, absolute-tick event list and a tempo map.
    events = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            events.append((tick, msg))
    events.sort(key=lambda e: e[0])
    tempo, last_tick, last_sec = 500000, 0, 0.0
    open_notes: dict[tuple[int, int], list] = {}
    out = []
    for tick, msg in events:
        sec = last_sec + mido.tick2second(tick - last_tick, mid.ticks_per_beat, tempo)
        last_tick, last_sec = tick, sec
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault((msg.channel, msg.note), []).append((sec, tick, msg.velocity))
        elif msg.type in ("note_off", "note_on"):
            stack = open_notes.get((msg.channel, msg.note))
            if stack:
                on_sec, on_tick, vel = stack.pop(0)
                out.append((PerfNote(msg.note, on_sec, sec, vel), (on_tick, tick)))
    out.sort(key=lambda x: (x[0].onset, x[0].pitch))
    return [n for n, _ in out], [t for _, t in out], mid.ticks_per_beat


def write_score_midi(path: str, notes, bpm: float = 100.0, tpq: int = 480) -> None:
    """Write ScoreNotes (quarter-note positions) as a quantized MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=tpq)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    ev = []
    for n in notes:
        on = int(round(Fraction(n.onset) * tpq))
        off = int(round(Fraction(n.onset + n.duration) * tpq))
        ev += [(on, 1, mido.Message("note_on", note=n.pitch, velocity=n.velocity)),
               (off, 0, mido.Message("note_off", note=n.pitch, velocity=0))]
    ev.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for t, _, msg in ev:
        tr.append(msg.copy(time=t - now))
        now = t
    mid.save(path)
