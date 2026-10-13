import json
import os
from fractions import Fraction as F

import numpy as np
import pytest

from jevquant import jev
from jevquant.candidates import counting, generate, generate_bar, snap
from jevquant.model import Chunk, PerfNote, ScoreNote
from jevquant.notation import bar_score, to_musicxml


def perfect_chunk(onsets, bpm=90.0):
    p = 60 / bpm
    truth = [ScoreNote(60 + i, F(o), F(1, 4)) for i, o in enumerate(onsets)]
    perf = [PerfNote(n.pitch, float(n.onset) * p, float(n.onset) * p + 0.1) for n in truth]
    return Chunk(perf=perf, truth=truth)


def test_snap_chooses_binary_or_ternary_per_beat():
    x = np.array([0.0, 0.26, 0.49, 0.74, 1.0, 1.34, 1.65])
    assert snap(x, "16|8t") == [F(0), F(1, 4), F(1, 2), F(3, 4), F(1), F(4, 3), F(5, 3)]


def test_counting():
    assert counting([F(0), F(1, 2), F(1), F(5, 4), F(3, 2), F(7, 4)], 3) == "1 + | 1 e + a | -"
    assert counting([F(0), F(1, 3), F(2, 3)]) == "1 trip let"


@pytest.mark.parametrize("onsets", [
    [0, F(1, 2), 1, F(3, 2), 2, F(5, 2), 3, F(7, 2)],
    [0, F(3, 4), 1, F(7, 4), 2, F(5, 2), 3],
    [0, F(1, 3), F(2, 3), 1, F(4, 3), F(5, 3), 2, 3],
])
def test_perfect_timing_is_recovered(onsets):
    ch = perfect_chunk(onsets)
    cands, _, idx = generate(ch)
    assert any([c.q[idx[i]] for i in range(len(onsets))] == [F(o) for o in onsets] for c in cands)


def test_bar_mode_recovers_offbeat_start():
    # first note on the "and" of beat 1; downbeats known from annotations
    onsets = [F(1, 2), 1, F(3, 2), 2, F(9, 4), F(5, 2)]
    ch = perfect_chunk(onsets, bpm=120)
    cands, _, idx = generate_bar(ch, 0.0, 1.5, beats=(3,))
    assert any([c.q[idx[i]] for i in range(len(onsets))] == [F(o) for o in onsets] for c in cands)


def test_musicxml_export_handles_triplets():
    notes = [ScoreNote(72, F(k, 3), F(1, 3)) for k in range(3)] + [ScoreNote(48, F(0), F(2))]
    xml = to_musicxml(bar_score(notes, "2/4"))
    assert "<time-modification>" in xml


def test_jev_client_caches(monkeypatch, tmp_path):
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return {"model": "jev-x", "answers": {"q": {"type": "noul", "noul": 0.9}},
                    "usage": {"input_tokens": 10, "output_tokens": 2}}

    monkeypatch.setattr(jev, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(jev.requests, "post", lambda *a, **k: calls.append(1) or Resp())
    q = {"q": {"type": "noul", "instructions": "?"}}
    a = jev.ask("s", q)
    b = jev.ask("s", q)
    assert a["answers"] == b["answers"] and not a["cached"] and b["cached"] and len(calls) == 1


@pytest.mark.skipif(not os.path.isdir("data/asap"), reason="ASAP data not fetched")
def test_asap_bar_alignment():
    from jevquant.asap import Piece
    ch = Piece("data/asap", "K331-3", "Stahievitch02").bar(3)
    assert ch.meta["timesig"] == "2/4" and ch.meta["match_rate"] > 0.9
    assert {t.onset for t in ch.truth if t} == {F(0), F(1, 2), F(1), F(3, 2)}


# ---------------------------------------------------------------- round 2: note values

from jevquant.notevalues import (OFFSETS, VALUES, build_state, decode_fit, decode_nearest,  # noqa: E402
                                 groups_of, value_name)


def bar_chunk(onsets, bar_q=3, bpm=100.0):
    ch = perfect_chunk(onsets, bpm)
    ch.meta.update(bar_seconds=bar_q * 60 / bpm, bar_quarters=str(bar_q), timesig=f"{bar_q}/4")
    return ch


def peaked(g, values, first="on_downbeat"):
    def dist(best, table):
        return {k: (0.9 if k == best else 0.1 / (len(table) - 1)) for k in table}
    out = {"first_start": dist(first, OFFSETS)}
    for k, v in enumerate(values):
        out[f"onset_{k + 1:02d}"] = dist(value_name(v), VALUES)
    return out


def test_decode_fit_follows_peaked_answers():
    onsets = [0, F(1, 2), 1, F(4, 3), F(5, 3), 2]
    g = groups_of(bar_chunk(onsets))
    vals = [b - a for a, b in zip(onsets, onsets[1:] + [F(3)])]
    assert decode_fit(g, peaked(g, vals)) == [F(o) for o in onsets]


def test_decode_fit_fills_the_bar():
    onsets = [0, F(1, 2), 1, 2]
    g = groups_of(bar_chunk(onsets))
    wrong = [F(1, 2), F(1, 2), F(1), F(2)]  # sums past the bar: the fit must repair the last value
    pos = decode_fit(g, peaked(g, wrong))
    ends = pos[1:] + [F(3)]
    assert len(pos) == 4 and all(value_name(b - a) for a, b in zip(pos, ends))  # every value is a real note value
    assert sum(b - a for a, b in zip(pos, ends)) + pos[0] == 3  # and together they fill the 3/4 bar exactly


def test_nearest_recovers_perfect_timing_and_state_lists_groups():
    onsets = [0, F(1, 4), F(1, 2), F(3, 4), 1, 2]
    ch = bar_chunk(onsets)
    g = groups_of(ch)
    assert decode_nearest(g) == [F(o) for o in onsets]
    assert len(build_state(ch, g, "raw")["onsets"]) == len(onsets)


# ------------------------------------------------------------------ round 3: passages

def _passage(onsets, ts, pickup, bpm=100.0, bass_every=None):
    """Perfectly timed passage: written ``onsets`` (quarters from the first note)."""
    from jevquant.phrase import bar_len
    p = 60 / bpm
    L = bar_len(ts)
    perf, truth = [], []
    for o in onsets:
        o = F(o)
        on_db = (o - pickup) % L == 0
        pitch = 48 if on_db else 72
        perf.append(PerfNote(pitch, float(o) * p, float(o) * p + 0.8 * p * 0.5, 90 if on_db else 60))
        truth.append(ScoreNote(pitch, o - pickup, F(1, 2)))
    return Chunk(perf=perf, truth=truth, meta={"timesig": ts, "qbpm": bpm, "id": "synthetic"})


def test_tracker_recovers_waltz_with_pickup():
    from jevquant.phrase import evaluate, hypotheses
    # 3/4, pickup of one quarter, then quarter-eighth-eighth-quarter patterns for 4 bars
    bar = [0, 1, F(3, 2), 2]
    onsets = [0] + [1 + 3 * b + x for b in range(4) for x in bar]
    ch = _passage(onsets, "3/4", F(1))
    hs, cl = hypotheses(ch, meters=("3/4",), qbpm=100.0)
    best = min(hs, key=lambda h: h.feats["dp_cost"] - 3 * h.feats["db_bass"])
    e = evaluate(ch, best, cl)
    assert e["gap_acc"] == 1.0
    assert any(evaluate(ch, h, cl)["exact"] for h in hs)


def test_starts_cover_half_beats_and_compound_eighths():
    from jevquant.tracker import starts
    assert starts(F(3), F(1)) == [F(0), F(1, 2), F(1), F(3, 2), F(2), F(5, 2)]
    assert starts(F(3), F(3, 2)) == [F(0), F(1, 2), F(1), F(3, 2), F(2), F(5, 2)]
    assert starts(F(4), F(2))[:3] == [F(0), F(1), F(2)]


def test_beat_unit():
    from jevquant.passage import beat_unit
    assert beat_unit("6/8") == F(3, 2) and beat_unit("3/8") == F(1, 2) and beat_unit("2/2") == F(2)
