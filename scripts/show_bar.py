"""Print one bar: performed notes next to their written (score) positions.

usage: python scripts/show_bar.py K332-1/ADIG01/17 [more ids...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.asap import Piece  # noqa: E402

_cache = {}
for bid in sys.argv[1:]:
    pc, pf, k = bid.split("/")
    if (pc, pf) not in _cache:
        _cache[(pc, pf)] = Piece("data/asap", pc, pf)
    ch = _cache[(pc, pf)].bar(int(k))
    m = ch.meta
    print(f"== {bid}  time sig {m['timesig']}  bar = {m['bar_quarters']} quarters in {m['bar_seconds']:.3f}s  "
          f"match_rate {m['match_rate']}  unmatched performed notes {m['extra_perf_notes']}")
    print("   pitch  played_at_s  written_position_quarters")
    for p, t in zip(ch.perf, ch.truth):
        print(f"   {p.pitch:5d}  {p.onset:10.3f}  {t.onset if t is not None else 'UNMATCHED'}")
    missing = len(m["score_notes"]) - sum(t is not None for t in ch.truth)
    print(f"   written notes never matched to a played note: {missing}")
