"""Write data/bar_pool.json: every annotated bar of the three Mozart pieces with
summary features, for choosing the evaluation bars."""
import json
import os
import sys
from collections import Counter
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.asap import PIECES, Piece, fetch, performers  # noqa: E402
from jevquant.candidates import generate_bar  # noqa: E402


def main(root="data/asap", out="data/bar_pool.json"):
    fetch(root)
    pool = []
    for pc in PIECES:
        for pf in performers(root, pc):
            P = Piece(root, pc, pf)
            for k in range(len(P.bars)):
                ch = P.bar(k)
                if not ch.perf:
                    continue
                sn = ch.meta["score_notes"]
                onsets = sorted({s.onset for s in sn})
                iois = Counter(str(b - a) for a, b in zip(onsets, onsets[1:] + [F(ch.meta["bar_quarters"])]))
                cands, _, idx = generate_bar(ch, 0, ch.meta["t_end"])
                nq = F(ch.meta["bar_quarters"])
                covered = any(c.feats["quarters_per_bar"] == nq and all(
                    tr is None or c.q[idx[i]] == tr.onset for i, tr in enumerate(ch.truth)) for c in cands)
                pool.append(dict(
                    id=f"{pc}/{pf}/{k}", piece=pc, performer=pf, bar=k, timesig=ch.meta["timesig"],
                    seconds=round(ch.meta["bar_seconds"], 3), perf_notes=len(ch.perf), score_notes=len(sn),
                    match_rate=ch.meta["match_rate"], ornament_notes=ch.meta["ornament_notes"],
                    written_onsets=[str(o) for o in onsets], ioi_counts=dict(iois),
                    has_triplets=any(o.denominator % 3 == 0 for o in onsets),
                    has_16ths=any(o.denominator == 4 for o in onsets),
                    has_32nds=any(o.denominator >= 8 for o in onsets),
                    candidates_cover_truth=covered,
                ))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(pool, open(out, "w"), indent=1)
    print(len(pool), "bars ->", out)


if __name__ == "__main__":
    main()
