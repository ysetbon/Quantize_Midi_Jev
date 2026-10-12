"""Final 30-bar eval set: the Sonnet-curated picks minus the bars its QA agent
flagged, each flagged bar replaced by a same-piece bar with the same rhythm
features that passes an automatic alignment check."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.bench import load_bars  # noqa: E402


def aligned(ch) -> bool:
    """Played times must follow written positions closely (linear fit residual
    < 0.2 beat, monotonic) and at most 2 played notes may be unmatched."""
    pts = [(float(t.onset), p.onset) for p, t in zip(ch.perf, ch.truth) if t is not None]
    if len(pts) < 3 or ch.meta["extra_perf_notes"] > 2:
        return False
    q, s = np.array(pts).T
    if len(set(q)) < 2:
        return True
    a, b = np.polyfit(q, s, 1)
    return bool(np.max(np.abs(s - (a * q + b))) / a < 0.2)


def main():
    cur = json.load(open("data/curation_raw.json"))
    pool = {b["id"]: b for b in json.load(open("data/bar_pool.json"))}
    keep = [i for i in cur["picks"] if i not in cur["flags"]]
    used = {(pool[i]["piece"], pool[i]["bar"]) for i in cur["picks"]}
    final, replaced = list(keep), {}
    feats = ("has_triplets", "has_16ths", "has_32nds")
    for bad in cur["flags"]:
        b = pool[bad]
        # prefer the same rhythm features; relax them one at a time if nothing aligned is left
        cands = []
        for k in range(len(feats), -1, -1):
            level = [c for c in pool.values() if c["piece"] == b["piece"] and (c["piece"], c["bar"]) not in used
                     and c["match_rate"] >= 0.95 and c["ornament_notes"] == 0 and c["perf_notes"] >= 3
                     and all(c[f] == b[f] for f in feats[:k]) and c not in cands]
            cands += sorted(level, key=lambda c: (c["performer"] != b["performer"], abs(c["bar"] - b["bar"])))
        for c in cands:
            if aligned(load_bars([c["id"]])[0]):
                final.append(c["id"])
                replaced[bad] = c["id"]
                used.add((c["piece"], c["bar"]))
                break
    auto_flags = [i for i in final if not aligned(load_bars([i])[0])]
    info = {"eval_ids": final, "replaced": replaced, "sonnet_qa_flags": cur["flags"],
            "auto_alignment_warnings": auto_flags,
            "reasons": {i: cur["picks"].get(i, {"rhythm_type": "replacement", "reason": f"replaces flagged bar"})
                        for i in final}}
    json.dump(info, open("data/eval_bars.json", "w"), indent=1)
    print(len(final), "bars; replaced:", replaced, "; auto warnings:", auto_flags)


if __name__ == "__main__":
    main()
