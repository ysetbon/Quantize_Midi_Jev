"""Run the Jev / baseline benchmark on the 30 curated bars.

Dev bars (for tuning the heuristic baseline) are a fixed random sample of the
pool that never shares a (piece, bar number) with the eval bars.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.bench import main  # noqa: E402

ev = json.load(open("data/eval_bars.json"))["eval_ids"]
pool = json.load(open("data/bar_pool.json"))
taken = {tuple(i.split("/")[::2]) for i in ev}
dev = [b["id"] for b in pool if (b["piece"], str(b["bar"])) not in taken and b["match_rate"] >= 0.9 and b["perf_notes"] >= 3]
dev = random.Random(0).sample(dev, 150)
summary = main(ev, dev)
print(json.dumps({k: summary[k] for k in ("selectors", "heuristic_weights", "coverage")}, indent=1))
