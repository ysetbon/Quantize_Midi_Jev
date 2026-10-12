"""Round 2 benchmark: Jev names each note's value; compare with code-only baselines.

Same 30 eval bars and dev bars as round 1; time signature given.
Writes results/notevalues_records.jsonl and a "notevalues" block in results/summary.json.
"""
import json
import os
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.bench import load_bars, score_positions  # noqa: E402
from jevquant.notevalues import (VALUES, ask_values, decode_argmax, decode_fit, decode_nearest,  # noqa: E402
                                 groups_of, timing_probs, truth_positions, value_name)

MODELS = ("jev-latest",)
STYLES = ("raw", "beats")


def values_of(pos: list[F], end: F) -> list[F]:
    return [b - a for a, b in zip(pos, pos[1:] + [end])]


def evaluate(ch, g, pos: list[F], vals: list[F]) -> dict:
    idx = [None] * len(ch.perf)
    for k, mem in enumerate(g.members):
        for i in mem:
            idx[i] = k
    s = score_positions(ch, pos, idx, g.bar_q)
    tp = truth_positions(ch, g)
    tvals = [(tp[k + 1] if k + 1 < len(tp) else g.bar_q) - tp[k]
             if tp[k] is not None and (k + 1 == len(tp) or tp[k + 1] is not None) else None for k in range(len(tp))]
    pairs = [(value_name(t), value_name(p) or str(p)) for t, p in zip(tvals, vals) if t is not None and value_name(t)]
    s["value_acc"] = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
    s["value_pairs"] = pairs
    s["fills_bar"] = sum(vals) + pos[0] == g.bar_q
    s["positions"] = [str(p) for p in pos]
    return s


def tune_code(dev):
    best = None
    for sigma in (0.08, 0.12, 0.18, 0.25, 0.35, 0.5):
        for simp in (0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.5):
            ex = sum(evaluate(ch, g, p := decode_fit(g, timing_probs(g, sigma, simp)), values_of(p, g.bar_q))["exact"]
                     for ch, g in dev)
            if best is None or ex > best[0]:
                best = (ex, sigma, simp)
    return best


def main():
    ev_ids = json.load(open("data/eval_bars.json"))["eval_ids"]
    pool = json.load(open("data/bar_pool.json"))
    taken = {tuple(i.split("/")[::2]) for i in ev_ids}
    dev_ids = [b["id"] for b in pool if (b["piece"], str(b["bar"])) not in taken and b["match_rate"] >= 0.9 and b["perf_notes"] >= 3]
    dev_ids = random.Random(0).sample(dev_ids, 150)  # same dev bars as round 1
    dev = [(ch, groups_of(ch)) for ch in load_bars(dev_ids)]
    n_dev, sigma, simp = tune_code(dev)
    print("code model tuned on dev:", dict(sigma=sigma, simplicity=simp, dev_exact=n_dev))

    ev = [(ch, groups_of(ch)) for ch in load_bars(ev_ids)]
    records = []

    def rec(ch, sel, s, info=None, extra=None):
        r = {"id": ch.meta["id"], "selector": sel, **s}
        if info:
            r.update(latency_s=info["latency_s"], input_tokens=info["input_tokens"], output_tokens=info["output_tokens"],
                     requests=info["requests"], model=info.get("model"), cached=info["cached"])
        if extra:
            r.update(extra)
        return r

    for ch, g in ev:
        p = decode_nearest(g)
        records.append(rec(ch, "code: nearest value", evaluate(ch, g, p, values_of(p, p[-1] + (p[-1] - p[-2] if len(p) > 1 else g.bar_q)))))
        p = decode_fit(g, timing_probs(g, sigma, simp))
        records.append(rec(ch, "code: timing model + bar fit", evaluate(ch, g, p, values_of(p, g.bar_q))))

    jobs = [(ch, g, style, m) for ch, g in ev for style in STYLES for m in MODELS]

    def run(job):
        ch, g, style, m = job  # sigma/simp from tuning above
        g2, probs, info = ask_values(ch, style, m)
        top = [VALUES[max(probs[f"onset_{k + 1:02d}"], key=probs[f"onset_{k + 1:02d}"].get)][0] for k in range(len(g.t))]
        p_arg = decode_argmax(g, probs)
        p_fit = decode_fit(g, probs)
        tm = timing_probs(g, sigma, simp)
        combo = {q: {v: probs[q].get(v, 0) * tm[q].get(v, 0) for v in probs[q]} for q in probs}
        combo = {q: {v: x / (sum(d.values()) or 1) for v, x in d.items()} for q, d in combo.items()}
        p_combo = decode_fit(g, combo)
        conf = statistics.mean(max(probs[f"onset_{k + 1:02d}"].values()) for k in range(len(g.t)))
        tag = f"{style}" + ("" if m == "jev-latest" else f", {m}")
        return [rec(ch, f"Jev ({tag}): top value per note", evaluate(ch, g, p_arg, top), info, {"mean_confidence": conf}),
                rec(ch, f"Jev ({tag}) + bar fit", evaluate(ch, g, p_fit, values_of(p_fit, g.bar_q)), info, {"mean_confidence": conf}),
                rec(ch, f"Jev ({tag}) x timing model + bar fit", evaluate(ch, g, p_combo, values_of(p_combo, g.bar_q)), info,
                    {"mean_confidence": conf})]

    with ThreadPoolExecutor(6) as ex:
        for out in ex.map(run, jobs):
            records += out

    with open("results/notevalues_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items()}, default=str) + "\n")

    summary = {}
    for sel in dict.fromkeys(r["selector"] for r in records):
        rs = [r for r in records if r["selector"] == sel]
        va = [r["value_acc"] for r in rs if r["value_acc"] is not None]
        s = {"n": len(rs), "exact": sum(r["exact"] for r in rs), "onset_acc": round(statistics.mean(r["onset_acc"] for r in rs), 3),
             "value_acc": round(statistics.mean(va), 3), "fills_bar": sum(r["fills_bar"] for r in rs)}
        if "latency_s" in rs[0]:
            lat = sorted(r["latency_s"] for r in rs)
            s.update(p50=round(statistics.median(lat), 3), p95=round(lat[int(round(0.95 * (len(lat) - 1)))], 3),
                     tokens=round(statistics.mean(r["input_tokens"] for r in rs)),
                     tokens_per_note_group=round(sum(r["input_tokens"] for r in rs) / sum(len(g.t) for _, g in ev)))
        conf = Counter((a, b) for r in rs for a, b in r["value_pairs"])
        s["confusion"] = [[a, b, n] for (a, b), n in conf.most_common()]
        summary[sel] = s
        print(f"{sel:40s} exact {s['exact']:2d}/30  notes {s['onset_acc']:.2f}  values {s['value_acc']:.2f}  fills {s['fills_bar']}  "
              + (f"p50 {s['p50']}s tok {s['tokens']}" if "p50" in s else ""))
    full = json.load(open("results/summary.json"))
    full["notevalues"] = {"selectors": summary, "code_model": {"sigma": sigma, "simplicity": simp, "dev_exact": n_dev},
                          "groups_per_bar": round(statistics.mean(len(g.t) for _, g in ev), 1)}
    json.dump(full, open("results/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
