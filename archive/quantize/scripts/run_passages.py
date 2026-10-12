"""Round 3 benchmark: ~4-bar passages, no bar lines given, BPM given.

Each window is pickup + 4 bars cut from a real performance (data/passages.json).
The quantizer must find the time signature (unless given), where bar 1 starts
(the pickup), every onset's written position and every note's written length.

Stage 1 (structure + onsets) methods:
  tracker      lowest-cost metrical tracker path (code only)
  code         conditional-logit scorer over all hypotheses, trained on dev windows
  jev          Jev's Choice over the code shortlist (8 options, two option orders)
  jev+code     code score + a * log P(Jev), a tuned on dev
  oracle       best option in the shortlist (ceiling for any selector)
Stage 2 (durations): see jevquant/durations.py.

Writes results/passages_records.jsonl and results/passages_summary.json.
"""
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant import durations as D  # noqa: E402
from jevquant.asap import PASSAGE_PIECES, fetch  # noqa: E402
from jevquant.passage import load_windows  # noqa: E402
from jevquant.phrase import (METERS, apply_scorer, combine, evaluate, feature_matrix, fit_scorer,  # noqa: E402
                             hypotheses, jev_rank, shortlist)

SETTINGS = ("ts_unknown", "ts_known")
METRICS = ("ts_ok", "upbeat_ok", "bar1_ok", "pickup_ok", "onset_acc", "gap_acc", "exact")


def _prep(ch):
    meters = METERS
    hs, cl = hypotheses(ch, meters, qbpm=ch.meta["qbpm"])
    return hs, cl, [evaluate(ch, h, cl) for h in hs]


def prepare(chunks, workers=4):
    with Pool(workers) as p:
        return p.map(_prep, chunks)


def labels(evs, idx):
    """Correct hypotheses: right time signature and bar lines, best onset accuracy among those."""
    import numpy as np
    good = [i for i in idx if evs[i]["pickup_ok"]]
    if not good:
        return None
    top = max(evs[i]["onset_acc"] for i in good)
    return np.array([float(i in good and evs[i]["onset_acc"] == top) for i in idx])


def subset(hs, evs, ch, setting):
    if setting == "ts_known":
        return [i for i, h in enumerate(hs) if h.ts == ch.meta["timesig"]]
    return list(range(len(hs)))


def train(chunks, prepped, setting):
    import numpy as np
    groups = []
    for ch, (hs, cl, evs) in zip(chunks, prepped):
        idx = subset(hs, evs, ch, setting)
        y = labels(evs, idx)
        if y is not None:
            groups.append((feature_matrix([hs[i] for i in idx]), y))
    return fit_scorer(groups)


def run_jev(chunks, prepped, setting, model, k=8):
    def one(j):
        ch, (hs, cl, evs) = chunks[j], prepped[j]
        idx = subset(hs, evs, ch, setting)
        opts = shortlist([hs[i] for i in idx], k)
        probs, r = jev_rank(ch, opts, cl, qbpm=ch.meta["qbpm"], model=model)
        return opts, probs, r
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, range(len(chunks))))


def picks_of(ch, hs, evs, opts, probs, setting, a):
    idx = subset(hs, evs, ch, setting)
    cand = [hs[i] for i in idx]
    ev_of = {id(hs[i]): evs[i] for i in idx}
    return {
        "tracker": min(cand, key=lambda h: h.feats["dp_cost"]),
        "code": max(cand, key=lambda h: h.code),
        "jev": max(opts, key=lambda h: probs.get(h.key, 0)),
        "jev+code": combine(opts, probs, a),
        "oracle": max(opts, key=lambda h: (ev_of[id(h)]["exact"], ev_of[id(h)]["pickup_ok"], ev_of[id(h)]["upbeat_ok"],
                                           ev_of[id(h)]["onset_acc"])),
    }, ev_of


def records(chunks, prepped, jev_out, setting, a):
    out = []
    for ch, (hs, cl, evs), (opts, probs, r) in zip(chunks, prepped, jev_out):
        idx = subset(hs, evs, ch, setting)
        picks, ev_of = picks_of(ch, hs, evs, opts, probs, setting, a)
        ceiling = max(evs[i]["onset_acc"] for i in idx if evs[i]["pickup_ok"]) if any(evs[i]["pickup_ok"] for i in idx) else 0
        for name, h in picks.items():
            e = ev_of[id(h)]
            rec = {"id": ch.meta["id"], "setting": setting, "method": name, "timesig": ch.meta["timesig"],
                   "pickup": ch.meta["pickup"], "pred_ts": h.ts, "pred_pickup": str(h.pickup),
                   "grid": h.cand.feats["grid"], **{m: e[m] for m in METRICS},
                   "ceiling_exact": any(evs[i]["exact"] for i in idx), "ceiling_onset_acc": ceiling}
            if name.startswith("jev"):
                ans = r["answers"]["notation"]
                rec.update(confidence=ans.get("confidence"), input_tokens=r["usage"]["input_tokens"],
                           latency_s=r.get("latency_s"), cached=r.get("cached"), model=r.get("model"))
            out.append(rec)
    return out


# ------------------------------------------------------------------ stage 2: durations

def duration_items(ch, h, cl):
    """(context, true written length) for notes whose onset this hypothesis got right."""
    pos = h.positions()
    ctxs = D.note_context(ch, h, cl)
    if h.ts != ch.meta["timesig"]:
        return ctxs, []
    items = [i for i, t in enumerate(ch.truth)
             if t is not None and t.duration > 0 and pos[cl.note_cluster[i]] == t.onset]
    return ctxs, items


def tune_duration_model(dev_items):
    best = None
    for sigma in (0.3, 0.45, 0.6):
        for ratio in (0.7, 0.8, 0.9):
            for legato in (0.5, 1.0, 1.5, 2.5):
                for cx in (0.0, 0.3, 0.8):
                    kw = dict(sigma=sigma, ratio=ratio, legato=legato, cx=cx)
                    ok = 0
                    for ch, ctxs, items in dev_items:
                        for i in items:
                            p = D.code_probs(ctxs[i], D.options(ctxs[i]), **kw)
                            ok += max(p, key=p.get) == ch.truth[i].duration
                    if best is None or ok > best[0]:
                        best = (ok, kw)
    return best[1]


def duration_run(items, kw, model, gate):
    """Per window: Jev answers for notes whose code confidence is below ``gate``."""
    def one(x):
        ch, h, ctxs, idx = x
        ask_idx = [i for i in idx if max(D.code_probs(ctxs[i], D.options(ctxs[i]), **kw).values()) < gate]
        if not ask_idx:
            return {}, {"input_tokens": 0, "requests": 0}
        return D.jev_durations(ch, h, ctxs, ask_idx, model=model)
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, items))


def duration_scores(items, jev, kw, b):
    tally = {m: [0, 0] for m in ("legato_rule", "code", "jev", "jev+code")}
    asked = tokens = 0
    for (ch, h, ctxs, idx), (probs, usage) in zip(items, jev):
        tokens += usage.get("input_tokens", 0)
        for i in idx:
            truth = ch.truth[i].duration
            opts = D.options(ctxs[i])
            pc = D.code_probs(ctxs[i], opts, **kw)
            pj = probs.get(i)
            asked += pj is not None
            comb = {v: pc[v] * (pj.get(v, 1e-4) ** b if pj else 1) for v in opts}
            for m, v in (("legato_rule", D.legato_rule(ctxs[i])), ("code", max(pc, key=pc.get)),
                         ("jev", max(pj, key=pj.get) if pj else max(pc, key=pc.get)), ("jev+code", max(comb, key=comb.get))):
                tally[m][0] += v == truth
                tally[m][1] += 1
    out = {m: round(a / max(1, n), 3) for m, (a, n) in tally.items()}
    out.update(notes=tally["code"][1], notes_asked_jev=asked, input_tokens=tokens,
               input_tokens_per_passage=round(tokens / max(1, len(items))))
    return out


def durations_stage(dev, p_dev, ev, p_ev, structure_of, model, gate=0.8):
    """Durations under (a) the best available structure and (b) the end-to-end pick."""
    def items(chunks, prepped, which):
        out = []
        for j, (ch, (hs, cl, evs)) in enumerate(zip(chunks, prepped)):
            h = structure_of(which, j, chunks is dev)
            ctxs, idx = duration_items(ch, h, cl)
            out.append((ch, h, ctxs, idx))
        return out
    d_or = items(dev, p_dev, "oracle")
    kw = tune_duration_model([(ch, ctxs, idx) for ch, h, ctxs, idx in d_or])
    j_dev = duration_run(d_or, kw, model, gate)
    b = max((0.0, 0.25, 0.5, 1.0, 2.0), key=lambda b: duration_scores(d_or, j_dev, kw, b)["jev+code"])
    res = {"code_model": kw, "jev_exponent_b": b, "gate": gate}
    for which in ("oracle", "jev+code"):
        it = items(ev, p_ev, which)
        res[f"structure={which}"] = duration_scores(it, duration_run(it, kw, model, gate), kw, b)
    return res


def summarize(recs):
    out = {}
    for setting in SETTINGS:
        for name in ("tracker", "code", "jev", "jev+code", "oracle"):
            rs = [r for r in recs if r["setting"] == setting and r["method"] == name]
            if not rs:
                continue
            s = {m: round(statistics.mean(float(r[m]) for r in rs), 3) for m in METRICS}
            s["n"] = len(rs)
            tok = [r["input_tokens"] for r in rs if r.get("input_tokens")]
            if tok:
                s["input_tokens_per_passage"] = round(statistics.mean(tok))
                lat = sorted(r["latency_s"] for r in rs if r.get("latency_s"))
                if lat:
                    s["latency_p50_s"] = round(statistics.median(lat), 3)
            out.setdefault(setting, {})[name] = s
        rs = [r for r in recs if r["setting"] == setting and r["method"] == "code"]
        if rs:
            out[setting]["ceiling"] = {"exact_in_candidates": round(statistics.mean(r["ceiling_exact"] for r in rs), 3),
                                       "best_onset_acc": round(statistics.mean(r["ceiling_onset_acc"] for r in rs), 3)}
    return out


def tune_a(chunks, prepped, jev_out, setting):
    best = None
    for a in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        rs = [r for r in records(chunks, prepped, jev_out, setting, a) if r["method"] == "jev+code"]
        score = sum(r["upbeat_ok"] + r["pickup_ok"] + r["onset_acc"] for r in rs)
        if best is None or score > best[0]:
            best = (score, a)
    return best[1]


def main(model="jev-latest"):
    fetch(pieces=PASSAGE_PIECES)
    ids = json.load(open("data/passages.json"))
    dev, ev = load_windows(ids["dev_ids"]), load_windows(ids["eval_ids"])
    p_dev, p_ev = prepare(dev), prepare(ev)
    all_recs, info, model_out = [], {}, {}
    for setting in SETTINGS:
        scorer = train(dev, p_dev, setting)
        for chunks, prepped in ((dev, p_dev), (ev, p_ev)):
            for hs, _, _ in prepped:
                apply_scorer(hs, scorer)
        j_dev = run_jev(dev, p_dev, setting, model)
        a = tune_a(dev, p_dev, j_dev, setting)
        j_ev = run_jev(ev, p_ev, setting, model)
        all_recs += records(ev, p_ev, j_ev, setting, a)
        if setting == "ts_unknown":
            chosen = {}
            for tag, chunks, prepped, jo in (("dev", dev, p_dev, j_dev), ("eval", ev, p_ev, j_ev)):
                for j, (ch, (hs, cl, evs), (opts, probs, _)) in enumerate(zip(chunks, prepped, jo)):
                    chosen[(tag, j)] = picks_of(ch, hs, evs, opts, probs, setting, a)[0]
            structure_of = lambda which, j, is_dev: chosen[("dev" if is_dev else "eval", j)][which]  # noqa: E731
            info["durations"] = durations_stage(dev, p_dev, ev, p_ev, structure_of, model)
        model_out[setting] = {"scorer": scorer, "jev_weight_a": a}
        info[setting] = {"jev_weight_a": a, "scorer": dict(zip(scorer["features"], [round(w, 3) for w in scorer["w"]])),
                         "dev": summarize(records(dev, p_dev, j_dev, setting, a))[setting]}
    os.makedirs("results", exist_ok=True)
    if info.get("durations"):
        model_out["durations"] = {k: info["durations"][k] for k in ("code_model", "jev_exponent_b", "gate")}
    json.dump(model_out, open("results/passages_model.json", "w"), indent=1)  # used by `python -m jevquant passage`
    with open("results/passages_records.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    summary = {"eval": summarize(all_recs), "durations": info.pop("durations", None), "settings": info,
               "n_eval": len(ev), "n_dev": len(dev)}
    json.dump(summary, open("results/passages_summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("eval", "durations")}, indent=1))
    return summary


if __name__ == "__main__":
    main()
