"""Next harmony + exact voicing benchmark on Bach chorales (code vs Jev vs combined).

For a sample of positions in held-out chorales: given the key and the recent
chords (exact notes), predict the next chord and its exact four-voice notes.

  code      trigram over chord labels; smoothest legal voicing
  jev       Jev's Choice over the code's top-15 chords; Jev's Choice over the
            code's top-8 voicings
  jev+code  log P(code) + a * log P(Jev), weights a (chords) and b (voicings) tuned on dev

Voicing questions are asked for the code's top-3 chords and for the true chord
(questions are answered independently, so the extra one does not leak into the
chord answer); the true-chord question measures voicing skill on its own.
Writes results/harmony_summary.json and results/harmony_records.jsonl.
"""
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from math import log

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevharmony.data import build, split_of  # noqa: E402
from jevharmony.model import (NGram, jev_suggest, label_of, name_of, note_name, soft, state_of,  # noqa: E402
                              tonic_pc, voicings)

N_CHORDS, N_VOICINGS, N_SPEC = 15, 8, 3


_LABELS: dict = {}


def labels(c):
    if c["id"] not in _LABELS:
        t = tonic_pc(c["tonic"])
        _LABELS[c["id"]] = [label_of(e["notes"], t) for e in c["events"]]
    return _LABELS[c["id"]]


def positions(chorales, n, seed):
    pts = []
    for c in chorales:
        labs = labels(c)
        pts += [(c, i) for i, lab in enumerate(labs) if i >= 4 and lab is not None and None not in labs[i - 3:i]]
    random.Random(seed).shuffle(pts)
    return pts[:n]


def prepare(ng, c, i, tau):
    t = tonic_pc(c["tonic"])
    labs = labels(c)
    hist = labs[:i]
    cands = ng.candidates(c["mode"])
    pc = ng.probs(c["mode"], [l for l in hist if l], cands)
    top = sorted(cands, key=lambda x: -pc[x])[:N_CHORDS]
    prev = c["events"][i - 1]["notes"]
    truth = labs[i]
    spec = top[:N_SPEC] + ([truth] if truth not in top[:N_SPEC] else [])
    vopts = {lab: voicings(lab, t, prev, N_VOICINGS) for lab in spec}
    vopts = {lab: vs for lab, vs in vopts.items() if vs}
    pv_code = {lab: soft({v: cost for v, cost in vs}, tau) for lab, vs in vopts.items()}
    ctx = {"tonic_pc": t, "prev_notes": prev, "state": state_of(c, i)}
    return dict(id=f"{c['id']}/{i}", ctx=ctx, top=top, pc=pc, vopts=vopts, pv_code=pv_code, truth=truth,
                true_notes=tuple(c["events"][i]["notes"]))


def tune_tau(ng, pts):
    best = None
    for tau in (0.5, 1.0, 2.0, 4.0):
        ll = 0.0
        for c, i in pts:
            p = prepare(ng, c, i, tau)
            pv = p["pv_code"].get(p["truth"], {})
            ll += log(max(pv.get(p["true_notes"], 0), 1e-4))
        if best is None or ll > best[0]:
            best = (ll, tau)
    return best[1]


def ask_all(prepped, model):
    def one(p):
        return jev_suggest(p["ctx"], p["top"], p["vopts"], model=model, seed=p["id"])
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, prepped))


def evaluate(prepped, answers, a, b):
    out = []
    for p, (pj, pvj, r) in zip(prepped, answers):
        top = p["top"]
        comb = {x: log(p["pc"][x]) + a * log(max(pj.get(x, 0), 1e-4)) for x in top}
        rank = {"code": sorted(top, key=lambda x: -p["pc"][x]), "jev": sorted(top, key=lambda x: -pj.get(x, 0)),
                "jev+code": sorted(top, key=lambda x: -comb[x])}

        def voice_rank(lab, method):
            pc_v = p["pv_code"].get(lab)
            if not pc_v:
                return []
            pj_v = pvj.get(lab)
            if method == "code" or not pj_v:
                key = lambda v: -pc_v[v]  # noqa: E731
            elif method == "jev":
                key = lambda v: -pj_v.get(v, 0)  # noqa: E731
            else:
                key = lambda v: -(log(pc_v[v]) + b * log(max(pj_v.get(v, 0), 1e-4)))  # noqa: E731
            return sorted(pc_v, key=key)

        rec = {"id": p["id"], "truth": name_of(p["truth"]), "truth_notes": " ".join(note_name(m) for m in p["true_notes"]),
               "truth_in_options": p["truth"] in top, "input_tokens": r["usage"]["input_tokens"],
               "latency_s": r.get("latency_s"), "chord_confidence": r["answers"]["next_chord"]["confidence"]}
        for m, rk in rank.items():
            rec[f"{m}_chord_top1"] = rk[0] == p["truth"]
            rec[f"{m}_chord_top3"] = p["truth"] in rk[:3]
            vt = voice_rank(p["truth"], m)  # voicing skill given the true chord
            rec[f"{m}_voicing_given_chord_top1"] = bool(vt) and vt[0] == p["true_notes"]
            rec[f"{m}_voicing_given_chord_top3"] = p["true_notes"] in vt[:3]
            vp = voice_rank(rk[0], m)  # full suggestion: predicted chord + its best voicing
            rec[f"{m}_exact_top1"] = rk[0] == p["truth"] and bool(vp) and vp[0] == p["true_notes"]
            rec[f"{m}_pred"] = f"{name_of(rk[0])}: {' '.join(note_name(x) for x in vp[0]) if vp else '-'}"
        out.append(rec)
    return out


def summarize(recs):
    s = {"n": len(recs), "truth_in_chord_options": round(statistics.mean(r["truth_in_options"] for r in recs), 3),
         "input_tokens_per_request": round(statistics.mean(r["input_tokens"] for r in recs)),
         "latency_p50_s": round(statistics.median(r["latency_s"] for r in recs if r["latency_s"]), 3)}
    for m in ("code", "jev", "jev+code"):
        s[m] = {k: round(statistics.mean(r[f"{m}_{k}"] for r in recs), 3) for k in
                ("chord_top1", "chord_top3", "voicing_given_chord_top1", "voicing_given_chord_top3", "exact_top1")}
    return s


def main(n_dev=120, n_test=240, model="jev-latest"):
    ch = build()
    train = [c for c in ch if split_of(c["id"]) == "train"]
    ng = NGram([(c["mode"], [l for l in labels(c) if l]) for c in train])
    dev_pts = positions([c for c in ch if split_of(c["id"]) == "dev"], n_dev, 1)
    test_pts = positions([c for c in ch if split_of(c["id"]) == "test"], n_test, 2)
    tau = tune_tau(ng, dev_pts)
    dev = [prepare(ng, c, i, tau) for c, i in dev_pts]
    test = [prepare(ng, c, i, tau) for c, i in test_pts]
    j_dev = ask_all(dev, model)
    grid = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    a = max(grid, key=lambda a: sum(r["jev+code_chord_top1"] + 0.5 * r["jev+code_chord_top3"] for r in evaluate(dev, j_dev, a, 1.0)))
    b = max(grid, key=lambda b: sum(r["jev+code_voicing_given_chord_top1"] for r in evaluate(dev, j_dev, a, b)))
    j_test = ask_all(test, model)
    recs = evaluate(test, j_test, a, b)
    summary = {"test": summarize(recs), "dev": summarize(evaluate(dev, j_dev, a, b)),
               "weights": {"a_chord": a, "b_voicing": b, "voicing_tau": tau},
               "setup": {"chord_options": N_CHORDS, "voicing_options": N_VOICINGS, "train_chorales": len(train)}}
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open("results/harmony_summary.json", "w"), indent=1)
    with open("results/harmony_records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
