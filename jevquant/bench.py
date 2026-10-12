"""Benchmark selectors on real ASAP bars.

Two settings:
  meter_unknown - the selector must also decide how many quarter beats the bar has
  meter_known   - the time signature is given (as when the score header is known);
                  only the rhythm inside the bar is decided
"""
from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F

from .asap import Piece
from .candidates import Candidate, generate_bar
from .selectors import (Heuristic, payload, sel_jev_blind, sel_jev_choice, sel_jev_rate, sel_min_error,
                        shortlist)

SETTINGS = ("meter_unknown", "meter_known")
JEV_SELECTORS = {"jev_choice": sel_jev_choice, "jev_rate": sel_jev_rate, "jev_blind": sel_jev_blind}


def load_bars(ids: list[str], root: str = "data/asap"):
    pieces: dict = {}
    out = []
    for bid in ids:
        pc, pf, k = bid.split("/")
        if (pc, pf) not in pieces:
            pieces[(pc, pf)] = Piece(root, pc, pf)
        ch = pieces[(pc, pf)].bar(int(k))
        ch.meta["id"] = bid
        out.append(ch)
    return out


def score(chunk, c: Candidate, idx: list[int]) -> dict:
    """Accuracy of candidate ``c`` against the chunk's ground truth (matched notes)."""
    return score_positions(chunk, c.q, idx, F(c.feats["quarters_per_bar"]))


def score_positions(chunk, group_pos: list[F], idx: list[int], n_pred: F) -> dict:
    """Accuracy of predicted onset-group positions (``idx`` maps note -> group)."""
    n_true = F(chunk.meta["bar_quarters"])
    scale = n_pred / n_true
    pairs = [(group_pos[idx[i]], t.onset) for i, t in enumerate(chunk.truth) if t is not None]
    strict = sum(p == t for p, t in pairs) / len(pairs)
    scaled = sum(p / scale == t for p, t in pairs) / len(pairs)
    return {"meter_ok": n_pred == n_true, "onset_acc": strict, "onset_acc_any_beat_unit": scaled,
            "exact": strict == 1.0 and n_pred == n_true}


def prepare(chunks, setting: str):
    """Candidates + shortlist for each bar (deterministic, no API calls)."""
    out = []
    for ch in chunks:
        cands, _, idx = generate_bar(ch, 0.0, ch.meta["t_end"])
        qpb = int(F(ch.meta["bar_quarters"])) if setting == "meter_known" else None
        options = shortlist(cands, ch.meta["id"], quarters_per_bar=qpb)
        out.append(dict(chunk=ch, idx=idx, cands=cands, options=options, setting=setting,
                        truth_in_candidates=any(score(ch, c, idx)["exact"] for c in cands),
                        truth_in_shortlist=any(score(ch, c, idx)["exact"] for c in options.values())))
    return out


def oracle(pr) -> str:
    return max(pr["options"], key=lambda l: (score(pr["chunk"], pr["options"][l], pr["idx"])["exact"],
                                             score(pr["chunk"], pr["options"][l], pr["idx"])["onset_acc"]))


def record(pr, selector: str, label: str, resp: dict | None = None) -> dict:
    rec = {"id": pr["chunk"].meta["id"], "setting": pr["setting"], "selector": selector, "label": label}
    rec.update(score(pr["chunk"], pr["options"][label], pr["idx"]))
    if resp is not None:
        rec.update(latency_s=resp.get("latency_s"), cached=resp.get("cached"), model=resp.get("model"),
                   input_tokens=resp["usage"]["input_tokens"], output_tokens=resp["usage"]["output_tokens"])
        best = resp["answers"].get("best")
        if best is not None:
            rec["confidence"] = best.get("confidence")
    return rec


def run_selector(name, fn, prepared, workers=8):
    def one(pr):
        out = fn(pr["options"], chunk=pr["chunk"], meter_known=pr["setting"] == "meter_known")
        label, resp = out if isinstance(out, tuple) else (out, None)
        return record(pr, name, label, resp)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prepared))


def summarize(records: list[dict]) -> dict:
    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault((r["setting"], r["selector"]), []).append(r)
    out: dict = {}
    for (setting, name), rs in groups.items():
        s = {
            "n": len(rs),
            "bars_exact": round(sum(r["exact"] for r in rs) / len(rs), 3),
            "onset_acc": round(statistics.mean(r["onset_acc"] for r in rs), 3),
            "onset_acc_any_beat_unit": round(statistics.mean(r["onset_acc_any_beat_unit"] for r in rs), 3),
            "meter_ok": round(sum(r["meter_ok"] for r in rs) / len(rs), 3),
        }
        lat = sorted(r["latency_s"] for r in rs if r.get("latency_s") is not None)
        if lat:
            s.update(latency_p50_s=round(statistics.median(lat), 3),
                     latency_p95_s=round(lat[int(round(0.95 * (len(lat) - 1)))], 3),
                     input_tokens_per_bar=round(statistics.mean(r["input_tokens"] for r in rs)),
                     output_tokens_per_bar=round(statistics.mean(r["output_tokens"] for r in rs)),
                     served_model=sorted({r["model"] for r in rs if r.get("model")}))
        out.setdefault(setting, {})[name] = s
    return out


def tune_heuristic(prepared) -> Heuristic:
    """Grid search of the heuristic weights on dev bars (never the eval bars)."""
    best = None
    for w_cx in (0.25, 0.5, 1.0, 1.5, 2.5):
        for w_tup in (0.0, 0.5, 1.5):
            for w_tempo in (0.0, 1.0, 3.0):
                for w_merge in (1.0, 3.0):
                    h = Heuristic(w_cx, w_tup, w_merge, w_tempo)
                    acc = sum(score(p["chunk"], p["options"][h(p["options"])], p["idx"])["exact"] for p in prepared)
                    if best is None or acc > best[0]:
                        best = (acc, h)
    return best[1]


def main(eval_ids, dev_ids, out_dir="results", models=("jev-latest", "jev-preview")):
    os.makedirs(out_dir, exist_ok=True)
    eval_chunks, dev_chunks = load_bars(eval_ids), load_bars(dev_ids)
    records, payloads, weights, coverage = [], {}, {}, {}
    for setting in SETTINGS:
        h = tune_heuristic(prepare(dev_chunks, setting))
        weights[setting] = h.__dict__
        ev = prepare(eval_chunks, setting)
        coverage[setting] = {
            "truth_in_candidates": round(sum(p["truth_in_candidates"] for p in ev) / len(ev), 3),
            "truth_in_shortlist": round(sum(p["truth_in_shortlist"] for p in ev) / len(ev), 3),
        }
        records += [record(p, "oracle", oracle(p)) for p in ev]
        records += run_selector("min_error", sel_min_error, ev)
        records += run_selector("heuristic", h, ev)
        for m in models:
            for name, fn in JEV_SELECTORS.items():
                records += run_selector(f"{name}[{m}]", lambda o, chunk, meter_known, fn=fn, m=m:
                                        fn(o, chunk, model=m, meter_known=meter_known), ev)
        for p in ev:
            payloads[f"{setting}:{p['chunk'].meta['id']}"] = {
                "options": payload(p["chunk"], p["options"], setting == "meter_known"),
                "truth_label": [l for l, c in p["options"].items() if score(p["chunk"], c, p["idx"])["exact"]],
            }
    bars = {}
    for ch in eval_chunks:
        m = ch.meta
        bars[m["id"]] = dict(piece=m["piece"], performer=m["performer"], bar=m["bar"], timesig=m["timesig"],
                             seconds=round(m["bar_seconds"], 3), notes=len(ch.perf),
                             written_onsets=sorted({str(s.onset) for s in m["score_notes"]}, key=F))
    json.dump(payloads, open(os.path.join(out_dir, "payloads.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = {"selectors": summarize(records), "heuristic_weights": weights, "coverage": coverage, "bars": bars}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary
