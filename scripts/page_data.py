"""Collect everything the results page needs into results/page_data.json."""
import json

S = json.load(open("results/summary.json"))
R = [json.loads(l) for l in open("results/records.jsonl")]
bars = json.load(open("results/bars_detail.json"))
NAMES = [
    ("oracle", "Best available option", "ceiling"),
    ("jev_choice[jev-latest]", "Jev, pick one", "jev"),
    ("jev_rate[jev-latest]", "Jev, rate each option", "jev"),
    ("jev_blind[jev-latest]", "Jev, raw timings only", "jev"),
    ("sonnet-5.5 (subagents)", "Sonnet 5.5 agents", "llm"),
    ("heuristic", "Plain-code rule (tuned)", "code"),
    ("min_error", "Smallest timing error", "code"),
]
board = {}
for st, sel in S["selectors"].items():
    rows = []
    for key, name, kind in NAMES:
        s = sel.get(key)
        if not s:
            continue
        row = {"key": key, "name": name, "kind": kind, "exact": round(s["bars_exact"] * s["n"]), "n": s["n"],
               "onset_acc": s["onset_acc"], "meter_ok": s["meter_ok"]}
        if "latency_p50_s" in s:
            row.update(p50=s["latency_p50_s"], p95=s["latency_p95_s"], tokens=s["input_tokens_per_bar"])
        rows.append(row)
    board[st] = rows

idx = {(r["setting"], r["id"], r["selector"]): r for r in R}
ids = sorted({r["id"] for r in R})
calib, hybrid = {}, {}
for st in ("meter_unknown", "meter_known"):
    js = [idx[(st, i, "jev_choice[jev-latest]")] for i in ids]
    calib[st] = []
    for lo, hi, lab in ((0, 0.5, "below 0.5"), (0.5, 0.8, "0.5 to 0.8"), (0.8, 1.01, "0.8 and up")):
        b = [r for r in js if lo <= r["confidence"] < hi]
        calib[st].append({"label": lab, "n": len(b), "exact": sum(r["exact"] for r in b)})
    hybrid[st] = sum((idx[(st, i, "jev_choice[jev-latest]")] if idx[(st, i, "jev_choice[jev-latest]")]["confidence"] >= 0.5
                      else idx[(st, i, "heuristic")])["exact"] for i in ids)
prev_same = all(idx[(st, i, "jev_choice[jev-latest]")]["label"] == idx[(st, i, "jev_choice[jev-preview]")]["label"]
                for st in ("meter_unknown", "meter_known") for i in ids)
jev_calls = sum(1 for r in R if r["selector"].startswith("jev_"))
jev_tokens = sum(r["input_tokens"] for r in R if r["selector"].startswith("jev_"))
out = {"board": board, "calib": calib, "hybrid": hybrid, "coverage": S["coverage"], "sonnet": S["sonnet_run"],
       "jev_preview_same_choices": prev_same, "jev_calls": jev_calls, "jev_input_tokens_total": jev_tokens,
       "served_model": sorted({r["model"] for r in R if r.get("model")}), "bars": bars}
json.dump(out, open("results/page_data.json", "w"))
print(json.dumps({k: v for k, v in out.items() if k != "bars"}, indent=1)[:3000])
