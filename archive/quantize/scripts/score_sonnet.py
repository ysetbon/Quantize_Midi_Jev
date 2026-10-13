"""Score the Sonnet 5.5 workflow answers against the same truth as Jev, and add
them to results/records.jsonl + results/summary.json.

usage: python scripts/score_sonnet.py <workflow output json>
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.bench import load_bars, prepare, score, summarize  # noqa: E402

wf = json.load(open(sys.argv[1]))
result = wf["result"] if isinstance(wf["result"], list) else json.loads(wf["result"])
answers = {a["key"]: a["choice"] for g in result for a in (g["answers"] or [])}
keymap = json.load(open("results/sonnet_key_map.json"))
agents = [a for a in wf["workflowProgress"] if a.get("type") == "workflow_agent"]
tok = [a["tokens"] for a in agents]
dur = [a["durationMs"] / 1000 for a in agents]

ids = json.load(open("data/eval_bars.json"))["eval_ids"]
chunks = load_bars(ids)
records = [json.loads(l) for l in open("results/records.jsonl") if '"sonnet' not in l]
for setting in ("meter_unknown", "meter_known"):
    for pr in prepare(chunks, setting):
        key = f"{setting}:{pr['chunk'].meta['id']}"
        anon = next(k for k, v in keymap.items() if v == key)
        lab = answers.get(anon)
        if lab not in pr["options"]:
            lab = None
        rec = {"id": pr["chunk"].meta["id"], "setting": setting, "selector": "sonnet-5.5 (subagents)", "label": lab}
        if lab is None:
            rec.update(meter_ok=False, onset_acc=0.0, onset_acc_any_beat_unit=0.0, exact=False, invalid=True)
        else:
            rec.update(score(pr["chunk"], pr["options"][lab], pr["idx"]))
        records.append(rec)
with open("results/records.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
summary = json.load(open("results/summary.json"))
summary["selectors"] = summarize(records)
summary["sonnet_run"] = {
    "agents": len(agents), "bars_per_agent": 10, "total_tokens": sum(tok),
    "tokens_per_bar_incl_agent_overhead": round(sum(tok) / 60),
    "agent_wall_time_s": {"min": round(min(dur), 1), "max": round(max(dur), 1), "mean": round(statistics.mean(dur), 1)},
    "seconds_per_bar": round(statistics.mean(dur) / 10, 2),
    "price_per_mtok_usd": {"input": 2.0, "output": 10.0},
}
json.dump(summary, open("results/summary.json", "w"), indent=1)
print(json.dumps(summary["sonnet_run"], indent=1))
for st in summary["selectors"]:
    print(st, summary["selectors"][st]["sonnet-5.5 (subagents)"])
