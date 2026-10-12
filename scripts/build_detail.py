"""Per-bar detail for the report: truth vs each method's notation (+ MusicXML)."""
import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jevquant.bench import load_bars, prepare  # noqa: E402
from jevquant.candidates import counting  # noqa: E402
from jevquant.notation import bar_score, quantized_notes, to_musicxml  # noqa: E402

SHOW = {"heuristic": "Plain-code rule", "jev_choice[jev-latest]": "Jev", "sonnet-5.5 (subagents)": "Sonnet 5.5",
        "min_error": "Smallest timing error"}

ev = json.load(open("data/eval_bars.json"))
records = [json.loads(l) for l in open("results/records.jsonl")]
NV = {(r["id"], r["selector"]): r for r in map(json.loads, open("results/notevalues_records.jsonl"))}
NV_SHOW = {"Jev (raw) + bar fit": "Jev, note by note", "Jev (raw) x timing model + bar fit": "Jev x timing model"}
rec = {(r["setting"], r["id"], r["selector"]): r for r in records}
chunks = load_bars(ev["eval_ids"])
out = []
for setting in ("meter_unknown", "meter_known"):
    for pr in prepare(chunks, setting):
        ch = pr["chunk"]
        m = ch.meta
        bid = m["id"]
        truth_on = sorted({s.onset for s in m["score_notes"]})
        item = out_item = next((o for o in out if o["id"] == bid), None)
        if item is None:
            n_q = int(F(m["bar_quarters"]))
            item = {
                "id": bid, "piece": m["piece"], "performer": m["performer"], "bar": m["bar"] + 1,
                "timesig": m["timesig"], "seconds": round(m["bar_seconds"], 3), "notes": len(ch.perf),
                "rhythm_type": ev["reasons"].get(bid, {}).get("rhythm_type", ""),
                "truth_counting": counting(truth_on, n_q),
                "truth_xml": to_musicxml(bar_score(m["score_notes"], m["timesig"])),
                "settings": {},
            }
            out.append(item)
        picks = {}
        for sel, name in SHOW.items():
            r = rec.get((setting, bid, sel))
            if r is None or r.get("label") is None:
                continue
            c = pr["options"][r["label"]]
            ts = f"{c.feats['quarters_per_bar']}/4"
            p = {"name": name, "exact": r["exact"], "onset_acc": round(r["onset_acc"], 3), "meter_ok": r["meter_ok"],
                 "counting": c.feats["counting"], "bpm": c.feats["bpm"], "timesig": ts, "grid": c.feats["grid"],
                 "err_ms": c.feats["mean_timing_error_ms"]}
            if "confidence" in r:
                p["confidence"] = r["confidence"]
            if sel in ("jev_choice[jev-latest]", "heuristic"):
                p["xml"] = to_musicxml(bar_score(quantized_notes(ch, c, pr["idx"]), ts))
            picks[sel] = p
        if setting == "meter_known":
            n_q = int(F(m["bar_quarters"]))
            for sel, name in NV_SHOW.items():
                r = NV.get((bid, sel))
                if r:
                    picks[sel] = {"name": name, "exact": r["exact"], "onset_acc": round(r["onset_acc"], 3), "meter_ok": True,
                                  "counting": counting([F(p) for p in r["positions"]], n_q), "timesig": m["timesig"]}
        item["settings"][setting] = {
            "truth_in_options": pr["truth_in_shortlist"], "n_options": len(pr["options"]), "picks": picks,
        }
json.dump(out, open("results/bars_detail.json", "w"))
print(len(out), "bars", os.path.getsize("results/bars_detail.json") // 1024, "KB")
