"""Command line.

  python -m jevquant quantize PERF.mid --annotations BEATS.txt --timesig 3/4 -o out.musicxml
      Quantize a performance bar by bar (downbeats from an ASAP-style annotation
      file: "<time>\\t<time>\\tdb..." / "b" rows) and write MusicXML.
  python -m jevquant passage PERF.mid --bpm 120 [--start 12.5 --end 21] [--timesig 3/4] -o out.musicxml
      Round 3: quantize a ~4-bar passage with no bar lines given.  Finds the time
      signature (unless given), the pickup, every onset and every note length.
      Uses the scorer trained by scripts/run_passages.py (results/passages_model.json).
  python -m jevquant bench
      Re-run the 30-bar benchmark (needs data/eval_bars.json; Jev answers are cached).
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F

from .asap import EARLY, read_annotations
from .candidates import generate_bar
from .midi_io import read_notes
from .model import Chunk, PerfNote
from .notation import piece_score, quantized_notes, to_musicxml
from .selectors import Heuristic, sel_jev_choice, shortlist

# weights tuned on the dev bars with the time signature given (results/summary.json)
RULE = Heuristic(w_cx=2.5, w_tup=0.5, w_merge=1.0, w_tempo=0.0)


def quantize(args) -> None:
    perf, _, _ = read_notes(args.midi)
    downbeats = [t for t, is_db, _ in read_annotations(args.annotations) if is_db]
    num, den = (int(x) for x in args.timesig.split("/"))
    qpb = F(num * 4, den)
    lo, hi = (int(x) for x in args.bars.split(":")) if args.bars else (0, len(downbeats) - 1)
    bars, log = [], []
    for k in range(lo, min(hi, len(downbeats) - 1)):
        t0, t1 = downbeats[k], downbeats[k + 1]
        notes = [PerfNote(n.pitch, n.onset - t0, n.offset - t0, n.velocity) for n in perf if t0 - EARLY <= n.onset < t1 - EARLY]
        if not notes:
            bars.append([])
            continue
        ch = Chunk(perf=notes, meta={"id": f"bar{k}", "bar_seconds": t1 - t0, "timesig": args.timesig})
        cands, _, idx = generate_bar(ch, 0.0, t1 - t0, beats=(int(qpb),) if qpb.denominator == 1 else (2, 3, 4, 6))
        options = shortlist(cands, ch.meta["id"], quarters_per_bar=int(qpb) if qpb.denominator == 1 else None)
        if args.selector == "jev":
            label, resp = sel_jev_choice(options, ch, model=args.model, meter_known=True)
            log.append({"bar": k, "choice": label, "confidence": resp["answers"]["best"]["confidence"],
                        "input_tokens": resp["usage"]["input_tokens"]})
        else:
            label = RULE(options)
            log.append({"bar": k, "choice": label})
        bars.append(quantized_notes(ch, options[label], idx))
    open(args.out, "w").write(to_musicxml(piece_score(bars, args.timesig)))
    print(json.dumps(log, indent=1))
    print(f"wrote {args.out} ({len(bars)} bars)", file=sys.stderr)


def passage(args) -> None:
    from math import floor

    from . import durations as D
    from .model import ScoreNote
    from .phrase import METERS, apply_scorer, combine, hypotheses, jev_rank, shortlist

    perf, _, _ = read_notes(args.midi)
    end = args.end if args.end is not None else float("inf")
    notes = [n for n in perf if args.start <= n.onset < end]
    t0 = notes[0].onset
    ch = Chunk(perf=[PerfNote(n.pitch, n.onset - t0, n.offset - t0, n.velocity) for n in notes],
               meta={"id": f"{args.midi}:{args.start}:{args.end}"})
    model = json.load(open(args.model_file))
    setting = "ts_known" if args.timesig else "ts_unknown"
    hs, cl = hypotheses(ch, (args.timesig,) if args.timesig else METERS, qbpm=args.bpm)
    apply_scorer(hs, model[setting]["scorer"])
    log: dict = {}
    if args.selector == "jev":
        opts = shortlist(hs)
        probs, r = jev_rank(ch, opts, cl, qbpm=args.bpm, model=args.model)
        h = combine(opts, probs, model[setting]["jev_weight_a"])
        log.update(jev_confidence=r["answers"]["notation"]["confidence"], structure_tokens=r["usage"]["input_tokens"])
    else:
        h = max(hs, key=lambda x: x.code)
    dm = model.get("durations", {"code_model": {}, "jev_exponent_b": 1.0, "gate": 0.8})
    ctxs = D.note_context(ch, h, cl)
    pc = [D.code_probs(c, D.options(c), **dm["code_model"]) for c in ctxs]
    ask_idx = [i for i, p in enumerate(pc) if max(p.values()) < dm["gate"]] if args.selector == "jev" else []
    pj, usage = D.jev_durations(ch, h, ctxs, ask_idx, model=args.model) if ask_idx else ({}, {"input_tokens": 0})
    pos = h.positions()
    bars: dict[int, list] = {}
    for i, n in enumerate(ch.perf):
        p = pos[cl.note_cluster[i]]
        comb = {v: pc[i][v] * (pj[i].get(v, 1e-4) ** dm["jev_exponent_b"] if i in pj else 1) for v in pc[i]}
        b = floor(p / h.L)
        on = p - b * h.L
        dur = min(max(comb, key=comb.get), h.L - on)
        bars.setdefault(b, []).append(ScoreNote(n.pitch, on, dur, n.velocity))
    seq = [bars.get(b, []) for b in range(min(bars), max(bars) + 1)]
    open(args.out, "w").write(to_musicxml(piece_score(seq, h.ts)))
    log.update(time_signature=h.ts, pickup_quarters=str(h.pickup), grid=h.cand.feats["grid"],
               notes_asked_jev=len(ask_idx), duration_tokens=usage["input_tokens"])
    print(json.dumps(log, indent=1))
    print(f"wrote {args.out} ({len(seq)} bars{', first is a pickup' if h.pickup else ''})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(prog="jevquant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("midi")
    q.add_argument("--annotations", required=True)
    q.add_argument("--timesig", required=True)
    q.add_argument("--bars", help="range of bar indices, e.g. 0:8")
    q.add_argument("--selector", choices=["jev", "rule"], default="jev")
    q.add_argument("--model", default="jev-latest")
    q.add_argument("-o", "--out", default="out.musicxml")
    p = sub.add_parser("passage")
    p.add_argument("midi")
    p.add_argument("--bpm", type=float, required=True, help="quarter notes per minute")
    p.add_argument("--start", type=float, default=0.0, help="seconds")
    p.add_argument("--end", type=float, default=None, help="seconds")
    p.add_argument("--timesig", help="e.g. 3/4; omit to let the quantizer decide")
    p.add_argument("--selector", choices=["jev", "code"], default="jev")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--model-file", default="results/passages_model.json")
    p.add_argument("-o", "--out", default="passage.musicxml")
    sub.add_parser("bench")
    args = ap.parse_args()
    if args.cmd == "quantize":
        quantize(args)
    elif args.cmd == "passage":
        passage(args)
    else:
        import runpy
        runpy.run_path("scripts/run_bench.py", run_name="__main__")


if __name__ == "__main__":
    main()
