"""Command line.

  python -m jevquant quantize PERF.mid --annotations BEATS.txt --timesig 3/4 -o out.musicxml
      Quantize a performance bar by bar (downbeats from an ASAP-style annotation
      file: "<time>\\t<time>\\tdb..." / "b" rows) and write MusicXML.
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
    sub.add_parser("bench")
    args = ap.parse_args()
    if args.cmd == "quantize":
        quantize(args)
    else:
        import runpy
        runpy.run_path("scripts/run_bench.py", run_name="__main__")


if __name__ == "__main__":
    main()
