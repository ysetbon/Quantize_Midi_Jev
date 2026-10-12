"""Suggest the next chord and its exact notes from the recent chords.

  python -m jevharmony --key "A major" --chords "E3 C#4 G#4 E5 | F#3 C#4 A4 D5 | A3 C#4 E4 C#5 | E3 D4 G#4 B4"

Each chord is four notes, bass to soprano.  Prints the top suggestions (chord
name, meaning, exact notes) from the code model combined with Jev, using the
weights tuned by scripts/run_harmony.py (results/harmony_summary.json).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from math import log

from music21 import pitch as m21pitch

from .data import build, split_of
from .model import NGram, jev_suggest, label_of, name_of, note_name, soft, tonic_pc, voicings, words_of

NGRAM_CACHE = "data/harmony_ngram.pkl"


def ngram() -> NGram:
    if os.path.exists(NGRAM_CACHE):
        return pickle.load(open(NGRAM_CACHE, "rb"))
    seqs = []
    for c in build():
        if split_of(c["id"]) == "train":
            t = tonic_pc(c["tonic"])
            seqs.append((c["mode"], [l for l in (label_of(e["notes"], t) for e in c["events"]) if l]))
    ng = NGram(seqs)
    pickle.dump(ng, open(NGRAM_CACHE, "wb"))
    return ng


def main() -> None:
    ap = argparse.ArgumentParser(prog="jevharmony")
    ap.add_argument("--key", required=True, help='e.g. "A major" or "G minor"')
    ap.add_argument("--chords", required=True, help='recent chords, bass to soprano, separated by "|"')
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--selector", choices=["jev", "code"], default="jev")
    ap.add_argument("--model", default="jev-latest")
    args = ap.parse_args()
    tonic_name, mode = args.key.split()
    tonic_name = tonic_name.replace("b", "-") if len(tonic_name) > 1 else tonic_name
    tonic = tonic_pc(tonic_name)
    chords = [[m21pitch.Pitch(n.replace("b", "-") if len(n) > 2 else n).midi for n in grp.split()]
              for grp in args.chords.split("|")]
    hist = [label_of(c, tonic) for c in chords]
    w = {"a_chord": 1.0, "b_voicing": 1.0, "voicing_tau": 1.0}
    if os.path.exists("results/harmony_summary.json"):
        w.update(json.load(open("results/harmony_summary.json"))["weights"])
    ng = ngram()
    cands = ng.candidates(mode.lower())
    pc = ng.probs(mode.lower(), [h for h in hist if h], cands)
    top = sorted(cands, key=lambda x: -pc[x])[:15]
    vopts = {lab: voicings(lab, tonic, chords[-1]) for lab in top[:5]}
    vopts = {lab: vs for lab, vs in vopts.items() if vs}
    pvc = {lab: soft({v: c for v, c in vs}, w["voicing_tau"]) for lab, vs in vopts.items()}
    pj, pvj, conf = {}, {}, None
    if args.selector == "jev":
        state = {"key": args.key, "style": "Bach chorale, four voices", "recent_chords": [
            {"chord": name_of(h) if h else "?", "meaning": words_of(h, tonic) if h else "unclear",
             "notes_bass_to_soprano": " ".join(note_name(m) for m in c)} for h, c in zip(hist, chords)]}
        pj, pvj, r = jev_suggest({"tonic_pc": tonic, "prev_notes": chords[-1], "state": state}, top, vopts,
                                 model=args.model, seed=args.chords)
        conf = r["answers"]["next_chord"]["confidence"]
    score = {x: log(pc[x]) + w["a_chord"] * log(max(pj.get(x, 1.0 if not pj else 0), 1e-4)) for x in vopts}
    out = []
    for lab in sorted(score, key=lambda x: -score[x])[:args.top]:
        pv = pvj.get(lab)
        best = max(pvc[lab], key=lambda v: log(pvc[lab][v]) + (w["b_voicing"] * log(max(pv.get(v, 0), 1e-4)) if pv else 0))
        out.append({"chord": name_of(lab), "meaning": words_of(lab, tonic),
                    "notes_bass_to_soprano": " ".join(note_name(m) for m in best)})
    print(json.dumps({"suggestions": out, "jev_confidence": conf}, indent=1))


if __name__ == "__main__":
    main()
