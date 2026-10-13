# next-harmony

Suggest the **next chord** and its **exact piano notes** from the chords you just played.

```bash
pip install -r requirements.txt
python3 -m nextharmony --key "A major" \
  --chords "E3 C#4 G#4 E5 | F#3 C#4 A4 D5 | A3 C#4 E4 C#5 | E3 D4 G#4 B4"
```

```json
{"suggestions": [
  {"chord": "I",  "meaning": "major triad on the tonic (A), root position, A in the bass", "notes_bass_to_soprano": "A3 E4 A4 C#5"},
  {"chord": "vi", "meaning": "minor triad on the submediant (F#), root position, F# in the bass", "notes_bass_to_soprano": "F#3 C#4 F#4 A4"},
  {"chord": "V",  "meaning": "major triad on the dominant (E), root position, E in the bass", "notes_bass_to_soprano": "E3 E4 G#4 B4"}
]}
```

Each input chord is four notes, bass to soprano, with chords separated by `|`.
- `--selector code` uses only the code models.
- `--selector jev` (the default) also asks TypeSafe's Jev model. Jev's weights come from `results/harmony_summary.json`; in the benchmark both were tuned to 0, so for now the output matches `--selector code`.
- Auth: in a claude.ai cloud environment the TypeSafe connection signs requests for you. Elsewhere, set `TYPESAFE_API_KEY`.

## How it works

1. **Name the recent chords** (`nextharmony/model.py`).
   - Exact notes become a chord label in the key: root scale degree, quality and inversion.
   - For example, `G2 B3 D4 F4` in C major is `V7`, a dominant seventh in root position.
2. **Rank candidate next chords.** A model of how often each chord follows the previous two (a trigram), trained on 256 Bach chorales, ranks every chord label seen at least 5 times.
3. **List exact voicings for each candidate.** Code builds every four-voice voicing that:
   - fits the soprano, alto, tenor and bass ranges
   - keeps sensible spacing between voices
   - includes every chord tone (a seventh chord may drop its fifth)
   - has no parallel fifths or octaves against the previous chord

   Voicings are ranked by voice-leading cost: how far each voice moves, inner-voice leaps, a doubled leading tone or seventh.
4. **Optionally, Jev re-ranks** the shortlists.
   - Chord options are described in words ("dominant seventh chord on the dominant, first inversion, B in the bass").
   - Voicings are given as note names plus each voice's motion.
   - Everything is one request. Code combines Jev's probabilities with its own scores, with weights tuned on held-out chorales.

## Benchmark

The data is 359 four-voice Bach chorales from the music21 corpus. Each chorale is in exactly one set:
- 256 train the chord model
- 34 tune the weights (dev)
- 69 are for testing

Run it with `python3 scripts/benchmark.py`; it writes `results/harmony_summary.json` and `results/harmony_records.jsonl`.

| 240 test positions | Next chord: top pick right | In top 3 | Voicing right (chord given) | Chord and all 4 notes right |
|---|---|---|---|---|
| **Code** (trigram + smoothest legal voicing) | **28%** | **47%** | **30%** | **8%** |
| Jev, choosing from code's top 15 chords / top 8 voicings | 13% | 26% | 28% | 5% |
| Jev + code, weights tuned on dev | 28% | 47% | 30% | 8% (both Jev weights tuned to 0) |

- **Ceilings:** the true next chord is among code's top 15 in 80% of positions. On dev chorales, Bach's exact voicing is among the top 8 legal voicings in 79%.
- **Jev adds nothing yet.** It knows Bach's chord habits less well than a trigram trained on them, and only ties with the voice-leading rules on voicings. It costs about 4k input tokens per request, 0.34 s median.

## Roadmap

1. **Resolve tendency tones in the voicing rules.** A chord's seventh should step down and the leading tone should rise. Today, after a `V7` the seventh can still move up.
2. **Better chord model.** Use a longer history and key changes, and train on more styles (DCML Mozart and Beethoven corpora, pop and jazz chord datasets).
3. **Piano voicings beyond four voices**: two hands, a span limit per hand and register, so the suggestions are playable.
4. **MIDI in, MIDI out**: read the recent chords from a MIDI file or keyboard, and write the suggestions back as MIDI.
5. **Where Jev may still help**: text-side choices such as style or genre from a title or tempo marking, and confidence flags for "unusual" spots. These need their own benchmark.

## Layout

```
nextharmony/   data (chorales -> exact-note chord events), model (labels, n-gram, voicings, Jev), jev (client + cache), __main__ (CLI)
scripts/       benchmark.py
results/       harmony_summary.json, harmony_records.jsonl
tests/         test_harmony.py
archive/       quantize/: the earlier goal of this repo (human MIDI -> sheet-music rhythm with Jev, rounds 1-3)
```

## History

This repository started as **Quantize_Midi_Jev**: testing whether Jev can turn human piano performances into notated rhythm. That work is kept in [`archive/quantize/`](archive/quantize/README.md). The finding: Jev matched simple code baselines but never beat them.
