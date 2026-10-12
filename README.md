# Quantize_Midi_Jev

Can TypeSafe's **Jev** model help turn short chunks of human piano MIDI into
sheet-music rhythm quickly, cheaply and well?

This repo tests that on **30 single bars of real Mozart performances**. Each
bar is a real pianist's MIDI with the matching written score. Jev is compared
with a tuned plain-code rule and with Claude Sonnet 5.5.

Results page: https://claude.ai/artifact/HfPhP3w3HrBJwq6JBmDqJq (private until you share it). It
renders every bar as notation for the written score, Jev's choice and the rule's choice.

## Short answer

| | Time signature given | Time signature unknown |
|---|---|---|
| **Jev, pick one** | **20/30** bars exact, 91% of notes | 14/30, 77% of notes |
| Plain-code rule, tuned on 150 other bars | 21/30, 92% | 17/30, 89% |
| Sonnet 5.5 (Claude Code subagents) | 19/30, 85% | 17/30, 84% |
| Best option in the list (ceiling) | 24/30, 94% | 20/30, 91% |
| Jev, raw note times only (no candidates) | 14/30, 87% | 4/30, 50% |

- **Fast:** yes. Jev takes 0.31–0.33 s median per bar (about 0.6 s at p95). The Sonnet agents took about 2 s per bar.
- **Cheap:** probably. Jev uses about 1,100 input tokens per bar (1,400 when the time signature is unknown), and output tokens are free. The Sonnet agents used about 8,900 tokens per bar. Jev's price per token isn't known here, so check it on your TypeSafe account.
- **Good:** roughly as good as a hand-tuned rule and Sonnet 5.5, not better.
  - **It doesn't work from raw input.** Jev can't do timing arithmetic from raw note times; it needs code to propose candidates and measure the error first.
  - **The time signature matters:** without it, Jev picks the wrong number of beats per bar in 20% of bars.
- **Confidence is useful when the time signature is unknown.** Every bar where Jev's confidence was below 0.5 was wrong (0/13). Bars at 0.5 or above were right 14 times out of 17.
  - Using Jev when confident and the rule otherwise gives 18/30, better than either alone.
  - The 0.5 cutoff was chosen after seeing these bars, so confirm it on more data.
- `jev-preview` currently serves the same model as `jev-latest` (`jev-1.13.0`).

Total Jev usage for the benchmark was 360 requests and about 551k input tokens.

### Round 2: Jev names every note's value

Here Jev gets the bar's notes directly, not ready-made options:
- pitches
- start time, in ms after the downbeat
- the gap to the next note, in ms
- how long each key was held

It answers one `choice` question per onset: whole note, dotted half, half, dotted quarter, quarter, dotted eighth, triplet quarter (1/6 of a whole note), eighth, triplet eighth (1/12), 16th, sextuplet 16th, 32nd, or grace note. One request covers a whole bar (`jevquant/notevalues.py`, `scripts/run_notevalues.py`). The time signature is given.

| 30 bars, time signature given | Bars exact | Notes right | Values right | Input tokens/bar |
|---|---|---|---|---|
| **Jev × code timing model, fit to bar** | **23/30** | 88% | 93% | 4,191 |
| **Jev, fit to bar** | **22/30** | 88% | 91% | 4,191 |
| Jev, top answer per note (no fit) | 18/30 | 79% | 87% | 4,191 |
| Jev given the gap in beats too, fit to bar | 19/30 | 86% | 87% | 4,352 |
| Code timing model (tuned on dev bars), fit to bar | 22/30 | 87% | 90% | – |
| Code, nearest value | 10/30 | 66% | 75% | – |

"Fit to bar" means a small dynamic program keeps Jev's per-note probabilities but forces the values to add up to exactly one bar. Jev's own top answers filled the bar in only 16 of 30 bars.

- **Accuracy:** naming notes is Jev's best mode. It got 22/30 on its own, vs 20/30 when picking whole-bar options. Combined with the code's timing model it got 23/30, the best of any method tested. Differences of one bar are within noise on 30 bars.
- **Input format:** raw milliseconds worked better than also giving the gap in beats (22 vs 19).
- **Mistakes:** Jev's errors are mostly 16th vs 32nd, sextuplet vs 16th, and grace notes vs 16ths.
- **Cost:** about 540 input tokens per note, or about 4.2k per bar, roughly 4× round 1. Speed is unchanged at about 0.36 s per bar.

## How it works

Jev answers classification questions (yes/no, choice, score); it can't write
notes. So the pipeline splits the work:

1. **Code proposes candidates** (`jevquant/candidates.py`).
   - Notes played within 45 ms of each other are grouped into one chord.
   - For each hypothesis of 2, 3, 4 or 6 quarter beats per bar, it fits a tempo that follows rubato, anchored at the annotated downbeats.
   - It snaps notes to several grids: eighths, sixteenths, triplets, binary-or-ternary per beat, cross-rhythms, 32nds and sextuplets.
   - It measures the timing error of each candidate in ms.
2. **The candidates are shortlisted** (`jevquant/selectors.py`).
   - Up to 8 options per bar, in a deterministic shuffle and labelled A–H.
   - Each option shows its tempo, its time signature, a counting line ("1 e + a | 1 trip let"), its mean and max error in ms, and its tuplet and merged-note counts.
3. **A selector picks one option.**
   - Jev, with one `choice` request per bar (`POST /v1/systemone`).
   - The plain-code rule.
   - Sonnet 5.5, given the identical JSON.
4. **Scoring** (`jevquant/bench.py`) compares every note's chosen position with its written position. Rolled chords and grace notes are scored at the chord's position.
5. **Sheet output** (`jevquant/notation.py`) writes MusicXML with two staves split at middle C.

### Data

The data comes from the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset): real Disklavier performances from MAESTRO and the e-competition, each with the written score, beat annotations and downbeat annotations.

| Piece | Time signature | Performers |
|---|---|---|
| K.331 iii (Alla Turca) | 2/4 | 1 |
| K.332 i | 3/4 | 4 |
| K.310 i | 4/4 | 5 |

The data is fetched with a sparse git clone into `data/` (gitignored).

YouTube isn't reachable from the cloud environment, and audio would need transcribing first anyway. Real Disklavier MIDI is a cleaner input.

### Workflow with Sonnet 5.5 subagents

1. **Curate:** three agents, one per piece, picked 10 rhythmically varied bars each from `data/bar_pool.json`.
2. **QA:** an agent per piece checked each bar's score alignment. It flagged five bars, which `scripts/select_bars.py` replaced with aligned bars of similar rhythm (`data/eval_bars.json`).
3. **Compete:** six agents answered the same 60 choices Jev answered (30 bars × 2 settings). Bar ids were anonymised so they couldn't look up the answer.

## Run it

```bash
pip install -r requirements.txt
python3 scripts/build_bar_pool.py          # fetch ASAP subset + list all bars
python3 scripts/run_bench.py               # baselines + Jev (cached in .jev_cache/)
python3 scripts/score_sonnet.py results/sonnet_workflow_output.json
python3 scripts/run_notevalues.py          # round 2: Jev names each note (cached)
python3 scripts/build_detail.py && python3 scripts/page_data.py && python3 scripts/build_page.py
python3 -m pytest -q tests

# quantize your own performance bar by bar (downbeats from an ASAP-style annotation file)
python3 -m jevquant quantize perf.mid --annotations perf_annotations.txt --timesig 3/4 --bars 0:8 -o out.musicxml
```

**Auth:** in this cloud environment the TypeSafe connection signs requests for you. Elsewhere, set `TYPESAFE_API_KEY`.

## Next steps that would move the numbers

- **Better candidates.** 20% of bars never had the written rhythm among the candidates (fast ornaments, 32nd runs, notes far off the beat), which caps every method. Better ornament handling and per-voice snapping would raise the ceiling.
- **Ask Jev narrower questions.** For example, ask the meter once per passage rather than once per bar. Also try breaking ties between the rule's top two options.
- **Confirm the confidence-routing result on a larger, held-out set of bars.**

## Layout

```
jevquant/   model, synth (synthetic data), midi_io, asap (real bars), candidates,
            jev (client + cache), selectors, bench, notation, __main__ (CLI)
scripts/    pool/selection, benchmark, Sonnet scoring, report page builders
results/    summary.json, records.jsonl, payloads.json, bars_detail.json, report page
examples/   K.332 i bars 1-8 quantized by Jev and by the rule (MusicXML)
tests/      unit tests
```
