# Round 3 strategy: quantize a ~4-bar passage with Jev

**Goal.** Input: about 4 bars of human piano MIDI plus the tempo in BPM (quarter notes per minute, as a DAW or MIDI file gives it). No bar lines are given. Output:

- the time signature
- the pickup (upbeat), i.e. where bar 1 starts
- the written position of every note
- the written length of every note

## What other Jev users do (research)

Jev is new: early access opened on 15 September 2026, and nobody has published a music-transcription use yet. The closest guidance comes from TypeSafe's own docs, cookbooks and integration guides. They all say the same thing: **code proposes, Jev picks**.

| Practice | Source | What it means here |
|---|---|---|
| "Jev is not a calculator… keep the arithmetic in code" | [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) | Code does tempo tracking, grid snapping and timing errors. Jev never sees a sum to compute. |
| "Semantic representations beat numeric ones… pass a named bucket" | same | Code turns numbers into words: "timing fit: tight", "new bass note on beat 1 in 4 of 4 bars", "held about an eighth (55% of the gap)". |
| "When the answer space is bounded, turn extraction into a Choice over the options" (Jev can't generate) | same, and [Pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md) | Code enumerates every reasonable reading. Jev chooses one; it never writes a rhythm. |
| Break broad judgements into narrow, atomic questions | [How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md) | Stage 1 is one question about bar structure and rhythm. Stage 2 is one question per note length. |
| Hierarchical decisions: beam search over Choice probabilities | [Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification.md) | Meter → pickup → rhythm → lengths form a tree. Keep several branches with their probabilities and combine them in code. |
| Re-rank a code shortlist | [Re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md), [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion.md) | A cheap code scorer ranks thousands of hypotheses. Jev re-ranks the top 8. |
| Combine atomic scores with weights you control | [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md) | Final pick = code score + a × log P(Jev), with *a* tuned on dev passages. Round 2 found Jev × code beat either alone. |
| Option order bias: Jev leans toward the first option | jaggedness doc; Steve Kinney's guide | Ask the same Choice twice with the options reversed, and average. |
| Put all questions in one request | [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md), [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions.md) | Every note-length question goes in one request (up to 48 per request). |
| Act on confidence; send unsure cases elsewhere | [Confidence](https://docs.typesafe.ai/confidence.md), [Confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md) | Jev is asked about a note's length only when the code model is unsure (top probability < 0.8). |
| Keep state focused; irrelevant detail costs accuracy | jaggedness doc, [State](https://docs.typesafe.ai/concepts/state.md) | Each note question carries only that note's context, in structured `instructions`. |

On the music side, the research says three things:

- **Find the bar position and the rhythm together, not bar by bar.** Metrical HMMs do this: Cemgil and Kappen; Nakamura et al.; [McLeod and Steedman's metrical alignment](https://github.com/apmcleod/met-align/blob/master/README.md), which handles anacrusis explicitly.
- **Strong beats show up in the notes.** Beat 1 tends to carry long notes, bass notes and accents, and phrases repeat (Steedman 1977; [London et al. 2009 on anacrusis](https://cdn.carleton.edu/uploads/sites/721/2021/12/London-Himberg-Cross-2009-MP.pdf)).
- **Note lengths are a separate problem.** Performed lengths drift far from the written ones, and score context (voices) matters more than timing ([Nakamura, Yoshii and Dixon 2017](https://arxiv.org/abs/1703.08144)).

End-to-end neural models such as [PM2S](https://ismir2022program.ismir.net/poster_335.html) and [Beyer and Dai's transformer, ISMIR 2024](https://arxiv.org/abs/2410.00210) are the state of the art. They would need training, whereas Jev works zero-shot.

## Can we give Jev "all the possibilities"? Yes: code enumerates them, Jev ranks a shortlist

The space is a tree, and code walks all of it:

```
time signature   2/4 3/4 4/4 2/2 3/8 6/8 9/8 12/8                                  8
 x first note's place in its bar (= pickup)   every half beat / every eighth     ~55 in total
   x rhythm grid   16ths | 16ths or triplets per beat | triplets | 32nds | sextuplets   5
     -> metrical tempo tracker (Viterbi) decodes the whole passage   ~275 paths per passage
```

1. **Metrical tracker, code** (`jevquant/tracker.py`). A Viterbi search whose state is *(position in the bar, local tempo)*.
   - The given BPM pins the tempo to ±12%, which leaves room for rubato but removes the "is it eighths at 120 or sixteenths at 60?" ambiguity.
   - Costs come from timing error (ms plus %), tempo change, metrical weakness of each landing position, and accents (bass, loudness, length) landing off the beat.
   - One run per (meter, pickup, grid) gives one complete reading with its own bar lines.
2. **Evidence and code score, code** (`jevquant/phrase.py`). For each reading, code computes:
   - how often beat 1 has a note
   - whether new bass notes, louder notes and long notes fall on beat 1 and on beats
   - whether bars repeat the same rhythm
   - syncopation, tuplet count, timing error and the tracker's cost

   A conditional-logit model trained on dev passages turns these into a score.
3. **Jev re-ranks the top 8, one request.** Each option is written for a musician:
   - the time signature
   - the pickup in words ("one quarter note before bar 1 (starts on beat 3)")
   - a counting line per bar ("bar 1: 1(bass)> + 2 e + a 3")
   - the evidence in words ("new bass note on beat 1: 4 of 4 bars")
   - the timing fit as a bucket

   One Choice, asked twice with the options in opposite order. Final pick = code score + a × log P(Jev).
4. **Note lengths** (`jevquant/durations.py`). For each note, code lists:
   - "until the next note in this hand" (legato)
   - shorter values followed by a rest
   - longer values (a held voice)

   A timing model scores them from how long the key was held, converted to beats through the passage's tempo map. Notes where the model is unsure get one Jev Choice each, all in one request. The options are described in words ("eighth note, then a rest of an eighth before the next note in this hand").

Design rules that follow from round 1 and round 2:
- **Never ask Jev for timing arithmetic.** Raw note times scored 4/30 bars in round 1.
- **Give Jev the evidence in words**, not ms tables.
- **Always combine Jev's probabilities with the code score.** It was the best method in round 2.
- **Use confidence to decide whether Jev is needed at all.**

Results: see `README.md` (Round 3) and `results/passages_summary.json`.
