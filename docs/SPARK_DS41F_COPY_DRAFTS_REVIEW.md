# Review: copy drafts in spark-ds41f rc17

2026-11-08 · SparkPipe

## Scope

The copy drafter is correct, deterministic and graph-safe. Its biggest limit is the 16-draft lookahead, which caps what long exact copies can earn.

This review covers rc17 patches 0583-0590 in `spark-ds41f/experiments/2026-10-26-spark-ds41f-candidate/rc17/vllm` and `rc-tp3-copy.json`: min_match 16, initial_drafts 8, max_drafts 16, beside DSpark's 5-token block, adaptive verification row cost 0.035, on the 3-node triangle's 59-row capture ladder. We read the code but did not run it, so no number below was measured on his hardware.

## How it works

Each step, a request whose context tail repeats earlier text proposes the tokens that followed the latest repeat. Those drafts replace DSpark's, and adaptive verification decides how many rows to spend.

1. `copy_match` (TileLang, one block per request × 32 chunks, no atomics) scans the request's own history. For each level (min_match, doubling up to 256), it finds the latest window end whose preceding tokens equal the context tail.
2. `copy_select` first updates the request's copy state from this step's verdict. The cap is initial × 2^doublings: a fully accepted copy adds a doubling, and a rejection or a DSpark step resets it to 0. Acceptance counts are decayed by 0.9.
3. It then takes the longest level's latest source and offers min(cap, total_len − source) drafts. Every draft carries the same confidence, (hits + 19) / (trials + 20). It also reports `alloc` for the async scheduler two steps ahead.
4. Adaptive verification ranks rows by survival = cumprod(confidences). It keeps the row count that maximizes expected tokens / (1 + 0.035 × rows).

## Strengths

It is built the way a drafter should be: on the GPU, in the graph, and unable to change the text.

- Deterministic: reductions don't depend on order, there are no atomics, and nothing syncs to the host. Both kernels replay inside the captured graph.
- Isolated: reads stay inside the request's own history row. The -1 key sentinel can never match a token id. Copy state resets when a new request takes the slot.
- Clean split: DSpark's 5-token block and the 16-row verification lookahead are separate knobs, and a step without a copy still verifies at DSpark's block.
- Well tested: numpy references cover contexts up to 20,000 tokens, vocabularies from 3 to 129,280 and any slot order. Tests also cover graph replay, stale-slot and cross-request isolation, and identical temperature-0 text with and without copy drafts.
- Budgeted: the benchmark gates a step's copy kernels at 1% of the release suite's step time.

## Findings and suggestions

The largest gain is a wider lookahead for long copies, then a confidence model that knows long copies keep going. One latent out-of-bounds write is worth guarding now. Line numbers refer to `vllm/v1/worker/gpu/spec_decode/tilelang_copy.py` as of patch 0590.

1. **The 16-draft lookahead caps long copies.** His own row cost (0.035 a row) puts the best frame far past 16 drafts once a copy is reliable. Under that model, the best frame at 99% conditional acceptance is 64 drafts, 1.5x the 16-draft rate. At 99.9% (whole-file rewrites, regenerated code) it is 256 drafts, 2.1x. The doubling ladder already reaches 256 (`MAX_DOUBLINGS` 16). What's missing is a wide step for one copying request (larger capture sizes or an uncaptured wide step) and a decode-row envelope that grows while it copies. The linear row cost should be measured at 64-256 rows before relying on it.
2. **One confidence for every draft (:237).** Survival is then p^k: at p = 0.95 it is 0.44 at depth 16 and 0.04 at depth 64. So the budget cuts long frames even when the source is a verbatim file. Copy acceptance depends on depth and on the matched level: a 256-token match keeps going far longer than a 16-token one. Decayed hit/trial counts per depth bucket (0-7, 8-15, 16-31, 32-63, 64+) and per matched level, kept in the same slot state, let the cumprod model this.
3. **One rejection resets the cap to 8 (:209).** Accepting 15 of 16 drops the next frame to 8. Setting the cap to max(initial, accepted + 1), or halving it, keeps most of the ramp.
4. **Only the latest source is tried (:216-223).** After an edit, the latest occurrence's continuation diverges while earlier occurrences still agree. Keeping a few candidates per level, and skipping one whose continuation was rejected at this position, recovers those frames.
5. **Copies run past stop tokens.** A source that ends a turn or a tool call proposes rows past the end of the reply. Truncating the offer at the request's stop tokens saves those rows.
6. **Short periods offer few drafts (:225).** offered = min(cap, distance), so a period-p repetition proposes only p drafts a step (one for a run of the same token). An overlapping copy, `history[start + col % distance]`, offers the full cap, the way LZ77 copies overlap.
7. **Latent out-of-bounds write (:198).** `copy_match` checks `slot >= 0` (:106), but `copy_select` reads and writes `int_state[slot]` and `float_state[slot]` without the check. The runner passes only real requests today. The same guard, or an assert, stops a future padded batch from writing out of bounds.
8. **The scan is O(context) per step.** `copy_match` reads the whole history every step: 4 MB per request at 1M int32 ids, 64 MB for 16 such requests. That fits the 1% gate today. If long-context concurrency grows, an append-only per-request n-gram index (hash buckets plus position chains) makes it O(1) per token.

## Compared with SparkPipe's lookup drafter

SparkPipe goes further on long copies: 255 drafts, depth-aware hazards and 64 candidates. His design is cleaner and cheaper per step: drafting in the graph, cheaper verify rows, and bit-identical text.

| | spark-ds41f copy drafts | SparkPipe lookup drafter (`runtime/model_batch_engine.c`) |
| --- | --- | --- |
| Where it runs | GPU, inside the captured graph, no host sync | Host, per request, planned before each frame |
| Match | Full-history scan per step, levels 16 to 256 | Append-only 3-gram hash index (4,096 buckets), tail extended to a 64-token match, minimum 8 |
| Sources tried | Latest at the longest level | Up to 64 candidates |
| Max drafts | 16 (initial 8) | 255 (256-row frames) |
| Confidence | One decayed estimate per request, cumprod over depth | Hazards per depth bucket and match class, engine-wide plus per request |
| Frame size | Expected tokens / (1 + 0.035 × rows), batch-wide | Expected tokens / measured frame cost, per request |
| Stop tokens | Not truncated | Frame truncated at stop tokens |
| Verify cost per row | 3.5% of a step (configured) | DSV4.1: ~2.1 ms, 6.8% of a 2-row step (31 ms at 2 rows, 564 ms at 256) |
| Greedy text vs plain decode | Identical at every frame size tested | Identical up to 8-row frames; wider frames differ in bits, accuracy held |
| Long-copy result | Not reported for wide frames | GLM whole-file edits 28 → 233 tok/s; K3 15 → 140 tok/s |

He can take the wide frames, depth hazards, candidates and stop truncation (findings 1-5). SparkPipe should take in-graph drafting and survival-ranked batch budgeting. Our bigger gap is verify cost per row, not drafting.
