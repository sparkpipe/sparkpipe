# Long verify frames

Lookup speculation for code edits: the engine drafts from the request's own
history and verifies up to 255 drafts in one frame of up to 256 rows. This
document is the contract for that path. Measured results are in
`PERFORMANCE_STATUS.md` (section "Speculation: long verify frames").

## What a frame is

- The engine finds the longest suffix match (up to 64 tokens, 64 candidates)
  of the request's committed tokens in its own history and drafts the tokens
  that followed the match.
- A verify submission carries one lane: the anchor row (the last committed
  token at position p) and one row per draft at p+1..p+d, wave-major, with the
  VERIFY flag. `tokens_per_sequence` is d+1 and the lane context is p+d+1.
- Limits:
  - `SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE` (256) per
    lane, separate from decode chains
    (`SPARK_MODEL_SERVING_ADAPTER_MAX_TOKENS_PER_SEQUENCE`, 8);
  - the descriptor's `max_speculative_token_count` (255 on GLM Full and K3
    TP16);
  - the descriptor's `max_verify_row_count`, the most rows one frame may
    carry (K3 256 because of its replay buffers, GLM Full 4096);
  - the completion carries at most 1024 tokens
    (`SparkModelServingAdapterCompletionTokenCapacity`).
- Drafts stop before an engine stop token, at the output budget, at the
  context limit and two tokens before a recurrent checkpoint
  (`cache_checkpoint_token_count`), so a stop or the budget can only be the
  last kept token and a checkpoint is always reached by a plain decode step.
- No frame while more than one lane is active, while a lane wants logprobs,
  and at most 8 rows while a prefill waits: a frame costs more than a batched
  decode step, and co-batched lanes would lose their chains.
- Sampled lanes draft too. Each verify row of a sampled lane is sampled with
  the lane's rule at its own position; the draw is keyed by seed, position and
  token, so the kept tokens are the ones plain sampled decode would emit.

## Kept rows

- Every rank runs the head on every row and holds every row's token after the
  argmax all-reduce. The stage adapter counts each lane's kept rows on the host
  (`SparkModelServingVerifyKeptRows`): one plus the number of leading drafts
  equal to the previous row's output.
- The completion holds exactly the kept tokens; the rest of the lane's stride
  is `SPARK_MODEL_SERVING_NO_TOKEN` (`SparkModelServingCompletionKeptRows`).
- The pipeline client requires the verify stride back and at least one kept
  token per lane. The engine re-derives the kept count from its own drafts and
  fails the request (VALIDATION_FAILED) before emitting anything if the counts
  differ.
- The continuation lease resumes at p + kept, so CONTINUE survives every
  verify round.

## KV commit

- The node commits a verify lane at exactly p + kept
  (`SparkKvLaneTransactionsFinishVerify`, `SparkKvPageCacheCompleteVerifyLane`).
- A verify lane carries one block identity for every block end in
  (p, p+d+1], digested from the committed tokens and the drafts. The node seals
  every full block below the kept length under those identities, including the
  last one when the kept length is a block multiple, and discards the pages
  past it. A rejected row never reaches an identity.
- The hub then records floor((p + kept) / B) * B as published, so the reply's
  blocks are reusable by the next prompt.
- There is no rewind: a lane resumes only at its next position.

## Recurrent state (K3)

- A verify step of a model with recurrent state runs with commit off. K3 binds
  replay buffers for the step (conv inputs, retention and write gate per row
  and KDA layer, at the rank's 6 heads; about 136 MB per rank for 256 rows).
- After the adapter counts kept rows, `SparkStageRunnerKeepRows` hands them to
  the model's `keep_rows` hook on the same stream. K3 folds the kept rows into
  the KDA state and conv windows with the forward's own conv, normalisation
  and delta-rule kernels through the wave's row indices, so the folded state
  equals a committed pass over the kept rows.
- A recurrent model without `keep_rows` refuses VERIFY.
- A verify frame drops the lane's staged checkpoint page: verify sealing does
  not stage, so a staged page from an earlier publish would otherwise receive
  a later state at release. A reply that ends on a block multiple inside a
  verify frame therefore keeps no reply-end resume point.

## Long context (GLM)

A GLM wave above 8 ordered rows used to all-gather the whole cached context's
latent in every layer (about 3.4 ms per 1K tokens of context per wave). A wave
of up to 256 rows (`GLM_SCATTER_ROWS`, the longest verify frame) now scatters
when the query, candidate and partial exchange costs less than that gather
(`SparkKvShardExchangeCostNs`), as K3 already chooses. Per peer and row the
scatter moves about 29 KB (4.6 KB of queries, the rank's 2,048 DSA candidates
at 8 bytes, 8.2 KB of partials) against about 72 bytes per context token for
the gather, so it wins above roughly 400 context tokens per row: about 26K
tokens for a 64-row frame, 100K for 256 rows. The scatter buffers for 256
rows take about 130 MB more per lane and rank than for 64.

## Frame size

`src/spark_speculation_frame.c` prices every frame; it replaces the doubling
cap.

- Acceptance: per match-length class (3-7, 8-15, 16-31, 32-63, 64 tokens of
  suffix match, greedy or sampled) and draft-index bucket (1, 2, 3-4, ...,
  129-255) the engine keeps decayed counts of drafts at risk and first
  rejections. The hazard of a bucket is (events + previous bucket's hazard) /
  (risk + 1), starting from a prior of 0.2, so an unseen long bucket inherits
  the shorter one's hazard and one long accept lets frames grow. Each request
  keeps its own table, blended with the engine's at weight 8. The expected
  tokens of d drafts are 1 plus the sum of survival products.
- Cost: the wall time of every submission that ran alone (one lane, nothing
  else in flight) feeds a table of frame cost per row bucket and context
  bucket (verify frames, and clean single-lane prefill waves as a prior), and
  the chain's per-token and single-step cost per context bucket. Missing cells
  interpolate in rows and borrow the nearest context bucket.
- Decision: among 1-16, 20, 24, ..., 255 and the feasible maximum, the frame
  with the most expected tokens per nanosecond, only if it beats the chain by
  5%. A class with no evidence gets one probe frame of up to 16 drafts.
- Caps: a waiting prefill caps a frame at two single steps. A lane below the
  highest priority the engine has served (chat-memory background work at 0
  while foreground carries 1) is capped at eight chained tokens, so a
  foreground arrival waits at most that long. Requests at the default priority
  are not capped: the cap held every hold-3 frame near 42 rows (8 x 35 ms).
  Replayed against the measured GLM costs, the uncapped policy picks frames
  near 200 rows.
