# Speculation subsystem

The one authority for how speculation is built and where it is going.
[README.md](../README.md#speculative-decoding) states the end state;
[TECHDEBT.md](../TECHDEBT.md#speculation) lists the open gaps. Code facts
were checked on 2026-11-11. Superseded designs are in `docs/archive/`
(`SPECULATION_PLAN.md`, `SPECULATION_TAPS.md`,
`SPECULATION_PROVIDER_DESIGN.md`, `SPECULATION_TREE_COMPOSITION_DESIGN.md`,
`SPECULATION_SUBSYSTEM_BOUNDARY.md`).

## Status

- Every model in the README model table verifies drafts taken from the
  request's own context. Its serving adapter declares
  `SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_SPECULATIVE_VERIFY`.
- Four lanes also ask a draft model on the hub: Kimi K3, Kimi K3.1 Pro,
  Qwen3.8-27B and DeepSeek V4.1 Flash (`draft_server` in
  `deployment/lanes.json`).
- Tree frames serve on the models whose adapter declares `TREE_VERIFY`
  ([TREE_VERIFICATION_DESIGN.md](TREE_VERIFICATION_DESIGN.md) has the list).
  Every other model verifies chains.
- Speculative output equals output without speculation, token for token.
  The release gate checks it on every lane
  ([Guarantee](#guarantee-and-its-limit)).

## Architecture as built

Speculation has two halves and one wire between them. The batch engine on
the API host decides what to draft and how many rows a frame carries. The
model's stage on the ranks computes the frame and keeps the rows the engine
accepted. Drafts from a draft model arrive over the DFT3 draft bridge. No
serving adapter holds speculation policy, and no family has a speculation
switch of its own.

### Batch engine: every draft is planned here

`runtime/model_batch_engine.c`. When a decode submission is built,
`SparkModelBatchPlanDrafts` plans the drafts of every lane in it. One lane
goes through `SparkModelBatchPlanDraft`; two to four lanes share one verify
frame through `SparkModelBatchPlanBatchDrafts`
(`SPARK_MODEL_BATCH_VERIFY_LANES`). A submission with more lanes, or an
adapter without the capability, gets a plain decode step.

A draft has two sources.

- **Lookup.** Each request keeps an index of its own tokens
  (`SparkSpeculationLookupIndex*` in `src/spark_speculation_lookup_draft.c`,
  3-token grams in 4,096 buckets). At the anchor it finds the most recent
  longest suffix match among up to 64 candidates, of up to 64 tokens, and
  drafts the tokens that followed it. A draft may run to 255 rows. The
  ranked form returns the next-best matches that diverge from the first,
  which become tree branches.
- **Draft model.** With a `draft_server`, and when the lookup match is
  shorter than 8 tokens, `SparkModelBatchPlanModelDraft` asks the draft
  model for up to `depth` tokens (at most 16). At most two lanes are
  drafted this way at once (`SPARK_MODEL_BATCH_MODEL_DRAFT_LANES`).

The ask never blocks the engine. `SparkDraftBridgeRequestTree` sends the
request and returns. A decode submission that still lacks its answer is
taken back and decode is held for that progress pass, while prefill and the
other work kinds dispatch. The engine adds the bridge's socket and time
limit to the descriptors its caller waits on, and collects the answer on the
next pass. One ask is outstanding at a time. Answers are kept for two
requests, keyed by slot, request, generation and position, so a lane whose
answer arrived while another lane was served still uses it. An ask that gets
no answer in time is abandoned and the lane decodes plainly.

### Frame choice: expected tokens per nanosecond

`src/spark_speculation_frame.c`. `SparkSpeculationFrameChoose` picks the
draft depth that maximizes expected tokens per nanosecond.

- Survival comes from hazard tables per match class and depth bucket, kept
  for the engine and for the request and decayed over frames. Lookup drafts
  are classed by match length; draft-model nodes use the drafter's
  confidence, calibrated per depth by what earlier frames observed against
  what they predicted.
- Cost comes from the frame times the engine measured, per row bucket and
  context bucket, for single frames and for shared batch steps.
- A lane whose speculative token rate falls under 0.97 of its plain rate
  stops asking the draft model for 64 tokens, doubling to 1,024 while it
  keeps losing. A failed or abandoned ask waits 8 tokens, doubling the same
  way.

### Tree frames

`src/spark_speculation_tree_frame.c` builds the frame layout
(`SparkSpeculationTreeLayoutBuild`) and resolves the accepted path
(`SparkSpeculationTreeResolvePath`). `SparkModelBatchPlanBranches` adds up
to three branches to the trunk, from the lookup's ranked alternates and the
draft model's runner-up tokens. It keeps a branch only when the branch raises the
frame's expected tokens per nanosecond and adds at least 0.05 expected
tokens (`SPARK_MODEL_BATCH_TREE_MIN_GAIN`). A tree is refused, and the trunk served
as a chain, when the adapter lacks `TREE_VERIFY`, when more than one lane is
active or when the lane wants logprobs. The frame contract, the key mapping
and the per-family state are in
[TREE_VERIFICATION_DESIGN.md](TREE_VERIFICATION_DESIGN.md).

### Model side: verify and keep

A verify frame is one submission: the anchor row and one row per draft
token. The stage computes every row in one pass, and the completion carries
the target's token for each row. The engine accepts the longest path whose
drafts equal the target's tokens and commits it plus the target's next
token. The stage then keeps the accepted rows (`keep_rows`) and folds its
recurrent and paged state to the last kept row. The chain frame contract is
[SPECULATION_LONG_FRAMES.md](SPECULATION_LONG_FRAMES.md); the submission
fields are in [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md).

A model supplies the verify frame in its stage and the fold of its own
state. It supplies no policy.

### Draft bridge: DFT3 client

`include/sparkpipe/spark_draft_bridge.h`, `ring/transport/draft_bridge.c`.
TCP with `TCP_NODELAY`, magic `DFT3`. `SparkDraftBridgeRequestTree` sends
the committed tokens, optional tap rows, a source mask, a depth and a time
budget, and returns at once. `SparkDraftBridgeReply` answers `PENDING` until
the reply is whole and then returns scored nodes (`token_id`,
`parent_index`, `depth`, `source_bit`, `score`); a tree may be 32 deep and
4,096 nodes. One exchange is open at a time. `SparkDraftBridgePoll` gives
the socket, the events and the time limit a caller waits for, and
`SparkDraftBridgeCancel` abandons the exchange.

### Draft server

`tools/block_draftd.py` serves DFT3 on the hub, one process per drafted
lane, started by `tools/hub_draftd.sh`. It serves one connection and one
request at a time. The host, its drafters and their store are in
[RTX5090_SPECULATION_NODE.md](RTX5090_SPECULATION_NODE.md).

`tools/draft_replay_server.py` is a DFT3 server for gates. It drafts the
continuation of any token stream an earlier request showed it, and can
delay, corrupt or withhold replies.

### Taps

A drafter that reads the target's hidden rows gets them from one rank. The
stage config's `taps` object names the rows (`entries`, a reduction and its
layers, or `final`), the tap rank and where they go: a TCP `stream` to the
drafter, or a `file`. The tap rank sends an `SPT2` hello with the model tag,
the pack identity, the row shape and a fingerprint of the tap set, then one
frame for each prefill wave, decode chain and verify frame with its request,
sequence, first position and rows (`include/sparkpipe/spark_tap_stream.h`,
`src/spark_tap_stream.c`). A verify frame sends only the rows that were
kept. A stream that is down is retried and serving goes on without it.
`tools/spec_offline/tapstream.py` reads the format.

## Control

| Where | Setting | Effect |
| --- | --- | --- |
| Lane record, `draft_server` | `host`, `port`, `target_model`, `depth` (1-16), `wait_ms` (1-1000), `min_confidence` | The engine asks this draft model. Without the object the lane drafts from lookup only. |
| Stage config, `taps` | `entries`, `stream` or `file`, `rank`, `window` | One rank sends hidden rows to the drafter or to a file. |
| Request, `"speculation": false` | | The request decodes plainly. |
| Adapter descriptor | `SPECULATIVE_VERIFY`, `TREE_VERIFY`, `tree_slot_grain` | What frames the model verifies. |

## Counters

The API's status and the engine's `batch_speculation` log line report
frames, drafted and kept tokens, tree frames and kept branches, shared batch
frames, and for the draft model its rounds, declines, unavailable and
abandoned asks, backoffs and mean wait. Each frame's `batch_step` line
carries its rows, drafts, kept rows, match class, expected tokens, predicted
and measured time and draft wait. `tools/release_capture.py` records the
counters per prompt next to the tokens.

## Guarantee and its limit

Every emitted token is the target model's own token from the verify frame.
Every verify-capable stage reproduces one-row decode bit for bit
(`tests/test_gemma4_verify_frames.cu`, `tests/test_ling_verify_frames.cu`,
`tests/test_glm52_verify_frames.cu`, `tests/test_hy4_verify_frames.cu`,
`tests/test_qwen38_verify_frames.cu`, `tests/test_mimo26_verify_frames.cu`
and the K3 and DeepSeek V4.1 Flash tests), so a speculative run emits the
tokens of a run without speculation.

The release gate holds each lane to that: `tools/release_capture.py`
captures every prompt with and without speculation and
`tools/release_compare.py` requires the tokens to be equal. A lane with a
draft model can also be captured against the replay drafter, once knowing
nothing and once replaying the stream it was shown.

The limit is acceptance, not output. A drafted token is kept only when it
equals the token the target emits at that row, so sampled output is
unchanged too, but fewer drafts survive as temperature rises.

## Direction

The deliverable is tokens per second and latency. Verify rows are nearly
free at small batch, since dense weights are read once per frame, so
several candidate continuations in one frame cost little.

Built: one tree from the lookup's alternates, the draft model's chain and
its runner-up tokens (`--alternates` in `tools/block_draftd.py`, one-node
branches), composed by expected gain.

Not built:

- A second draft model on one lane. Drafters approximate the same target,
  so their errors correlate; the agreement between two drafters on recorded
  streams decides whether the second adds branches the first lacks.
- Drafting one round ahead, so the round trip hides behind the frame in
  flight.
- Trees on the MLA and DSA families, which need per-path key mapping.

## Removed on 2026-11-11

An earlier design put speculation in each serving adapter: a seam with a
source mask per family (`SPARK_<FAMILY>_SPECULATORS`), a policy engine for
acceptance, and n-gram, recorded, reference and relay draft sources under
it. The per-family adapters that called it went on 2026-10-11, and the batch
engine has planned every draft since. The seam, the policy engine, those
draft sources, the drafter mix, the static tree header and the tools that
replayed them were deleted with their tests; nothing in the serving path
called them.
