# Tree verification over the production speculation path

Status: design, 2026-11-09. Nothing in this document is built. It turns the
"Direction: multi-drafter trees" section of
[`SPECULATION_UNIFIED_DESIGN.md`](SPECULATION_UNIFIED_DESIGN.md) and sections
2.3 to 2.4 of [`SPECULATION_PLAN.md`](SPECULATION_PLAN.md) into a wiring plan
for the code that serves today: the batch engine's lookup and model drafts,
the stage adapter's verify frame, the model `keep_rows` hook and the attention
kernels. The contract for chain frames in
[`SPECULATION_LONG_FRAMES.md`](SPECULATION_LONG_FRAMES.md) stays in force; a
chain is a tree with one child per node, and every rule below reduces to that
document when it is.

## 1. Why a tree, and why from the sources we have

A verify frame today is a chain: the anchor row and one row per draft at
positions p+1 to p+d. Only one continuation is checked per frame, so when the
lookup drafter and the model drafter disagree at a position, the engine picks
one (`SparkModelBatchPlanDraft` prefers lookup above an 8-token match,
otherwise the model) and the other's tokens are never verified. At batch 1
and 2 verify rows are nearly free: dense weights are read once per frame
whatever the row count, and K3 already verifies up to 255 lookup rows in one
frame. A tree spends those rows on alternatives instead of depth.

The gain is at the first divergence. With one chain the expected accepted
tokens are the sum over depth of the path survival; with a tree it is the sum
over every node of its path survival, so a second candidate at depth 1 with
survival s₂ adds s₂ times its own expected continuation, at the cost of its
rows. Published tree verifiers (SpecInfer, EAGLE-2) gain half a token to one
token per frame over the best chain at equal row budgets; the gain model in
the unified design gives the same order. That is the target to measure
against, not a multiple.

The two sources that pay first cost nothing new:

- **Lookup alternates.** `SparkSpeculationLookupIndexFind` already scores up
  to 64 candidate matches and returns the longest. The second-longest match
  that diverges from the first is a branch.
- **The model drafter's own alternates.** DFlash2's selector keeps a top-16
  candidate set per position and the keyed-Gumbel coupling in
  `tools/spec_offline/block_drafter.py` scores 64 candidates; today
  `tools/block_draftd.py` emits only the argmax chain. The DFT3 protocol
  already carries trees (`SparkDraftBridgeNode.parent_index`, up to 32 deep
  and 4096 nodes), and `SparkDraftBridgeProposeTree` already receives them;
  the engine cuts at the first node whose parent is not the previous row
  (`SparkModelBatchPlanModelDraft`).

A second draft model is a third source behind these two. Drafters
approximate the same target, so their errors correlate; the agreement matrix
of section 8, phase P0, decides whether a second model adds branches the
first two do not. The seam's source mask (`SPARK_SPECULATION_SEAM_SOURCE_*`
in `include/sparkpipe/spark_speculation_seam.h`) and the node's `source_bit`
already name sources, so acceptance is attributed per source without new
fields.

## 2. The tree frame

A tree frame is one verify submission for one lane. It extends the chain
frame of `SPECULATION_LONG_FRAMES.md` as follows.

- **Node table.** Row 0 is the anchor: the last committed token at position
  p. Every other row is a node with `parent_row` < row, `depth` =
  parent's depth + 1, a token id and a source bit. Rows are in depth order,
  so a node's parent is always an earlier row. `SparkSpeculationTreeNode`
  (`include/sparkpipe/spark_speculation_tree.h`) already has this shape:
  `parent_row`, `depth`, `candidate_index`, `child_row_base`, `child_count`,
  all 8-bit, so a plan holds up to 255 rows; `SparkSpeculationTreePlan`
  caps at 64. The engine builds a plan in request state next to
  `draft_tokens`.
- **Semantic position.** A node's position is p + depth. Rotary embedding,
  the sampling noise key, the attention window and the stop and budget
  rules all use it. Siblings share a position.
- **Storage slot.** A node's KV is written at slot p + row, unique per row,
  so siblings never overwrite each other. Today position and slot are the
  same number (`scratch_row_positions[row] = position + wave`, and the
  kernels store at `position % page_slots`); the frame splits them into two
  per-row arrays. A chain has slot = position, so chain frames are unchanged.
- **Lane context.** The lane's context is p + rows, as the page allocation
  for a 256-row chain frame already is. Attention does not use it as the key
  range for tree rows (section 3).
- **Limits.** The row budget is the frame cost model's, as for chains
  (`SparkSpeculationFrameChoose`), within `max_verify_row_count`,
  `SPARK_SPECULATION_TREE_MAX_PLAN_ROWS` and the completion's 1024 tokens.
  Depth is bounded by `SPARK_MODEL_RESIDENT_DRAFT_MAX_DEPTH` (8) for model
  nodes and by the lookup match for lookup nodes, as today.
- **Block ends.** Chain drafts stop two tokens before a recurrent checkpoint
  and the lane carries one block identity per block end in (p, p+d+1],
  digested from the chain's tokens. A tree would need one identity per path.
  Rule: only the trunk (the path the engine would have verified as a chain)
  may cross a block end; every other branch stops before the next block end.
  The trunk's identities are computed as today; an accepted off-trunk path
  never reaches a block end inside the frame, so sealing is unchanged. The
  same rule applies to the recurrent checkpoint.
- **Stops and budget.** A node after an engine stop token, past the output
  budget or past the context limit is not added, per path, as for chains.
- **Sampled lanes.** Each row is sampled with the lane's rule at its semantic
  position with noise keyed by (seed, position, token), as today. Two
  siblings therefore draw against the same noise, so at most one of them can
  equal the parent's output, and a sampled tree never accepts two children
  of one node.

## 3. Verify on the device

Everything that treats a row as a row is unchanged: embedding, norms, dense
and expert projections, routing, the head and its argmax all-reduce. Four
things read history and change.

**Full attention (GQA, MLA).** A node attends to the committed keys and to
its ancestors only. The committed keys are slots 0 to p, or the window's
tail of them (`SparkAttentionFirstKey` on the semantic position). The
ancestors are a per-row list of slots, at most depth−1 entries. The CUDA
decode row kernel already takes a per-row selected-position list
(`LmGqaAttentionDecodeRow`'s `selected_positions` and `selected_count`),
used for sparse selection; a tree row runs the committed range as today
followed by its ancestor list as selected positions, with one online-softmax
state across both. The Metal decode kernel reads `positions[row]` both as the
store slot and as the key range (`attention.metal`, the key length is
`positions[row] + 1`); it gets a `slots` buffer and an ancestor list the same
way. Rotary embedding uses the semantic position in both.

**Sparse selection (DSA, the compressor and indexer of DeepSeek V4.1 Flash,
GLM 5.3 Full and A.X K2).** The indexer scores a row against candidate
blocks and the row's own recent keys. A node's candidate set is the
committed blocks plus its ancestors, never a sibling, so the selection runs
per path: the indexer post and compressor read the ancestor slots, as the
DSV4 scratch-KV work did row by row
(`docs/archive/DSPARK_SCRATCH_KV_HANDOFF.md`, "per-row causal attention").
A pool whose last member is a rejected node is recomputed at commit, as the
plan already states for chains.

**Recurrent layers (KDA in K3 and Solar Open 2, gated delta in the Ling and
Qwen3.8 stages).** A chain verify runs with `commit` off and saves each row's
convolution input, decay and write gate; `keep_rows` then replays the kept
rows into the state. For a tree, a node's recurrent read is its parent's
state, not the previous row's: the per-sequence scan takes `parent_row`
instead of row−1, and the causal convolution window of a node is its
ancestors' inputs. Both kernels already iterate rows in order within a
sequence, so the change is the index they read, not the arithmetic. After
acceptance `keep_rows` receives the accepted path's row list rather than a
prefix count, and the fold is the same gather and replay over those rows.

**Taps.** The VERIFY tap frame sends the kept rows' hidden states to draftd.
It sends the accepted path's rows, in path order, under the same flags.

Row equality (G-ROWEQ) holds per history: a node row must equal the
one-row decode of its own path, not of the trunk. The pin for a tree frame
is therefore: every root path of the tree, verified alone as a chain frame,
produces the same token at every row as the tree frame does. That pin runs
on the mock adapter and on every family's gate corpus before a tree frame
serves.

## 4. Acceptance and commit

- **Resolution.** The kept path is the longest root path on which every
  node's token equals its parent's output; the bonus token is the output at
  the last accepted node. `SparkSpeculationPolicyResolveVerifierTree`
  (`src/spark_speculation_policy.c`) is that function on the host and
  `SparkSpeculationTreePlanResolve` is its plan-form twin; the stage adapter
  calls the host rule where `StageServingVerifyKept` calls the chain rule
  today, with the node table carried in the submission. A device kernel that
  resolves on the GPU, as section 2.4 of the plan wants for one-graph rounds,
  is a later optimisation, not a prerequisite.
- **Completion.** The lane's completion holds the accepted path's tokens
  followed by the bonus token and then `SPARK_MODEL_SERVING_NO_TOKEN`, so
  `SparkModelServingCompletionKeptRows` and the pipeline client are
  unchanged. The engine re-derives the path from its own node table and the
  returned tokens and fails the request with VALIDATION_FAILED on any
  mismatch, as `SparkModelBatchAcceptVerifiedLane` does for chains.
- **KV compaction.** The accepted nodes sit at slots p + row, not at
  positions p+1 to p+kept−1. After resolution the node gathers those rows'
  KV (and, for sparse families, index-pool entries) into the positions, per
  layer, before the lane resumes at p + kept. The DSV4 scratch-KV work did
  this copy ("copy rows 1..accepted from scratch to main per layer"). Its
  cost is kept rows times layers times the slot bytes, under a megabyte per
  frame for every served family. Rejected slots are overwritten by later
  frames, as rejected chain rows are today. A chain frame, whose slots equal
  its positions, skips the gather.
- **Hazards and calibration.** Acceptance is observed per source bit along
  the accepted path, so the lookup class hazards, the model depth
  calibration and the model-extension counters keep their meaning; a branch
  that was not on the accepted path counts as rejected at its first node.
- **Publication.** The engine publishes floor((p + kept) / B) × B as today;
  the block-end rule of section 2 guarantees only trunk identities are
  needed.

## 5. Composition in the batch engine

Composition lives where the two sources already meet,
`SparkModelBatchPlanDraft` and `SparkModelBatchExtendModelDraft`, not in the
unused seam and policy engine. The policy engine's resolver is reused as the
host twin; the seam stays out of the production path until the cleanup in
the unified design decides its fate.

1. Collect candidates: the lookup's ranked matches, the model's chain and,
   once draftd emits them, the model's alternates as a DFT3 tree.
2. Build the trunk: the chain the engine would verify today, priced by
   `SparkModelBatchPlanFrameCost` as now.
3. Add branches by marginal value: for each candidate node not on the tree,
   the value is its path survival (the product of the per-edge survivals the
   hazard tables already provide per class and depth bucket, calibrated for
   model nodes as `draft_path` is now) times one plus its expected
   continuation, divided by the frame cost increase of its rows. Add nodes
   while the frame's expected tokens per nanosecond rises, under the row
   budget and the block-end rule.
4. Refuse the tree and fall back to the trunk when the adapter lacks the
   tree capability (section 6), when more than one lane is active, or when
   the lane wants logprobs, mirroring the chain rules.

The cost model needs one addition: expected tokens of a tree are the sum of
node path survivals plus one, which the hazard tables already support, since
survival is per depth bucket and class rather than per row index.

## 6. Capability and refusal

A new adapter capability bit, `TREE_VERIFY`, next to `SPECULATIVE_VERIFY`,
says that the model's verify frame honours per-row slots, ancestor lists and
a path-form `keep_rows`. An adapter without it receives only chain frames,
as today. The stage runner refuses a tree frame on a model that lacks it
with the same named status it uses for unsupported verify, so a
misconfiguration fails loudly rather than verifying siblings against each
other.

## 7. Per-family work

| Family | Verify today | Tree work |
| --- | --- | --- |
| Kimi K3 | chain, lookup + DFlash2 | MLA decode: ancestor slots as selected keys. KDA: parent-indexed scan and conv window; path-form `keep_rows` over the existing replay buffers. |
| DeepSeek V4.1 Flash | chain, lookup + DSpark | MLA and compressor/indexer per path over ancestor slots; the scratch-KV precedent already splits anchor from draft rows. |
| GLM 5.3 Full, A.X K2 | chain, lookup only | DSA indexer per path; MLA as K3. A.X K2's DSpark drafter is a second model source once wired. |
| K-EXAONE 2.0 | chain, lookup only | GQA with a 128-token window on semantic positions; no recurrent state. Its DSpark drafter likewise. |
| Solar Open 2, Ling, Qwen3.8 stages | chain (Solar), none (others) | Generic device stage: GQA ancestor keys; gated-delta and KDA parent-indexed scan; `SparkDeviceGatedDeltaKeep` takes a row list. Families without verify need chain verify first. |
| Metal (Qwen3.8-27B, Gemma, Ling) | chain in `tools/metal_stage_run.c` | `attention.metal`: separate `slots` from `positions`, ancestor list; `SparkDeviceStageRunnerKeepRows` by path. |

The common GQA decode kernel and the generic device stage cover most
families at once; K3 and the DSA families each need their own attention
change. Shared code stays model-neutral: the node table, slot and position
arrays, ancestor lists and the resolver carry no family names.

## 8. Phases and gates

Each phase is a pull request with its gate; later phases do not start on a
failed gate.

- **P0, agreement matrix.** Replay recorded K3 and DSV4.1 Flash traces
  through lookup and the model drafter together and record, per position,
  whether each source's token equals the target's and whether the two agree.
  The output is the fraction of positions where exactly one source is right.
  That fraction bounds the tree's gain over the better chain; if it is below
  a few percent on the ledger corpora, stop here. Tooling exists in
  `tools/spec_offline/` (acceptance against recorded taps).
- **P1, engine and mock.** Node table in request state, composition from
  lookup alternates and the model chain, slot and position arrays in the
  submission, the host resolver, path-form completion checking and hazards
  per source. The mock adapter verifies trees; gates are the mock engine
  tests and the pin that a tree frame resolves identically to the same tree
  resolved by `SparkSpeculationPolicyResolveVerifierTree`.
- **P2, generic device stage and Metal.** Slot and position split, ancestor
  keys in the GQA decode kernels, parent-indexed recurrent scan, path-form
  `keep_rows`, the `TREE_VERIFY` capability. Gate: on Solar Open 2 and the
  Metal Qwen3.8-27B tool, every root path of a tree frame matches its chain
  frame token for token on the gate corpus, and a chain frame is bitwise
  unchanged.
- **P3, K3.** MLA ancestor keys and the KDA tree scan. Gate: the same path
  pin on K3 at TP16, and B1 and B2 tokens per second on the speculation
  receipts' workloads against the chain, in one hold. Serve only if the tree
  wins on the ledger corpora; the frame cost model falls back to chains
  where it does not.
- **P4, drafter alternates.** draftd returns the selector's alternates as a
  DFT3 tree within the same wait budget; the lookup index returns its
  second match. Gate: acceptance per source bit on the agreement-matrix
  traces, and no increase in draft wait.
- **P5, sparse families.** DSV4.1 Flash, GLM 5.3 Full and A.X K2 with
  per-path selection; A.X K2 and K-EXAONE DSpark drafters as model sources
  once their chain verify serves.

## 9. Out of scope

- Drafter training. On-policy distillation of a drafter (SpecOPD, Draft-OPD)
  improves a source's acceptance and would plug in as a better checkpoint on
  the hub; it changes nothing here.
- A candidate-restricted head. Verify cost is per row through every layer
  and does not depend on the vocabulary; restricting the head to candidate
  tokens saves a few percent at most and needs a full-vocabulary bonus row
  anyway.
- Device-side resolution and one-graph rounds (plan section 2.4). Worth it
  once a tree frame serves and the host round trip is measured on the step
  trace.
