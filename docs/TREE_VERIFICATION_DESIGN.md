# Tree verification over the production speculation path

Status: phases P1 and P2 of section 8 are built and gated, and the branches
of P4 are in the engine and the draft server; P3 and P5 are not built. This
page was written as the design on 2026-11-09, for the code that serves: the
batch engine's lookup and model drafts, the stage adapter's verify frame, the
model `keep_rows` hook and the attention kernels.
[`SPECULATION_UNIFIED_DESIGN.md`](SPECULATION_UNIFIED_DESIGN.md) describes the
subsystem as built. The contract for chain frames in
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
  and 4096 nodes), and `SparkDraftBridgeReply` already receives them;
  the engine cuts at the first node whose parent is not the previous row
  (`SparkModelBatchPlanModelDraft`).

A second draft model is a third source behind these two. Drafters
approximate the same target, so their errors correlate; the agreement matrix
of section 8, phase P0, decides whether a second model adds branches the
first two do not. A node's `source_bit` already names its source, so
acceptance is attributed per source without new fields.

## 2. The tree frame

A tree frame is one verify submission for one lane. It extends the chain
frame of `SPECULATION_LONG_FRAMES.md` as follows.

- **Node table.** Row 0 is the anchor: the last committed token at position
  p. Every other row is a node with `parent_row` < row, a token id and a
  source bit; its depth is its parent's plus one. The submission carries
  `row_parent_rows` and `row_slots` beside the tokens and positions
  (`SPARK_MODEL_SERVING_SUBMISSION_FLAG_TREE`, at most 256 rows), and
  `SparkSpeculationTreeLayoutBuild`
  (`include/sparkpipe/spark_speculation_tree_frame.h`) derives everything
  else from the parents and the tokens. The engine builds the table in
  request state next to `draft_tokens`.
- **Trunk and branches.** The trunk is the chain of first children, the
  path the engine would have verified as a chain. Every other child of a
  trunk node starts a branch, and a branch never forks again: a tree is a
  trunk with single-level branches, at most eight. Siblings carry distinct
  tokens. The rows list the trunk first and then each branch's rows
  consecutively, so a recurrent state replayed row by row is the parent's
  state exactly when a row does not start a branch.
- **Semantic position.** A node's position is p + depth. Rotary embedding,
  the sampling noise key, the attention window and the stop and budget
  rules all use it. Siblings share a position.
- **Storage slot.** A trunk node's KV is written at slot p + depth, as a
  chain's is. A node of branch b is written at slot p + depth + b × stride,
  where stride is max_depth + 1 rounded up to the adapter's
  `tree_slot_grain`: on a position-sharded KV cache that grain is the shard
  grain times the degree, so a branch slot has the same owner rank as the
  semantic key it stands in for. Today position and slot are the same
  number; the frame splits them into two per-row arrays. A chain has
  slot = position, so chain frames are unchanged.
- **Lane context.** The lane's context is the highest slot plus one, which
  the layout reports; the validator refuses a frame whose
  `context_token_count` differs from it. Attention does not use it as the
  key range for tree rows (section 3).
- **Limits.** The row budget is the frame cost model's, as for chains
  (`SparkSpeculationFrameChoose`), within `max_verify_row_count`,
  `SPARK_SPECULATION_TREE_FRAME_MAX_ROWS` (256) and the completion's 1024
  tokens. Depth is bounded by `SPARK_MODEL_RESIDENT_DRAFT_MAX_DEPTH` (16) for model
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

**Full attention (GQA).** A node attends to the committed keys and to its
ancestors only, which the slot scheme turns into a per-row key mapping: a
row at depth d reads semantic keys 0 to p + d (or the window's tail of
them, `SparkAttentionFirstKey` on the semantic position), and key q lives
at slot q when q < `fork_position[row]` and at slot q + `slot_offset[row]`
otherwise. A trunk row's fork position is one past its own position, so
its mapping is the identity. The CUDA sharded GQA kernels
(`inference/kernels/gqa_shard.cuh`) take the two per-row arrays and walk
the committed range and then the branch range through one online-softmax
state; the tiled kernel sweeps the local key range up to the last branch
slot and admits a key to a row when it lies in the row's committed range or
in its branch range. The Metal decode kernel (`attention.metal`) maps each
key the same way and stores at `slots[row]`. Rotary embedding uses the
semantic position in both. The MLA and DSA attention paths (K3, DeepSeek
V4.1 Flash, GLM, A.X K2, Ling 3.0) do not take the mapping yet and stay
chain-only.

**Sparse selection (DSA, the compressor and indexer of DeepSeek V4.1 Flash,
GLM 5.3 Full and A.X K2).** The indexer scores a row against candidate
blocks and the row's own recent keys. A node's candidate set is the
committed blocks plus its ancestors, never a sibling, so the selection runs
per path: the indexer post and compressor read the ancestor slots, as the
DSV4 scratch-KV work did row by row
(`docs/archive/DSPARK_SCRATCH_KV_HANDOFF.md`, "per-row causal attention").
A pool whose last member is a rejected node is recomputed at commit, as the
plan already states for chains.

**Recurrent layers (KDA in Solar Open 2, gated delta in Qwen3.8).** A chain
verify runs with `commit` off and saves each row's convolution input, decay
and write gate; `keep_rows` then replays the kept rows into the state. For a
tree, the causal convolution builds a node's taps from its ancestors'
inputs (`parent_rows`), falling back to the committed window beyond the
anchor, and reads its inputs from the verify slab since the chain form
runs in place. The delta rule scans rows in order as before; at a branch's
first row it loads the state its parent left, which the parent's row saved
into a per-branch snapshot (`state_loads`, `state_saves`, eight snapshots
per layer), so the arithmetic of every row is the arithmetic of its own
chain. After acceptance `keep_rows` receives the accepted path's rows in
`sequence_row_indices` with `sequence_row_begin` delimiting them, rather
than a prefix count, and the fold is the same gather and replay over those
rows; it also moves the kept branch slots to the trunk positions p + k in
every KV layer (`LmKvShardPathMoveKernel`, `spark_kv_move_paged`), so the
lane continues as a chain.

**Taps.** The VERIFY tap frame sends the kept rows' hidden states to draftd.
It sends the accepted path's rows, in path order, under the same flags.

Row equality (G-ROWEQ) holds per history: a node row must equal the
one-row decode of its own path, not of the trunk. The pin for a tree frame
is therefore: every root path of the tree, verified alone as a chain frame,
produces the same token at every row as the tree frame does. Trunk rows
are bitwise the chain's rows everywhere. Branch rows are bitwise their
chain's rows on Metal and on the scalar CUDA kernel, whose key loops run in
semantic order; the tiled CUDA kernel tiles the local slot range, so a
branch row's keys fall into other tiles than its chain's and the attention
sum differs by rounding order, as a row inside a different batch would.
`tests/test_ling_verify_frames.cu` and `tests/cuda/gqa_shard_cuda.cu` pin
both statements; the family gate corpora pin the tokens.

## 4. Acceptance and commit

- **Resolution.** The kept path is the longest root path on which every
  node's token equals its parent's output; the bonus token is the output at
  the last accepted node. `SparkSpeculationTreeResolvePath`
  (`src/spark_speculation_tree_frame.c`) is that function on the host; the
  stage serving adapter calls it for a tree frame where
  `StageServingVerifyKept` applies the chain rule, with the node table
  carried in the submission. A device kernel that resolves on the GPU, for
  one-graph rounds, is a later optimisation, not a prerequisite.
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
`SparkModelBatchPlanDraft` and `SparkModelBatchPlanBranches`.

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

The adapter capability bit `TREE_VERIFY`, next to `SPECULATIVE_VERIFY`,
says that the model's verify frame honours per-row slots, the key mapping
and a path-form `keep_rows`; it comes with the descriptor's
`tree_slot_grain`, which the stage serving adapter checks against the
runner's KV shard at open. An adapter without the bit receives only chain
frames, as today. The stage runner refuses a tree frame on a model whose
geometry lacks `tree_verify` with `SPARK_STATUS_UNSUPPORTED`, so a
misconfiguration fails loudly rather than verifying siblings against each
other. `docs/INTERFACE_CONTRACTS.md` has the submission contract.

## 7. Per-family work

| Family | Verify | Tree |
| --- | --- | --- |
| Solar Open 2 (CUDA, TP16) | chain | tree: sharded GQA key mapping, KDA parent taps and branch snapshots, path keep with slot relocation; `TREE_VERIFY`, `tree_slot_grain` 16. |
| Qwen3.8-27B (Metal) | chain in `tools/metal_stage_run.c` | tree: `attention.metal` key mapping, conv parent taps, delta snapshots, `SparkDeviceStageRunnerKeepPath` with `spark_kv_move_paged`; `--tree-branches` in the tool. |
| Ling 3.0 (CUDA) | chain | chain only: its latent attention has no key mapping; the stage refuses a tree step. |
| Kimi K3, DeepSeek V4.1 Flash, GLM 5.3, A.X K2 | chain | chain only: MLA and DSA selection would need per-path key mapping. |
| K-EXAONE 2.0, Gemma 4, MiMo (CUDA) | chain | the shared GQA kernels take the mapping; the stages lack `tree_verify`, the relocation and the capability until wired. |
| Qwen3.8 Flash (Metal, n-gram embedding) | none | refused: the n-gram hash assumes the previous row is the predecessor. |

The common GQA kernels and the generic device stage cover the GQA families
at once; the MLA and DSA families each need their own attention change.
Shared code stays model-neutral: the layout, slot and position arrays, the
key mapping and the resolver carry no family names.

## 8. Phases and gates

Each phase is a pull request with its gate; later phases do not start on a
failed gate.

- **P0, agreement matrix.** Replay recorded K3 and DSV4.1 Flash traces
  through lookup and the model drafter together and record, per position,
  whether each source's token equals the target's and whether the two agree.
  The output is the fraction of positions where exactly one source is right.
  That fraction bounds the tree's gain over the better chain. The
  implementation went ahead without it (2026-11-09); the frame cost model
  only adds a branch whose expected tokens pay for its rows
  (`SPARK_MODEL_BATCH_TREE_MIN_GAIN`), so a corpus where the sources
  always agree keeps serving chains. The measurement remains worth taking
  from the speculation receipts.
- **P1, engine and mock.** Done: node table in request state, composition
  from lookup alternates and the model chain, slot and position arrays in
  the submission, the host resolver, path-form completion checking and
  hazards per node. The mock adapter verifies trees;
  `tests/test_model_batch_engine_mock.c` and
  `tests/test_speculation_tree_frame.c` are the gates.
- **P2, generic device stage and Metal.** Done for Solar Open 2 and the
  Metal Qwen3.8-27B tool: slot and position split, the key mapping in the
  GQA kernels, parent taps and branch snapshots in the recurrent kernels,
  path-form `keep_rows` with slot relocation, the `TREE_VERIFY` capability.
  Gates, all run on 2026-11-09: `tests/test_ling_verify_frames.cu` and
  `tests/cuda/gqa_shard_cuda.cu` on a Spark; `tests/test_metal_kernels.c`
  (tree attention, conv, delta and the KV move bit for bit) and
  `metal_stage_run --mode spec` on M5U0, token-identical on all six
  Qwen3.8-27B prompts with `--tree-branches 2` (korean accepted 14 tokens
  off the trunk) and with `--tree-branches 0`. The tool adds every
  available alternate, so on the rewrite prompt, where chains already
  accept most drafts, trees cost rows without gain (155 against 182 tok/s);
  the engine adds a branch only when the frame cost model says it pays.
- **P3, K3.** MLA ancestor keys and the KDA tree scan. Gate: the same path
  pin on K3 at TP16, and B1 and B2 tokens per second on the speculation
  receipts' workloads against the chain, in one hold. Serve only if the tree
  wins on the ledger corpora; the frame cost model falls back to chains
  where it does not.
- **P4, drafter alternates.** The lookup index returns ranked matches
  (`SparkSpeculationLookupIndexFindRanked`) and the engine verifies their
  diverging suffixes as branches; draftd's latent drafter returns the
  selector's runner-up at every step and emits them as one-node branches
  of the DFT3 tree within the same wait budget. Gate: acceptance per source
  bit on the agreement-matrix traces, and no increase in draft wait.
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
- Device-side resolution and one-graph rounds. Worth it
  once a tree frame serves and the host round trip is measured on the step
  trace.
