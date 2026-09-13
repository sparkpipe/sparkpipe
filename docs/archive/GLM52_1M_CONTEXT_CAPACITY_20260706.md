# 1M Context: Capacity Layer, Corrected Map, and the Two Remaining Projects - 2026-07-06

## Retraction and corrected architecture map

My audit claimed no full-context latent or index-key pool existed. Wrong at
the architecture level: spark_glm52_kv_cache.h defines a physical block
arena (ALLOCATED/RESIDENT flags, host-side block indices, per-lane counts),
the DSA kernels (DsaScoreKernel, DsaSelectRadixTopk, SelectedBlockBuild,
KvFragmentPrefetch/Save) resolve slots through block tables against
kv_block_count/cache_token_capacity, and the service derives its block pool
from a context-token constant. The paged tier exists. What pinned the
system to a 2048-token window and 64k contexts was CONFIGURATION: the
builder's 32-blocks-per-sequence constant, sequence-major cache sizing, a
selected-count-sized key_index_cache, and the 65536 service constant.

That last one was also a latent bug class: DsaScoreKernel addresses
key_index_cache by resolved pool slot, but the builder sized the buffer at
b x SELECTED_TOKEN_COUNT - consistent only while capacity equals the
window. Any capacity raise without this PR's reshape would have been
out-of-bounds.

## What this PR changes (verified mechanical)

One shared capacity source in spark_glm52_kv_cache.h:
SPARK_GLM52_KV_CONTEXT_TOKENS = 1048576 (per-sequence context ceiling) and
SPARK_GLM52_KV_POOL_TOKENS = 4194304 (total pooled KV tokens per stage),
with a preprocessor divisibility check. Consumers derive:

- builder: blocks-per-sequence = CONTEXT/64 = 16384; mla_cache and
  key_index_cache become pool-shaped (POOL_TOKENS x 576 and
  POOL_TOKENS x 128), slot-addressed as the kernels already assume;
  node cache_token_capacity/kv_block_count = pool totals; block tables
  b x 16384. The JIT selected-set buffers (key_nope/value at
  b x 2048 x head-dims) are correct as-is and untouched.
- service backend: CONTEXT_TOKENS aliases the shared constant, which
  scales its derived block pool and prefix bindings.

Per-stage memory at these defaults: mla pool 27.6 GB (4M x 1152 B x 6
layers), index-key pool 6.1 GB, block tables 64 MB at B1024, JIT buffers
~2 GB, request token storage 512 MB host. With ~60 GB FP8 weights: ~96 GB
of 128 GB committed. Capacity: 4 concurrent 1M-context users, or ~62 at
64k, or mixes - the pool is the product knob. B4/B64/B1024 share the same
footprint; batch size costs nothing extra, exactly as requested.

## REQUIRED on spark2 (the growth path is the unverified piece)

1. A real >2048-token decode: prefill ~8k tokens, decode, verify the model
   attends early context (needle retrieval). This exercises the service
   block allocator growing lanes past 32 blocks (lane_capacity handling,
   KvFragmentPrefetch behavior at depth) - the one path I could not
   execute from here.
2. Verify dsa_candidate_count derives from context_length (not a 2048
   constant) at the score/select call sites during that run.
3. Full make test + the existing token-match gates (allocation reshape
   should be bit-neutral at old workloads; the 27.6 GB pool memset at
   startup is one-time).

## Project 1: DSA-sparse prefill (kills the N^2 that matters)

Full-attention chunked prefill is O(N^2) at 512-dim x 64 heads: ~27 PFLOP
at 64k, unusable at 1M. DSA-sparse prefill is O(N^2) only in the INDEXER
(128-dim single-vector dots: ~256 TFLOP at 1M, tens of seconds once) plus
O(N x 2048) attention - two orders of magnitude off the wall.

The trap that forbids the naive kernel: per-query selected attention with
no KV sharing reads 2048 x 1152 B per query = ~14 TB/stage at 1M. Real
implementations exploit selection overlap between adjacent queries. Two
candidate strategies, both reusing the existing score/radix/build kernels:
(a) tile-union - 16-query tiles attend the union of their selections
through the WMMA prefill kernel with a gathered slot list; (b)
chunk-shared selection - one selection per chunk prefix plus the causal
in-chunk window. Both are approximations of per-query DSA; the grounding
prerequisite is the reference semantics (upstream DSA prefill behavior)
and an accuracy gate (long-context needle fixtures) before either ships
to a public product. This is the next kernel project and it is mine.

## Project 2: dspark serving integration (and the MTP question)

Grounded: the serving stack has zero dspark references; topology sidebands
exist only for DSA index sharing. The five tap layers live on five
different ranks, so integration = a new sideband kind exporting tap hidden
states to the final rank (the sideband/transport machinery is the
template), the backend's TapOutputPointers/StageLane wiring in the
adapter, and a Draft -> MTP-accept loop in the serving engine - the MTP
draft/accept/commit buffers already exist in the builder. Order of work:
first validate MTP itself end-to-end (draft acceptance rate > 0 and
committed tokens match greedy decode - the doubt is warranted until
measured), then the tap sideband, then Draft v1 per-lane, then batched
Draft v2. Speculation is accuracy-neutral by construction (rejected
drafts fall back to verifier tokens), which makes it the safest large
multiplier: 2-3x decode on top of everything above.

## Priority order

1. This PR's spark2 verification (the >2048 decode test).
2. DSA-sparse prefill (TTFT is the public-site killer at any long context).
3. MTP validation, then dspark sideband + draft loop.
4. Absorbed attention v2 (64 heads/block, dynamic smem): 4x fewer latent
   bytes in the now-default decode path.
5. Web-side work resumes after 1-3, per the stated priority.


## Addendum: DSA-sparse prefill implementation blueprint (grounded 2026-07-06)

Grounding completed this session. The indexer tensors loaded by the builder
(indexer.wq_b [index_query_dim x query_a_dim], indexer.wk
[index_key_dim=128 x hidden], indexer.weights_proj, indexer.k_norm) are the
published lightning-indexer architecture, and the semantic reference for
prefill selection is the repo's own decode scoring: the prefill kernel must
compute the identical per-head weighted formula per query row that
DsaScoreWarpCandidateKernel (sole score launch site, .cu:13554) computes for
the single decode query. Only one scoring launch exists; the 17xxx-region
DSA functions are per-kernel launcher wrappers; the production prefill
attention is still the full-causal WMMA kernel. The N^2 stands.

The 14 TB objection dissolves under selection locality: adjacent prefill
queries' top-2048 sets overlap heavily, so per-query latent gathers hit L2
and DRAM traffic collapses toward union-once per chunk (working set per
1024-query chunk ~ a few thousand unique rows x 1152 B, well inside L2).
Therefore v1 implements EXACT per-query semantics with the simple kernel -
no tile-union approximation - and spark2 measures whether locality delivers.

Pipeline per prefill chunk of C tokens at positions [p, p+C), reusing the
absorbed decode machinery with row = query:

1. Index keys for the chunk tokens are already written to the pooled
   key_index_cache by the existing prefill-side DsaKeyNormRopeStore path
   (launcher wrapper present; verify the call site writes during prefill -
   decode correctness already requires it).
2. New DsaScorePrefillTileKernel: grid (candidate_span, query_row_in_tile,
   sequence), body = the decode score math with the query pointer offset
   per row and the causal candidate bound = the row's absolute position.
   Writes a [tile_rows x context] score workspace (16 rows x
   SPARK_GLM52_KV_CONTEXT_TOKENS x 4 B = 64 MB builder allocation).
3. DsaSelectRadixTopkKernel reused per row (its launcher at the 17828-area
   wrapper) over the workspace rows, emitting per-row selected indices
   (16 x 2048 x 4 B).
4. Absorbed attention per tile: AbsorbedQueryProjectKernel over the chunk's
   query rows (active_sequence_count = C), then AbsorbedAttentionKernel
   with the tile's rows presented as sequences (per-row sparse indices,
   context_lengths[row] = position), then AbsorbedValueApplyKernel once per
   chunk. Prefill needs its own C-row staging for query_latent/rope/output
   (the decode slot buffers are b-sized and live during interleaved decode):
   C=1024 rows x 64 x 512 x 2 B = 64 MB latent staging + 8 MB rope + 32 MB
   output, builder-allocated.
5. Hook: a reserved_execution_flags bit routes the bulk-prefill attention
   branch to the new orchestrator; OPT-IN until the long-context needle
   gate passes on spark2, full-causal WMMA remains the default and the
   fallback.

Cost model at 1M: indexer O(N^2) at 128-dim fp8-weight dots ~ tens of
seconds once; attention O(N x 2048) absorbed; versus minutes-to-hours of
full-causal 512-dim x 64-head attention. Validation: (a) short-context
equivalence - with context < 2048 the selection is the full prefix, so
sparse prefill must reproduce the dense prefill hidden within tolerance
(tests/test_glm52_exact_pp13_prefill_hidden.py comparison mode), (b)
long-context needle retrieval at 32k/128k versus the dense path, (c) TTFT
measurement at 64k before/after.


## Implementation part 1 landed + a blocking finding for the needle test

Landed (inert until the orchestrator consumes them): the
EXECUTION_DSA_SPARSE_PREFILL flag, node-context buffer fields, and the
builder's state-level prefill staging (scores 16 x CONTEXT_TOKENS f32,
per-row selected/context/sequence arrays, chunk staging for query_a /
index heads / index weights / normalized hidden / low-column scratch,
row capacity 1024). Zero behavior change; both TUs gated, host tests
green.

BLOCKING FINDING, independent of sparse prefill: index keys are written
ONLY by the decode-step indexer (LaunchDsaIndexerDecode is the sole
caller of the key store; no prefill variant exists). Prefilled tokens
therefore have no entries in key_index_cache. The old 2048-window
config masked this (PRESELECTED prefix selection never read keys), but
with the pooled capacity, decode DSA selection over long prefilled
context scores against unwritten keys. The >2048 needle test will fail
on THIS before it tests anything else, unless run in PRESELECTED mode.
The fix is the same prefill indexer pass sparse prefill needs anyway
(chunk-row key projection + KeyNormRopeStore over prompt positions and
slots), which is step one of part 2.

Part 2 remaining, specified: (a) refactor LaunchDsaIndexerDecode to take
its six buffer pointers + row count (decode passes slot fields, prefill
passes the new staging + prompt buffers; key store needs a per-row
prompt slot mapping - either a small fill kernel from block table +
prompt positions or a keystore-kernel block-table resolve variant);
(b) add const uint32_t *row_sequence_indices (NULL = identity) to
DsaScoreWarpCandidateKernel and AbsorbedAttentionKernel, applied only
to block_table/first_block_token_offset resolution - all other indexing
already means "row"; (c) the orchestrator: per chunk, indexer pass,
AbsorbedQueryProject in place on prompt_query_latent (rows are already
(seq x stride + token) x 64 + head at 512-wide with nope in [0,192) -
verified identical to the decode layout), then per 16-row tile at
share-source layers: score into the staging (per-row causal bound via
row context lengths), radix per row (kernel reused verbatim - grid is
already one block per row), absorbed attention with tile-offset
pointers, selected indices persisted across the share group's layers;
UvApply once per chunk into prompt_attention_output; (d) hook in the
bulk-prefill attention branch behind the flag, WMMA full-causal remains
default and fallback.


## Superseding finding: decode candidate space is capped at 2048

The builder sets node->dsa_candidate_count to SELECTED_TOKEN_COUNT
(spark_glm52_pp13_node_context_builder_cuda.cu:953), and the radix
select clamps context_length to it. Decode DSA selection therefore
scores only the FIRST 2048 tokens of any context - a needle past 2048
is invisible by construction, independent of the index-key gap. The old
32-block window made this consistent; under pooled 1M capacity it is
the correctness wall for long-context decode.

The decode fix is not a constant bump: the per-slot dsa_token_scores
buffer uses a uniform seq-major stride of dsa_candidate_count, and
1M x B1024 is 4TB. It needs per-sequence score offsets (prefix sums of
context lengths, pool-bounded at 16MB total) plus a service-driven
per-step candidate bound, and belongs to its own effort.

Prefill is fixed on this branch: the sparse-prefill orchestrator now
derives its candidate bound from the chunk extent
(prompt_token_offset + prompt_token_stride, validated against
KV_CONTEXT_TOKENS), so selection spans the full prefix and launch grids
scale with actual context.

## Indexer pass decoupled from the sparse flag

LaunchDsaPrefillIndexerPass (RowSetup + attention-norm + q_a + the
shared indexer rows core) now runs on every index-share FULL layer
during paged bulk prefill, flag or no flag. Prefilled tokens get index
keys unconditionally, unblocking decode DSA over prefilled context by
default; SHARED layers skip it and reuse the group's row arrays and
selections from the state-level staging. The sparse attention
orchestrator consumes the pass outputs and starts at the absorbed
project.


## Branch-and-bound DSA pre-score filtering (proposal adopted)

Adopted the other dev's summary + bound layer onto this branch after
verification: the per-head interval bound is a true upper bound of the
decode score formula (negative head weights contribute at most zero and
are correctly omitted), the ordered-BF16 min/max compares are exact
over cache values, block indexing matches ResolveCacheSlot
((first_block_token_offset + token) / block_tokens against the
per-sequence table row), and candidate counts are zeroed per launch.
Excluded from the zip: module/production-runner serving changes,
which belong to a separate effort.

Fixes applied on adoption: the summary builder gained a dirty-flag
parameter (null = full rebuild) because the full-scan launcher cannot
run per step; the shared indexer rows core now marks dirty blocks from
the slot mapping after every key store (covers decode and prefill from
one site), and the prefill indexer pass rebuilds dirty summaries and
clears the flags per chunk. Summary buffers are per layer and sized by
the PHYSICAL pool (65536 blocks, 16 MiB min+max per layer), not the
16384 per-context logical blocks in the proposal's estimate. Stale or
unwritten slots can only loosen bounds, never break exactness.

Third independent 2048 wall found and fixed while integrating: the
service backend hardcoded MAX_BLOCKS_PER_SEQUENCE 32u, capping
allocator growth at 2048 tokens regardless of the builder capacity.
Now derived from SPARK_GLM52_KV_CONTEXT_TOKENS /
SPARK_GLM52_KV_BLOCK_TOKENS, with block tokens single-sourced in
kv_cache.h and the firmware constant deriving from it. The ifndef
guard around the capacity constants is removed.

The masked exact scorer (next step) must NOT write -FLT_MAX into a
dense per-candidate row: that preserves the uniform-stride score
buffer and the 4 TB wall. It must compact surviving blocks into a
candidate token list, score into a compact row, and radix-select over
it with an index indirection back to absolute tokens. The seed
threshold comes from exactly scoring the previous IndexShare selection
and taking the kth score, which keeps pruning exact. Decode-side
summary maintenance is already in place via the mark-dirty site in the
indexer core; the decode rebuild call lands with the masked scorer.
