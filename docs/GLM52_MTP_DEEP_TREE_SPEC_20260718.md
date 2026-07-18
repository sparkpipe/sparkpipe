# GLM-5.2 MTP deep tree specification (16 rows)

Date: 2026-07-18. Status: SPEC. Host-side groundwork is merged (table-driven
tree engine, adaptive floor); the CUDA wiring below lands only after the
multirow linear-plan fix passes its hardware A/B, because this design assumes
verify rows are near-free up to bucket 16.

## Why

The current tree is depth 3, branch 1-2-2, 6 rows, ceiling 4 committed per
cycle. Measured 2.9 committed per cycle is 72% of that ceiling: acceptance is
strong, the tree is the binding constraint. Post-fix, rows cost ~nothing up to
the prepared bucket of 16. Deepening the tree raises tokens per ring
traversal, which is the only lever below the one-traversal cycle floor
(see the information argument in
`docs/GLM52_PP13_MULTIROW_LINEAR_PLAN_FIX_20260718.md` history: no outcome
bits exist before rank12 finishes, so cross-cycle overlap is impossible).

## Host groundwork already merged

`include/sparkpipe/spark_glm52_mtp_tree.h` is now table-driven:
`SparkGlm52MtpTreeNode {parent_row, depth, candidate_index, child_row_base,
child_count}` with a generic resolver, proven bit-identical to the legacy
depth-3 ladder by `tests/test_glm52_mtp_tree.c` (exhaustive 3^11 token
assignments plus 200k randomized plus helper-by-helper equivalence).
Key identity the generalization rests on: the wire `path_id` IS the deepest
accepted verifier row index, `fallback_row == path_id`, tail candidate and
parent are node-table lookups. The wire semantics do not change with tree
size; only the valid range grows.

The request API also gained the adaptive MTP floor: per-slot EMA (alpha 1/4)
of committed tokens per verify cycle; below 1.15 tokens/cycle the slot's
draft budget drops to zero (plain decode), re-probing every 16 committed
tokens. Suppressed slots drop arriving drafts with NOT_FOUND (engine already
tolerates it). This bounds worst-case MTP at plain-decode throughput minus
one cycle of hysteresis, for any tree size.

## The 16-row topology

Row 0 is the committed-context input row. Rows 1..15 carry candidates 0..14
(candidate_index = row - 1). Depth-heavy spine with early breadth where
divergence is most likely; children of each node are contiguous rows.

```text
row parent depth cand children      role
1   0      1     0    3,4           d1 primary
2   0      1     1    5             d1 alternate
3   1      2     2    6,7           d2 primary (of primary)
4   1      2     3    8             d2 alternate (of primary)
5   2      2     4    9             d2 (of d1 alternate)
6   3      3     5    10,11         d3 primary spine
7   3      3     6    -             d3 alternate
8   4      3     7    12            d3 (of d2 alternate)
9   5      3     8    -             d3 (alternate line tail)
10  6      4     9    13            d4 primary spine
11  6      4     10   -             d4 alternate
12  8      4     11   -             d4 (secondary line tail)
13  10     5     12   14            d5 spine
14  13     6     13   15            d6 spine
15  14     7     14   -             d7 spine
```

Constants: CANDIDATE_COUNT 5->15, VERIFIER_ROW_COUNT 6->16,
MAX_COMMITTED_TOKEN_COUNT 4->8, CONTEXT_EXTENSION 3->7,
RESOLUTION_COUNT 6->16. Expected committed per cycle at the measured
per-depth acceptance: ~4.5-5.5 against the new ceiling of 8.

## Drafting at rank12

Chained MTP drafting becomes 7 depth-serial passes over the internal-node
frontier: batch sizes 1,2,3,2,1,1,1 (parents {0},{1,2},{3,4,5},{6,8},{10},
{13},{14}); each pass runs the MTP layer plus head once for the batch and
takes top-k per parent where k is that parent's child_count. With
bucket-prepared plans each pass is one set of batched GEMMs; expected tail
~25 ms at depth 7. Future headroom: fuse passes.

## KV bookkeeping deltas

Multiple rows share position offsets (d1:2, d2:3, d3:4, d4:3, d5..d7:1), so
transient KV row slots = 2+3+4+3 minus the four canonical spine rows = 8;
TRANSIENT_BLOCK_COUNT 2->8, SHADOW_TOKEN_COUNT follows.
CANONICAL_POSITION_COUNT 3->7. ANCESTOR_COPY_COUNT: the resolution patch
list becomes table-derived per accepted path (max path length 7, each
non-spine hop copies one transient into canonical); size the constant to the
table maximum and generate the per-path patch lists host-side from the node
table, uploaded with the work packet, so the clone/patch kernels stop
hardcoding rows.

## Wire and struct deltas (internal hardfork, zero-drift release)

- `SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT` 6->15 resizes
  `mtp_draft_token_ids` in the request slot, dispatch, final event, and
  cuda-resident IPC completion; `REQUEST_API_MTP_MAX_DRAFT_TOKEN_COUNT`
  follows automatically.
- Work-control lane `mtp_resolution_path_id` stays uint16; range 0..15.
- `rows_per_lane` in verify packets: 6->16; scheduler cost model and
  `MtpOutranksPlainDecode` thresholds re-derived from the A/B numbers.
- KV budget per lane: MAX_COMMITTED+1 headroom checks update via the
  constant.

## CUDA touchpoints

1. `spark_glm52_sm121_required_decode_stage.cu`: verifier row loops, the
   MtpTree clone/patch/shadow kernels (consume uploaded per-path tables
   instead of DEPTH constants), `maximum_speculative_rows_per_lane`
   validation, final-token candidate row capacity.
2. `spark_glm52_pp13_node_context_builder_cuda.cu`: tree row position and
   embedding setup, transient block allocation, resolution application
   (already table-driven through the header helpers), drafting frontier
   passes at the final rank.
3. Archive stage plans: `maximum_speculative_rows_per_lane` 6->16 and
   `final_token_candidate_row_capacity` re-validated at load.

## Gate

Same discipline as the linear-plan fix: rerun the retained MTP cycle prompt;
token parity is NOT expected to hold (deeper tree changes committed
sequences under greedy only when alternates fire; a fresh parity hash is
retained instead against a plain-decode oracle of the same release). Accept
when committed/cycle >= 4.0 sustained on the benchmark prompt and B1
tokens/s >= 13, with the adaptive floor never engaging on the oracle prompt.
