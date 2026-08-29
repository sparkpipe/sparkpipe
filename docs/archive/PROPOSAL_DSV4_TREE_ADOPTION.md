# DSV4 lane bookkeeping → neutral speculation tree adoption

Proposal — SPECULATION subsystem agent, for the DSV4 Flash/Pro sessions to
implement. Unified-tree paths; every claim grep/read-verified and cited
`file:line` against `origin/unified` (HEAD 5da43a5, post step-4).

Base: `include/sparkpipe/spark_speculation_tree.h` (neutral machinery),
`model-families/glm52/include/sparkpipe/spark_glm52_mtp_tree.h` (the pinned
reference shape), `tests/test_glm52_mtp_tree.c` (the equivalence-pin pattern),
`docs/SPECULATION_AUDIT.md` step 2.

---

## 1. What the DSV4 cluster actually does today

The DSV4 resident stage runs a DEGENERATE speculation tree: a single greedy
draft stream (the Markov chain), verified as a linear chain of 7 draft
positions + 1 anchor = 8 verify rows. There is no branching — one candidate per
position.

- Verify geometry is driven by `SPARK_DSV4_MODEL_DSPARK_SPEC_STEP 7u`
  (`spark_dsv4_model.h:43` = the contract's `serving_block_size`,
  `dsv4_flash_authoritative.json:97`): forward block = 7, Markov chain = 7
  iterations, verify rows = 8.
- `SPARK_DSV4_MODEL_DSPARK_BLOCK_SIZE 5u` (`spark_dsv4_model.h:37` =
  the contract's `block_size`, `dsv4_flash_authoritative.json:82`) is a
  SEPARATE constant: it is the serving-facing `max_speculative_token_count`
  (`spark_dsv4_serving_adapter.c:296`) and the DSpark reference block size,
  and it is NOT used anywhere in the draft/verify path. Do not build the tree
  from 5 — the acceptance loop iterates SPEC_STEP = 7 (`:3583`).

The hand-rolled bookkeeping lives in
`modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c`:

- Lane state: `dspark_lane_ready/anchor/position` arrays
  (`:375-377`); `dspark_host_draft_tokens[SPEC_STEP]` = 7 draft
  tokens (`:205`); `dspark_verify_rows` / `dspark_verify_accept`
  (`:203-204`).
- Acceptance (greedy Leviathan) is a 6-line longest-prefix loop
  (`:3582-3586`), then: emitted = `1 + accepted` and lane advance
  (`:3589-3596`), and the next anchor is the target's token at the
  divergence point `host_output_token_ids[accepted]`
  (`:3633-3637`). `host_output_token_ids[0..7]` is the target's 8
  per-row outputs (the 8th is the bonus token), copied D2H from
  `output_token_ids` over `continuation->rows` = 8
  (`:3544-3549`).

So the "tree" today is: 7 candidates (one per depth), 8 verifier rows, no
branching, longest-prefix accept, bonus token = the verifier's row at the stop
point.

---

## 2. DSV4 state → tree concept mapping

| DSV4 state (module.c) | Neutral tree concept |
| --- | --- |
| `dspark_host_draft_tokens[0..6]` (`:205`) | `candidate_token_ids[0..6]` → `SPARK_SPECULATION_TREE_CANDIDATE_COUNT = 7` |
| `host_output_token_ids[0..7]` (`:3544-3549`) | `verifier_token_ids[0..7]` → `SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT = 8` |
| longest-prefix loop (`:3582-3586`) | `SparkSpeculationTreeResolve` walk (`spark_speculation_tree.h:174-228`) |
| `accepted` (0..7) | `resolution->accepted_token_count` |
| `1 + accepted` (`:3589-3592`) | `resolution->committed_token_count` |
| `host_output_token_ids[accepted]` (`:3633-3634`) | `verifier_token_ids[resolution->fallback_row_index]` |
| `dspark_lane_ready/anchor/position` (`:375-377`) | NOT in the tree — stays as lane arm/disarm + tap-store state |
| `dspark_verify_rows/accept` (`:203-204`) | NOT in the tree — stays as verify-expansion staging bookkeeping |

The tree captures ONLY the resolve (longest-prefix + fallback/bonus row). Tap
capture, verify expansion (`SparkDsv4ModuleExpandDsparkVerify`,
`:2427`), the pad fallback (`SparkDsv4ModulePadDuplicateRows`,
`:2492`), the Markov draft (`SparkDsv4ModuleDsparkDrive`,
`:4214`), and the lane store all stay exactly as they are.

---

## 3. The DSV4-shaped tree (derived)

Shape constants (`model-families/dsv4/include/sparkpipe/spark_dsv4_speculation_tree.h`,
new, mirroring `spark_glm52_mtp_tree.h:14-59`):

```c
#define SPARK_SPECULATION_TREE_CANDIDATE_COUNT       7u
#define SPARK_SPECULATION_TREE_VERIFIER_ROW_COUNT    8u
#define SPARK_SPECULATION_TREE_MAX_COMMITTED_TOKEN_COUNT 8u
#define SPARK_SPECULATION_TREE_CONTEXT_EXTENSION     7u
#define SPARK_SPECULATION_TREE_VOCAB_COUNT           SPARK_DSV4_MODEL_VOCAB_COUNT
#define SPARK_SPECULATION_TREE_VERIFIER_INPUT_ROW    0u
#define SPARK_SPECULATION_TREE_RESOLUTION_NONE       0u
```

Node table — a linear chain of 8 rows; node fields are
`{parent_row, depth, candidate_index, child_row_base, child_count}`
(`spark_speculation_tree.h:41-48`):

```c
#define SPARK_SPECULATION_TREE_NODE_ROWS \
    { \
        {0u,0u,0u,1u,1u}, \
        {0u,1u,0u,2u,1u}, \
        {1u,2u,1u,3u,1u}, \
        {2u,3u,2u,4u,1u}, \
        {3u,4u,3u,5u,1u}, \
        {4u,5u,4u,6u,1u}, \
        {5u,6u,5u,7u,1u}, \
        {6u,7u,6u,0u,0u} \
    }
```

Then `#include "sparkpipe/spark_speculation_tree.h"`.

Derivation and validation (against `SparkSpeculationTreeTopologyIsValid`,
`spark_speculation_tree.h:121-172`):

- Root row 0: depth 0, parent 0 (`:125-127`) ✓.
- Rows 1..7 carry candidate_index 0..6, each used exactly once → the
  `candidate_seen_mask` covers `(1<<7)-1` (`:146-147`) ✓.
- Each row's depth = parent depth + 1; max depth 7 = CONTEXT_EXTENSION;
  max depth + 1 = 8 = MAX_COMMITTED_TOKEN_COUNT (`:136-150`) ✓.
- Child links: row r (0..6) has `child_row_base = r+1`, `child_count = 1`;
  row 7 is a leaf (`:151-170`) ✓.

Resolve walk semantics (`spark_speculation_tree.h:200-227`) with
`candidate_token_ids = dspark_host_draft_tokens` and
`verifier_token_ids = host_output_token_ids`:

- compares `verifier[i] vs candidate[i]` for i = 0..6, descending one row
  per match — exactly the loop at `module.c:3582-3586`.
- full reject → path 0, accepted 0, committed 1;
  match 0..k-1 → path k, accepted k, committed k+1;
  full accept → path 7, accepted 7, committed 8.
- `fallback_row_index == path_id == accepted`, so
  `verifier_token_ids[fallback_row_index] == host_output_token_ids[accepted]`
  (the next anchor). For a linear chain `path_id == depth`, so the
  position aliases are trivial; no extra helper is needed.

---

## 4. Byte-identical pinning strategy

The pin is on BEHAVIOR (emitted stream, lane advance, KV context, next anchor),
not on the C text. Two layers, following the GLM52 template:

1. Host equivalence pin (new test, before touching the module): reimplement the
   current acceptance as `LegacyDsv4Resolve` (a verbatim copy of the
   longest-prefix loop + `committed = accepted + 1` +
   `fallback = accepted`), then exhaustively and randomized-compare it
   against `SparkSpeculationTreeResolve` over a small alphabet — the exact
   pattern of `tests/test_glm52_mtp_tree.c:104-161` (legacy reimplementation)
   and `:220-253` (exhaustive-alphabet + randomized compare). Also
   `_Static_assert` the shape constants (7/8/8/7/129280), mirroring
   `tests/test_speculation_tree_pin.c:11-21`, and assert
   `SparkSpeculationTreeTopologyIsValid() == 1u`.
2. End-to-end CUDA re-pin: after the module swap, run the existing DSV4 host
   tests + `modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh`
   unchanged and require identical receipts. Because the resolve emits the same
   `accepted` / `1 + accepted` / `lane_next_positions` /
   `context_token_count` increments, the KV rollback and lane store are
   byte-identical.

The bucket-8 behavior is untouched: the runtime gate
`SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u && state->dspark_enabled != 0u`
(`module.c:1989-1992`, `:2531-2532`, `:5607-5610`), the
verify expansion, and the pad fallback are all outside the resolve and remain as-is.

---

## 5. Step-by-step migration (DSV4 sessions)

1. **Additive family header.** Add
   `model-families/dsv4/include/sparkpipe/spark_dsv4_speculation_tree.h`
   (shape + node table from §3). Nothing consumes it yet — zero behavior change.
2. **Pin before rewiring.** Add `tests/test_dsv4_speculation_tree_pin.c`
   (§4 layer 1). It proves the neutral resolve reproduces the current acceptance
   for every input, so step 3 cannot regress it.
3. **Surgical swap.** In `SparkDsv4ModuleContinueHeadMax`
   (`module.c`), replace the loop (`:3582-3586`) with
   `SparkSpeculationTreeResolve(slot->dspark_host_draft_tokens, slot->host_output_token_ids, &resolution)`;
   set `accepted = resolution.accepted_token_count`; read the next anchor
   as `slot->host_output_token_ids[resolution.fallback_row_index]` (identical,
   since fallback_row_index == accepted). Leave `1 + accepted`, the lane
   advance, and the tap-store publish (`:3599-3638`) unchanged.
4. **Re-pin the CUDA path** (§4 layer 2). Green before/after is the gate.
5. **Optional cleanup.** Drop the now-dead loop; `dspark_host_draft_tokens`
   stays as the candidate array (no storage change).

---

## 6. Out of scope (stays in the DSV4 model dir, unchanged)

Draft forward kernels + Markov bias + confidence head (per-model, audit
`SPECULATION_AUDIT.md:101-106`); tap capture + lane store; verify
expansion + pad fallback; the runtime bucket gate. These are not touched by
this adoption.

---

## Report

Changed path: `docs/PROPOSAL_DSV4_TREE_ADOPTION.md` (new). Verified by
grep/read of the DSV4 module cluster, `spark_dsv4_model.h`, the Flash
contract, the serving adapter, the neutral tree header, and the two tree pin
tests; the node table above was hand-checked against
`SparkSpeculationTreeTopologyIsValid` and the resolve walk. One flag for
the DSV4 sessions: the tree is 7-candidate / 8-row (driven by
`DSPARK_SPEC_STEP 7`), not 5 — `DSPARK_BLOCK_SIZE 5` is the serving
adapter's `max_speculative_token_count` and is a different constant.

---

## Coordinator verification note (2026-08-17)

The 7-candidate/8-row finding is CONFIRMED: the Flash draft/verify path is
sized by SPARK_DSV4_MODEL_DSPARK_SPEC_STEP 7 (draft token arrays at
module.c:205-213, draft block launch at :1887), not by DSPARK_BLOCK_SIZE 5.
The 0813 Pro table in docs/KERNEL_CONTRACT_CARDS.md (block size 5) is the
Pro drafter's own contract value and is unaffected. DSV4 sessions building
the tree must use 7 candidates / 8 verifier rows.
