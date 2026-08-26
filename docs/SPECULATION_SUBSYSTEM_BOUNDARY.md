# Speculation subsystem boundary

Proposal doc — SPECULATION subsystem agent. Unified-tree paths; the clone
workspace root is `.agents/speculation/`. Every claim is grep/read-verified
and cited `file:line` against the unified branch.

Base docs: `docs/SPECULATION_AUDIT.md` (audit), `docs/DRY_CONSOLIDATION_PLAN.md`
(DRY plan), `docs/PACKER_CORE_PLAN.md` (packer core).

---

## 1. Complete inventory of the subsystem on `unified`

### 1.1 Neutral drafter config tables — `model-families/common/include/sparkpipe/spark_dspark_drafter.h`

The one-backend/many-shapes table (129 lines). One of
`SPARK_DSPARK_TARGET_GLM52` / `SPARK_DSPARK_TARGET_K3` /
`SPARK_DSPARK_TARGET_DSV4_PRO_0813` selects a per-model table
(`spark_dspark_drafter.h:6-8`, `:87-89` error otherwise).

- GLM52 table aliases measured model constants (`:22-40`).
- K3 table (`:41-63`): block 7, 5 draft layers, aux {7,23,51,67,83}, 64 Q /
  16 KV heads, head dim 64, intermediate 14336, markov 256, mask 163824, rope 10000.
- DSV4 Pro 0813 table (`:64-86`): 3 draft layers at taps {58,59,60}, block 5,
  markov 512, noise 128799; draft attn heads/intermediate are declared `0u`
  "not declared yet" (`:82-85`).
- Shape-independent ABI (`:91-128`): ABI version 3, confidence milli 1000,
  default min confidence 350 / realtime 250, policy/sequence/draft/verify flags.

K3's source of truth for the table is `inference/llms/kimi_k3/dspark.h`
(read from the released DSparkDraftModel checkpoint, `dspark.h:1-8`); the
side-by-side GLM52-vs-K3 diff is `dspark.h:15-31` (KV heads 64 vs 16 is the
four-fold GQA divergence, `:29-31`).

### 1.2 Verifier kernels — `inference/kernels/speculate.cuh`

The shared verify/accept kernel contract (130 lines). Header states the split —
"the drafter is a policy and the verifier is a kernel" (`:1-13`), why acceptance
is exact not probabilistic (`:15-21`), why KV rollback is the part that bites
(`:23-28`).

- `LmSpeculativeVerifyGreedyKernel` (`:42-64`): longest-prefix walk + bonus
  token + rollback (`:58-63`).
- `LmSpeculativeVerifySampledKernel` (`:76-129`): modified rejection with the
  residual `max(0, p_target - p_draft)` resample (`:105-128`) and rollback (`:128`).

Audit marks it "should stay as-is; every model uses it"
(`SPECULATION_AUDIT.md:24-27`).

### 1.3 Speculation tree machinery — `include/sparkpipe/spark_speculation_tree.h`

Neutral node-topology / resolve machinery (228 lines). Shape is per-model: the
includer must define `SPARK_SPECULATION_TREE_CANDIDATE_COUNT`,
`..._VERIFIER_ROW_COUNT`, `..._MAX_COMMITTED_TOKEN_COUNT`,
`..._CONTEXT_EXTENSION`, `..._VOCAB_COUNT`, `..._NODE_ROWS` before including
(`:14-31`). `SparkSpeculationTreeResolve` does the longest-prefix walk
(`:174-228`); `SparkSpeculationTreeTopologyIsValid` validates the node table
(`:121-172`).

### 1.4 Dispatch-policy core — `include/sparkpipe/spark_speculation_policy.h` + `src/spark_speculation_policy.c`

Neutral dispatch-policy core (audit step 3). Header owns the descriptor struct
layouts under neutral names (`:44-169`) and 10 public prototypes (`:173-216`);
comment states layouts are byte-identical to the first adopting ABI and shapes
resolve through `spark_dspark_drafter.h` (`:3-13`). Source (`spark_speculation_policy.c`,
799 lines) is a pure textual rename (`:1-4`): `SparkSpeculationPolicyBuildDefaultModelContract`
(`:13-56`), `...ValidateModelContract` (`:58-108`), plus 10 public entry
points and 13 static helpers (initialize/mark-taps-ready/ensure-draft/get-draft/
complete-verify/resolve-verifier-tokens/cancel-sequence, `:287-799`).

The model-specific remainder is `modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_dispatch_policy.c`
(92 lines): the hidden tap plan + PP stage geometry only (`:1-4`,
`SparkGlm52DsparkBuildDefaultHiddenTapPlan` `:12-48`,
`SparkGlm52DsparkValidateHiddenTapPlan` `:50-92`).

### 1.5 GLM52 DSpark backend + tap transport

- Backend module `modules/glm52_dspark_draft_backend/`: header
  `include/sparkpipe/spark_glm52_dspark_draft_backend.h` (166 lines; backend
  struct with 64 weight slots, tap/block/context/markov/confidence device
  arenas, `:67-128`) and source
  `source/spark_glm52_dspark_draft_backend.cu` (2344 lines; audit
  `SPECULATION_AUDIT.md:18`). Kernel inventory (grep-verified): RMS-norm rows
  (`:323`), add-bf16 (`:371`), swiglu (`:389`), gather-stage-taps (`:408`),
  scatter-context (`:432`), build-query-block (`:465`), head-norm-rope
  (`:490`), block-attention (`:556`), gather-markov (`:702`), argmax
  (`:730`), confidence (`:799`). Weight roles (`:29-41`) include
  markov W1/W2 and confidence + confidence bias.
- GLM52 ABI surface `model-families/glm52/include/sparkpipe/spark_glm52_dspark.h`
  (124 lines): selects `SPARK_DSPARK_TARGET_GLM52` (`:17`), includes the neutral
  headers (`:18-19`), keeps legacy constant/type/function aliases (`:21-97`),
  and the GLM52-only hidden-tap-plan structs (`:99-120`).
- Tap transport (already shared, audit `SPECULATION_AUDIT.md:28`):
  `ring/sideband.h` (189 lines) — `LM_SIDEBAND_HIDDEN_TAP` kind (`:49`),
  `LmTapPlan` with `consumer_rank` placement field (`:84-91`);
  `include/sparkpipe/spark_hidden_transport.h` (461 lines) —
  `SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_DSPARK_HIDDEN_TAP 2u` (`:135`) and
  packet sideband fields (`:162-177`); impl `ring/transport/hidden_transport.c`.

### 1.6 DSV4 module cluster — `modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_module.c`

The resident-stage DSpark lane logic (structurally the fourth DSpark drafter,
welded into the stage; audit `SPECULATION_AUDIT.md:36-44`):

- `SparkDsv4ModuleDsparkDrive` (`:4214-4271`): markov chain — embedding
  gather on `markov_w1`, bias accum on `markov_w2`, argmax per step.
- `SparkDsv4ModuleRunDsparkDraft` (`:4320-4518`): the 3-layer draft forward —
  main-proj + norm prelude (`:4335-4342`), `[anchor, noise x (block-1)]`
  embed (`:4343-4348`), per-layer attn + MoE (`:4364-4511`).
- `SparkDsv4ModuleRunDsparkHead` (`:4273-4310`): HC head mix/reduce + final
  norm + lm_head shard linear.
- `SparkDsv4ModuleExpandDsparkVerify` (`:2427-2485`): stages `SPEC_STEP+1`
  rows (anchor + drafts) so the bucket replay is one verify frame.
- `SparkDsv4ModulePadDuplicateRows` (`:2492-2515`): pads a staged single row
  to bucket width (duplicate rows).
- Lane bookkeeping: `dspark_lane_ready/anchor/position` state arrays
  (`:375-377`), armed on accept (`:3633-3637`), consumed on submission
  (`:2542-2566`). Acceptance (greedy Leviathan) `:3568-3598`.
- Draft kernels are per-model in
  `source/spark_dsv4_dspark_kernels.cuh` (419 lines): DSpark attention
  (`:17`), markov bias accum (`:243`), argmax (`:284`), tap mean (`:339`),
  expand streams (`:383`).
- Bucket coupling: the whole cluster is gated on
  `SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u` (bucket 8)
  at `:1989`, `:2531`, `:5607` (audit `SPECULATION_AUDIT.md:54-60`).
  Pro shares the same module via `SPARK_DSV4_PRO_BUILD` (`Makefile.pro:45`;
  `spark_dsv4_resident_decode_stage_cuda.cu:1922,2751,2763,3366`).

### 1.7 MTP tree header — `model-families/glm52/include/sparkpipe/spark_glm52_mtp_tree.h`

GLM52 tree SHAPE pinning (107 lines): candidate 5, verifier rows 6, exec steps 3,
max committed 4, context extension 3 (`:14-18`), `NODE_ROWS` topology
(`:49-57`), then `#include spark_speculation_tree.h` (`:59`) and legacy
`SPARK_MODEL_MTP_TREE_*` / `SparkMtpTree*` aliases (`:61-107`).

### 1.8 Per-model draft configs

- **K3**: `inference/llms/kimi_k3/dspark.h` (130 lines), config-only today
  (no backend/verifier — audit `SPECULATION_AUDIT.md:14`). Constants
  `:38-74`; verify-budget rationale `:94-129`; `K3_DSPARK_TRIM_MIN_BATCH 8u`
  (`:122`).
- **DSV4 Pro GA 0813**: contract `model_contracts/dsv4_pro_authoritative.json`
  `"dspark"` block (`:32-42`): layer_count 3, block_size 5, noise 128799,
  target layers {58,59,60}, markov_rank 512. Mirrored in the generated header
  `model-families/dsv4/include/sparkpipe/spark_dsv4_pro_model.h:14-18` and the
  alias surface `spark_dsv4_pro_model_aliases.h:50-55`. Flash constants live in
  `spark_dsv4_model.h:36-43` (note Flash markov rank 256 vs Pro 512 —
  `spark_dsv4_model.h:39`).
- **Qwen38 MTP**: `mtp_layer_count: 1` (`model_contracts/qwen38_authoritative.json:20`),
  explicitly "MTP execution not enabled by the first driver revision"
  (`:11`); packed-not-served — serving adapter hardcodes
  `SPARK_QWEN38_STAGE_MTP = "0"` (`spark_qwen38_serving_adapter.c:329`),
  Makefile defaults `MTP_LAYER_COUNT ?= 0` (`modules/qwen38_resident_decode_stage/Makefile:41`),
  MTP tensors packed under sentinel layer `SPARK_QWEN38_STAGEPACK_MTP_LAYER (UINT32_MAX-1u)`
  (`spark_qwen38_stagepack_format.h:27`), and the three-frame MTP modifier
  contract documented in `spark_qwen38_max_resident_decode_stage_firmware.h:355-397`.

---

## 2. Boundary contract

### 2.1 The speculation subsystem OWNS (shared cross-model machinery)

Landed neutral code under shared paths; changes land through the coordinator.

1. `model-families/common/include/sparkpipe/spark_dspark_drafter.h` — the
   per-model config tables and the shape-independent ABI.
2. `include/sparkpipe/spark_speculation_tree.h` — candidate/verifier/step
   counts, node topology, resolution walk.
3. `include/sparkpipe/spark_speculation_policy.h` + `src/spark_speculation_policy.c`
   — the dispatch-policy core (proposal budget, verify batching, credit/window
   decisions, rollback handling; audit `SPECULATION_AUDIT.md:50-53`). Step 3
   is IN FLIGHT: the neutral proposal exists in the working tree; the
   coordinator lands it (Makefile wiring, GLM52 pin byte-identical, CI —
   `DRY_CONSOLIDATION_PLAN.md:33-40`). Pin tests already present:
   `tests/test_dspark_drafter_pin.c`, `tests/test_speculation_policy_pin.c`,
   `tests/test_speculation_tree_pin.c`.

### 2.2 Stays in model dirs (speculation REVIEWS, does NOT edit)

- **GLM52** (`model-families/glm52/`, `modules/glm52_dspark_draft_backend/`):
  `spark_glm52_dspark.h` (legacy aliases + hidden tap plan), the 2344-line
  backend `.cu`, the tap-plan remainder `spark_glm52_dspark_dispatch_policy.c`,
  `spark_glm52_mtp_tree.h` (the pinned shape + aliases). Owner: GLM52 session.
- **DSV4** (`model-families/dsv4/`, `modules/dsv4_resident_decode_stage/`):
  the module cluster (DsparkDrive/RunDsparkDraft/RunDsparkHead/ExpandDsparkVerify/
  PadDuplicateRows), `dspark_lane_*` bookkeeping, `spark_dsv4_dspark_kernels.cuh`,
  the bucket-8 `#if` coupling, `spark_dsv4_pro_model.h` + the aliases header.
  Owners: DSV4 Flash / DSV4 Pro sessions.
- **K3**: `inference/llms/kimi_k3/dspark.h` (config only). Owner: K3 session.
- **Qwen38**: MTP module machinery (packed, not served). Owner: Qwen38 session.

### 2.3 The CUDA-kernels agent implements (kernel contracts)

`inference/kernels/` is the CUDA-kernels lane (charter `.agents/AGENT_CHARTER.md:19-21`).

- **Verifier kernels** — already shared and stable: `inference/kernels/speculate.cuh`
  (greedy + sampled acceptance, KV rollback). The speculation subsystem relies on
  this contract and leaves it untouched (audit `SPECULATION_AUDIT.md:24-27`,
  `:91`).
- **Draft forward kernels** (draft attention, Markov chain/bias, confidence
  head) are kernel contracts and stay PER-MODEL for now — not to be generalized
  until three models converge on one implementation (audit
  `SPECULATION_AUDIT.md:101-106`). Today they are: GLM52's 11 kernels in
  `spark_glm52_dspark_draft_backend.cu`, DSV4's 5 kernels in
  `spark_dsv4_dspark_kernels.cuh`, K3's none (config only). When a model or
  subsystem agent needs a new/parameterized draft kernel, it files a contract
  (shapes, dtypes, precision route, target number) with the CUDA-kernels agent.

### 2.4 Already-correctly-shared infra (not speculation-owned, but relied on)

- Tap transport: `ring/sideband.h` + `include/sparkpipe/spark_hidden_transport.h`
  (+ `ring/transport/hidden_transport.c`) — audit `SPECULATION_AUDIT.md:28`.

---

## 3. Remaining consolidation steps (with exact owners)

1. **Audit step 5 — one replicated-draft packer rule.** Owner: **coordinator**
   (DRY plan item 5, `DRY_CONSOLIDATION_PLAN.md:47-48`; PACKER_CORE_PLAN
   `docs/PACKER_CORE_PLAN.md:55-64`). Speculation supplies the spec: a
   per-family sentinel table (`DRAFT_SENTINELS = {"dsv4": (0xFFFFFFFB, 3),
   "qwen38": (0xFFFFFFFE, 1)}`, `PACKER_CORE_PLAN.md:41`) and the
   `draft_layer_bounds` / `draft_rows_replicated` helpers. Today the rule is
   duplicated with two incompatible encodings: DSV4 sharder
   `MTP_LAYER_FIRST = 0xFFFFFFFB`, `MTP_LAYER_COUNT_MAX = 3` with the
   replicated branch in `plan_entry` (`tools/dsv4_tp16_stagepack.py:54-55`,
   `:336-340`; also `tools/dsv4_stagepack.py:94-95`), vs qwen38 packer
   `MTP_LAYER = 0xFFFFFFFE`, `MTP_LAYERS = 1` with the
   `MTP_LAYER` refs in `build_inventory` (`tools/qwen38_stagepack.py:55`,
   `:80`, `:278`) — see `PACKER_CORE_PLAN.md:17`.

2. **Qwen38 MTP adoption.** Owner: **Qwen38 session** (model agent owns
   `modules/qwen38_resident_decode_stage/`). Speculation proposes: when MTP
   execution is enabled, the qwen38 decode stage adopts
   `include/sparkpipe/spark_speculation_tree.h` (shape: 1 candidate, single
   MTP layer) instead of growing its own parallel bookkeeping
   (`SPECULATION_AUDIT.md:73-77`, `:49`). The three-frame modifier contract
   already documented at `spark_qwen38_max_resident_decode_stage_firmware.h:355-397`
   becomes the consumer of the neutral tree.

3. **DSV4 lane bookkeeping → the tree.** Owner: **DSV4 sessions** (Flash and/or
   Pro; the cluster is shared via the `SPARK_DSV4_PRO_BUILD` alias surface).
   Speculation proposes: the `dspark_lane_*` arrays + lane anchors
   (`spark_dsv4_resident_decode_stage_module.c:375-377`, `:3633-3637`) are the
   degenerate single-candidate tree; they adopt `SparkSpeculationTreeResolution`
   / the neutral node table instead of growing parallel arrays
   (`SPECULATION_AUDIT.md:73-77`, `:49`).

Adjacent (not in this deliverable's three steps, but still open from the audit):
**step 4 — un-couple the spec step from the bucket** (runtime program parameter,
delete the `#if SPARK_BATCH_BUCKET == ...` guards at
`spark_dsv4_resident_decode_stage_module.c:1989,2531,5607`; owner: DSV4
sessions, coordinated; `SPECULATION_AUDIT.md:82-86`). **Step 3 — dispatch-policy
split** is the in-flight handoff described in §2.1.

---

## Report

Changed path: `docs/SPECULATION_SUBSYSTEM_BOUNDARY.md` (new). How verified:
grep/read of every cited file against the unified clone; line numbers quoted
above are from the current checkout. Needed from others: (1) coordinator to
land the in-flight step-3 policy proposal and gate step-5 packer rule; (2) DSV4
and Qwen38 sessions to own the tree-adoption migrations; (3) CUDA-kernels agent
stands by for any draft-forward-kernel contract once three models converge.
