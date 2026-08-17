# Speculation paths audit

Cross-model audit of every speculative-decode path in the unified tree
(2026-08-17), and the generalization plan. Scope: the six model sessions'
DSpark/MTP machinery as merged on `unified`.

## What exists today

| Model | Drafter | Draft source | Verify/accept | Tree/bookkeeping | Maturity |
| --- | --- | --- | --- | --- | --- |
| DSV4 Flash | DSpark | 3 checkpoint draft layers (40-42), spec step 7 | module cluster: DsparkDrive/RunDsparkDraft/RunDsparkHead/ExpandDsparkVerify + `speculate.cuh` kernels | lane anchors, dspark_lane_*, verify rows | integrated, bucket-8 coupled |
| DSV4 Pro (GA 0813) | DSpark | 3 draft layers (58-60), markov 512, confidence heads, main-proj | same module via `SPARK_DSV4_PRO_BUILD` aliases | same | integrated via the alias surface |
| GLM 5.2 | DSpark + MTP | separate small model + one MTP layer | `glm52_dspark_draft_backend` (2344-line .cu) + `speculate.cuh` | `spark_glm52_mtp_tree.h` (5 candidates, 3 exec steps) | most complete |
| Kimi K3 | DSpark (planned) | config-only today: `inference/llms/kimi_k3/dspark.h` | none yet | none | contract stage |
| Qwen 3.8 Max | MTP | 1 decoder layer, packed not served | none yet | none | packed only |
| MiniMax H3 | MTP | 3 MTP layers in contract | none | none | contract stage |

Line budget: GLM52 backend 2344 + dispatch policy 877 + dspark header 294 +
mtp tree 245 = ~3.7K lines; DSV4 module cluster + 419-line kernels; K3 config
130 lines; the shared verifier `inference/kernels/speculate.cuh` is 130 lines.

## Already correctly shared

- `inference/kernels/speculate.cuh`: the verifier/acceptance kernels (greedy
  and sampled with the residual correction) plus the KV rollback contract.
  Its header comment states the right split: "the drafter is a policy and the
  verifier is a kernel." This file should stay as-is; every model uses it.
- Tap transport: `ring/sideband.h` + `include/sparkpipe/spark_hidden_transport.h`.
- Serving-adapter capability flags and batch-tuning tables.
- The Flash/Pro commonization pattern (`SPARK_DSV4_PRO_BUILD` alias surface):
  one module serves two DSV4 variants. This pattern is the template for the
  rest of the plan.

## Duplication and model-specific drift

1. **Three parallel DSpark drafters.** The GLM52 backend implements the full
   drafter (tap capture, low-rank Markov bias, confidence head, draft
   attention). K3's `dspark.h` documents - convincingly - that the same
   backend already models K3's drafter; only the constants differ (block size
   8 vs 7, aux layer ids, GQA 16 KV heads vs 64, intermediate 12288 vs 14336,
   mask token 154856 vs 163824). DSV4 Pro's 0813 drafter (markov 512,
   confidence heads, main-proj) is a third instance of the same shape, and
   DSV4 Flash's module cluster is a fourth, structurally identical but welded
   into the resident stage with bucket-8 `#if` coupling.
2. **Tree bookkeeping is GLM52-named.** `spark_glm52_mtp_tree.h` (candidate
   count, verifier rows, execution steps, committed tokens, context
   extension) is generic speculation-tree structure wearing a model name;
   DSV4 re-implements the degenerate version with lane anchors and
   dspark_lane_* arrays, and Qwen38's MTP will need the same bookkeeping next.
3. **The 877-line dispatch policy has a generic core.** Proposal budget,
   verification batching, credit/window decisions, and rollback handling are
   model-neutral; the GLM52-specific parts are the tap sites and the
   row shapes.
4. **Bucket coupling.** DSV4 ties speculation to
   `SPARK_BATCH_BUCKET == SPARK_DSV4_MODEL_DSPARK_SPEC_STEP + 1u` (bucket 8)
   with preprocessor guards around call sites AND definitions (the unified
   branch already had to balance these for the gate's bucket-1024 archive
   build). The spec step belongs in the program/contract at runtime, not in
   per-bucket `#if` blocks - the same direction as TECHDEBT's "one program
   catalog for B1-B1024".
5. **Drafter pack rules repeat.** "Draft tensors replicate full-width to
   every rank" is implemented three times (flash TP16 sharder MTP block, the
   same block for Pro, the GLM52 packer).

## Generalization plan (staged, GLM52 case pinned byte-identical at each step)

1. **Neutralize the GLM52 backend** (`spark_glm52_dspark_draft_backend` ->
   `spark_dspark_draft_backend`): move every `SPARK_GLM52_MODEL_*` /
   `spark_glm52_dspark.h` constant behind a per-model config table selected
   by the package. GLM52's table reproduces today's values exactly, so its
   B12x receipts stay valid. K3 and DSV4 Pro then point at the same backend
   with their own tables - K3 gets a working drafter with zero new kernels.
2. **Extract the speculation tree** from `spark_glm52_mtp_tree.h` into
   `include/sparkpipe/spark_speculation_tree.h`: candidate/verifier/step
   counts and committed-token/context-extension bookkeeping as model-neutral
   types with per-model size constants. GLM52 consumes it unchanged; DSV4's
   lane logic and Qwen38's MTP adopt it instead of growing parallel arrays.
3. **Split the dispatch policy**: the model-neutral core (proposal budget,
   verify batching, rollback decisions) moves to shared code as
   `spark_speculation_policy.h`; the GLM52-specific remainder shrinks to tap
   sites and shapes.
4. **Un-couple the spec step from the bucket**: make the speculative step
   count a runtime program parameter, delete the `#if SPARK_BATCH_BUCKET ==
   ...` guards in the DSV4 module (the unified branch's guard-balancing work
   becomes unnecessary), and fold the pad/verify expansion into the common
   verifier path.
5. **One replicated-draft rule**: a single shared helper for "draft tensor
   rows replicate to every rank" used by the flash/pro sharders and the GLM52
   packer.

Ordering by ROI: 1 (K3 and Pro speculation land nearly free; kills the
biggest duplication) -> 4 (removes the guard cluster the CI already tripped
over) -> 2 -> 3 -> 5. Every step keeps `speculate.cuh` untouched and pins
the GLM52 case byte-identical before moving on.

## Explicitly NOT generalized

- The drafter forward kernels themselves (draft attention, Markov chain,
  confidence head) stay per-model until three models have converged on one
  implementation; parameterizing them prematurely would freeze the DSV4/GLM52
  hot paths before the hardware receipts agree.
- MTP as an algorithm stays distinct from DSpark only at the draft-source
  policy; everything downstream of the draft is already the same verifier.
