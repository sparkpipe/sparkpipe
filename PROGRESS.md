# PROGRESS — gemma4 driver lane

Branch `lane/gemma4-driver` @ origin/main 8f3a6f2. DESIGN.md untracked (per
brief). NEVER pushed (manager owns GitHub). Working dir /Users/mac/batch-gemma4.
Queue note: LIVE queue tool is
/Users/mac/sparkpipe-qwen38-lane/tools/spark_queue.py (v2); the in-repo
tools/spark_queue.py subcommand set is DEAD — use the v2 tool for every
sparka task.

## Commits (all on lane/gemma4-driver, in order)

- 99b4b78 port_family.py mechanical port (qwen4_flash donor -> gemma4, all 12
  files non-zero counts; tool copied byte-exact from batch-ling).
- a13cddf shared kernels: muse LmHeadRmsNormKernel appended byte-exact to
  inference/kernels/norm.cuh (weight_or_null + bf16-round-then-multiply);
  inference/kernels/attn.cuh + project.cuh now byte-identical to
  lane/laguna-driver (defaulted scale on LmRopePair/Rotate; LmRopePerHeadKernel
  +3 trailing params). No existing call site changes (all defaults).
  NOTE: DESIGN.md located the extended rope in attn.cuh; on the laguna branch
  it lives in project.cuh:428 — coded against the real signature.
- bba39c3 two-contract model headers (dense 31B defines + 26B MoE alias arm,
  dsv4 flash/pro pattern; dense arm #errors if MOE_BLOCK set without MOE_BUILD).
- c9ef0ea gemma stagepack format (family enum, 23 kinds, 120-byte common wire
  layout proved field-by-field; kv replication law in NarrowShape; nvfp4 seam
  = payload size math only; loader sentinels 100-103 keep the shared MTP hooks
  fail-closed) + gemma firmware ABI (two-pool KV, per-layer norm tables).
- 9b7ad5b cuda.cu carve 2587 -> ~470 lines: gqa.cuh triple consumption
  (window 1024, store, decode, qk_scale 1.0), two-pool geometry dispatch
  (sliding KV_HEADS 1|2, full 1 — covers every approved topology incl. TP16
  replication), rope 6-arg theta / 8-arg inv_freq table, head
  DirectArgmax+MaxLocPack (complement key = HF smallest-token tie-break)
  +TpCombineU64Max, bf16 grouped experts (LmRouteBuild + GroupedScalarLinear),
  family-local EmbeddingGatherShardedScaled (73.5/53.0) + GatedGelu.
- fb77152 module.c carve/rewrite 2533 -> ~1380: gemma sandwich (post-norm on
  sublayer OUTPUT before residual add), sliding/full attention bodies,
  k_eq_v ordering (v_norm reads kraw before k_norm overwrites), FFN + MoE
  router chain (router norm on the RESIDUAL, softmax temp 1, renormalising
  top-8, per-expert scale folded at pack), head emit, PP hidden transport;
  SPARK_FAIL at new host error sites; prefill fails closed (decode-only v1).
- ccaf958 arm Makefiles (31B TP16xPP1 default; 26B MoE TP4xPP4 fallback with
  stages {8,8,7,7}; GEMMA4_MODEL_REVISION defaults pending-warm-download and
  the adapter #errors if unset) + serving adapter on the SHARED qwen38
  template (arm-conditional ids/targets/stage tables; bf16 expert codec).
- 6885cda standalone gemma4_pack_synthesize.c (the shared qwen synth template
  hard-codes the donor lm_head+MTP tail); PER_EXPERT_SCALE reclassified
  per-layer; donor-ported validator/oracle/shell DELETED (they extern the
  deleted kernel set); AC1 grep clean (see gate below).

## Acceptance criteria status

| AC | status | evidence |
|---|---|---|
| 1 port + deletion | DONE | 99b4b78 + deletion commits; grep gate: zero Gdn\|Chunk\|Hc[A-Z]\|Indexer\|Ple\|Screened\|Mtp tokens in module+family code EXCEPT SparkGemma4ModuleBindMtp/ExpectedMtpBits (the shared spark_pack_load_common.h hook NAMES, required; implementations fail-closed) and the five gdn_* common wire field names in the header layout proof (shared SparkStagePackHeaderCommon field names) |
| 2 both arms compile | DONE for host code | cc -std=c11 -Wall -Wextra -Werror -fsyntax-only vs cuda_stub: module.c + serving_adapter.c + stagepack_format.h green in BOTH arms (dense + -DSPARK_GEMMA4_MOE_BUILD); .cu compile is an nvcc item for the sparka session |
| 3 header bindings test | NOT STARTED | test file pending (BINDINGS tables trivially derivable from spark_gemma4_model.h / contracts below) |
| 4 stagepack + synth | MOSTLY DONE | format header compiles both arms with geometry asserts; standalone synth compiles both arms, count assertion passes (payload emission proven before GB-scale write stopped); full-pack runs land with the sparka session |
| 5 oracle | NOT STARTED | spark_gemma4_reference.c was deleted with the donor port; rewrite per DESIGN section 7 (constants 73.5/53.0, qk 1.0, inv_freq 1e6**(-i/256) i<128 else 0, theta 1e4, eps 1e-6, window 1024, eos {1,106}) |
| 6 CUDA validation tier | NOT STARTED | harness .cu + validate.sh need a gemma-specific rewrite + a sparka GPU job through the v2 queue (see queue note); boundary matrix list in DESIGN section 7 stands |
| 7 offline gates | NOT RUN | dry-law/code-size/contract --check need contracts + test (AC3) first; manifest LAST |
| 8 blocked-on-download | BLOCKED (by design) | warm dirs not landed at session end; revision pins still pending-warm-download |

## Data files still to land (next session, in order)

1. model_contracts/gemma4_31b_authoritative.json + gemma4_26b_a4b_authoritative.json
   (serving adapter ids spark.gemma4.serving-adapter.tp16.v1 / .tp4pp4.v1;
   tp_degree 16/4; source_revision "pending-warm-download"; softcap 30 inert;
   embed scale 73.5/53.0; per-expert + router-scale pack folds; kv
   replication law; PP4 stage lists {8,8,7,7} for 26B fallback)
   + references/gemma4/ (config.json x2, modeling_gemma4.py,
   modeling_rope_utils.py, index x2 — fetch from HF when network allows).
2. tests/test_gemma4_model_header.py (two BINDINGS tables parameterised over
   the contracts; qwen4_flash test is the pattern).
3. model-families/gemma4/name_map.json + tensor_patterns.json (layer
   dispatch: layer % 6 == 5 -> full; tensor census per DESIGN section 1).
4. examples/model_descriptions/gemma4_31b / gemma4_26b_a4b firmware JSONs
   (session-port table envs left UNSET: pending fleet renumber, fail-closed).
5. validation/spark_gemma4_reference.c (oracle) +
   validation/spark_gemma4_resident_decode_stage_cuda_validation.cu +
   validate .sh; then sparka: offline gates cpu-class, validator GPU (v2
   queue tool, budgets: node tooling <=1GB RSS, GPU <=10GB).
6. tools/gemma4_stagepack.py real-weight packer (fires when
   /mnt/model-warm/gemma-4-31b-it / gemma-4-26b-a4b-it land; poll with
   short-timeout ssh stat, never blind-sleep): shard maps implement
   router-scale fold into router proj, per-expert scale fold into expert
   down rows, sliding k|v row fusion, full-layer kv-head replication
   (rank r -> global head r/(tp/kv)), layer_scalar assert-all-ones
   fail-closed, revision pin capture.
7. PACKAGE_MANIFEST.json + SHA256SUMS regenerated LAST; code-size ratchet
   re-measured EXACT (module.c ~1380 + cuda.cu ~470 vs donor 2533+2587 =
   the ~62% net deletion; deleted subsystems listed in commit messages).

## Topology (operator ruling chain, settled)

31B: TP16 x PP1, single stage, 16 sparks — kv replication proven (muse
precedent): sliding kv 16/16 exact, full kv 4 replicated x4, per-rank
KV_HEADS=1; gqa.cuh static_asserts hold (2048B/1024B slots); v=v_norm(raw k)
determinism chain safe (column-parallel kraw on identical post-reduce
residual -> per-head-local norm/rope -> bitwise-identical replicas).
26B: TP4 x PP4 stated default (16 sparks, experts 8/rank, uniform ~3.2GB);
TP16 table ready (kv 8 x2 / 2 x8 replication — same proof) — one-table
switch, module is topology-agnostic.
Ports: ALL session bases env-driven, NO frozen defaults (fail-closed when
unset) — pending the coredev fleet renumber (route-kind offsets +256/+512/
+768 make the region above 64700 unusable for degree-16; width-aware ledger
in earlier revision of this file; 26B TP4 fallback cells must also wait).

## Blockers / flags for the manager

- Warm download still absent at session end: real-pack validation, HF
  numerical comparison, revision pins, fleet bring-up all BLOCKED (AC8).
- Cross-lane: muse LmHeadRmsNormKernel + laguna rope hunks are copied
  byte-exact into this tree; merge converges trivially. Neither lane
  announced post-approval signature changes (DESIGN open question 2).
- The ported validation harness deletion (6885cda) means AC5/AC6 start from
  zero — deliberate: a donor harness externing deleted kernels cannot be
  "finished", only replaced.
