# DSV4 Pro driver — top speed + TOP-10 improvements

Owner: dsv4-pro MODEL agent · clone of `unified` @ afb43a8 · proposal only (no
commits, no pushes). Area: the DeepSeek V4 Pro driver — GA 0813 migration (3 draft
layers, markov 512), codec selectability (FP8 experts, KV FP8), first-light BF16
activation path. Every claim grep/read-verified against this tree.

---

## Top-speed assessment (what exists / measured / missing)

**What exists (verified).**
- **GA geometry is pinned end-to-end.** The authoritative contract
  `model_contracts/dsv4_pro_authoritative.json:32-42` and the generated header
  `model-families/dsv4/include/sparkpipe/spark_dsv4_pro_model.h:13-18` agree:
  7168 hidden / 61 layers / 384 experts / top-6 / 128 heads / 512 head-dim,
  MTP=3 draft layers, DSpark block 5, target taps {58,59,60}, markov rank 512,
  noise 128799. The neutral drafter table carries the same shape
  (`spark_dspark_drafter.h:64-86`; draft attention heads/intermediate are still 0).
- **GA packer + module loader are GA-complete.** `tools/dsv4_pro_stagepack.py:175-214`
  emits the full MTP record set (main_proj/main_norm, 3 packed draft layers,
  markov_w1/w2, confidence_proj, hc_head); the module loads them
  (`spark_dsv4_resident_decode_stage_module.c:1066-1074`); the stagepack format
  carries the kind ids (`spark_dsv4_stagepack_format.h:107-115`).
- **Codec plumbing is landed, kernels are not.** `Makefile.pro:4-20` exposes
  `PRO_EXPERT_CODEC=mxfp4|fp8_e4m3` and `PRO_KV_CODEC=bf16|fp8_e4m3`; the header
  switches codecs on those flags (`spark_dsv4_pro_model.h:47-62`). The FP8-expert
  pack converter exists (`tools/dsv4_pro_expert_requant.py`) and the KV FP8 layout is
  specified (`tools/devcycle/dsv4_pro_kv_codec_plan.md:24-45`), but the expert
  kernels are MXFP4-hardwired (`spark_lm_kernels.cuh:2440,2747,3064`) and the KV
  path still stores BF16 with quant error *simulated*, not real FP8 bytes
  (`spark_dsv4_resident_decode_stage_cuda.cu:211,1051-1084`).
- **First-light BF16 path is the shipped default.** Non-expert activations are
  BF16 (`spark_dsv4_pro_model.h:63`, contract `_first_light_note` :71), matching the
  Flash-validated kernel set; routed-expert activations stay FP8-E4M3 (:64).

**What is measured.**
- Single-spark GPU validation only: val4 (0+4) and valtail (57+4) slices both
  **PASS**; valtail produced the first real Pro token 48774
  (`tools/devcycle/dsv4_pro_single_spark_receipts.md:8-29`). DRAM bandwidth
  measured 250-273 GB/s (`dsv4_pro_performance_estimate.md:6-9`).
- **No 16-rank decode has run.** The contract is `NOT_MEASURED / production_ready
  false` (`dsv4_pro_authoritative.json:73-77`); the staged deployment is
  PREVIEW-baseline, not GA (`dsv4_pro_ga_migration.md:101`). All decode numbers
  (12-13 tok/s main-only, 45-60 tok/s with DSpark) are estimates
  (`dsv4_pro_performance_estimate.md:39-47`), cross-checked against Flash TP4's
  measured 40.46 tok/s (:10-13).

**What is missing (the gap).**
- **GA DSpark execution is refused.** `spark_dsv4_resident_decode_stage_module.c:47-48`
  ("GA DSpark execution remains refused until a native pass lands"); no
  `SparkDsv4DSparkLaunch*` Pro-draft kernel exists in the tree (grep-negative).
  The DSpark kernels named in `dsv4_pro_ga_migration.md:116-124` (R11 log) are not
  present at afb43a8. Without them decode is weight-bound at ~12-13 tok/s.
- **FP8 expert kernel variant + FP8-aware sharder** (selectability matrix rows
  "MISSING", `dsv4_pro_selectability.md:56-57`).
- **Real FP8 KV read/write kernels** (write sites + 3 read sites,
  `dsv4_pro_kv_codec_plan.md:30-37`).
- **Duplicate admission loops and two KV-cache systems** (cross-model proposals
  below target dsv4 first).

---

## TOP-10 improvements (ranked)

### 1. Land the GA DSpark speculative execution pass (the native pass)
- **What:** implement the Pro draft kernels + module wiring in
  `dsv4_pro_ga_migration.md:126-163`: tap-mean capture of layers 58-60,
  main_proj+norm (3x7168→7168), rolling main-KV window, draft attention over
  main-KV + causal draft KV, 3 draft-layer mHC forward, markov bias + confidence
  head + shared screened-argmax head; 1 main + 5 draft tokens per step.
- **Why right:** buys **accurate-slow → match SOTA, then exceed SOTA** on decode:
  12-13 tok/s → ~45-60 tok/s at 50-70% acceptance — the GA checkpoint's built-in
  answer to decode rate (`dsv4_pro_performance_estimate.md:42-47`). This is the
  whole point of the 0813 migration; no other item moves a level this much.
- **Δ code:** +900…1400 (kernels ≈ +600, module chain ≈ +300, acceptance ≈ +100, tests).
- **Owner:** dsv4-pro (module + acceptance); CUDA-KERNELS (kernels 1-3 per
  `dsv4_pro_ga_migration.md:70-88`).
- **First step:** pin the still-zero drafter shapes in the contract — draft
  attention heads/intermediate/dim (`spark_dspark_drafter.h:82-85`) — then file
  the kernel contract cards against `docs/KERNEL_CONTRACT_CARDS.md:27-49`
  (Pro 0813 = block 5 / 3 draft layers / markov 512) and re-pin valtail.

### 2. Reconcile the two DSV4 KV-cache systems onto the common arena (KV_SEAM §3.3)
- **What:** adopt `docs/PROPOSAL_KV_SEAM.md:433-446`: fill the token-free
  `SparkKvModelTable` from the common-core consumer `spark_dsv4_paged_cache.c`
  (405 lines), keep `spark_dsv4_cache_plan.c/.h` (1037+126) as the *offline*
  sizing calculator only, and retire the duplicate model-specific arena
  (`spark_dsv4_cache_arena.c/.h`, 102+44) as a serving layout.
- **Why right:** **DRY win** — one DSv4 page layout instead of two; the JIT
  invariants are already enforced generically (`dsv4_pro_kv_cache_audit.md:9-27`).
- **Δ code:** −600…−700 net (delete arena layout + plan overlap).
- **Owner:** dsv4-pro (table fill); KV subsystem supplies `SparkKvModelTable`
  (approved as single fill point, `PROPOSAL_KV_SEAM.md:556`).
- **First step:** make `spark_dsv4_paged_cache.c` the only runtime layout and
  emit `SparkKvModelTable`; keep `tests/test_dsv4_cache_plan.c` +
  `tests/test_dsv4_paged_cache.c` green.

### 3. Adopt the admission/policy core (ADMISSION_CORE phases C & D, dsv4 first)
- **What:** collapse the three dsv4 admit sites into the shared core:
  `SparkDsv4ServingAdmit` (`spark_dsv4_serving_adapter.c:1207-1322`),
  `SparkDsv4StageRunnerAdmit` (`spark_dsv4_stage_runner.c:356-414`), and
  `SparkDsv4ResidentDecodeStageAdmit` (`..._module.c:5871-5946`) → a policy
  table + `SparkAdmissionEvaluateShape`/`EvaluateAndApply`.
- **Why right:** **DRY win** (removes ~230 lines of duplicated ladder) **plus a
  correctness fix** — the dsv4 wrappers map every rejection to BUSY/CAPACITY,
  losing DEADLINE and UNSUPPORTED_SHAPE (`PROPOSAL_ADMISSION_CORE.md:240-248`).
  dsv4 is the ordered first migrator (:412) because it also has the JIT-KV
  predicate + third call site.
- **Δ code:** −180…−200 net.
- **Owner:** dsv4-pro (own adapter/runner/module); scheduler supplies the core.
- **First step:** fill dsv4's `SparkAdmissionPolicyTable` (ceilings
  `PROPOSAL_ADMISSION_CORE.md:312-317`, JIT-KV predicate from
  `_module.c:5854-5937`) and swap the serving-side call.

### 4. Fold the Pro packer into the Flash packer (DRY H2)
- **What:** `tools/dsv4_pro_stagepack.py` is a geometry-override wrapper that
  `importlib`-loads the Flash packer (`dsv4_pro_stagepack.py:23-31`) and
  redefines geometry functions. Fold both into one parameterized packer with a
  MODEL_GEOMETRY table (the pattern already landed for the sharder pair).
- **Why right:** **DRY win** — one packer, geometry as data; removes the fragile
  importlib indirection and keeps Flash/Pro byte-identical where they overlap.
- **Δ code:** −100…−140 net (drop the wrapper module).
- **Owner:** dsv4-pro (offered, `docs/DRY_CONSOLIDATION_PLAN.md:9-13`); coordinator gates.
- **First step:** extract a `MODEL_GEOMETRY` dict into `dsv4_stagepack.py` and
  re-point `dsv4_pro_stagepack.py` at it; pin `tests/test_dsv4_stagepack.py`
  and `tests/test_dsv4_compressor_emission_source.py` byte-identical.

### 5. Generate identity (module-id / description sha) from the contract (M1/M2)
- **What:** the aliases header hardcodes the module id, target, and two
  description sha256s (`spark_dsv4_pro_model_aliases.h:60-63`); Makefile.pro
  rebuilds the id fragment by hand (`Makefile.pro:22`). Generate all of them
  from `dsv4_pro_authoritative.json` (contract → module-id/description JSON →
  sha), matching the generator already used for the #defines.
- **Why right:** **DRY win** — one source of truth; a geometry/codec edit can no
  longer desynchronize the id and the pinned shas (`DRY_CONSOLIDATION_PLAN.md:52-55`).
- **Δ code:** −40…−60 net.
- **Owner:** dsv4-pro (coordinator for the shared generator library, :49-51).
- **First step:** emit the module id + description shas in
  `tools/generate_dsv4_contracts.py` and delete the hand-maintained literals.

### 6. Real FP8-E4M3 KV cache (write/read conversion)
- **What:** convert the two write sites (cache scatter + KvEmission, incl. index)
  and three read sites (sparse attn, indexer score, draft attention) from the
  current quant-error *simulation* (`spark_dsv4_resident_decode_stage_cuda.cu:211,
  1051-1084`) to emitting/reading real E4M3+UE8M0 bytes (block 64, rope tail BF16).
- **Why right:** **match SOTA** on KV-codec fidelity — the reference itself
  quantizes KV to E4M3+UE8M0 (`dsv4_pro_kv_codec_plan.md:18-20`); gains 1.75×
  capacity per entry (583 vs 1024 bytes, :24-27), unlocking the native 1M-token
  context (`spark_dsv4_pro_model.h:33`). Neutral on the weight-bound decode path
  (`dsv4_pro_performance_estimate.md:69`), but it is the large-context lever.
- **Δ code:** +150…+250 (write ≈ +60, read dequant ≈ +120, tests).
- **Owner:** dsv4-pro (write/read sites); CUDA-KERNELS (block-scaled E4M3 dot).
- **First step:** build the `kv_fp8` module variant and run the val4 slice
  gate (`dsv4_pro_kv_codec_plan.md:49-53`) — compressed-history-only first, then
  measure the window.

### 7. FP8-E4M3 expert kernel variant + FP8-aware sharder
- **What:** (a) a fused expert W13/W2 kernel with FP8-E4M3 loads + F32 per-128
  scales instead of the MXFP4 loaders (`spark_lm_kernels.cuh:2440,2813,3122`),
  the only "MISSING" kernel row in `dsv4_pro_selectability.md:56`; (b) an
  FP8-aware sharder (the current column slice assumes 2-elements-per-byte, :57).
- **Why right:** completes the **codec-selectability** axis the task owns. Honest
  framing: it is an *accuracy* knob, NOT a speedup — FP8 doubles expert DRAM
  traffic (1 B vs 0.5 B/element, `dsv4_pro_performance_plan.md:91-93`), so it
  does not buy a throughput level; it buys the ability to choose the codec.
- **Δ code:** +240…+340 (kernel ≈ +200, sharder ≈ +40, requant already exists).
- **Owner:** CUDA-KERNELS (kernel); dsv4-pro (sharder).
- **First step:** write the W13/W2 FP8 loaders against the existing
  codec-agnostic grouped-tile schedule and smoke on the already-converted val4
  FP8 pack (`dsv4_pro_selectability.md:60-69`).

### 8. Weight read-ahead generalization (P8)
- **What:** extend the read-ahead that today only overlaps WQ_B with the
  attn-side reduce to prefetch the *next* layer's routed-expert tensors once the
  current layer's gate output is known (gate runs before FFN).
- **Why right:** hides expert DRAM latency behind the FFN-side collective on the
  weight-bound path — the dominant cost (`dsv4_pro_performance_plan.md:80-87`,
  `dsv4_pro_inference_path_audit.md:124`). A decode-level buy without new kernels.
- **Δ code:** +80…+120.
- **Owner:** dsv4-pro.
- **First step:** after the baseline ring receipt, add the next-layer prefetch
  hint behind the existing read-ahead queue and measure per-stage service time.

### 9. 2-token decode chains + control-plane pacing (P2a + P5)
- **What:** issue `chain_step_count == 2` resident chains (module already
  validates multi-step, `dsv4_pro_performance_plan.md:28-34`) and raise
  `maximum_messages_per_rank_per_progress`/`maximum_new_submissions_per_progress`
  to cut the per-token control round trips.
- **Why right:** halves per-token TCP traffic (32→16) and lets layer collectives
  cover 2 rows at 2× payload / same latency — expected 30-60 → 45-110 tok/s
  (`dsv4_pro_performance_plan.md:26-40`). This is the cheap interim before the
  DSpark pass (item 1) lands. Buys a decode level; config-only for P5, small
  module change for P2a.
- **Δ code:** +50…+150.
- **Owner:** dsv4-pro.
- **First step:** flip P5 knobs on the ring (config-only, :57-62), then gate a
  2-token chain behind a token-stream hash comparison.

### 10. Adopt the neutral speculation tree (Pro-shaped 5/6) — after item 1
- **What:** per `docs/PROPOSAL_DSV4_TREE_ADOPTION.md`, replace the hand-rolled
  longest-prefix accept loop (`_module.c:3582-3586`) with
  `SparkSpeculationTreeResolve`. Caveat: that proposal's 7-candidate/8-row shape
  is the *Flash* degenerate tree (`PROPOSAL_DSV4_TREE_ADOPTION.md:209-216`); the
  **Pro** tree must be 5-candidate/6-row (block 5, `spark_dsv4_pro_model.h:15`),
  derived after item 1 lands.
- **Why right:** **DRY win** (drop the duplicated accept loop; the neutral
  resolve already reproduces it byte-identically) and unblocks future *branching*
  drafts — a step toward **exceed SOTA** once acceptance stops being a linear chain.
- **Δ code:** −10 in module + ~40 family header + ~60 test (net +90, then −10).
- **Owner:** dsv4-pro (speculation agent's proposal as the base).
- **First step:** add
  `model-families/dsv4/include/sparkpipe/spark_dsv4_speculation_tree.h` with a
  5-candidate/6-row node table and pin it against `SparkSpeculationTreeTopologyIsValid`.

---

## Cross-model items with landed specs (where they apply to this model)

- **`docs/PROPOSAL_ADMISSION_CORE.md`** → item 3 (dsv4 migrates first; table at
  :312-317, mapping bug at :240-248, phases C/D at :396-404).
- **`docs/PROPOSAL_KV_SEAM.md`** → item 2 (DSv4 §3.3 at :433-446;
  `SparkKvModelTable` fully approved as single fill point at :556).
- **`docs/PROPOSAL_DSV4_TREE_ADOPTION.md`** → item 10 (Flash 7/8 shape at :209-216;
  Pro re-derives as 5/6).
- **`docs/KERNEL_CONTRACT_CARDS.md`** → items 1, 6, 7. Pro 0813 drafter card =
  block 5 / 3 draft layers / markov 512 / noise 128799 (:27-49); heads/intermediate
  "zero until the Pro session pins them" (:48-49) — pinning those is item 1's first step.
