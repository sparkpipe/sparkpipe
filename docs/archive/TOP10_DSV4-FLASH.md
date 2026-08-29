# DeepSeek V4 Flash — top-10 improvements (dsv4-flash MODEL agent)

Proposal/docs only. No commits, no pushes. Every claim grep/read-verified and
cited file:line against `origin/unified` @ `afb43a8` (this workspace clone).

---

## Top-speed assessment (what exists / measured / missing)

**Area.** The Flash driver lives entirely in
`modules/dsv4_resident_decode_stage/` (module 6243 L, CUDA 3436 L, serving
adapter 1526 L, stage runner 521 L) with model facts in
`model-families/dsv4/`. No `inference/llms/dsv4_*` dir exists. The DSpark
drafter is 5 kernels in `spark_dsv4_dspark_kernels.cuh` (419 L); the verify
loop is in `spark_dsv4_resident_decode_stage_module.c`.

**What is measured (the ladder).** Merged-main `ed1d731d` = **33.55 tok/s**
mean (33.5505, best 33.6468) — `PERFORMANCE_STATUS.md:322-355`. The unmerged
resident-decode-chain branch (`a14c2e1`) = **38.18 tok/s** (38.1059 mean /
38.1757 best) — `PERFORMANCE_STATUS.md:280-297`. Retained no-spec floor =
33.6647 (`PERFORMANCE_STATUS.md:356`); selected lean stack = 40.4553 branch
(`PERFORMANCE_STATUS.md:97-166`); device-predicated compressor = 33.9911
branch (`PERFORMANCE_STATUS.md:384-397`). **Target = 50 tok/s**
(`PERFORMANCE_STATUS.md:276-278,288`; `KERNEL_CONTRACT_CARDS.md:283`).
Projections (not measured): TP8→60, TP16→83 tok/s
(`PERFORMANCE_STATUS.md:531-537`).

**The binding constraint.** No-spec B1 is pinned: the exact-token gate rejects
any reassociation of the target math, and the measured floor is **130 serial
collectives per token ≈ 5.5 ms/token idle** (`DSPARK_DSV4_FLASH_DESIGN.md:9-10`;
48.32 us/collective → 6.28 ms regression in the removed full-graph bridge,
`PERFORMANCE_STATUS.md:508-513`). DSpark is the one lever that survives both:
verify over k+1 rows streams weights once and runs the 130 collectives once per
k+1 tokens (`DSPARK_DSV4_FLASH_DESIGN.md:13-23`).

**What exists but is NOT measured.** The drafter is implemented but every
D4-D-01..05 card is `NOT_MEASURED` (`KERNEL_CONTRACT_CARDS.md:250-347`); no
speculation path has any decode number (`KERNEL_CONTRACT_CARDS.md:11-13`). The
contract now says `packed:true, execution_supported:true, serving_block_size:7`
(`model_contracts/dsv4_flash_authoritative.json`) — the design doc §2 still
shows the stale `false/false` (`DSPARK_DSV4_FLASH_DESIGN.md:28-42`). The
greedy Leviathan accept is the hand-rolled loop at `module.c:3582-3586`
(`dspark_enabled=1u` at `module.c:880`), gated behind a build-time
`SPARK_BATCH_BUCKET == SPEC_STEP+1 == 8` build (`module.c:1989-1992,2531-2532,5607-5610`);
the default build is bucket 1024, the measured B1 runs are bucket 1
(`Makefile:25`). TP4×PP4 is tooling + doc, `MEASURED=False` / "analytical
estimate" (`tools/dsv4_tp4_pp4_perf_estimate.py`; `DSV4_FLASH_TP4_PP4.md`).

**What is missing.** (1) any measured DSpark decode number (k-sweep, tok/step);
(2) the predeclared collective program that replaces 130 host submissions; (3)
a 16-node TP4×PP4 measurement; (4) the three landed cross-model adoptions below.

---

## TOP-10 improvements (ranked by Solutions / code-size²)

### 1. Adopt the admission/policy core (dsv4 first) — **DRY win**

- **What.** Collapse dsv4's three admission sites — serving-adapter
  `SparkDsv4ServingBuildCacheAdmission`+`EvaluateAdmission` call sites
  (`spark_dsv4_serving_adapter.c:1207-1311`), `SparkDsv4StageRunnerAdmit`
  (`spark_dsv4_stage_runner.c:356-414`), and the module ladder
  `SparkDsv4ResidentDecodeStageAdmit` (`module.c:5871-5946`) — onto
  `SparkAdmissionEvaluateShape` + a `SparkAdmissionPolicyTable` config with
  the JIT-KV prepare/commit/abort tail (`module.c:5910-5937`).
- **Why.** DRY: three re-implementations of the same descriptor/flags/slot/KV
  ladder, plus the stage-runner's lossy
  `BUSY ? BUSY : CAPACITY_EXCEEDED` mapping that drops the DEADLINE and
  UNSUPPORTED_SHAPE distinctions (`spark_dsv4_stage_runner.c:408-410`). dsv4 is
  the specified **first** adopter (most complex: JIT-KV predicate + third call
  site) — `PROPOSAL_ADMISSION_CORE.md:412-415`.
- **Code delta.** dsv4 net **≈ −200 lines** (three ladders → one table + tail;
  shared core is scheduler-owned and amortized across 4 models).
- **Owner.** dsv4-flash (its three sites); SCHEDULER agent supplies the core.
- **First step.** Land `include/sparkpipe/spark_admission.h` (Phase A), then
  replace the stage-runner + module admits with the table (§2.2 of the proposal),
  pinned by `tests/test_dsv4_stage_runner.c:36-40,220,230-232`.

### 2. Reconcile dsv4's two KV systems and fill `SparkKvModelTable` — **DRY win**

- **What.** dsv4 ships both a model-local `spark_dsv4_cache_plan.c` (1037 L) +
  `cache_arena.c` and the common-core `spark_dsv4_paged_cache.c` (405 L)
  wrapping `SparkKvCacheArena`+`SparkKvPageCache`. Decide cache-plan = offline
  sizing calculator, paged-cache = serving page-layout mapping, and emit the one
  `SparkKvModelTable` fill.
- **Why.** DRY: two arena implementations for one page layout. `SparkKvModelTable`
  is **FULLY APPROVED** as the single token-free fill point
  (`PROPOSAL_KV_SEAM.md:533-556`); dsv4 is the named adopter at §3.3
  (`PROPOSAL_KV_SEAM.md:431-446`).
- **Code delta.** net **≈ −300 to −500 lines** (dedupe the arena/copy path; the
  offline sizing calculator stays).
- **Owner.** dsv4-flash (model dir); KV-CACHE agent owns the table type.
- **First step.** Fill `SparkKvModelTable` from
  `spark_dsv4_resident_decode_stage_firmware.h:17` (block 128) +
  `spark_dsv4_pool_layout.h:16-40`, pinned by `tests/test_dsv4_cache_plan.c` +
  `tests/test_dsv4_paged_cache.c`.

### 3. Adopt the neutral speculation tree (drop the hand-rolled accept) — **DRY/structural**

- **What.** Replace the 6-line greedy longest-prefix loop
  (`module.c:3582-3586`) with `SparkSpeculationTreeResolve`; add
  `spark_dsv4_speculation_tree.h` (7-candidate / 8-row linear chain) + a pin
  test. The 7/8 shape is confirmed (driven by `DSPARK_SPEC_STEP 7`, not
  `DSPARK_BLOCK_SIZE 5`).
- **Why.** DRY: removes a model-local reimplementation of the shared resolve walk;
  "shared machinery once, model facts in tables." Byte-identical behavior pin
  makes it zero-risk (`PROPOSAL_DSV4_TREE_ADOPTION.md:137-163`).
- **Code delta.** net **≈ +140 lines** (+60 header +~100 pin test, −6 loop).
- **Owner.** dsv4-flash; SPECULATION agent owns the neutral tree.
- **First step.** Add the additive family header + `tests/test_dsv4_speculation_tree_pin.c`
  (steps 1-2 of the proposal), then the surgical swap (step 3).

### 4. Measure DSpark decode end-to-end (P4/P5) — **performance level: sub-80% SOTA → match/exceed SOTA**

- **What.** Build the `SPARK_BATCH_BUCKET=8` driver and run the already-written
  drafter + verify path: k-sweep (5/7/8/10), per-position acceptance, tok/step,
  vs the 40.4 control.
- **Why.** The single highest-value lever: it amortizes 130 collectives + one
  weight stream over k+1 rows. Community on this exact hardware class: **123.13
  tok/s @ k=7** (spec) vs 104.17 no-spec (`DSPARK_DSV4_FLASH_DESIGN.md:21-23`).
  Projected on our engine: ~60-70 tok/s at production acceptance, ~120 tok/s at
  community tok/step (`DSPARK_DSV4_FLASH_DESIGN.md:112-117`). Buys the step from
  **below-80% SOTA (33.55) toward match SOTA (level 4) or exceed SOTA (level 5)**.
  The code already exists — this is measurement + wiring, not new kernels.
- **Code delta.** **≈ 0 to +10 lines** (build flag + measurement harness; kernels
  already present at `KERNEL_CONTRACT_CARDS.md:250-347`).
- **Owner.** dsv4-flash.
- **First step.** Gate 1 first — O24/O128 exact-token hash vs the no-spec control
  (`DSPARK_DSV4_FLASH_DESIGN.md:137-140`), then the P4 k-sweep with warmups +
  10 measured runs (`DSPARK_DSV4_FLASH_DESIGN.md:143-148`).

### 5. Switch greedy → sampled (probabilistic) draft verification — **performance level multiplier**

- **What.** Feed draft logits as f32 probabilities into the existing
  `LmSpeculativeVerifySampledKernel` (`inference/kernels/speculate.cuh:76-130`,
  card V-02) instead of the greedy argmax-only path.
- **Why.** Single biggest community win: probabilistic acceptance **34.3% vs
  26.5%** (2.86→3.40 tok/step) (`DSPARK_DSV4_FLASH_DESIGN.md:62`). A direct
  multiplier on item 4's tok/step, therefore on the same level jump. The kernel
  already exists; only the draft-probability wiring is missing.
- **Code delta.** **≈ +60 to +80 lines** (emit draft logits for the ratio; reuse
  V-02 — do not re-implement).
- **Owner.** dsv4-flash (+ CUDA-KERNELS if V-02 needs a draft-prob input shape).
- **First step.** Emit the draft head's f32 logits row (already computed for
  argmax at `module.c:4248-4259`) alongside the greedy argmax, then call V-02.

### 6. Replace the 130-collective control plane with a predeclared program — **performance level: removes the network floor**

- **What.** Predeclare the 130 TP collectives per token as one graph program
  instead of 130 host submissions + callbacks (the full-graph bridge that
  wrapped them regressed 17.46% and was removed —
  `PERFORMANCE_STATUS.md:499-525`; the working graph-island controller is the
  sole path pending a predeclared program, `PERFORMANCE_STATUS.md:523-525`).
- **Why.** This is the measured no-spec ceiling: ~5.5 ms/token idle is the 130
  serial collectives (`DSPARK_DSV4_FLASH_DESIGN.md:9-10`). Removing host
  round-trips per collective buys the no-spec path toward the 50 target — a
  level jump on the **below-80% SOTA → 90% SOTA** axis without speculation.
- **Code delta.** net **≈ −150 to −300 lines** (delete 130 submissions/callbacks;
  add the prebuilt program).
- **Owner.** dsv4-flash requests; COORDINATOR/SCHEDULER lands the shared
  collective-program machinery (it is shared code, review-only for me).
- **First step.** Pin the graph-island controller as the sole path (already done),
  then replace its 130 submissions with one prebuilt collective schedule —
  pinned byte-identical on the 128-token control hash
  (`PERFORMANCE_STATUS.md:360-362`).

### 7. TP4×PP4 stage-3 DSpark draft placement (P6) — **performance level: batch/pipeline occupancy**

- **What.** Place the draft forward on PP stage 3 (layers 33-42, which own the
  target layers 40-42), with the verify running pipeline-wide as a (k+1)-row
  micro-batch and draft state round-tripping the stage-3 boundary
  (`DSPARK_DSV4_FLASH_DESIGN.md:123-133`).
- **Why.** Turns the TP4×PP4 16-Spark layout (`DSV4_FLASH_TP4_PP4.md:9-27`) into
  the DSpark-under-PP differentiator ("vLLM does not support this upstream" —
  `DSPARK_DSV4_FLASH_DESIGN.md:128-130`); B4 fills the pipeline and B8
  shared-prefix combines occupancy with draft revisit slack
  (`PERFORMANCE_STATUS.md:539-543`). Buys the batch/aggregate-throughput level.
- **Code delta.** **≈ +150 to +250 lines** (stage-3 draft dispatch + boundary
  round-trip). No measured 16-node number yet (`MEASURED=False`,
  `tools/dsv4_tp4_pp4_perf_estimate.py`).
- **Owner.** dsv4-flash; SCHEDULER for the cohort/micro-batch boundary.
- **First step.** Wire the stage-3 draft-forward dispatch on the TP4 ranks of PP
  stage 3, keeping the other stages' micro-batch slots starve-free
  (`DSPARK_DSV4_FLASH_DESIGN.md:170`).

### 8. Finish compressor post-launch predication — **small level + DRY**

- **What.** Predicate the remaining RMSNorm/RoPE/Hadamard/quantize/scatter
  compressor post work on the boundary token; the device-predicated compressor
  candidate already cut 269→62 launches and measured 33.9911 (branch)
  (`PERFORMANCE_STATUS.md:384-403`), but ~221 useless launches per token are
  still listed (`TECHDEBT.md:74-77`).
- **Why.** DRY/small level: deleting zero-emission launches is a solution at
  zero cost and is already a measured +0.97% over the 33.66 floor.
- **Code delta.** **≈ −100 lines** (remove zero-emission launch sites).
- **Owner.** dsv4-flash.
- **First step.** Merge the de1beb2 device-predicated compressor, then predicate
  the remaining compressor post launches with the bitwise sm_121a probe
  (`PERFORMANCE_STATUS.md:405-408`) as the gate.

### 9. Replace ~780 CUDA event record/wait per token with data-hazard edges — **performance level: host-overhead reduction**

- **What.** Emit true data-hazard dependency edges instead of blanket
  record/wait events; specifically KV post-processing and query projection must
  not inherit unrelated attention/projection barriers (`TECHDEBT.md:78-79`).
- **Why.** ~780 events/token is measured host overhead on the decode critical
  path; removing inherited barriers is a level contributor toward 50 with no
  numeric change (bit-exact gate still applies).
- **Code delta.** net **≈ −150 lines** (−300 events, +150 edge program).
- **Owner.** dsv4-flash (module + CUDA); CUDA-KERNELS for any graph edges in
  shared `inference/kernels/`.
- **First step.** Trace the KV-post and query-projection launch chain, then
  re-wire their dependencies to true hazards — gate on the 128-token hash.

### 10. Profile + fuse the draft GEMM epilogues (Marlin lesson) — **performance level: draft speed**

- **What.** Profile the 3 draft-layer GEMMs + 5-way heads + rank-256 Markov
  update; replace repeated global reductions with fused atomic accumulation
  (`DSPARK_DSV4_FLASH_DESIGN.md:94-99`; the GLM52 quantized draft went
  6.5→44.6 tok/s on this one change).
- **Why.** Draft speed is the first profile target (`DSPARK_DSV4_FLASH_DESIGN.md:167`);
  a faster draft directly raises tok/step and thus the speculative level. The
  D4-D-* cards are contract-complete but unmeasured (`KERNEL_CONTRACT_CARDS.md:250-347`).
- **Code delta.** **≈ +80 to +120 lines** (fused epilogue; fewer kernels).
- **Owner.** dsv4-flash (+ CUDA-KERNELS against updated D4-D-* cards).
- **First step.** CUPTI the 5 drafter kernels to find the small-M reduction
  hotspot, then file a kernel contract update for the fused epilogue per
  `KERNEL_CONTRACT_CARDS.md:399-409`.

---

## Cross-model specs that apply to dsv4 (landed, to adopt)

- `PROPOSAL_ADMISSION_CORE.md` — dsv4 first adopter (item 1).
- `PROPOSAL_KV_SEAM.md` — §3.3 dsv4 reconcile + `SparkKvModelTable` fill
  (item 2), table APPROVED (`PROPOSAL_KV_SEAM.md:533-556`).
- `PROPOSAL_DSV4_TREE_ADOPTION.md` — 7/8 tree swap (item 3), confirmed.
- `KERNEL_CONTRACT_CARDS.md` — dsv4's cards are D4-D-01..05 (drafter) + shared
  V-01/V-02 (verifier); all `NOT_MEASURED`, update `current_measured` when
  items 4-5 land.
