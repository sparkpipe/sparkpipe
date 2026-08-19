# Qwen 3.8 27B driver — top-10 improvements

Lane: qwen38-27b MODEL agent. Covers the migration off deprecated-frozen
qwen36, the TP4 band (spark0-3), and the perf targets (the qwen36 TP4
phase-2 plan re-based to the 3.8 27B checkpoint). Every claim grep/read
verified against this clone at HEAD afb43a8; cited file:line. Proposal/docs
only — no commits, no pushes, no shared/model edits in the real tree.

---

## Top-speed assessment (≤1 page)

**What exists.** A fully wired, frozen 27B driver under the *qwen36* name —
`model-families/qwen36/`, `modules/qwen36_resident_decode_stage/` (~9k LOC:
module 2307 + cuda 2208 + serving adapter 1724 + tp 403 + validation 1209 +
reference 567), `inference/llms/qwen_3_6/`, `tools/qwen36_stagepack.py` (692).
Geometry pinned: hidden 5120, 64 layers, 16×(3 GDN→1 attn), GDN 16/48 heads,
attn 24/4, dense FFN 17408, MTP 1 (`spark_qwen36_model.h:7-25`;
`qwen36_authoritative.json:15-44`). The TP4 band deployment artifacts are
already renamed to "qwen38" on the *runtime* side while still building the
*qwen36* module: `qwen38_tp4_build.sh:11,22` (`make -C modules/
qwen36_resident_decode_stage`), `qwen38_tp4_deploy.sh:4` (runtime root
`qwen38.bf16.tp4`), deploy env `SPARK_QWEN36_STAGE_MTP=1 …
SPECULATIVE_DRAFT_COUNT=2` (`qwen38_tp4_deploy.sh:34`).

**What is measured.** TP4 B1 on the spark0-3 band (4×GB10): plain 83.0 ms,
spec D=2 74.8 ms = **13.2 tok/s**, weight-stream floor 59.8 ms, practical
BF16 ceiling ~76-78 ms — i.e. ~92% plain / ~98% spec of ceiling
(`QWEN36_TP4_PERF.md:6,19-28`). Entropy of the real weights: 10.52 bits
order-0 = 1.52× lossless headroom (`QWEN36_TP4_PERF.md:35-36`). These are
**Qwen 3.6 27B** weights (54.5 GB BF16, 27.27B params;
`QWEN36_BF16_SPEED.md:6-8`). There is **no Qwen row in PERFORMANCE_STATUS.md**
(case-insensitive grep: 0 matches) — the only ledger is the doc.

**What is missing.** (1) No Qwen 3.8 27B authoritative contract or family
facts — the directive requires them and forbids reusing the 3.8 Max constants
(`COORDINATION.md:154-155`; `TECHDEBT.md:132-135`). (2) The adapter advertises
four conflicting identities: adapter id `…tp4.v1`, serving model
`Qwen/Qwen3.8-27B`, driver model `alibaba.qwen3.6-27b…`, stage
`qwen36_resident_decode_stage` (`spark_qwen36_serving_adapter.c:55-62`).
(3) `must_work_targets.json:59-76` still pins `qwen36_27b_bf16`.
(4) `fleet_registry.json:15-22` `qwen27b` entry mixes a `…pp16` runtime
root with `topology: TP4` and a "verify" owner. (5) No qwen38-27b entry in
`tools/perf_estimate.py:212` / `tools/nvme_kv_estimate.py:137`, and no kernel
contract cards in `KERNEL_CONTRACT_CARDS.md`. (6) `qualification/ds4_eval/
README.md:54` still lists the Qwen 3.6-27B-FP8 reference profile.

The migration is **half-done**: runtime/scripts renamed, model id touched, but
family, module, stagepack, contract, registry, and must-work target are still
3.6. Everything below closes that gap, DRY wins before perf levels.

---

## TOP-10 improvements (ranked)

### 1. Pin the Qwen 3.8 27B checkpoint → authoritative contract + family facts
- **What.** New `model_contracts/qwen38_27b_authoritative.json` +
  `model-families/qwen38_27b/` + `inference/llms/qwen_3_8_27b/config.h`; replace
  the `qwen36_27b_bf16` must-work target (`must_work_targets.json:59-76`) in the
  same PR, per `COORDINATION.md:150-153`.
- **Why.** This is the *accurate-slow* rung (level 1): there is no 3.8 27B
  driver at all today, and no perf level exists until the geometry is
  checkpoint-pinned. It is mandated as one gate-breaking PR and blocks #2-#10.
  Do NOT copy the 3.8 Max header (`COORDINATION.md:154-155`).
- **Delta.** +~250 lines (contract ~90, family header ~65, config.h ~70,
  `test_must_work_targets.py` edits).
- **Owner.** qwen38-27b (me); coordinator lands.
- **First step.** Obtain the exact checkpoint id+revision, pin hidden/layers/
  GDN/attn/FFN/MTP in the JSON, emit the family header, update
  `tests/test_must_work_targets.py:12,47`.

### 2. Finish the rename qwen36 → qwen38_27b; fix the 4-way identity drift + fleet registry
- **What.** A real `qwen38_27b` family/module/adapter/stagepack identity; collapse
  the conflicting ids at `spark_qwen36_serving_adapter.c:55-62`; fix
  `fleet_registry.json:15-22` (pp16 runtime_root vs TP4 topology, "verify"
  owner) to the real TP4 root the scripts already use.
- **Why.** DRY/structural: one model should have one name; today the same
  artifact is `qwen36`/`qwen3.6-27b`/`Qwen3.8-27B` in four adjacent lines,
  and the registry disagrees with the build scripts. Deleting the fork is a
  zero-cost solution and removes the trap that a future editor re-reads 3.6
  geometry as "3.8-27B".
- **Delta.** net ~0 lines (renames + a config table), minus the drift.
- **Owner.** qwen38-27b (me) for model dirs; coordinator for the registry entry.
- **First step.** Create `model-families/qwen38_27b/`, point the family
  conformance gate at the #1 contract, rename the serving adapter ids.

### 3. Start the module from the admission core (PROPOSAL_ADMISSION_CORE.md phase D)
- **What.** The new qwen38-27b module fills `SparkAdmissionPolicyTable` and calls
  `SparkAdmissionEvaluateShape` instead of cloning the qwen36 admit ladder
  (`spark_qwen36_resident_decode_stage_module.c:2031-2116`, serving
  `spark_qwen36_serving_adapter.c:981-1010`). The proposal explicitly orders
  qwen38 to *start* from the core rather than migrate (`PROPOSAL_ADMISSION_CORE.md:412-418`).
- **Why.** DRY win: the admit ladder is re-implemented 4× across models
  (`PROPOSAL_ADMISSION_CORE.md:20-32`); the table is ~3 ceilings + 2 flag bits.
  It also fixes the confirmed rejection→status bug that collapses DEADLINE/
  UNSUPPORTED_SHAPE to CAPACITY_EXCEEDED (`PROPOSAL_ADMISSION_CORE.md:238-247,504-512`).
- **Delta.** −~85 lines vs carrying the qwen36 ladder (table ≈ 15 lines).
- **Owner.** qwen38-27b (me) for the table; scheduler agent supplies the core.
- **First step.** Fill the table (max_active/max_input/max_sequence_positions +
  PREFILL_SINGLE_SLOT|DECODE_EQUALS_SLOTS), no predicate, no cost tail.

### 4. Fill the shared KV table (PROPOSAL_KV_SEAM.md §3.5), re-based to the 27B
- **What.** Fill `SparkKvModelTable` for qwen38-27b's **16** full-attention
  layers (not the Max's 23): 2×4 KV heads×256×bf16 = 2048 el/token = 4 KiB/
  token/layer, 48 GDN layers carry no KV. Re-base qwen36's
  `SparkQwen36ModuleOpenKvTier` as the reference (`PROPOSAL_KV_SEAM.md:464-478`).
- **Why.** DRY win: one token-free table replaces the ad-hoc per-model fills
  (only k3 ships one today; `PROPOSAL_KV_SEAM.md:292-296`). Unlocks the
  TP 1/N head-sharding capacity fix and the DRIVER_OWNS_KV path. §3.5 is
  written for the Max geometry and must be re-based to 16/48 — a real
  correction, not a copy.
- **Delta.** −~50 lines (one table) + a `SparkQwen38_27bPageCopy` primitive.
- **Owner.** qwen38-27b (me) + kv-cache agent (table type).
- **First step.** Fill the table for 16 attn layers; mirror `tests/test_k3_kv_cache.c`
  to prove the seam is crossed token-free.

### 5. Adopt the neutral speculation tree + shared verifier for the MTP chain
- **What.** Replace the hand-rolled qwen36 MTP accept loop
  (`spark_qwen36_serving_adapter.c:1280`) and the 3-frame decode-draft/verify/
  replay bookkeeping (`:82-96,1201-1210`) with `SparkSpeculationTreeResolve`
  (the `PROPOSAL_DSV4_TREE_ADOPTION.md` pattern) and `inference/kernels/
  speculate.cuh` (V-01/V-02, `KERNEL_CONTRACT_CARDS.md:350-395`).
- **Why.** DRY win: qwen36's MTP speculation is 0-match against the shared
  verifier/tree (grep `speculate.cuh|SparkSpeculationTree` in the module = 0);
  it re-implements the longest-prefix walk the neutral tree already owns
  (`SPECULATION_AUDIT.md:24-26`). Also the missing-cards gap: qwen38-27b has
  **no** cards in `KERNEL_CONTRACT_CARDS.md`; request the MTP drafter + reuse
  V-01/V-02 for verify.
- **Delta.** −~60 lines (drop the hand-rolled accept/restore) + a tree-shape
  header mirroring `spark_glm52_mtp_tree.h`.
- **Owner.** qwen38-27b (me) + speculation agent (tree).
- **First step.** Add the MTP tree-shape header, pin the legacy accept vs
  `SparkSpeculationTreeResolve` equivalence before rewiring (per
  `PROPOSAL_DSV4_TREE_ADOPTION.md:136-157`).

### 6. Re-base the perf targets + land a PERFORMANCE_STATUS row
- **What.** Add a qwen38-27b geometry to `tools/perf_estimate.py:212` (PERF) and
  `tools/nvme_kv_estimate.py:137` (MODELS); re-base the phase-2 plan
  (`QWEN36_TP4_PERF.md:53-66`) onto the 3.8 27B weights; record a
  `Qwen38-TP4-E2E-PASS` B1 receipt in `PERFORMANCE_STATUS.md`.
- **Why.** Makes the level visible and the "N× single-spark" accounting honest
  (`QWEN36_TP4_PERF.md:8-13`). The current 13.2 tok/s and 1.52× entropy are
  3.6 datapoints, not release evidence (`COORDINATION.md:160-161`) — the re-base
  needs the real 3.8 27B weights.
- **Delta.** +~10 lines (2 table entries + 1 doc row), gated on a TP4 receipt.
- **Owner.** qwen38-27b (me).
- **First step.** Re-run the TP4 B1 gate on 3.8 27B packs; add the
  `qwen38_27b` entry to `PERF`/`MODELS` and the receipt.

### 7. Lossless weight-stream codec + decompress-in-GEMM (the +25-30% at quality parity)
- **What.** ANS codec over the BF16 weight stream (measured 1.52× headroom;
  `QWEN36_TP4_PERF.md:35-51`), decoded in the small-batch GEMM tile staging so
  per-spark stream falls 13.5→8.9 GB and B1 ~63 ms (~15.9 tok/s).
- **Why.** **Perf level 90% SOTA → match SOTA** on single-stream decode at
  identical quality: the driver is already ~92-98% of the *BF16* ceiling
  (`QWEN36_TP4_PERF.md:27`); shrinking the bytes is the only way past it
  without quantizing. This is the single largest latency lever that does not
  trade quality.
- **Delta.** +~200 lines (ANS codec + tile-staging decode path).
- **Owner.** qwen38-27b (me, propose) + CUDA-KERNELS agent (implement).
- **First step.** File a Part-1 contract card (shapes/dtypes/precision route +
  target number) per `KERNEL_CONTRACT_CARDS.md:399-408`.

### 8. Sharded-delta B64 collective (ship only the changed columns)
- **What.** B64 collective sends 1280 of 5120 changed hidden columns instead of
  the full 655,360 bytes; B64 step ~404→~150 ms (`QWEN36_TP4_PERF.md:56-57`).
- **Why.** **Perf level 80%→90% SOTA on B64 aggregate**: TP4 shards columns, so
  the delta of a token step is a thin slice; shipping it whole is pure waste
  (`QWEN36_TP4_PERF.md:8-13`).
- **Delta.** +~120 lines (delta detection + reduced all-reduce).
- **Owner.** qwen38-27b (me, propose) + CUDA-KERNELS/scheduler agents.
- **First step.** Contract for the delta-encode kernel + the reduced collective
  shape; confirm the hidden-transport backend's multi-outstanding path first.

### 9. B1 head-shadow path + GDN small-kernel fusions (close the last ~1-3 ms)
- **What.** Screen the B1 full-vocab head read 2.8→~1.5 ms and fuse the GDN
  small-kernel branches (~470 µs/layer above the ~215 µs GEMM floor) —
  `QWEN36_TP4_PERF.md:25-28,59-60`.
- **Why.** **Perf level: 92%→~98% of the practical ceiling** on B1 latency
  (level 3→4 at the margin); these are the only BF16-only headroom left.
- **Delta.** net ~0 lines (fusions + a screened head path; may delete more than
  it adds).
- **Owner.** qwen38-27b (me) + CUDA-KERNELS agent.
- **First step.** Profile the head read and the GDN branch overheads on the
  3.8 27B packs to confirm the numbers re-base unchanged.

### 10. bf16-state GDN kernel variant (halve the 310 MB/seq/token stream)
- **What.** A named bf16-pool delta-rule VARIANT (never a silent constant flip),
  guarded by the existing static_assert (`qwen_3_6/config.h:96-98`).
- **Why.** **Perf level on B64+ (state-bound knee)**: GDN state is 310 MB/seq/
  token fp32 and overtakes weights past ~B95 (`QWEN36_BF16_SPEED.md:11-13,45-51`);
  bf16 halves the stream and is "the biggest single lever the model has at
  B64+" (`QWEN36_BF16_SPEED.md:20-23`). It is a numerics question on a
  compounding recurrence, so it lands as a precision-contract variant.
- **Delta.** +~90 lines (variant + contract card + pinning test).
- **Owner.** qwen38-27b (me, contract) + CUDA-KERNELS agent (variant).
- **First step.** Numerics study (per-token compounding drift) → precision
  contract card → variant kernel.

---

## What I need
- **Coordinator:** land the #1 contract PR (gate-breaking); fix the
  `fleet_registry.json` `qwen27b` entry; approve the shared packer base so
  qwen36/qwen38/qwen38-27b don't fork three ways (`DRY_CONSOLIDATION_PLAN.md:14-17`).
- **CUDA-KERNELS agent:** cards for #7/#8/#10 (decompress-in-GEMM, sharded-delta
  collective, bf16-state GDN).
- **Scheduler / kv-cache / speculation agents:** confirm the shared core/table/
  tree each of #3/#4/#5 consumes (already proposed in their docs; sign-off noted
  where landed).
- **The exact Qwen 3.8 27B checkpoint id+revision** — everything re-bases from it.

## Report
Changed path: `docs/TOP10_QWEN38-27B.md` (new, proposal only). Verified by
grep/read of COORDINATION.md, the qwen36 family/module/adapter/stagepack, the
TP4 build/deploy scripts, the four landed cross-model specs
(PROPOSAL_ADMISSION_CORE / PROPOSAL_KV_SEAM / PROPOSAL_DSV4_TREE_ADOPTION /
KERNEL_CONTRACT_CARDS), the perf estimators, QWEN36_TP4_PERF.md and
QWEN36_BF16_SPEED.md, fleet_registry.json, must_work_targets.json, and
PERFORMANCE_STATUS.md (no qwen rows). No commits, no pushes.
