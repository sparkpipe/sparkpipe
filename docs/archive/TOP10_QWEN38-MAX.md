# Qwen 3.8 Max (2.4T-A95B) — top-speed assessment + TOP-10 improvements

Owner: qwen38-max MODEL agent · Area: resident stage, TP-sharded KV (head-parallel
attention), GDN state paging, packed-not-served MTP. Proposal/docs only — no
commits, no pushes. Every claim grep/read-verified against `afb43a8`; cited
`file:line`.

---

## Top-speed assessment (<=1 page)

### What exists (landed, compile-verified)

- **Resident decode stage** `modules/qwen38_resident_decode_stage/` is complete for
  decode: stage-pack load + coverage validation, GDN + gated attention + routed
  FP8-E4M3 MoE + shared expert, argmax head, hidden-transport PP contract. GDN +
  attention kernels are math-verified against the pinned reference
  (`docs/QWEN38_MAX_MATH_AUDIT.md`). Decode-only by design: "Prefill, MTP and
  speculation fail closed" (`spark_qwen38_resident_decode_stage_module.c:4-5`).
- **TP machinery landed, half-activated.** Expert-sharded MoE (routed experts split
  `tp_rank×512/tp`) + one residual all-reduce per layer via
  `SparkTpDeviceCollectiveSubmitBf16` (`module.c:1137`, `:1538-1546`); head-parallel
  attention kernels already take `(tp_degree,tp_rank)` (`module.c:1307-1308`). But the
  attention/GDN **linear views are still full-width** — "head-parallel attention/GDN
  slicing is the next increment" (`module.c:1185-1189`) — so KV is still 4×-replicated.
- **KV tier wired, dormant.** Module reads logical/physical page budgets, opens a
  pluggable store, pages KV blocks with round-robin eviction + dirty write-back
  (`module.c:611-682`); work-control GDN record path is built and tested
  (`model-families/qwen38/src/spark_qwen38_work_control.c:75-80,200-210,260-264`).
- **MTP packed-not-served.** MTP layer weights are bound + coverage-verified
  (`stagepack_format.h:27,60-63`; `module.c:406-445,523-526`); firmware declares the
  draft view + 8 draft tokens (`..._firmware.h:53,55,429`) but no draft path runs.

### What is measured

Single-spark real-FP8-pack numbers (`docs/QWEN38_MAX_PERF.md:7-17`,
`QWEN38_MAX_KERNEL_SPEED.md`): B=1 ≈ 8.44 ms/layer → **~1.29 tok/s** per request
(92 layers); replicated B=256 ≈ 10.5 tok/s aggregate before the m-loop win, ~39
tok/s at TP4xPP4 (`QWEN38_MAX_PERF.md:47`). Grouped-expert m-loop landed −69% MoE
at B=256 (`QWEN38_MAX_KERNEL_SPEED.md:38-52`). No qwen38 entry exists in
`PERFORMANCE_STATUS.md` (grep-empty); the fleet TP4xPP4 end-to-end window has not run.

### What is missing (the honest gaps)

1. **Admission is a stub.** `SparkQwen38ResidentDecodeStageAdmit` returns
   `SPARK_STATUS_UNSUPPORTED` (`module.c:1242-1251`); the serving wrapper collapses
   every non-BUSY rejection to `CAPACITY_EXCEEDED`, dropping DEADLINE/UNSUPPORTED
   (`spark_qwen38_serving_adapter.c:841-870`, the ternary at `:868`).
2. **KV is 4×-replicated** (TP), head-parallel kernels exist but attention linear
   views are unsliced and the residual all-reduce is blocked on
   `SparkTpDeviceCollectiveCreate → CAPACITY_EXCEEDED` (`docs/QWEN38_MAX_TP16.md`).
3. **GDN state does not page.** Record size is a 4096-byte placeholder
   (`module.c:35,663,678`) while the real per-lane state is
   128×128×128 fp32 = **8 MiB/layer/lane** (`module.c:1600`), ×69 GDN layers ≈ 552
   MiB/lane — the dominant resident cost, not in the tier.
4. **Prefill refused** (`module.c:4`, `serving_adapter.c:964`) → prompts run as N
   decode steps (~1.3 prompt-tok/s).
5. **MTP weights are dead weight** — packed and bound, never served.

**Bottom line: accurate-but-slow (METRIC level 1), structurally near ready to
leave it.** The math is correct and the machinery (TP collective, KV tier,
work-control, MTP bind) is built; the work left is activation + sizing + the
capacity decision, not new foundations.

---

## TOP-10 improvements (ranked)

### 1. Fill the admission policy table (adopt PROPOSAL_ADMISSION_CORE.md)
- **What:** Replace the stub `SparkQwen38ResidentDecodeStageAdmit` (`module.c:1242-1251`)
  with a `SparkAdmissionPolicyTable` + `SparkAdmissionEvaluateShape`, and collapse
  `SparkQwen38ServingAdmit` (`serving_adapter.c:841-870`) to the shared
  `SparkAdmissionRequestFromSubmission` + `SparkAdmissionEvaluateAndApply`.
- **Why it is right:** DRY win — qwen38 is Phase-D order (4) of the landed proposal:
  "it *starts* from the core instead of migrating: fill the table, get a real admit
  for free" (`docs/PROPOSAL_ADMISSION_CORE.md` §3.2). Also fixes the rejection-mapping
  bug (loses DEADLINE/UNSUPPORTED_SHAPE) at `serving_adapter.c:868`.
- **Code-size delta:** serving wrapper −30, module admit +~15 (table) −6 (stub) → net ≈ **−20 lines**.
- **Owner:** qwen38-max agent (module + serving adapter).
- **First step:** Write the table (`max_active_sequence_count` = firmware max,
  `max_sequence_positions` = 262144, `DECODE_EQUALS_SLOTS` + `PREFILL_SINGLE_SLOT`
  flags) and pin `SparkAdmissionEvaluateShape` against the existing serving fixture.

### 2. Fill SparkKvModelTable (adopt PROPOSAL_KV_SEAM.md §3.5)
- **What:** Ship the qwen38 fill of the token-free `SparkKvModelTable`: 23 full-
  attention layers, 64 Q / 4 KV heads, head dim 256, token record 2×2048 bf16
  (`spark_qwen38_model.h:47-48,53-54`), plus a `SparkQwen38PageCopy` primitive.
- **Why it is right:** DRY win — the landed seam makes the four adopters symmetric;
  today qwen38 links `stage_kv_client.c`/`kv_store.c` but ships no fill
  (`docs/QWEN38_MAX_KV.md:42-48`). It is the prerequisite for items 3–4 and for
  DRIVER_OWNS_KV → JIT-KV parity.
- **Code-size delta:** +1 model-family header (~40 lines) + one copy primitive (~20) → **+~60 lines**.
- **Owner:** qwen38-max agent.
- **First step:** Add `model-families/qwen38/include/sparkpipe/spark_qwen38_kv_table.h`
  mirroring `spark_k3_kv_geometry.h`, then a `test_qwen38_kv_cache.c` on the
  `test_k3_kv_cache.c` pattern proving zero model symbols cross the seam.

### 3. Size the GDN-state record and page it through the tier
- **What:** Replace the 4096-byte placeholder (`module.c:35,663,678`) with the real
  lane record (128 value heads × 128×128 fp32 per GDN layer = 8 MiB/layer,
  `module.c:1600`), stage it into `kv_gdn_staging`, and emit it via the already-built
  work-control GDN record path (`work_control.c:200-210` PUT / `:260-264` restore).
- **Why it is right:** This is the named "GDN state paging" gap and the dominant
  resident cost (~552 MiB/lane at full width, `docs/QWEN38_MAX_KV.md:82`). Paging
  it converts per-lane residency from ~half a GiB to the KV window, buying the
  resident capacity that every throughput level above "accurate-slow" requires.
- **Code-size delta:** module +~40 (record sizing + stage/restore call), work-control 0 → **+~40 lines**.
- **Owner:** qwen38-max agent.
- **First step:** Fix `state->kv_plan.gdn_record_bytes` to the true per-lane bytes and
  allocate `kv_gdn_staging` to match, then A/B the tier round-trip in
  `test_qwen38_work_control`.

### 4. Activate head-parallel attention (the TP-sharded KV fix)
- **What:** Slice the attention linear views per rank (query/key/value row slices +
  o_proj row-parallel) so each rank projects its 16 Q heads and its one KV head,
  and let the already-landed head-parallel kernels (`module.c:1307-1308`) write a
  per-rank 1/4 KV cache (the 4×-capacity win, `docs/QWEN38_MAX_KV.md:122-125`).
- **Why it is right:** Performance level — removes the 4× KV replication *and* the
  attention collectives (T2 in `QWEN38_MAX_PERF.md`), moving the driver from
  "accurate-slow" toward the TP16 "match/exceed SOTA" schedule (same all-reduce,
  but KV capacity ×4).
- **Code-size delta:** module +~150–250 (rank-sliced view fills + dispatch), kernels ~+100 → **+~250–350 lines**.
- **Owner:** qwen38-max agent (module) + CUDA-KERNELS agent (head-sliced projections).
- **First step:** Land the strided o_proj + q/k/v row-slice views and re-enable
  `tp_degree>1` attention in the two-process smoke (currently expert-only).

### 5. Unblock SparkTpDeviceCollectiveCreate (CAPACITY_EXCEEDED)
- **What:** Resolve the create failure that blocks every `tp_degree>1` collective:
  compare the qwen38 config field-by-field with dsv4's working
  `SparkDsv4ModuleInitializeTpCollective` and check the backend .so buffer/queue
  sizing against `max_active_sequence_count 512 / credit_count 2 / route_count 1`
  (`docs/QWEN38_MAX_TP16.md` "Blocked" section).
- **Why it is right:** It gates items 4 and 10; it is config/backend-contract
  reconciliation, not new design, so it buys the whole TP16 path at near-zero code.
- **Code-size delta:** ~0–20 lines (config/threshold match) → **+~20 lines**.
- **Owner:** qwen38-max agent + transport owner.
- **First step:** Reproduce the two-process smoke with dsv4's rail-hosts/threshold
  topology fields and log which `Validate*`/credit-plane step diverges.

### 6. Prefill attention kernels + module prefill acceptance
- **What:** Add chunked causal GQA over `[base, base+token_count)` that writes paged
  K/V blocks and emits per-position head outputs, then accept `PrefillFrameView`
  (already declared, `firmware.h`; adapter already advertises `CAPABILITY_PREFILL`
  at `serving_adapter.c:204`).
- **Why it is right:** Performance level — prefill today is decode-step-bound
  (~1.3 prompt-tok/s) and is the other half of serving; the phase-2 estimate is
  ~1K–5K tok/s (`docs/QWEN38_MAX_PHASE2.md:2`, `QWEN38_MAX_PERF.md:54`).
- **Code-size delta:** kernels +~200–300, module +~60 → **+~260–360 lines**.
- **Owner:** qwen38-max agent + CUDA-KERNELS agent.
- **First step:** Land the chunked-causal GQA + paged K/V-write kernel and its
  module frame path; gate on the refused-prefill smoke flipping to status 0.

### 7. Capacity decision: MXFP4 requantize routed experts
- **What:** Pack routed experts as MXFP4-E2M1 (codec code 4 already accepted by the
  firmware) instead of the vendor FP8, the documented capacity decision
  (`docs/QWEN38_MAX_AUDIT.md` §4).
- **Why it is right:** 2.31 TiB FP8 = ~148 GB/rank > ~107 GB usable → the fully-
  resident TP4xPP4/TP16 plan does not fit; MXFP4 ≈ 74 GB/rank fits *and* unlocks the
  existing sm121 native B1 expert kernels (double win, `QWEN38_MAX_PERF.md:206`).
  Buying "resident at all" is the precondition for every level above 1.
- **Code-size delta:** packer change only (small), module kernel-path switch → **+~30–60 lines**.
- **Owner:** qwen38-max agent (packer) + coordinator (quality stance decision).
- **First step:** Emit an MXFP4 variant pack and A/B the two-step decode output
  against the FP8 pack for identical routing + logits within the certified bound.

### 8. B=1 tensor-core expert path (FP8 kernel or sm121 MXFP4 adoption)
- **What:** Give the B=1..8 scalar MoE a tensor-core path: either a new FP8_E4M3
  B1 grouped kernel, or (post item 7) adopt `SparkLmSm121B1ExpertW13/W2Kernel`.
- **Why it is right:** Performance level — B=1 streams ~500 MB/layer of expert
  weights per token (the ~1.29 tok/s root cause); the documented lever is "the next
  5-10x at B=1" (`QWEN38_MAX_PERF.md:25-26`). This is the single largest
  "accurate-slow → match SOTA" single-stream move.
- **Code-size delta:** common kernel +~150–250 (coordinated) → **+~150–250 lines**.
- **Owner:** CUDA-KERNELS agent (common code) + qwen38-max (consume).
- **First step:** File the kernel contract (shape FP8_E4M3 block-128, B=1/2/4) per
  `docs/KERNEL_PLAYBOOK.md`; measure scalar vs tile at B<8 to pick the crossover.

### 9. Serve the packed MTP layer (speculation)
- **What:** Turn the bound-but-dormant MTP weights (`module.c:406-445,523-526`) into a
  draft: a 1-layer MTP forward on the `MTP_DRAFT_AFTER` frame (`firmware.h:53,55,429`),
  verified by the shared `LmSpeculativeVerifyGreedyKernel` (`docs/KERNEL_CONTRACT_CARDS.md` V-01).
- **Why it is right:** Performance level — the MTP is already packed and bound, so the
  weight cost is sunk; serving it buys ~2-3× per-request tokens/s (standard MTP gain),
  the "match/exceed SOTA" per-request lever. qwen38 currently has **no drafter cards** —
  they must be requested via the card-editing process (`KERNEL_CONTRACT_CARDS.md` §4).
- **Code-size delta:** draft forward + verify wiring +~500 (largest item) → **+~500 lines**.
- **Owner:** qwen38-max agent + SUBSYSTEM speculation agent (tree/verify seam).
- **First step:** Declare the qwen38 MTP drafter shape card (block 8, hidden 8192,
  heads 64/4, vocab 248320) and confirm the MTP layer reuses target-layer kernels.

### 10. Finish TP16 scale-out (GDN channel slicing + rank-local packs + T4)
- **What:** Complete the T1–T4 schedule: GDN channel slicing (per-rank q/k/v
  contiguous views), rank-local packs, and the split-ring/async/gpudirect overlap
  (`QWEN38_MAX_PERF.md` §2, `QWEN38_MAX_TP16.md` "Next increments").
- **Why it is right:** Performance level — this is the "exceed SOTA" ceiling:
  ~450–500 tok/s output / ~6 tok/s B=1 at TP16 (vs ~39 tok/s replicated today),
  each rank streaming 1/16 the weights (`QWEN38_MAX_PERF.md:1b0,§4B`).
- **Code-size delta:** module +~200–300 (GDN slicing + dispatch), packer rank-local → **+~250–350 lines**.
- **Owner:** qwen38-max agent (+ transport owner for the gpudirect spec).
- **First step:** Land the per-rank GDN qkv contiguous views + runtime conv channel
  count, then the tp=2/4 → tp=16 smoke ladder before the fleet window.

---

## Cross-model items that already have landed specs (applied above)

- **PROPOSAL_ADMISSION_CORE.md** → item 1 (qwen38 is the "start from the core"
  Phase-D adopter; its stub at `module.c:1242-1251` is cited verbatim in the proposal).
- **PROPOSAL_KV_SEAM.md** → items 2 & 4 (§3.5 names qwen38's exact target: fill the
  table, wire the store client, activate head-parallel 1/tp KV sharding).
- **KERNEL_CONTRACT_CARDS.md** → items 8 & 9 (shared verifier kernels V-01/V-02 apply
  to MTP; qwen38 has no drafter cards yet and must request them per §4). The
  FP8-E4M3 expert shape has no card yet — item 8 files it.
- **PROPOSAL_DSV4_TREE_ADOPTION.md** → does not apply directly (it is DSV4's
  DSpark/Markov chain); qwen38's MTP item 9 should instead consume the neutral
  `spark_speculation_tree.h` machinery that proposal demonstrates, reusing the
  shared verifier kernels rather than the DSV4 drafter.