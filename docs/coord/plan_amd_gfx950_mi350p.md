# SparkPipe's first AMD target — implementation plan for `rocm.gfx950.mi350p`

Status: **PLAN** (engineering execution plan, not a contract). Written against
the frozen hwiface v1 contract + REV1 and the current unified head. Where this
plan paraphrases contract documents, the frozen text wins; reopening any frozen
decision is a coordinator act (`hwiface_v1_freeze.md`, freeze semantics).

Authority order: `SPEC.md` > `hwiface_v1_freeze.md` > `hwiface_v1.md` >
`.agents/coord/hwiface_amd_contract_summary.md` >
`.agents/coord/hwiface_amd_impl_guide.md` > **this plan**.

Fact tags used throughout, matching repo convention:

- *(measured)* — read out of this tree today (file:line cites in Appendix A).
- *(analytical)* — engineering judgment derived from the contract.
- *(advisor-provided)* — gfx950/MI350P hardware facts; unverified locally.

---

## 0. Summary

SparkPipe's first AMD deployment is DSV4 resident decode on MI350P
(gfx950, CDNA4), implemented as **one static archive**,
`dsv4_rocm_gfx950_mi350p`, behind the frozen seven-island ABI
(E0, L1–L5, F1) plus a `spark_hw_rocm_*` runtime-primitive family over HIP.
The work decomposes into four workstreams that map onto the frozen step gates
S6/S7/S8 with no gate changes:

| Workstream | Frozen step | Focus |
|---|---|---|
| A. HIP runtime backend | S6 | `spark_hw_rocm_*` archive, descriptor, fail-closed guard |
| B. MXFP4 native layouts | spans S6–S8 (designed up front) | packed-layout strategy below the seam |
| C. One complete layer bring-up | S7 | E0 + L1–L5 + F1 end-to-end, TP1 |
| D. RCCL TP4 | S8 | NCCL-backend analog of `spark_tp_device_collective.h` |

Everything links statically against the **same unmodified `dsv4_core`
objects** the CUDA deployment uses; there is no hot-path indirection anywhere
(R2), and deployments are homogeneous per vendor (R6/R7).

---

## 1. What binds this plan (do-not-renegotiate list)

From the frozen contract, restated as one line each — full text governs:

1. Seven islands exactly (E0/L1/L2/L3/L4/L5/F1); island internals free (R1);
   capture topology entirely target-internal.
2. Static link, direct calls, missing symbol = link error; same core objects
   both targets (R2/F3 rule 2).
3. Handles opaque — move, never inspect (R3). Sealed-route batches are fixed-
   offset slot-workspace views allocated at slot creation, never dynamically
   allocated (freeze §F2).
4. No numeric constant crosses either direction of the seam (R4/F4); the
   `spark_lm_kernels.cuh` landmine rule: no ROCm TU may evaluate a
   `SPARK_LM_SM121_*` selector.
5. Aggregation/seal policy is core-owned; the backend realizes an already-fixed
   logical route (R5).
6. DSV4-only scope for v1 seams (R6); homogeneous deployments only — no
   mixed-vendor collective, ever (R6/R7).
7. Tolerances come from the shared recipe, identical numbers to CUDA's (Q4);
   token ids match CUDA exactly with deterministic lowest-index tie-break;
   reduction trees fixed per shape bucket (C2/C3).

This plan adds sequencing, technical mappings, and decision points below those
frozen lines. It does not add primitives, islands, descriptor fields, or
tolerances.

---

## 2. Dependency audit — tree state today *(all measured)*

Nothing AMD-facing has landed yet. This is the honest starting position and it
shapes the sequence:

1. **No neutral primitive header exists**: no `spark_hw*` file anywhere in
   the tree; `include/sparkpipe/` contains no `spark_hw_iface.h`.
2. **No link-unit split exists**: no `dsv4_core` /
   `dsv4_cuda_sm121_gb10` archives; the module still builds as
   `libdsv4_resident_decode_stage.a` with the CUDA source inside.
3. **No ROCm toolchain hooks exist**: zero `rocm`/`hipcc`/`gfx950`
   references in `Makefile`, `sources.mk`, or `tools/devcycle`.
4. **The landmine header is unsplit**: `spark_lm_kernels.cuh` still carries
   its SM121 selectors inline (122 MXFP4/E8M0 references in one file).
5. **The fleet is all-CUDA GB10** (CUDA 13 on every Spark,
   `COORDINATION.md` fleet facts). No MI350P node is registered.

Consequence: **S6 cannot start on code alone.** The AMD track consumes these
CUDA-track deliverables (owners: hwiface/CUDA sessions via coordinator queue):

- [ ] `spark_hw_iface.h` landed verbatim per freeze §F2 (S1).
- [ ] Island entries named in the host module; GB10 constants moved into the
      descriptor (S2) — verified by C1 byte-identity.
- [ ] Build split producing `dsv4_core` + `dsv4_cuda_sm121_gb10` under SPEC
      §2 content addressing (S2/S3), with P1 baseline captured *before* S1
      (freeze §F6) so the refactor's no-regression gate stays falsifiable.
- [ ] `spark_lm_kernels.cuh` split/guard such that no ROCm TU can evaluate a
      `SPARK_LM_SM121_*` selector (freeze §F4 landmine rule) — explicit S6
      precondition; if absent when S6 starts, escalate through the coordinator
      queue rather than including the header defensively.
- [ ] MI350P host provisioned: ROCm release pinned, RCCL included, reachable
      like the existing sparks; added to `tools/devcycle/fleet_registry.json`
      as a new node class (it cannot join the GB10 bands — different vendor,
      homogeneous-only rule).

*(Box limitation, precedent `.agents/coord/hwiface_evidence_r2c2_rocm_probe.hip`):
the authoring workstation has no ROCm toolchain. Until hardware lands, all
AMD-side work is limited to `hipcc --offload-arch=gfx950 -fsyntax-only`
probes executed elsewhere or plain-C++ include-surface proofs locally. Every
evidence note must classify honestly what ran where — same discipline the R2-C2
probe used.*

---

## 3. Workstream A — HIP runtime backend (S6)

### 3.1 Environment pinning *(analytical, to be confirmed on hardware)*

- Toolchain: ROCm release with gfx950 (CDNA4) support;
  `hipcc --offload-arch=gfx950`. Pin the exact release string in the build
  receipt; treat it as part of the artifact identity story (rebuild with a new
  ROCm ⇒ new bytes ⇒ new validation identity, which SPEC §2 handles naturally).
- Archive compiles as `.hip`; island entries and `spark_hw_rocm_*` symbols
  are plain externs with C linkage.
- RCCL is **not** an S6 dependency; do not link it until S8 (guide §0).

### 3.2 Primitive mapping

The normative mapping worksheet is impl-guide §3.2 (HIP substrate per frozen
§F2 family). This plan adds only the engineering deltas that worksheet does
not settle:

| Item | Decision for this plan |
|---|---|
| Memory pool | `spark_hw_memory_pool_alloc/free` wraps `hipMalloc`/`hipFree` behind the opaque `SparkHwMemory` handle first; migrating to stream-ordered pool backing later is internal-free (R1 analog for primitives). No pool API change ever surfaces. |
| Host callback | `spark_hw_queue_enqueue_host_callback` → `hipLaunchHostFunc` is THE external completion mechanism; prove it under a stress loop before any island work (guide §3.4.2) — silent completion deadlocks are the highest-cost failure mode downstream. |
| Sync discipline | `queue_synchronize` on failure paths only; success path completes exclusively via the single stream-ordered callback. Any sync-on-success found later is a defect, not a tuning issue (frozen semantic note, freeze §F2). |
| Events | created with timing disabled; carry L4/L5 fork-join and cross-queue order only. |
| Graphs | relaxed-mode capture preserved via the flag argument; instantiate/launch/exec-destroy map 1:1 onto HIP graphs; topology freely chosen (§3.3 v1). |
| Read-ahead | `spark_hw_read_ahead` is a non-blocking kick; contents target-defined (E0 weight read-ahead start). |

### 3.3 Descriptor and guard

Fill once in `ModuleInitialize`, never consulted hot-path (frozen):

- `target_id = "rocm.gfx950.mi350p"` (frozen string, freeze §F3);
- `multiprocessor_count` = queried CU count; `max_dynamic_shared_bytes` =
  the gfx950 LDS limit (**not** GB10's 101376); `wavefront_lanes = 64`;
- `capability_major/minor`: pin from the actual device query on hardware,
  then freeze as constants of this archive (advisor-provided values are not
  trusted blindly — measure, then pin).

Fail-closed guard mirrors the CUDA pattern *(measured by guide §3.3:*
`SparkDsv4RequireNativeSm121` at `.cu:139`, called :161/:2785, cached per
executor thread, plus an image-level PTX trap*)*: device-attribute check that
fails closed on anything but gfx950, cached per executor thread, plus the
image-level analog of the PTX trap. Two layers, both fail closed.

### 3.4 Build integration *(analytical; names follow measured patterns)*

- New make path emitting `dsv4_rocm_gfx950_mi350p.a` — a **normal static
  archive** (thin forbidden, SPEC §2) containing the `.hip` objects, the
  `spark_hw_rocm_*` implementations, the gfx950 constants, and the guard.
- Module-target-key analog follows the existing flash key
  `cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16`
  *(measured, module Makefile:9)* →
  `rocm.gfx950.mi350p.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16`.
  The batch-variant ladder *(measured:*
  `MODULE_BATCH_VARIANT_BUCKETS ?= 1 2 4 8 ... 1024` in `Makefile.pro`*)
  carries over unchanged — buckets are portable-core policy, not sweepable.
- Publication follows SPEC §2 unchanged: hash-and-copy exact bytes, validate
  once per `module-target-key + validation-key`, read-only on pass, atomic
  activate. Model compilation/driver loading never validates (SPEC §2).

### 3.5 Bring-up order within S6 *(analytical, from guide §3.4)*

1. Descriptor + guard + memory/copy/memset families + queue/event, smoke
   kernel through the queue abstraction.
2. Host-callback completion stress loop.
3. Graph capture/instantiate/launch round-trip on a toy capture (relaxed).
4. Landmine-split verification, then declare S6 ready for its C2 gate.

### 3.6 S6 acceptance

- Every §F2 primitive present and exercised or honestly classified
  unexercised in the milestone note.
- C2 exactness on integer islands through the primitive layer.
- Validation record published per SPEC §2 (cold build included).

---

## 4. Workstream B — MXFP4 native layouts

MXFP4 is the headline lever of this port and is designed **before** S7 coding
starts, even though its payoff lands at S8 scale.

### 4.1 Measured format facts (the starting point)

1. Expert weights are already MXFP4 E2M1 in the flash deployment:
   format code `SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 = 3u`
   *(lm_kernels.cuh:48)*; the flash module-target-key literally says
   `expert_mxfp4` *(Makefile:9)*; the shared linear machinery runs
   bf16/fp8/mxfp4 from one body *(`.cu`:167 comment)*.
2. The stage pack stores **e8m0 scales, one byte per block per row:
   fp8 weights block 128 columns, fp4 experts block 32**, other tensors
   scale-free *(stagepack_format.h:329–330)*.
3. Activations are FP8 E4M3 with UE8M0 scale codec — enforced by static
   asserts in the CUDA TU *(`.cu:40–42`)*.
4. Logical weight inventory is shared (boundary row 8); **packed layout +
   dequant path are target-private** (v1 §5.2 table; guide §4 L4).

### 4.2 Why this is the lever *(advisor-provided hardware fact + analytical bridge)*

gfx950/CDNA4 provides native block-scaled MXFP4 matrix operations
(advisor: "native MXFP4"). The pack's block structure — E2M1 nibbles with one
E8M0 scale byte per 32-element block — is the block-scaled operand shape that
class of hardware consumes. The analytical conclusion, to be confirmed against
the toolchain's actual operand-layout documentation on hardware: **the AMD
target may be able to feed expert weights to MFMA without the dequant-to-a-wide-
format staging the CUDA side performs**, removing both bandwidth and ALU cost
from the L4 hot loop.

### 4.3 Layout decisions owned by this workstream (below the seam only)

| # | Decision | Plan position |
|---|---|---|
| B1 | Direct-MFMA operand layout | Derive nibble packing order, K/N major-ness, and scale-byte placement from the gfx950 block-scaled instruction's expected operand layout. Goal state: **zero-repack** — load stagepack bytes as-is. If the pack's byte order mismatches the operand layout, prefer a K-major repack at load time over a hot-path shuffle. *(analytical; confirm against ISA docs on hardware)* |
| B2 | If repack is unavoidable | Happens once at pack-load time inside the target archive, never on the hot path, never portable-side. Repacked artifacts are derived data keyed by the content-addressed pack hash; they are cache, not contract. |
| B3 | Per-shape-bucket paths | Decode buckets are core policy; the target selects its own tile/shape/dequant-strategy per bucket (F4 sweep grant). Fix the reduction tree **per bucket**, document it once, stop changing it — C3 tolerances assume a fixed tree (guide §4 L1). |
| B4 | Activation-side operands | Activations arrive bf16 / FP8-E4M3(UE8M0 codec). Decide conversion points (A-operand to MFMA class) per GEMM shape; keep them target-internal. |
| B5 | Landmine interaction | All `SPARK_LM_SM121_NATIVE_WEIGHT_FP8/_MXFP4` format codes and tile selectors stay unreachable from ROCm TUs (F4); the ROCm archive defines its own format-code usage over the neutral machinery after the header split. |

### 4.4 Validation consequences

- L4 grouped-GEMM output remains **C3 within the shared tolerance set** — the
  MXFP4 choice changes nothing about tolerance authority (Q4).
- Route realization stays **C2 bit-exact** and is completely independent of
  weight layout: indices/group offsets/expert assignment are integer work fed
  by core-computed routes (freeze §F1 L4 split verdict).
- KV payload note: flash config today is `kv_bf16` *(measured,
  Makefile:9)*; whether L3 emission quantizes is decided by the L3
  emitted-field table at recipe time (guide §5 precondition 2) — the MXFP4
  weight layout does not preempt that table.

---

## 5. Workstream C — One complete layer bring-up (S7)

Scope is frozen (guide §5): exactly E0 + one layer through L1→L5 (forked L4/L5
legs joined pre-hcPost) + F1, end-to-end, **TP1**. Not attention-scale, not
grouped-MoE-scale, no RCCL.

### 5.1 Preconditions (blocking, checkable)

1. S6 accepted; landmine split verified landed (§2 checklist above).
2. L3 emitted-field table published (per-field: unrounded copy ⇒ C2,
   quantized ⇒ C3) — authored by the AMD side, recipe-time artifact.
3. Golden frame list per bucket published — also AMD-side, also due now.
4. Recipe ID registered for `rocm.gfx950.mi350p` carrying the shared
   tolerance set (identical numbers to CUDA's, Q4).

### 5.2 Bring-up order *(analytical)*

Rationale: exercise the two cheapest proof classes first so plumbing bugs are
found while the surface is small, then accumulate C3 complexity.

1. **E0 + L5 first.** E0 proves integer exactness through the whole primitive
   stack; L5 is the simplest C3 accumulation island and doubles as the
   tolerance-plumbing shakeout (guide §4 L5 recommends exactly this).
2. **F1 skeleton next** — final hcPost + screened argmax with the engineered
   tie case constructed deliberately *(analytical: construct equal-score
   candidates; don't wait to observe a natural tie)*, proving lowest-index
   tie-break survives wave64 reductions before attention complexity arrives.
3. **L1 then L4**, L4 driven with synthetic sealed-route batches over the
   slot-workspace views (no aggregation machinery needed at TP1 scope).
4. **L2/L3 last** (indexer + paged-cache attention + ring-page emission — the
   widest kernel surface), uncaptured first.
5. Wrap stable sequences in HIP graphs only after uncaptured correctness;
   choose exec boundaries freely (topology is target-internal).
6. Full-layer oracle run: `tools/verify_dsv4_ga_reference_fixture.py`
   against `qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128`
   (`after_layer_2.bf16le` + `prompt_tokens.u32le` + manifest)
   *(measured: all present)* — same fixture as CUDA.

### 5.3 Acceptance (frozen, restated as checklist)

- [ ] Every C2 half bit-exact: E0 gather; L3 page addresses / emit counters /
      ring indices (+ unrounded payload fields per the field table); L4 route
      realization; F1 token ids / maxloc pack-unpack / feedback integers.
- [ ] Every C3 island within shared tolerances: L1, L2, L4 grouped-GEMM, L5,
      F1 scores and quantized outputs.
- [ ] Absolute invariants: token ids match CUDA exactly; tie-break
      demonstrated on ≥1 engineered tie case.
- [ ] Evidence artifacts: validation record keyed
      `module-target-key + validation-key`; golden-frame hashes; milestone
      note honestly classifying what ran on MI350P hardware vs syntax-probe.

Explicitly out of S7 scope: performance (row-9 qualification, first exercised
at S8), TP>1, mixed-vendor anything.

---

## 6. Workstream D — RCCL TP4 (S8)

### 6.1 Surface facts *(all measured off `include/sparkpipe/spark_tp_device_collective.h`, ABI 12)*

- Backend enum has exactly two values: `_HIDDEN_TRANSPORT` (0) and
  `_NCCL` (1) (:30–31). ROCm implements the **NCCL-kind backend over RCCL**
  (NCCL-aligned API — advisor fact).
- Operations: `ALL_GATHER` and `ALL_REDUCE_SUM_BF16` (:32–33).
- Algorithm mask names three algorithms: recursive-doubling,
  counter-rotating split-ring, direct all-to-all (:34–41); topology struct
  carries rank hosts, rail count, step rail indices (:171–187).
- The TP4 combine function's contract is written down: reproduce the
  recursive TP4 BF16 tree — round(0+1), round(2+3), then round(local+remote)
  (:144–154).
- Submissions are non-blocking, carry `void *cuda_stream` (:112), a
  stream-ordered-completion flag (:47–50), and an optional per-submission
  element-count override that **only the NCCL-kind backend honours** while the
  hidden-transport tier keeps the pre-registered frame (:98–108).
- Credit model: ≤4 steps × 64 credits, send/receive device+transport pointer
  pairs (:21–24, :63–75); phase machine PHASE_FREE…PHASE_RELEASE_PENDING
  (:52–61); failure injection and operation-phase queries (:286–299);
  memory mode probed device vs mapped-host (:254–257).

### 6.2 Mapping plan *(analytical unless noted)*

| Surface element | RCCL realization |
|---|---|
| Communicator | `ncclCommInitRankConfig` over the homogeneous TP4 rank set; rank order = TP rank order; unique `collective_identifier` reused as the communicator rendezvous context. One communicator per deployment collective instance — no runtime negotiation beyond init. |
| Streams | `void *cuda_stream` slots take `hipStream_t`; enqueue `ncclAllReduce`/`ncclAllGather` on that stream so ordering composes with `spark_hw_queue` semantics. |
| `ALL_REDUCE_SUM_BF16` | `ncclAllReduce(..., ncclBfloat16, ncclSum, ...)`. Element override maps directly onto nccl's per-call count (the header already anticipates exactly this, :103–107). |
| `ALL_GATHER` | `ncclAllGather` with the same stream discipline. |
| Credit bindings | Register the binding buffers with RCCL at create time (buffer registration where the installed RCCL provides it; otherwise rely on its internal caching). Bindings stay opaque above the seam. |
| Completion | Stream-ordered callbacks via the same `hipLaunchHostFunc` mechanism as the primitive family; preserve the submission phase machine and completion-honesty fields exactly. |
| Memory mode | Target `MEMORY_MODE_DEVICE` first; mapped-host is a GB10-style unified-memory tier *(analytical: not the MI350P default assumption — verify on hardware before relying either way).* |

### 6.3 Decisions this workstream must record at S8 design time

1. **Algorithm-mask honesty.** The three named algorithms are part of the
   config surface. Decide per algorithm: honor via RCCL selection controls, or
   return `UNSUPPORTED` at apply-topology/create time. **Never silently
   substitute** — a substituted algorithm would corrupt per-target performance
   qualification (boundary row 9) and any ordering assumptions.
2. **Combine-TP4 tree.** Keep the documented recursive tree contract
   (:144–154) as the reference semantics. First implementation: three staged
   RCCL ops reproducing round(0+1)/round(2+3)/round(local+remote) on the
   submission stream. A custom fused kernel over RCCL transport is a later
   optimization, judged by the same tree-order contract so cross-target
   comparability holds.
3. **Homogeneity enforcement.** `Create` fails closed unless the rank set is
   single-vendor AMD (R6/R7). There is no mixed mode and none is planned.
4. **Failure paths.** Map `RequestFailure`/`RequestOperationFailure` onto
   RCCL abort/error handling; phases must reach a terminal state honestly.

### 6.4 TP4 acceptance

- [ ] C2 halves unaffected and re-proven at TP4 (route realization, counters).
- [ ] Collective results bit-comparable under the BF16 sum ordering contract
      the combine functions document.
- [ ] Per-target perf qualification executed here for the first time: own
      thresholds, own baseline artifact (the P1-AMD analog), buckets matching
      the portable bucket set. Capture baselines **before** optimization churn.

---

## 7. Milestones, gates, evidence

| Step | Scope | Blocking preconditions | Gate | Evidence |
|---|---|---|---|---|
| S6 | `spark_hw_rocm_*` archive + descriptor + fail-closed guard | §2 checklist (header, split, landmine, hardware) | C2 exactness on integer islands | validation record; smoke-classification note |
| S6.x | MXFP4 layout decision set (B1–B5) resolved on hardware | S6 environment live | documented decision note per §4.3 | layout memo with ISA-doc citations |
| S7 | One complete DSV4 layer, TP1 | S6 accepted; L3 field table; golden frames; recipe ID | C2 halves bit-exact + C3 in shared tolerances + tie-break demo | golden-frame hashes; record; milestone note |
| S8 | Attention full path, grouped MoE, RCCL TP4 | S7 accepted; §6.3 decisions recorded | C2 + own perf qualification | thresholds + baseline artifacts |

Dependency call-out repeated from §2: S6's critical path runs through
CUDA-track S1–S4 deliverables and MI350P provisioning, both outside this
plan's ownership. Track them in the coordinator queue, not by forking core.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No local ROCm toolchain (box limitation) | Syntax-only probes remotely/local-C++; real bring-up on the MI350P node; honest ran-where classification in every evidence note. |
| Host-callback completion deadlocks surfacing late | Stress-loop proof as the second S6 task, before any island code (guide §3.4.2). |
| Wave64 reorders breaking C3 or the argmax tie-break | Reduction tree fixed per bucket and documented once; engineered tie case at S7 step 2; tie-break treated as an integration invariant, not a numerics nicety (v1 §7). |
| LDS/register budget forcing HcMix re-tiling | Re-derived wholly inside the target (F4 sweep grant); GB10 tiles never imported; descriptor carries the LDS limit, kernels receive it as an argument. |
| Block-scaled operand layout mismatch vs stagepack byte order (B1) | Confirm on hardware before committing to zero-repack; fall back to load-time repack keyed by pack hash (B2). Never a hot-path transform. |
| RCCL version drift vs NCCL-aligned assumptions | Pin the RCCL/ROCm release in the build receipt; probe capabilities at create/init time (never submit-time probing, R2). |
| `spark_lm_kernels.cuh` split slipping | Treat as hard S6 blocker; escalate via coordinator queue; never include defensively (landmine rule). |
| MI350P scheduling vs fleet exclusivity model | Register the node class in `fleet_registry.json`; homogeneous-only means it never shares a deployment with GB10 ranks. |

## 9. Explicitly out of scope

- qwen/glm/k3 seam generalization (migration step 7, post-v1; REV1 grants
  naming dispositions only).
- Mixed-vendor collectives (forbidden, R6/R7).
- Performance claims relative to CUDA (row-9 qualification is per-target and
  self-owned; nothing in this plan asserts speed).
- Any edit to portable-core sources, contract documents, or tolerances —
  coordinator acts, all of them.

---

## Appendix A — measured-fact index

| Fact | Source |
|---|---|
| No `spark_hw*` headers; no rocm/hipcc/gfx950 build refs; no dsv4_core split | tree globs/greps, this session |
| Flash module target key incl. `expert_mxfp4`, `kv_bf16` | `modules/dsv4_resident_decode_stage/Makefile:9` |
| Batch-bucket ladder 1..1024 | `modules/dsv4_resident_decode_stage/Makefile.pro:28` |
| `SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 = 3u` | `model-families/common/include/sparkpipe/spark_lm_kernels.cuh:48` |
| e8m0 scales, fp8 block 128 / fp4 block 32 | `modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h:329–330` |
| Activation codec static asserts (FP8_E4M3_UE8M0) | `source/spark_dsv4_resident_decode_stage_cuda.cu:40–42` |
| Shared linear over bf16/fp8/mxfp4 | same file :167 (comment) |
| SM121 guard sites :139/:161/:2785 | guide §3.3 (measured by hwiface iter22) |
| Collective ABI 12, backends, ops, algorithms, TP4 tree, credit/phase machine | `include/sparkpipe/spark_tp_device_collective.h` (lines per §6.1) |
| Oracle fixture present (`after_layer_2.bf16le`, `prompt_tokens.u32le`, manifest) + verifier | `qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128/`, `tools/verify_dsv4_ga_reference_fixture.py` |
| Fleet all-CUDA (CUDA 13, GB10), registry/tier model | `COORDINATION.md` fleet facts |
| Islands, signatures, target strings, sweep lists, gates S6/S7/S8 | `hwiface_v1_freeze.md`, `hwiface_v1.md`, `hwiface_amd_contract_summary.md`, `hwiface_amd_impl_guide.md` |
