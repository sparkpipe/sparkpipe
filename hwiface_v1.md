# SparkPipe hardware-agnostic model–device interface — v1 (FROZEN, consolidated)

Status: **FROZEN.** This document is the complete, self-contained contract the
AMD dev implements against. It supersedes, consolidates, and freezes the three
predecessor documents — `hwiface_v0.md` (evidence base), `hwiface_v1.md`
(advisor fold), `hwiface_v1_freeze.md` (freeze + REV1) — into one text.
Revision history:

- **v0 → v1 fold:** advisor boundary table made normative (§2); binding rules
  R1–R7 (§1); validation classes C1–C3 (§7).
- **Freeze REV1:** qwen36 naming extensions (§9), DSV4 decisions untouched.
- **REV2 (this consolidation):** every open finding of
  `.agents/coord/audits/AUDIT_HWIFACE_V1_FREEZE.md` (A1–A13; A14–A18 were
  PASS) is resolved in place, tagged *(REV2, closes AN)*, and summarized in
  the ledger (§10). Erratum ownership question answered: hwiface owns its
  freeze; this consolidation IS that patch. A13 timing question answered: no
  `p1-baseline-*` artifact existed in the tree when REV2 was written
  (re-measured 2026-08-24), so the §7 bucket re-pin precedes any S1 capture.

Freeze semantics (unchanged): every decision below is binding for the first
`rocm.gfx950` implementation. Reopening any of it is a coordinator act that
revises this document; because tolerances and validator arguments are ordered
contract inputs (SPEC §2/§7), most reopenings also create a new validation
identity downstream. Nothing here is left "to be decided during
implementation".

Provenance of facts: numbers tagged *(measured)* were read out of unified head
(`spark_dsv4_resident_decode_stage_module.c` = 6314 lines, 61
`extern cudaError_t SparkDsv4Launch*` declarations + 93 call sites = 154
launch references, 0 direct `<<<` launches in the host module). gfx950
hardware facts remain **advisor-provided** (`docs/coord/advisor_amd.md`,
user-supplied; unverified locally).

---

## 1. Binding rules

1. **Semantic execution-island granularity.** The interface names islands —
   attention block, routed-MoE block, shared expert, head, cache transition —
   never one entry per launch function. The advisor makes the reason binding:
   a per-launch surface (**61** functions on DSV4 alone) would freeze CUDA's
   fusion boundaries into the ABI and forbid a ROCm island from fusing them.
   A target may implement an island with 1 kernel or 40.
2. **Static linking per target, no hot-path backend vtable.** Target chosen at
   **package compile time**: `cuda.sm121.*` links CUDA symbols,
   `rocm.gfx950.*` links ROCm symbols, and **both link the same host/control
   source**. Direct calls, no backend branches, no dispatch tables, no
   registration, no submit-time capability probing. Opaque handles resolve via
   static linking — the only indirection anywhere is the SPEC §3 module ABI at
   the scheduler boundary. A missing target symbol is a link error, never a
   runtime search.
3. **Opaque handles.** Queues, events, graphs, memory, cache pages, sealed
   route batches: opaque target handles. The portable core moves them; it
   never inspects them.
4. **GB10-specific constants parameterized.** Shared-memory limits, tile/shape
   selectors, SM counts, capability checks live in the target descriptor or
   target-internal code (§5). No target numeric constant in portable-core
   code; no portable-core constant silently assumed by a second target.
5. **Expert aggregation split.** The **common core decides**: which submitted
   frames may combine into one expert batch; queue keys/generations; max wait;
   batch-bucket selection; when a layer batch seals; cancellation/fairness.
   The **backend receives an opaque sealed-route batch** and implements
   counting/grouping/grouped GEMM. No routing policy crosses the seam in
   either direction (§3.4).
6. **Homogeneous deployments only.** CUDA and AMD are separate, homogeneous
   deployments; **no mixed-vendor collective**. The collective seam (§4.4) is
   per-deployment uniform: one transport implementation per link unit.
7. **DSV4-first scope.** v1 neutral primitives and islands are defined for
   DSV4 only. Generalizing the seams to qwen/glm/k3 happens after migration
   steps 1–6 land (§8); nothing in v1 may leak DSV4 geometry into primitive
   signatures.

---

## 2. Boundary table (normative)

Folded verbatim from the advisor (`docs/coord/advisor_amd.md`) and annotated.
**Shared** = model-specific but target-free code, owned by the portable core.
**Hardware-target** = owned by the per-target implementation archive.

| Shared (portable core) | Hardware-target (per-target archive) | Lands in | Seam artifact |
|---|---|---|---|
| Model geometry and layer order | CUDA/HIP allocation, streams and events | firmware header + core init vs `spark_hw_memory/queue/event` | target descriptor (§4.1) |
| Batch buckets and lane lifecycle | CUDA Graph/HIP Graph creation and replay | stage runner / admission vs graph primitives | opaque `SparkHwGraphExec`; capture topology target-internal (§3.3) |
| Expert queue admission and sealing policy | Device-resident queue representation | common core (rule 5) vs routed-MoE island impl | opaque sealed-route batch (§3.4, §4.3) |
| Logical routing and grouped-expert contract | Count/scan/scatter kernels | L4 input contract (route weights/indices) vs L4 internals | L4 island boundary |
| KV and prefix-cache policy | Physical KV layout and attention kernels | `spark_dsv4_paged_cache.c` (0 CUDA refs today) vs L2/L3 impls | page-table handle + cache-page opacity |
| TP/PP operation ordering | NCCL, RCCL or custom transport implementation | portable orchestration of collectives vs transport impl | collective seam (§4.4) |
| Speculation and sampling semantics | Device kernels and target-specific fusion | F1 semantics (markov bias, screened argmax, feedback) vs F1 kernels | F1 island boundary |
| Logical weight tensor inventory | CUDA- and AMD-specific packed layouts | stagepack loader + pool layout vs target packers | logical tensor inventory in the pack; physical layout target-private |
| Numerical oracle and tolerances | Per-target performance qualification | C1/C2/C3 classes (§7) vs P1 perf gate per target (§7, §8) | validation recipe ID per target |

Reading of the table: every row's **left cell already has a named portable
home**; every row's right cell is contained behind an island boundary or an
opaque handle. No row requires a shared abstraction beyond what §3–§5 define —
the table validates the island cut rather than reshaping it.

---

## 3. Contract sketch: execution islands

An **island** is one semantic unit of device work with a fixed input/output
buffer set and fixed handle dependencies. It is the unit of (a) graph capture
and replay, (b) cross-target validation, and (c) the port. Boundary data
between islands is the boundary packet row (`hc_mult × hidden` bf16).

### 3.1 Final island set — seven islands, determinism class, declared surface

The set below is the **DSV4 instantiation** and is frozen as written: **no
splits, no merges within the DSV4 instantiation** *(REV2, closes A5 — scoped:
family-specific slot instantiation may vary island count per §9, e.g. dense
families instantiate one `layer.ffn` where DSV4 has L4+L5)*. State channels
are declared per island; anything not declared here must not persist across
frames.

| # | Island | Declared inputs | Declared outputs | Handles | State channels | Class |
|---|--------|-----------------|------------------|---------|----------------|-------|
| E0 | `prologue.embed` (stage 0 only) | token ids, embedding view | boundary packet | queue, memory | — | **C2** (gather/index is integer) |
| L1 | `layer.boundary_norm_project` | boundary packet, layer weights | normalized hidden, packed Q/KV/indexer shards | queue, memory | slot workspace | **C3** (norm/projection accumulation; shard *permutation* verified via downstream C2 per §3.5) |
| L2 | `layer.attention` | L1 outputs, cache pages, freqs, positions | attention delta, q/kv/score states | queue, event, memory | slot workspace, cache pages | **C3** |
| L3 | `layer.cache_transition` | q/kv/score states, positions, page tables | updated pages, emitted entries, emit counters | queue, memory | **cache pages and/or recurrent-state pools** *(amended by §9 REV1.2 for hybrid-recurrence families)* | **split verdict:** page addresses, ring indices, emit counters = **C2**; emitted payload bytes = C2 if copied unrounded, else C3 (per emitted-field table, §7) |
| L4 | `layer.moe_routed` | normalized hidden, sealed-route batch (opaque), routed weights | FFN accumulator, expert offsets | queue, event, memory | slot workspace | **split verdict:** route indices / group offsets / expert assignment = **C2** (core-decided, backend-realized); grouped-GEMM output = **C3** |
| L5 | `layer.moe_shared` | normalized hidden | FFN accumulator (partial) | aux queue, fork/join event pair, memory | slot workspace | **C3** |
| F1 | `head.final` | final boundary packet, head weights | output token ids (+scores), resident token/lane feedback | queue, event, memory | slot workspace | **split verdict:** output token ids + maxloc pack/unpack + feedback integers = **C2** (exact argmax, lowest-index tie-break); scores and quantized head outputs = **C3** |

L1–L5 repeat per layer; L4/L5 are forked legs joined before the block hcPost;
L2/L3 are sequential today but are separate islands so a target may reorder,
fuse, or re-pipeline them. **Q3 decided:** F1 stays whole — the DSpark
markov/confidence heads remain inside `head.final`; `draft.speculate` is
not created (rationale: they share F1's reduction/quantize/screened-argmax
machinery and nothing interleaves between head sub-stages; §9 REV1.3 confirms
multi-pack drafter loading without touching this).

### 3.2 Mapping onto today's TP captures

`PROJECTION` ≈ L1; `ATTENTION` ≈ L2+L3; `FFN` ≈ L4+L5; `FINAL` ≈ F1
(today: `3L+1` captured islands per pipeline slot).

### 3.3 Capture topology is out of the ABI

If the ABI mandated a capture count it would do to graph fusion what a
per-launch vtable does to kernel fusion: freeze one target's internal
decomposition as everyone's contract. Resolution: **the ABI names semantic
islands only**; CUDA keeps `3L+1` today and may change it later without an
ABI change; ROCm picks its own topology. The `cuda_graph_count` node-context
contract stays target-owned.

### 3.4 Routed-MoE island under the aggregation split (rule 5)

At the L4 boundary: core owns combine policy, queue keys/generations, max
wait, bucket selection, the seal point, cancellation/fairness. Backend owns
device-resident representation of the sealed batch, counting, scan/offsets,
scatter, grouped-GEMM scheduling, MXFP4 dequant strategy. The sealed-route
batch crosses as an opaque handle over the same logical route weights/indices
on every target; the *logical* route is core-computed and identical across
targets.

### 3.5 Split-verdict comparison mechanics *(REV2, closes A2 + A8)*

- **C2 is not a raw-buffer hash of whole island outputs.** For split-verdict
  islands, C2 compares exactly the fields named in the applicable field table
  (§7) via recipe-defined extraction; target-private padding/alignment inside
  a declared output buffer is excluded from comparison.
- **L4's C2 probe surface is the island's declared `expert offsets` output
  buffer.** Core-side route inputs are excluded from cross-target comparison
  as tautological (same `dsv4_core` objects compute them on every target).
- **L1 shard permutation** is verified via the canonically ordered downstream
  C2 observables (L3 page addresses / ring indices / emit counters), so a
  permutation difference can neither mask nor fabricate a pass.
- **L3's C2 quantities are integer-pure:** page addresses, ring indices, and
  emit counters depend only on declared integer inputs (positions, page
  tables, shape buckets, token counts) — never on an L2 float score. If
  implementation work ever reveals a float-driven emission decision, that
  decision moves into L3's split-verdict field table explicitly at recipe
  time (a §7 realization, not a silent class change) *(REV2, closes A4)*.

---

## 4. Neutral runtime primitives

One header, `spark_hw_iface.h`, owned by the interface (not by either
target). All symbols C-linkage, provided only by the per-target archive.
Handles are opaque pointers.

### 4.0 Status model *(REV2, closes A9)*

```c
typedef enum SparkHwStatus {
    SPARK_HW_OK = 0,
    SPARK_HW_NOT_READY,       /* async op not yet complete (query only) */
    SPARK_HW_INVALID,         /* bad handle/argument */
    SPARK_HW_UNSUPPORTED,     /* capability/descriptor check failed */
    SPARK_HW_EXHAUSTED,       /* capacity refusal: pool/pinned OOM, graph
                                 resources; retryable, feeds core admission/
                                 backpressure policy */
    SPARK_HW_LOST             /* device/context lost; instance is fail-closed */
} SparkHwStatus;
```

Allocation failure returns `SPARK_HW_EXHAUSTED` (never `INVALID`);
pressure handling is core policy and needs a distinguishable code. Device
loss returns `SPARK_HW_LOST` from any primitive on affected handles; there
is no in-place recovery in v1 — the core tears the instance down. Targets
must not invent further mappings.

### 4.1 Target descriptor

```c
typedef struct SparkHwTarget {
    const char *target_id;                /* frozen strings, §6 */
    uint32_t    abi_version;              /* 1 */
    uint32_t    multiprocessor_count;     /* GB10: queried via cudaDeviceGetAttribute
                                             and passed into launches (measured);
                                             gfx950: CU count */
    uint64_t    max_dynamic_shared_bytes; /* GB10: 101376 (measured);
                                             gfx950: its LDS limit */
    uint32_t    capability_major, capability_minor; /* GB10: 12.1 */
    uint32_t    wavefront_lanes;          /* GB10: 32; gfx950: 64 (advisory, informational) */
} SparkHwTarget;
```

*Erratum applied (REV2, closes A7):* earlier drafts said the descriptor also
carries "target-selected tile/shape selectors (native decode shape, expert
W13/W2 tile-N)". **It deliberately carries none**: tile/shape selection
resolves *inside* the target (§5 landmine rule). Read once in
`ModuleInitialize`; never consulted in the hot path. `wavefront_lanes` is
informational: C3 reduction-order guarantees (§7) are keyed to the target's
internal launch configuration per shape bucket, **never derived from this
field**, which must feed no gate *(REV2, closes A12)*.

### 4.2 Primitive families — exact signatures

Names, argument shapes, and semantics are frozen; the list is exactly the
CUDA surface module.c uses today *(measured: 23 cudaMemcpyAsync + 2 memset +
16 pinned alloc/free + 2 mapped + malloc/free + weight read-ahead)*.

```c
typedef void SparkHwMemory;      /* opaque pool/workspace backing */
typedef void SparkHwQueue;
typedef void SparkHwEvent;
typedef void SparkHwGraphExec;

/* Memory */
SparkHwStatus spark_hw_memory_pool_alloc(SparkHwMemory **mem, uint64_t bytes);
SparkHwStatus spark_hw_memory_pool_free(SparkHwMemory *mem);
SparkHwStatus spark_hw_host_pinned_alloc(void **host, uint64_t bytes, uint32_t flags);
SparkHwStatus spark_hw_host_pinned_free(void *host);
SparkHwStatus spark_hw_host_device_pointer(void *host, void **device);
SparkHwStatus spark_hw_copy_async(void *dst, const void *src, uint64_t bytes,
                                  SparkHwCopyKind kind /*H2D|D2D|D2H*/, SparkHwQueue q);
SparkHwStatus spark_hw_memset_async(void *dst, uint32_t value, uint64_t bytes, SparkHwQueue q);
SparkHwStatus spark_hw_read_ahead(SparkHwQueue q, void *sink_u32,
                                  uint32_t word_capacity, void *context);

/* Queue */
SparkHwStatus spark_hw_queue_create(SparkHwQueue *q, uint32_t flags, int32_t priority);
SparkHwStatus spark_hw_queue_destroy(SparkHwQueue q);
SparkHwStatus spark_hw_queue_query(SparkHwQueue q);          /* SPARK_HW_NOT_READY allowed */
SparkHwStatus spark_hw_queue_synchronize(SparkHwQueue q);    /* failure paths ONLY (see note) */
SparkHwStatus spark_hw_queue_enqueue_host_callback(SparkHwQueue q, void (*fn)(void*), void *arg);

/* Event — timing NOT required (today: cudaEventDisableTiming) */
SparkHwStatus spark_hw_event_create(SparkHwEvent *e, uint32_t flags);
SparkHwStatus spark_hw_event_destroy(SparkHwEvent e);
SparkHwStatus spark_hw_event_record(SparkHwEvent e, SparkHwQueue q);
SparkHwStatus spark_hw_queue_wait_event(SparkHwQueue q, SparkHwEvent e);

/* Graph — relaxed capture mode preserved by the flag argument */
SparkHwStatus spark_hw_graph_capture_begin(SparkHwQueue q, uint32_t mode /*relaxed*/);
SparkHwStatus spark_hw_graph_capture_end(SparkHwQueue q, void *graph_out);
SparkHwStatus spark_hw_graph_instantiate(void *graph, SparkHwGraphExec *exec);
SparkHwStatus spark_hw_graph_launch(SparkHwGraphExec exec, SparkHwQueue q);
SparkHwStatus spark_hw_graph_destroy(void *graph);
SparkHwStatus spark_hw_graph_exec_destroy(SparkHwGraphExec exec);
```

**Synchronization note *(REV2, closes A10).*** `queue_synchronize` appears on
failure paths only; a successful frame completes externally via
`enqueue_host_callback` (the single stream-ordered callback). This prohibition
binds the **frame path only**: `queue_destroy`, `memory_pool_free`,
`graph_destroy`, and `graph_exec_destroy` may internally synchronize to
drain in-flight frame work and stream-ordered callbacks during teardown after
the core has quiesced the instance.

**Weight read-ahead contract *(REV2, closes A11).*** For
`spark_hw_read_ahead(q, sink_u32, word_capacity, context)`: (1) enqueues a
stream-ordered, advisory bulk prefetch of target-resident weight regions
associated with `context` on `q` (today's weight-read-ahead kick);
correctness never depends on it completing. (2) On completion writes at most
`word_capacity` `uint32` words of target-defined occupancy/status into
`sink_u32` (host-pinned), ordered after the prefetch on `q`; fewer words
is allowed, contents target-defined. (3) May be called any time between
instance initialize and destroy, including before graph capture; bad handles
return `SPARK_HW_INVALID`; a failed kick degrades to no-op and must never
fail the frame.

Island entries themselves are plain externs resolved by static link (§6);
they are not part of this primitive header.

### 4.3 Sealed-route batch lifecycle (closes N2)

Measured basis: route/grouping buffers are slot-workspace fields
(`slot->moe_scores_f32`, `slot->moe_indices_u32`, `slot->grouped_rows_u32`,
`slot->expert_offsets_u32`, `slot->group_tile_prefix_w1/w2_u32`,
`slot->route_packed_row`, `slot->route_source_token`), allocated once per
slot at creation. Therefore: (1) the sealed-route batch is **not dynamically
allocated** — it is a fixed offset view into the slot workspace determined by
the shape bucket; (2) core-side seal bookkeeping publishes readiness, L4
consumes within the frame, nothing persists except through declared state
channels; (3) contents below the L4 entry are target-defined, the logical
route is core-computed and identical across targets.

### 4.4 Collective seam (closes N1)

No new primitive family. ROCm implements the existing
`include/sparkpipe/spark_tp_device_collective.h` surface — already neutral
(backend enum `_HIDDEN_TRANSPORT`/`_NCCL`; opaque `void *` stream handles;
Create / ProbeMemoryMode / CreditStepCount / CreditBindingRouteCount /
ApplyTopology / SliceTopology / RequestFailure / RequestOperationFailure /
OperationPhase …) — over RCCL, selected at package compile time like every
other target fact. Homogeneous deployments only (rule 6).

---

## 5. GB10 parameterization sweep list and landmine rule

Audit result *(measured 2026-08-22)*: **no GB10 numeric constant is reachable
from portable-core sources today.** Two sections plus one landmine rule.

**(a) Migrates during S1–S2**

| Constant / fact | Value today | Home today | Destination | AMD sweep note |
|---|---|---|---|---|
| dynamic shared-memory limit | 101376 B | `.cu` comment :1924 + `cudaFuncAttributeMaxDynamicSharedMemorySize` :2745/2760/2773 | descriptor `max_dynamic_shared_bytes`; kernels receive limit as argument | gfx950 sets its own LDS limit; HcMix tiling re-derived inside target, never imported |
| multiprocessor count | runtime query | module.c (`state->multiprocessor_count`, passed into launches) | descriptor `multiprocessor_count` | CU count; core passes through, never assumes |
| capability gate | 12.1 | `.cu:139` `SparkDsv4RequireNativeSm121` (+ PTX image trap) | target-internal fail-closed guard | gfx950 equivalent fails closed on non-gfx950 |
| `SPARK_DSV4_HC_MIX_TILE` | 4096u | `.cu:1911` | stays target-internal | re-derive under LDS budget |
| `SPARK_DSV4_HC_ELEMENT_TILE` | 256u | `.cu:28` | target-internal | free |

**(b) Already target-internal — must not leak portable-side**
(cites are symbol-level by rule; line numbers drift, symbol values do not
*(REV2, adopting the auditor's line-drift observation)*)

| Constant | Value | Home | Rule |
|---|---|---|---|
| `SPARK_LM_SM121_B1_EXPERT_W13_TILE_N` | 32u | `model-families/common/include/sparkpipe/spark_lm_kernels.cuh` | SM121-only; see landmine rule |
| `SPARK_LM_SM121_B1_EXPERT_W2_TILE_N` | 128u | same header | ditto |
| `SPARK_LM_SM121_B1_EXPERT_BLOCKS_PER_SM` | 2u | same header | ditto |
| `SPARK_LM_SM121_B1_EXPERT_W2_BLOCKS_PER_SM` | 4u | same header | ditto |
| `SPARK_LM_SM121_NATIVE_TILE_N` | (header) | same header | ditto |
| `SPARK_LM_SM121_NATIVE_WEIGHT_FP8/_MXFP4` | format codes | same header | packed-format policy is target-side |
| `SparkLmSm121NativeDecodeShape` row set | {1,5,7,8,16,32,64,1024} | same header | shape policy named "Sm121"; becomes target-selected selector |
| `SparkLmSm121ExpertW13TileN/W2TileN` | f(rows) | same header | ditto |
| `SPARK_LM_HC_MIX_ROWS_PER_WARP` | 3u | `.cu:1912` | warp-shape-dependent; target-internal |

**Landmine rule (binding, extended — REV2, closes A1).**
`spark_lm_kernels.cuh` is a *model-families common* header carrying SM121
selectors. Restated family-agnostically: **no non-CUDA translation unit may
evaluate a `SPARK_LM_SM121_*` selector.** The obligation binds **all current
and future includers** of the header, enforced at the header split whichever
model family triggers it first, and no later than migration step S6. Measured
includers 2026-08-24:
`modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu`,
`modules/qwen38_resident_decode_stage/source/spark_qwen38_resident_decode_stage_cuda.cu`,
`modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_cuda.cu`
(the k3/glm52 decode-stage TUs exist but do not include it yet; the rule binds
them the day they do). Before S6 the header must be split or guarded such that
neutral kernel machinery stays shared while SM121 tile/shape/format selection
moves under target control — a ROCm build that silently took CUDA's tile
choices would be a correctness and performance hazard the linker cannot catch.
Scoping note: the remediation trigger attaches to the **header split**, not to
the DSV4 backend step alone, so a step-7 qwen ROCm build cannot hit the hazard
first.

**Sweep ranges granted to the AMD dev without touching portable code:** all
of section (b) plus LDS limit, CU count, tile/shape selectors, wave size —
everything below island entries and inside the descriptor. Portable-core
constants (geometry, buckets, seal policy, tolerances) are **not** sweepable.

---

## 6. Static-link resolution

Target-id strings (frozen; appear in descriptor + model description + library
records): **`cuda.sm121.gb10`**, **`rocm.gfx950.mi350p`**.

Link units (all static archives, SPEC §2 kinds; thin forbidden):

| Link unit | Contains | Linked into |
|---|---|---|
| `dsv4_core` | portable core: island orchestration, admission/seal/lane policy, paged-cache policy, stagepack/pool bookkeeping, serving adapter config — zero target headers | every deployment |
| `dsv4_cuda_sm121_gb10` | today's `.cu` (63 wrappers + kernels), `spark_hw_cuda_*` impl, GB10 constants, SM121 capability guard | deployments resolving `cuda.sm121.gb10` |
| `dsv4_rocm_gfx950_mi350p` | `.hip` island implementations, `spark_hw_rocm_*` impl, gfx950 constants, gfx950 fail-closed guard | deployments resolving `rocm.gfx950.mi350p` |
| transport unit (existing) | `tp_device_collective` implementation, backend chosen at compile time | TP>1 deployments |

Resolution rules (frozen):

1. Stage JSON resolves module ID + exact target; the generated orchestrator
   links `dsv4_core` + exactly one target unit (SPEC §4 step 5). Missing
   symbol = link error, never runtime search (R2).
2. Both targets link the **same** `dsv4_core` object files. Any need to fork
   core sources per target is a freeze violation by definition.
3. Capability checks stay **fail-closed inside the target** (mirroring
   `SparkDsv4RequireNativeSm121`: major==12 && minor==1, cached per executor
   thread; image-level PTX-guard analog). gfx950 fails closed on non-gfx950.
4. Publication follows SPEC §2 unchanged: content-addressed bytes, one
   validation record per `module-target-key + validation-key`, cold build.

**gfx950 / MI350P hardware facts (advisor-provided) and their consequences:**
MI350P, gfx950, CDNA4 → own descriptor instance, fail-closed like SM121;
64-lane wavefronts → wave-size-dependent reductions target-internal, C3 order
guarantees keyed to internal launch config per bucket (§4.1, §7); native
MXFP4 → stays below the seam (logical inventory unchanged, packing/dequant in
AMD archive); different LDS/register/tiling budget → own descriptor limits,
HcMix tiling re-derived internally; HIP graphs → map 1:1 onto the graph
family; RCCL NCCL-aligned → collective seam per §4.4, homogeneous-only.

Deliverable shape for the AMD side: a `.hip` translation unit implementing
the same **island-level** entry surface (free to fuse today's 61 wrappers —
e.g. one kernel for all of L1), a `spark_hw_rocm_*` primitive archive, a
gfx950 descriptor, and its own validator recipe ID + reference-fixture run.

---

## 7. Validation plan

Three distinct claims, kept separate so a cross-target difference is triaged,
not argued about.

- **C1 — same-target regression identity (guards the refactor).** After every
  migration step the CUDA module reproduces bit-identical outputs: same stage
  pack, deterministic frame sequence, bucket ⇒ identical output token ids and
  boundary buffers. SHA-256 per island-boundary buffer against golden hashes;
  mismatch blocks the step.
- **C2 — cross-target exact identity.** Integer/index/addressing surfaces —
  per the §3.1 split verdicts and §3.5 comparison mechanics — are
  bit-identical across targets. Hard acceptance gate for `rocm.gfx950`.
- **C3 — cross-target bounded identity.** Accumulation islands compared
  bit-wise, accepted within per-island tolerances carried as ordered validator
  arguments; absolute invariants: output token ids match exactly (deterministic
  lowest-index tie-break), reduction order fixed **per shape bucket** by the
  target's internal launch configuration (§4.1).

**Class authority vs recipe-time field tables *(REV2, closes A3).*** The
per-island classes in §3.1 are frozen; reclassification is a freeze revision.
Field-table assignments made **before S7** under an already-frozen split
verdict (e.g. deciding whether an L3 emitted payload byte-range is copied
unrounded ⇒ C2, or accumulated ⇒ C3) are **realizations**, not
reclassifications. Assignments made or changed **after S7** are revisions and
follow freeze semantics. Golden frame list per bucket and the L3/L4/F1 field
tables are recipe-time artifacts due before S7 (first full-layer ROCm
milestone).

**Mechanics.** Reference fixture
`qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128` +
`tools/verify_dsv4_ga_reference_fixture.py` as numerical oracle; hash-pinned
validator sources, cold build per SPEC §7; per-island probes exist only in
the validation build, forbidden in the serving image; per-target publication
keyed by `module-target-key + validation-key`. Tolerance authority (Q4):
numbers are owned by the validation recipe as ordered arguments; both targets
are judged against the same shared-oracle tolerance set; any change creates a
new validation identity — tighten or loosen alike. Targets may add stricter
private checks without identity change.

**P1 — same-target performance non-regression gate.** Pinned bench command
*(measured: exists in tree)*: `tools/devcycle/bbench.sh B` — no-spec
B-batch decode benchmark for the lean DSV4 TP4 driver; one discarded warmup +
one measured O128 run; prints aggregate decode rate. Baseline requirements,
frozen:

- Capture **before migration step S1** on the pre-split driver, per bucket
  **B in {1, 8, 1024}** *(REV2, closes A13 — re-pinned from {1, 8, 64}: the
  tree's own qualified decode-shape set is {1, 8, 1024}, per
  `spark_lm_kernels.cuh` "DSV4 calls only this section for its qualified
  B1/B8/B1024 shapes" [re-measured 2026-08-24]; B64 dropped as a
  possibly-unqualified noise anchor, and the largest qualified bucket is now
  covered so a regression manifesting only at B1024 cannot pass P1). Store
  raw outputs as `p1-baseline-<date>.txt` next to the validation records;
  append-only. Verified 2026-08-24: no such artifact existed when REV2 was
  written, so this re-pin precedes any capture and nothing needs re-capture.
  bbench.sh itself is B-agnostic; the authoritative bucket list lives here.
- P1 pass = refactored aggregate decode rate within run-to-run noise of the
  baseline (two repeat runs each side, compare medians). A miss blocks the
  step exactly like a C1 hash mismatch.
- P1 is same-target (CUDA) only; cross-target speed is the AMD side's
  per-target performance qualification (boundary row 9).

---

## 8. Migration sequence (advisor's 7 steps mapped)

Each early step leaves the CUDA module **byte-identical (C1) and not slower
(P1)** and revalidated.

| Advisor step | Plan step(s) | Gate |
|---|---|---|
| 1. Neutral runtime primitives for DSV4 only | introduce `spark_hw_iface.h` + `spark_hw_cuda` mechanical wrapper; module.c switches to opaque handles | C1 byte-identical |
| 2. Split DSV4 portable-core / device-contract / cuda-impl | name islands in the host module; move GB10 constants into the descriptor; split build into portable-core + CUDA target link units under SPEC §2 content addressing | C1 byte-identical |
| 3. Byte-identical + no CUDA perf regression | continuous acceptance across all refactor steps | **C1 + P1** |
| 4. gfx950 runtime backend | implement `spark_hw_rocm_*` + descriptor + ROCm island entries | ROCm recipe validates once |
| 5. One complete DSV4 layer | first ROCm milestone: one full layer through L1–L5+F1 end-to-end | C2 exact on index islands; C3 bounded elsewhere |
| 6. Attention, grouped MoE, TP4 | expand ROCm scope to L2/L3, L4 grouped GEMM, RCCL collective | C2 + P1-AMD |
| 7. Generalize seams for qwen/glm/k3 | post-v1 until 1–6 land (§9 governs then) | n/a |

Deadlines pinned by REV2: landmine header split **before S6** (§5); field
tables due **before S7** (§7); P1 baseline due **before S1** (§7).

---

## 9. Family extensions (carried from freeze REV1 — effective at step 7 per R7)

Scope rule: these dispositions let other model families instantiate frozen
slots mechanically; they change **nothing frozen for DSV4** — seven-island
set, every C-class, all `spark_hw_*` signatures, target-id strings, sweep
lists, tolerances stand exactly as written above. No DSV4 validation identity
is created or invalidated by them.

- **REV1.1 Dense-FFN row.** For a dense-FFN family (qwen36 today), the FFN
  stage instantiates as **one island `layer.ffn`** (normalized hidden + FFN
  weights → updated hidden; queue, memory; slot workspace; **C3**), not the
  L4/L5 pair — no routing means no sealed-route subject for rule 5, and L5's
  distinctive fork/join machinery vanishes. DSV4 keeps
  `layer.moe_routed`/`layer.moe_shared` frozen as written. If a family
  adds routing, the full L4/L5 machinery applies unchanged.
- **REV1.2 Hybrid-recurrence state channel.** L3's state channels read
  "**cache pages and/or recurrent-state pools**". For hybrid-recurrence
  families (qwen36 GDN), the per-lane float recurrence state + conv tails in
  the pool is a declared state channel alongside cache pages: written by L2
  and the verify-walk snapshot/restore pair, transitioned by L3. Split
  verdict applies identically: pool addresses, lane cursors/ring indices,
  emit counters = **C2**; stored float payloads = **C3**. Snapshot/restore is
  a **bit-exact copy obligation on both channels regardless of class**, and
  the copied state is **lane-local: a restore never crosses lanes**.
  Physical pool layout stays target-private behind opaque handles.
- **REV1.3 Drafter second pack under F1-stays-whole.** A stage may carry more
  than one logical tensor inventory, each published as its own
  content-addressed pack artifact (main pack + drafter pack, today
  `SPARK_QWEN36_DSPARK_PACK_PATH`). Every pack's inventory is **logical and
  target-free** — §5 landmine/sweep rules apply per pack. Drafter device work
  (MTP argmax rows, head-style quantize/screened argmax, selector machinery)
  stays inside F1 and inherits F1's split verdict unchanged. Loading a second
  pack creates no second validation identity by itself.
- **Rejected variants (recorded so they are not re-proposed):** a
  `draft.speculate` island; renaming/generalizing the DSV4 MoE island names;
  a family-specific `spark_hw_*` header fork ahead of step 7; treating this
  section as license for pre-step-7 implementation work.

---

## 10. Dispositions ledger

| Item | Disposition | Where |
|---|---|---|
| Q1 capture topology | Closed — ABI names islands; topology target-internal | §3.3 |
| Q2 TP collective seam | Closed — existing neutral header + RCCL; homogeneous only | §4.4 |
| Q3 head-island scope | Closed — F1 unsplit; `draft.speculate` not created | §3.1, §9 |
| Q4 tolerance authority | Closed — recipe-owned ordered arguments; shared-oracle set; change ⇒ new identity | §7 |
| Q5 advisor fold | Closed — boundary table normative | §2 |
| N1 collective signatures | Closed — reuse existing header, no new family | §4.4 |
| N2 sealed-route lifecycle | Closed — slot-workspace views, allocated at slot create | §4.3 |
| N3 P1 baseline artifact | Closed — pinned command + buckets + pre-S1 capture | §7 |
| A1 landmine blast radius | Closed — family-agnostic rule; all includers bound; header-split trigger | §5 |
| A2 L4 C2 observable | Closed — `expert offsets` output buffer named; route inputs tautological | §3.5 |
| A3 class-freeze vs field tables | Closed — pre-S7 realizations, post-S7 revisions | §7 |
| A4 L3 integer purity | Closed — C2 quantities integer-only; escape hatch via field table | §3.5 |
| A5 splits/merges scope | Closed — scoped to DSV4 instantiation; families vary per §9 | §3.1 |
| A6 L3 state-channel annotation | Closed — row amended in place | §3.1 |
| A7 descriptor erratum | Closed — selector fields removed from struct; erratum noted | §4.1 |
| A8 C2 comparison mechanics | Closed — field-table extraction; padding excluded; canonical ordering | §3.5 |
| A9 status expressiveness | Closed — `SPARK_HW_EXHAUSTED` + `SPARK_HW_LOST` added; mappings frozen | §4.0 |
| A10 teardown drain | Closed — destroy/free may synchronize; prohibition binds frame path only | §4.2 |
| A11 read-ahead semantics | Closed — three-clause contract | §4.2 |
| A12 wavefront_lanes | Closed — informational; never feeds a gate; C3 order keyed to launch config | §4.1 |
| A13 P1 bucket list | Closed — re-pinned to {1, 8, 1024} pre-capture; verified no baseline exists | §7 |
| A14–A18 | PASS (audit) — no action required | — |

Precedence: if any predecessor document (`hwiface_v0.md`, the v1 advisor
fold, `hwiface_v1_freeze.md`) disagrees with this text, **this text
governs.**
