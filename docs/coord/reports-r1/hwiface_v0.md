# SparkPipe hardware-agnostic model–device interface — v0 draft

Status: **v0 for review.** The AMD implementer builds against the frozen **v1**,
not this document. Everything here is negotiable until v1 freezes; the design
rules below are not.

Inputs consumed: `SPEC.md` (firmware contract), `modules/dsv4_resident_decode_stage/`
(first port target). `.agents/coord/advisor_amd.md` was **not present** at draft
time; the boundary table is therefore not yet folded in — v1 must incorporate it.

---

## 0. Evidence base (what the first port target actually is)

Measured facts about `modules/dsv4_resident_decode_stage/` that this design is
grounded in:

- `source/spark_dsv4_resident_decode_stage_module.c` — **6314-line** host module.
  It contains **61 `extern cudaError_t SparkDsv4Launch*` declarations** and
  **93 launch call sites** — the **154 launch references**. Every launch goes
  through one of those 61 named wrappers; the host module never launches a
  kernel directly (`<<<` appears 0 times in it).
- `source/spark_dsv4_resident_decode_stage_cuda.cu` — 3440 lines holding the 63
  wrapper definitions and all device code. It is explicitly **SM121-only,
  fail-closed** (`SparkDsv4RequireNativeSm121` checks `major==12 && minor==1`;
  device code is independently trapped by `LM_SM121_NATIVE_COMPUTE_PTX` on any
  non-SM121 image).
- TP execution already runs as **fixed stage-local graph islands**:
  `3 × layer_count + 1` captured islands per pipeline slot
  (`SPARK_DSV4_TP_GRAPH_ISLAND_PROJECTION / _ATTENTION / _FFN` per layer plus
  `_FINAL`). The eager TP1 path runs the same sequence uncaptured
  (`SparkDsv4ModuleRunLocalLayer`). **The island decomposition already exists in
  code; v0 names it semantically and regularizes it.**
- The MoE block already forks: the shared expert runs on an auxiliary stream
  (`SparkStageModuleCudaForkBegin/Join`) in parallel with gate/route + routed
  experts, joined before the pair-reduce.
- Runtime primitive inventory in the host module: `cudaMemcpyAsync` ×23,
  `cudaStreamSynchronize` ×13, pinned host alloc/free ×16 (+2 mapped-device
  pointers), graph begin-capture/end-capture/instantiate/launch/destroy,
  event create/record/wait/destroy, stream create/query/wait-event, and
  `cudaLaunchHostFunc` (the one stream-ordered host callback from which every
  accepted frame completes externally — submit never synchronizes a successful
  frame).
- GB10-specific constants baked into device code today: the **101376 B dynamic
  shared-memory limit** (Pro's 28672-float HcMix staging = 112 KB exceeds it, so
  `SPARK_DSV4_HC_MIX_TILE = 4096` tiles the staging), the SM121 **native decode
  shape** selector and **expert tile-N selectors**
  (`SparkLmSm121NativeDecodeShape`, `SparkLmSm121ExpertW13TileN`,
  `SparkLmSm121ExpertW2TileN`, `SPARK_LM_SM121_NATIVE_TILE_N`), and the
  multiprocessor count (already queried at runtime via
  `cudaDeviceGetAttribute` for split heuristics).
- Host-side logic that is already target-free: `spark_dsv4_paged_cache.c`
  (0 CUDA references), `spark_dsv4_stage_runner.c` (0), the stagepack loader and
  pool layout, and `spark_dsv4_serving_adapter.c` (only the config string
  `cuda_graph_count`, no CUDA API).

SPEC.md already fixes the frame this interface must live in: one exact target
per stage resolved at compile time (§1), content-addressed link units validated
once and statically linked by a generated orchestrator with **direct calls and
no per-operation indirect dispatch** (§2, §4, §5), and an orchestrator that
never learns attention/MoE/KV internals (§6, §8). The hardware interface below
is the **module-internal** seam between portable model firmware and per-target
device implementation; it does not touch the scheduler boundary.

---

## 1. Design rules (binding for v1)

1. **Semantic execution-island granularity.** The interface names islands like
   *attention block*, *routed-MoE block*, *shared expert*, *head*, *cache
   transition* — never one vtable entry per launch function. A target may
   implement an island with 1 kernel or 40; the contract fixes the island's
   inputs, outputs, and handles, not its internal launch count.
2. **Static linking per target, no hot-path function pointers.** The model
   description resolves one exact target (`cuda.sm121.gb10` today,
   `rocm.gfx950.*` next). The portable core calls target implementation symbols
   directly; the generated stage driver links exactly one target's units
   (SPEC §4). The only indirection anywhere is the SPEC §3 module ABI at the
   scheduler boundary. Dispatch tables, registration, and capability probing in
   the hot path are forbidden.
3. **Opaque handles.** Queues, events, graphs, and memory are opaque target
   handles (`void *`-backed, target-defined contents). The portable core moves
   them; it never inspects them. This is the same opacity rule the orchestrator
   applies to dispatch slots (SPEC §6).
4. **GB10-specific constants parameterized.** Shared-memory limits, tile/shape
   selectors, SM counts, and capability checks move into a target descriptor
   read once at initialization. No target-specific numeric constant may appear
   in portable-core code, and no portable-core constant may be silently assumed
   by a second target.

---

## 2. Contract sketch: execution islands

An **island** is one semantic unit of device work with a fixed input/output
buffer set and fixed handle dependencies. It is the unit of (a) graph capture
and replay, (b) cross-target validation, and (c) the port. Islands are ordered
within a decode step exactly as today's launch sequence is.

Boundary data between islands is the **boundary packet row**: `hc_mult × hidden`
bf16 (the four hyper-connection streams, which never collapse between layers —
only `hc_pre` inside a block and the head reduction do). Everything else a
target keeps in private slot workspace is invisible above island level.

### 2.1 Island set

| # | Island | Contents (today's kernels) | Inputs | Outputs | Handles touched |
|---|--------|----------------------------|--------|---------|-----------------|
| E0 | `prologue.embed` (stage 0 only) | embedding gather, stream expand, weight read-ahead kick | token ids, embedding view | 4-stream boundary packet | queue, memory |
| L1 | `layer.boundary_norm_project` | hcPost fold of prior block, hcEnter, RMSNorm, attention input projections (+ TP shard pack/unpack) | boundary packet, layer weights | normalized hidden, packed Q/KV/indexer shards | queue, memory |
| L2 | `layer.attention` | query-head RMS + rope, indexer core + topk, sparse attention over paged cache, output projection `wo_a/wo_b` | L1 outputs, cache pages, freqs, positions | attention delta, q/kv/score states | queue, event, memory, **cache pages** |
| L3 | `layer.cache_transition` | compressor step, KV emission into ring pages, short-window cache scatter | q/kv/score states, positions, page tables | updated cache pages, emitted entries, emit counters | queue, memory, **cache pages** |
| L4 | `layer.moe_routed` | gate/route (bias or hash pin), grouped routed expert up/down over stacked views, pair reduce | normalized hidden, route weights | FFN accumulator | queue, event, memory |
| L5 | `layer.moe_shared` | shared expert `w1/w2/w3` | normalized hidden | FFN accumulator (partial) | **auxiliary queue**, event pair (fork/join), memory |
| F1 | `head.final` | final hcPost + head reduction, shadow/certified FP8 quantize, screened argmax (optionally sharded), maxloc pack/unpack, markov bias, token feedback | final boundary packet, head weights | output token ids (+scores), resident token/lane feedback | queue, event, memory |

L1–L5 repeat per layer (`first_layer_index .. first_layer_index+layer_count`).
L4 and L5 are concurrent legs forked after L1's normalization and joined before
the block's hcPost; L2 and L3 are sequential today (round-major publication
keeps the 128-slot ring causal) but are **separate islands** so a target may
reorder, fuse, or re-pipeline them without renegotiating the interface.

### 2.2 Mapping onto the existing TP graph islands

Today's captured islands per pipeline slot are `PROJECTION, ATTENTION, FFN`
per layer + `FINAL` (= `3L+1`). v0's semantic set relates as:

- `PROJECTION` ≈ L1 (unchanged);
- `ATTENTION` ≈ L2 + L3 (v0 **splits the cache transition out** so the ROCm
  target can own it independently — on CUDA the two may remain one capture
  unit);
- `FFN` ≈ L4 + L5 (v0 **names the fork legs as islands**; today they are one
  capture with an internal fork);
- `FINAL` ≈ F1.

Capture topology (which semantic islands are fused into one captured graph
unit, and how many graph execs exist per slot) is **target-internal**. The ABI
names semantic islands; it does not mandate `3L+1`, `4L+1`, or any capture
count. The existing `cuda_graph_count` node-context contract stays a
target-owned quantity validated by the target's own initialization.

### 2.3 Island contract rules

- Each island is submitted as one unit on one queue, or replayed as one
  captured graph unit; a target never interleaves another island into the
  middle of one.
- Cross-island ordering is queue order plus explicit events (the L4/L5 fork
  join). No implicit global synchronization between islands.
- Islands own no hidden cross-frame state except through the two declared
  state channels: **resident slot workspace** (opaque, allocated at slot
  creation) and **cache pages** (via the page-table handle). Anything else
  must appear in the island's declared inputs/outputs.
- Determinism class per island is part of the contract (§5): which islands must
  reproduce identical bytes across targets, and which are tolerance-bounded.

---

## 3. Neutral runtime primitive list

All primitives are C-linkage `spark_hw_*` symbols, provided **only** by the
per-target implementation archive. Handles are opaque. The list is exactly the
CUDA surface the module uses today — nothing aspirational.

**Target descriptor** (read once at `ModuleInitialize`, never in the hot path):

```c
typedef struct SparkHwTarget {
    const char *target_id;            /* "cuda.sm121.gb10" | "rocm.gfx950..." */
    uint32_t    abi_version;
    uint32_t    multiprocessor_count; /* today: cudaDeviceGetAttribute */
    uint64_t    max_dynamic_shared_bytes;  /* GB10: 101376 */
    uint32_t    capability_major, capability_minor;  /* GB10: 12.1 */
    /* target-selected tile/shape selectors, e.g. native decode shape,
     * expert W13/W2 tile-N — resolved inside the target, not the core */
} SparkHwTarget;
```

**Memory** — `spark_hw_memory` (opaque):
- `pool_alloc / free` — device pool backing for slot workspace and resident
  weights (today: `cudaMalloc` + module pool layout);
- `host_pinned_alloc / free` + `host_device_pointer` — pinned staging and its
  mapped device address (today: `cudaHostAlloc` / `cudaHostGetDevicePointer`);
- `copy_async` (H2D / D2D / D2H), `memset_async` (today: 23 + 2 sites);
- `read_ahead` — bulk weight prefetch (today: `SparkDsv4LaunchWeightReadAhead`).

**Queue** — `spark_hw_queue` (opaque):
- `create(flags, priority)` / `destroy` (priority from stage configuration
  `stream_priority`; `stream_ordered` flag from the program contract);
- `query`, `synchronize` (failure path only — a successful frame is never
  synchronized);
- `enqueue_host_callback` — the single stream-ordered host callback that
  produces external completion (today: `cudaLaunchHostFunc`).

**Event** — `spark_hw_event` (opaque):
- `create(flags)`, `record(queue)`, `queue_wait(queue, event)`, `destroy` —
  the L4/L5 fork-join and any cross-queue ordering (today: 4 API forms).

**Graph** — `spark_hw_graph_exec` (opaque):
- `capture_begin(queue)`, `capture_end(queue)`, `instantiate`, `launch(queue)`,
  `destroy` — island capture/replay, shape-bucketed per pipeline slot
  (today: the full `cudaGraph*` set used by the TP islands).

**Linking rule.** The portable core declares these plus the island entry
symbols as plain externs. The model compiler resolves the stage's exact target
and links the portable core with exactly that target's implementation archive
(SPEC §4 step 5); a missing symbol is a link error, never a runtime search.
No registry, no per-launch function pointers, no capability negotiation at
submit time (SPEC §5, §8).

---

## 4. DSV4 split plan: portable core vs. CUDA implementation

### 4.1 Portable core (stays host C, zero target headers)

- Module state machine, `ModuleInitialize/Execute/Admit/Snapshot/Destroy`
  bodies, frame validation (`SparkDsv4ModuleValidate*`), boundary bounce and
  TP-continuation logic;
- island orchestration: the ordered L1→L5, F1 sequence, buffer wiring, fork/join
  structure, graph replay decisions — expressed over `spark_hw_*` and island
  entries;
- `spark_dsv4_paged_cache.c` (already 0 CUDA references), pool layout,
  stagepack loader, hash-table verification, freq computation;
- slot staging bookkeeping, admission/snapshot counters;
- `spark_dsv4_batch_tuning.h` and the firmware header's geometry-free views
  (they are already target-free);
- `spark_dsv4_stage_runner.c`, `spark_dsv4_serving_adapter.c` (config only).

### 4.2 Target implementation, CUDA (`cuda.sm121.gb10`)

- `spark_dsv4_resident_decode_stage_cuda.cu`: all 63 `SparkDsv4Launch*`
  wrappers and device kernels; SM121 fail-closed guards become the target
  descriptor's capability check (same fail-closed semantics, checked at
  initialize, cached per executor thread as today);
- the `spark_hw_cuda_*` archive implementing §3 over the CUDA runtime;
- GB10 constants live here and in the descriptor, nowhere else;
- the existing validator (`validation/spark_dsv4_resident_decode_stage_cuda_validation.cu`
  + recipe script) stays the CUDA target's validation recipe.

### 4.3 Target implementation, ROCm (`rocm.gfx950.*`) — the AMD side

- a `.hip` translation unit implementing the **same island entry surface**
  (island-level, not 154 per-launch shims — a ROCm island may fuse today's
  per-launch wrappers arbitrarily, e.g. one kernel for L1);
- a `spark_hw_rocm_*` archive implementing §3 over HIP/ROCm runtime;
- a gfx950 descriptor with its own shared-memory limit, SM count, tile
  selectors, and capability check;
- its own validator recipe ID and reference-fixture run (§5).

### 4.4 Ordered migration (each step leaves the CUDA module byte-identical and revalidated)

1. Introduce the `spark_hw_*` header + `spark_hw_cuda` archive as a mechanical
   wrapper over the current CUDA calls; module.c switches from `cudaStream_t`
   etc. to opaque handles. No behavior change; C1 validation (§5) must pass
   byte-identical.
2. Move GB10 constants (101376 B limit, HcMix tile choice, native decode
   shapes, tile-N selectors, SM count) into the descriptor; kernels receive
   limits as arguments. Compile-time tile templates stay **inside** the target
   as target-internal dispatch — allowed, since they never cross the seam.
3. Name the islands in the host module: factor the existing
   `LaunchTp{Projection,Attention,Ffn,Final}Island` bodies and the eager
   `RunLocalLayer` into the L1/L2+L3/L4+L5/F1 island functions; elevate the
   cache-transition launches (`CompressStep`, `KvEmission`, `CacheScatter`)
   behind one island entry in both paths.
4. Split the build into portable-core link unit + CUDA target link unit;
   publish both under the existing SPEC §2 content-addressed contract (new
   bytes ⇒ new identity ⇒ validated once, cold build). Module ID and target
   field are unchanged for CUDA.
5. Freeze v1 from review of this document + the advisor boundary table. AMD
   implements `rocm.gfx950` against the frozen ABI only.

---

## 5. Byte-identical validation plan

"Byte-identical" is three distinct claims; the plan keeps them separate so a
cross-target difference is triaged, not argued about.

**C1 — same-target regression identity (guards the refactor).**
After every migration step, the CUDA module must reproduce **bit-identical**
outputs to the pre-step module: same stage pack, same deterministic frame
sequence, same bucket ⇒ identical output token ids **and** identical boundary
buffers. Harness: a validation-build frame driver replays a fixed frame set per
batch bucket (B1/B8/B64 at minimum), SHA-256-hashes each island-boundary buffer
(boundary packets, FFN accumulators, emitted cache entries, output token ids),
and compares against the previous step's golden hashes. Any mismatch blocks the
step.

**C2 — cross-target exact identity (integer/index/addressing islands).**
`prologue.embed`, route indices and expert offsets in `moe_routed`, page
addressing and emit counters in `cache_transition`, maxloc pack/unpack and
token feedback in `head.final` produce **bit-identical bytes across targets**.
These paths are integer or single-rounding; there is no architectural excuse
for divergence. This is a hard acceptance gate for `rocm.gfx950`.

**C3 — cross-target bounded identity (accumulation islands).**
Sparse attention, HcMix/RMSNorm reductions, FP8/FP4 linear outputs, MoE
combine: bit patterns are compared and recorded, and acceptance is bounded by
per-island tolerances that are part of the **ordered validator arguments**
(SPEC §2 — changing a tolerance changes validation identity, exactly as the
contract requires). Two invariants are still absolute for C3 islands:
output **token ids must match exactly** (argmax with deterministic
lowest-index tie-break), and reduction order must be fixed **per shape bucket**
(already true: TP islands replay bucket-width shapes, so the tree is fixed).

**Mechanics.**

- Reuse the existing machinery: reference fixture
  `qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128` +
  `tools/verify_dsv4_ga_reference_fixture.py` as the numerical oracle; the
  per-target recipe script pattern (hash-pinned validator sources, cold build
  per SPEC §7) for both targets.
- Per-island probes (boundary-buffer capture) exist **only in the validation
  build**, selected by the recipe; they are forbidden in the serving image
  (SPEC §8: no validation machinery in the serving process).
- Publication: each target's link unit gets its own validation record keyed by
  module-target-key + validation-key; an unchanged CUDA artifact is never
  retested, and the ROCm artifact validates exactly once against its own
  recipe.
- The v1 freeze must include: the C2/C3 island classification table, the C3
  tolerance set, and the golden frame list per bucket. Those three are the
  deliverable's testable core.

---

## 6. Open questions for v1 review

1. **Capture topology vs. semantic islands.** Does `cache_transition` become
   its own captured graph unit (CUDA capture count `3L+1` → `4L+1`), or stay
   fused into the attention capture on CUDA while standing alone on ROCm?
   Proposal: keep `3L+1` on CUDA (capture topology is target-internal); the ABI
   only names semantic islands.
2. **TP collective seam.** `SparkTpDeviceCollective` (ring/all-reduce over the
   Spark fabric) is a second hardware seam. In scope for hwiface v1, or a
   separate contract? The ROCm port cannot close without an answer.
3. **Head island scope.** Do the DSpark markov/confidence heads stay inside F1
   or become a separate draft island (`draft.speculate`)? Affects the island
   count and the C2/C3 classification.
4. **C3 tolerance authority.** Who owns the tolerance numbers — module author
   or validator recipe — and may a target tighten but never loosen them?
   Proposal: frozen in v1, changes create a new validation identity by
   construction.
5. **Advisor boundary table.** `.agents/coord/advisor_amd.md` was absent; fold
   its boundary rows into v1 before freezing.
