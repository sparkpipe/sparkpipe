# Proposal: one admission/policy core (consolidation 3.1)

Owner: SCHEDULER subsystem agent · Workspace: `.agents/scheduler` (clone of `unified`)
Follows the landed `docs/SCHEDULER_SUBSYSTEM_BOUNDARY.md` (§3.1). Proposal only — no
commits, no pushes, no shared/model edits. Every claim cited `file:line` and
verified with grep/read against this tree at `HEAD a947d65`.

---

## 0. Scope

Today admission is one shared *type system* with four re-implemented *loops*:

| Layer | Where it lives | Status |
| --- | --- | --- |
| Request/decision/rejection **types** | `include/sparkpipe/spark_model_driver.h:184-234` | shared, keep |
| Inline **helpers** (validate/init/evaluate/apply) | `include/sparkpipe/spark_model_driver_support.h:109-280` | shared, keep |
| **Generated** per-driver admit + decision merge | `runtime/pack/driver_compiler.c:692-875` | shared string-emitted C |
| **Cross-endpoint cost selection** | `src/spark_orchestrator.c:507-622` | shared, one impl |
| **Serving-side builder + admit call** (per model) | `Spark<Model>ServingAdmit` ×4 + `SparkDsv4StageRunnerAdmit` | **duplicated** |
| **Module-side shape/capacity admit** (per model) | `Spark<Model>ResidentDecodeStageAdmit` ×4 | **duplicated** |

The four serving-side wrappers are near-identical boilerplate
(`spark_glm52_serving_adapter.c:885-916`, `spark_qwen36_serving_adapter.c:981-1010`,
`spark_qwen38_serving_adapter.c:841-870`, plus dsv4's
`spark_dsv4_serving_adapter.c:1207-1311` and `spark_dsv4_stage_runner.c:356-414`).
The four module-side admits are the same decision ladder over different constants
(`spark_glm52_resident_decode_stage_module.c:1554-1591`,
`spark_dsv4_resident_decode_stage_module.c:5871-5946`,
`spark_qwen36_resident_decode_stage_module.c:2031-2116`,
`spark_qwen38_resident_decode_stage_module.c:1242-1251` — the last is a stub
returning `SPARK_STATUS_UNSUPPORTED`).

This proposal extracts the model-neutral loop into one core and shrinks each
adapter to a config table plus a model-specific tail. It does **not** change the
wire types, the generated-driver ABI, or the DRY law surface.

---

## 1. The exact API

### 1.1 Request descriptor, decision output, rejection reasons (reused, not re-invented)

The core consumes the existing ABI types verbatim. There is no new request or
decision struct; adding one would fork the driver ABI (`SPARK_MODEL_DRIVER_ABI_VERSION 12u`,
`spark_model_driver.h:11`) for no gain.

**Request descriptor** — `SparkModelDriverAdmissionRequest`
(`spark_model_driver.h:194-215`):

```c
typedef struct SparkModelDriverAdmissionRequest {
    uint32_t descriptor_bytes;
    uint32_t program_id;
    uint64_t submission_id;
    uint64_t control_generation;
    uint64_t transaction_id;
    uint64_t request_generation;
    uint64_t step_generation;
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t deadline_time_ns;      /* priority/deadline/quanta: deadline   */
    uint32_t active_slot_count;     /*                                   : slots */
    uint32_t new_token_count;       /*                                   : quanta */
    uint32_t priority;              /*                                   : priority */
    uint32_t frame_flags;           /* PREFILL / CACHE_RELEASE (h:42-44) */
    uint32_t admission_flags;       /* CACHE_PREPARE/COMMIT/ABORT (h:45-51) */
    uint32_t cache_lane_count;
    const SparkModelDriverCacheLane *cache_lanes;  /* JIT-KV joint surface */
    SparkModelDriverResidencyToken residency;
} SparkModelDriverAdmissionRequest;
```

**Decision output** — `SparkModelDriverAdmissionDecision`
(`spark_model_driver.h:217-234`): `accepted`, `rejection_reason`,
`driver_dispatch_slot` (+ cookie/generation), `estimated_queue_delay_ns`,
`estimated_service_time_ns`, `endpoint_cost`, `residency_match_score`,
`device_memcpy_bytes`, `host_staging_bytes`, `private_queue_pressure`,
`available_dispatch_slot_count`.

**Rejection reasons** — `SparkModelDriverAdmissionRejection`
(`spark_model_driver.h:184-192`): `ACCEPTED`, `REJECTED_BUSY`,
`REJECTED_KV_CAPACITY`, `REJECTED_DEADLINE`, `REJECTED_HOST_STAGING_REQUIRED`,
`REJECTED_UNSUPPORTED_SHAPE`.

### 1.2 Priority / deadline / quanta inputs

All three already ride the request; the core adds no fields, it only **enforces**
them uniformly:

* **priority** — `request->priority` (`spark_model_driver.h:209`). Source today:
  `SparkModelServingSubmission.priority` (`spark_model_serving_adapter.h:193`) and
  `SparkModelBatchSubmitRequest.priority` (`spark_model_batch_engine.h:71`). The
  core does not order by it (ordering is the batch engine's aged/priority/fill
  passes, `runtime/model_batch_engine.c:1492-1601`); the core only *carries* it to
  the driver and to the orchestrator's cost model.
* **deadline** — `request->deadline_time_ns` (`spark_model_driver.h:206`). The core
  centralizes the monotonic check that today exists only in residentd
  (`node/model_residentd.c:966-985`) and is not applied at admission time at all.
* **quanta** — the bounded work quantum is `new_token_count` (rows this admission
  covers), ceilinged by the program's `max_new_tokens`
  (`spark_model_description.h:38`, enforced in generated code at
  `driver_compiler.c:836-840`); its lifetime unit is one decode step
  (`spark_topology_switch.h:176`, `scheduler/topology_switch.c:350-354`). The core
  names these two ("row quantum" + "step quantum") so a future preemption/ring-window
  hook (§4.2) can price a window in the same units.

### 1.3 The new core surface — `include/sparkpipe/spark_admission.h`

Header-only, model-neutral (no token passes `tests/test_dry_law.py:28-32`'s
regex — no `glm52`/`k3`/`qwen`/`dsv4`/`deepseek`/`mimo25`/`kimi` in
paths or text under `include/sparkpipe/`/`runtime/`/`src/`). Six symbols:

```c
/* --- policy table: the per-model config, model-neutral type --- */
#define SPARK_ADMISSION_ABI_VERSION 1u

typedef uint32_t SparkAdmissionPolicyFlags;
#define SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT 0x00000001u
#define SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS  0x00000002u
#define SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG  0x00000004u
#define SPARK_ADMISSION_POLICY_KNOWN_FLAGS     (SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |      SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS  |      SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG)

/* model-specific cost tail (bytes, from owns_embedding/owns_final_head/etc.). */
typedef void (*SparkAdmissionCostFunction)(
    void *context, const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);
/* model-specific capacity/shape tail (e.g. JIT-KV prepare/commit/abort). */
typedef SparkStatus (*SparkAdmissionPredicateFunction)(
    void *context, const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

typedef struct SparkAdmissionPolicyTable {
    uint32_t abi_version;                 /* SPARK_ADMISSION_ABI_VERSION */
    uint32_t descriptor_bytes;            /* sizeof(table)               */
    uint32_t max_active_sequence_count;   /* active_slot_count ceiling   */
    uint32_t max_input_row_count;         /* new_token_count ceiling     */
    uint64_t max_sequence_positions;      /* KV sequence_position bound; 0 = none */
    SparkAdmissionPolicyFlags flags;      /* prefill/decode slot rules   */
    SparkAdmissionPredicateFunction predicate;  /* NULL = shape accepted  */
    void *predicate_context;
    SparkAdmissionCostFunction cost;             /* NULL = no cost fields */
    void *cost_context;
} SparkAdmissionPolicyTable;

/* --- builders: the four field-copy loops collapse here --- */
SparkStatus SparkAdmissionRequestFromSubmission(
    uint32_t program_id,
    const SparkModelServingSubmission *submission,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request);

SparkStatus SparkAdmissionRequestFromFrame(
    uint32_t program_id,
    const SparkModelDriverFrame *frame,
    const SparkModelDriverCacheLane *cache_lanes,
    uint32_t admission_flags,
    SparkModelDriverAdmissionRequest *request);

/* --- evaluate/apply: the admit + validate + status + apply loop --- */
SparkStatus SparkAdmissionEvaluate(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

SparkStatus SparkAdmissionEvaluateAndApply(
    const SparkModelDriverInterface *driver_interface,
    void *driver_instance,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverFrame *frame,
    SparkModelDriverAdmissionDecision *decision);

/* --- module-side table-driven gate (what ResidentDecodeStageAdmit becomes) --- */
SparkStatus SparkAdmissionEvaluateShape(
    const SparkAdmissionPolicyTable *table,
    uint32_t available_dispatch_slot_count,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision);

/* --- cost + merge: promote the two private arithmetic blocks to shared --- */
uint64_t SparkAdmissionDecisionCost(const SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkAdmissionMergeDecision(
    SparkModelDriverAdmissionDecision *destination,
    const SparkModelDriverAdmissionDecision *source);
```

Semantics of each:

* **Builders** replace `SparkBuildDriverAdmissionRequest`
  (`src/spark_orchestrator.c:488-505`), `SparkDsv4ServingBuildCacheAdmission`
  (`spark_dsv4_serving_adapter.c:1207-1238`), and the inline field-copies in the
  four serving admits. They carry the one invariant that is easy to get wrong —
  "cache lanes present ⇔ admission_flags present ⇔ generation fields present"
  (`spark_model_driver_support.h:114-129`) — into a single validated path.
* **`SparkAdmissionEvaluate`** is the already-existing
  `SparkModelDriverEvaluateAdmission` (`spark_model_driver_support.h:244-261`)
  made the canonical entry (it validates → admits → maps via
  `SparkModelDriverAdmissionDecisionStatus` `228-242`). The four serving
  wrappers currently bypass it and hand-roll a *lossy* mapping (§1.5).
* **`SparkAdmissionEvaluateAndApply`** = evaluate + `SparkModelDriverApplyAdmissionDecision`
  (`spark_model_driver_support.h:263-280`); the orchestrator already does this
  pair (`spark_orchestrator.c:679-680`).
* **`SparkAdmissionEvaluateShape`** is the common module-side ladder: ABI/descriptor
  check → `frame_flags` whitelist → active/new-token ceilings from the table →
  the two slot-rule flags → `max_sequence_positions` KV bound → BUSY on
  `available_dispatch_slot_count == 0` → optional `predicate` tail → optional
  `cost` tail → accept. This is exactly the ladder each
  `Spark<Model>ResidentDecodeStageAdmit` repeats (compare glm52
  `spark_glm52_resident_decode_stage_module.c:1564-1590` with qwen36
  `spark_qwen36_resident_decode_stage_module.c:2046-2112`).
* **`SparkAdmissionDecisionCost`** promotes the orchestrator's static
  `SparkDriverAdmissionDecisionCost` (`src/spark_orchestrator.c:507-523`) to the
  shared header so the orchestrator and any future hook share one formula.
* **`SparkAdmissionMergeDecision`** promotes the string-emitted
  `SparkGeneratedMergeAdmissionDecision` (`driver_compiler.c:721-748`) out of the
  generator into a compiled symbol; the generator then emits a one-line call
  instead of 28 lines of arithmetic (shrink, not behavior change).

### 1.4 The cost model (two layers, both now shared)

1. **Per-module cost** — a module admit sets `host_staging_bytes`,
   `device_memcpy_bytes`, `estimated_service_time_ns`,
   `estimated_queue_delay_ns`, `endpoint_cost`, `residency_match_score`
   (see glm52 `1588-1589`, dsv4 `5944-5946`, qwen36 `2113-2116`). The
   generated driver merges them across the program's modules (saturating-add the
   byte/time/cost fields, min the residency score, max the queue delay) —
   `driver_compiler.c:735-747`. That merge becomes `SparkAdmissionMergeDecision`.
2. **Cross-endpoint cost** — `SparkAdmissionDecisionCost` folds the merged
   decision into one scalar for the orchestrator's lowest-cost route selection
   (`spark_orchestrator.c:601-608`): `endpoint_cost + estimated_queue_delay_ns +
   estimated_service_time_ns + (private_queue_pressure << 20) +
   (host_staging_bytes << 2) + device_memcpy_bytes − residency_match_score`
   (`spark_orchestrator.c:512-522`). Unchanged; only relocated.

### 1.5 Rejection → status mapping (a real bug the core fixes)

`SparkModelDriverAdmissionDecisionStatus` (`spark_model_driver_support.h:228-242`)
maps: ACCEPTED→OK; BUSY or **DEADLINE**→BUSY; UNSUPPORTED_SHAPE→UNSUPPORTED; else
CAPACITY_EXCEEDED. The four serving wrappers and the dsv4 stage runner instead do
`BUSY ? BUSY : CAPACITY_EXCEEDED` (`spark_glm52_serving_adapter.c:913`,
`spark_qwen36_serving_adapter.c:1008`, `spark_qwen38_serving_adapter.c:868`,
`spark_dsv4_stage_runner.c:408-410`), which **loses the DEADLINE and
UNSUPPORTED_SHAPE distinctions** at the adapter boundary. The core routes all four
through `SparkAdmissionEvaluate`, so the richer mapping applies uniformly.

---

## 2. The per-model adapter surface (what shrinks, to what)

### 2.1 `Spark<Model>ServingAdmit` → one `SparkAdmissionEvaluateAndApply` call

Every serving-side wrapper reduces to: build the request with
`SparkAdmissionRequestFromSubmission` (or `FromFrame`), then
`SparkAdmissionEvaluateAndApply`. No per-model fields remain — the program id,
deadline, priority, row count, flags, and residency all come from the submission
(or frame) which the builders read generically. dsv4's cache-lane building
(`SparkModelServingAdapterBuildDriverCacheLanes`, `spark_model_serving_adapter.c:469-517`)
is already shared and stays as the caller of `FromSubmission`.

The dsv4 stage runner (`spark_dsv4_stage_runner.c:356-414`) is the *third*
serving-side site; it collapses to the same call with `FromFrame` + a dispatch
struct (its `dispatch` fields already mirror the frame fields,
`spark_dsv4_stage_runner.c:370-388`).

### 2.2 `Spark<Model>ResidentDecodeStageAdmit` → a config table + a tail

Each module admit becomes:

```c
/* in the model's own *_module.c — model dir, may name the model */
static const SparkAdmissionPolicyTable kAdmissionTable = {
    .abi_version = SPARK_ADMISSION_ABI_VERSION,
    .descriptor_bytes = sizeof(SparkAdmissionPolicyTable),
    .max_active_sequence_count = <MODEL>_MAX_ACTIVE_SEQUENCE_COUNT,
    .max_input_row_count       = <MODEL>_MAX_INPUT_ROW_COUNT,
    .max_sequence_positions    = <MODEL>_MAXIMUM_CONTEXT_TOKENS,
    .flags = SPARK_ADMISSION_POLICY_FLAG_...,   /* prefill/decode slot rules */
    .predicate = <MODEL>_AdmissionPredicate,    /* JIT-KV tail, or NULL     */
    .predicate_context = state,
    .cost = <MODEL>_AdmissionCost,              /* staging/memcpy formula    */
    .cost_context = state,
};

SparkStatus Spark<Model>ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    Spark<Model>ModuleState *state = module_state;
    uint32_t available = SparkStageModuleSlotCountFree(state->slot_states,
                                                       state->pipeline_slot_count);
    SparkStatus status = SparkAdmissionEvaluateShape(&kAdmissionTable, available,
                                                     request, decision);
    if (status != SPARK_STATUS_OK && <model increments its rejected_count>)
        atomic_fetch_add_explicit(&state->rejected_count, 1u, memory_order_relaxed);
    return status;
}
```

The per-model **data** that remains is precisely the table: three ceilings, three
flag bits, and (only where genuinely model-specific) a predicate and a cost
function. What disappears is the repeated descriptor/flags/active/new-token/KV/
BUSY ladder.

Mapping today → table:

| Field | glm52 | qwen36 | dsv4 | qwen38 (stub) |
| --- | --- | --- | --- | --- |
| `max_active_sequence_count` | `resident_sequence_capacity` (`1569`) | `state->max_active_sequence_count` (`2066`) | descriptor | (unimplemented) |
| `max_input_row_count` | `SPARK_GLM52_..._MAX_INPUT_ROW_COUNT` (`1569`) | `state->max_active_sequence_count` (`2068`) | descriptor | (unimplemented) |
| `max_sequence_positions` | `state->max_sequence_positions` (`1575`) | `SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS` (`2086,2089`) | `SparkDsv4ModuleAdmissionFitsKv` (`5905`) | (unimplemented) |
| `flags` | DECODE_EQUALS_SLOTS (`1569`) | PREFILL_SINGLE_SLOT + DECODE_EQUALS_SLOTS (`2069-2071`) | via `AdmissionShapeSupported` (`5895`) | (unimplemented) |
| `predicate` | NULL | NULL | JIT-KV prepare/commit/abort (`5910-5937`) | (unimplemented) |
| `cost` | `1588-1589` | `2113-2116` | `5944-5946` | (unimplemented) |

Note the `ALLOW_DISPATCH_FLAG` divergence: glm52 whitelists
`DRIVER_DISPATCH_SLOT_VALID` in `frame_flags` (`1569`) while qwen36 rejects it
(`2063-2064`). The flag bit makes that a data difference, not a code fork.

### 2.3 JIT-KV cache admission — joint contract, marked not decided

dsv4's module admit has a prepare → commit/abort → require-committed sequence
(`spark_dsv4_resident_decode_stage_module.c:5910-5937`, with helpers
`SparkDsv4ModulePrepareCacheAdmission` / `ResolveCacheAdmission` /
`RequireCommittedCacheAdmission` `5854-5869`). The flags it consumes
(`SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE/COMMIT/ABORT`,
`spark_model_driver.h:45-51`) and the lane struct (`SparkModelDriverCacheLane`,
`spark_model_driver.h:94-108`) are already model-neutral shared types. The core's
stance:

* The core **passes through** `admission_flags` + `cache_lanes` verbatim; it does
  not interpret prepare/commit/abort semantics.
* The **`predicate`** slot in the table is exactly where a JIT-KV model plugs its
  prepare/commit/abort logic. For non-JIT-KV models the slot is NULL and
  `cache_lane_count == 0` is enforced by `SparkModelDriverAdmissionRequestIsValid`
  (`spark_model_driver_support.h:114-129`).
* The prepare idempotence + "must not publish/release" rule
  (`spark_model_driver.h:89-93`) and the physical-page mapping are **the kv-cache
  agent's lane**. §5 records the joint contract to reconcile; this proposal does
  not decide it.

---

## 3. Migration sequence + byte-identical pinning

### 3.1 Host tests that pin admission behavior today (must stay green)

* `tests/test_dry_law.py` — the gate for the core itself. New file
  `include/sparkpipe/spark_admission.h` + any `runtime/`/`src/` code must pass
  `MODEL_TOKEN` (`28-32`); COMMON includes `include/sparkpipe`, `runtime`,
  `src` (`13-27`).
* `tests/test_driver_compiler.c` — pins the **generated** admit: the exact emitted
  condition `(frame_flags & PREFILL) == 0 && new_token_count > 7` (`139`), the
  reject-then-accept behavior (`183-193`), and that generated code does **not**
  re-define a local decision validator (`138`, i.e. it must call the shared
  `SparkModelDriverAdmissionDecisionIsValid`). Any change to the merge/emit
  arithmetic is pinned byte-identical by the `strstr` at `139`.
* `tests/test_model_serving_adapter.c:463-503` — pins
  `SparkModelDriverAdmissionRequestIsValid` over the JIT-KV flag/lane invariants
  (the exact rules the builders must preserve).
* `tests/test_dsv4_stage_runner.c:26-50, 218-233` — pins the stage-runner admit
  call: `submission_id/control_generation/transaction_id/request_generation/
  step_generation` are copied into the request (`36-40`), admit_count increments
  exactly once per submit (`220`), and an invalid decision is ABI_MISMATCH (`230-232`).
* `tests/test_model_pipeline_client.c:1038, 1094, 1501-1502` — pins the engine's
  `CloseAdmission` + `admission_open` + `admitted_count/rejected_count`
  (the batch-engine admission gate is *outside* this proposal but shares the
  "admission" name; the core must not disturb it).
* `tests/test_model_resident_deadline.c:62, 90-91` — pins the monotonic
  `deadline_time_ns` expiry + completion queueing that `SparkAdmissionEvaluate`
  will centralize at admission time (no behavior change to residentd itself).
* `tests/test_topology_switch.c:324-363` — pins `SparkTopologySwitchAdmissionsOpen`
  close-at-Begin / reopen-when-resident; the topology switch is a *consumer* of the
  same "admission closed" idiom but is not folded into this core (it stays in
  `scheduler/topology_switch.c`).

### 3.2 Ordered steps (each lands green independently)

**Phase A — core lands, no callers change (behavior-neutral).** Coordinator lands
`include/sparkpipe/spark_admission.h` (+ `src/spark_admission.c` for the two
promoted non-inline symbols). `SparkAdmissionDecisionCost` is a pure move of
`SparkDriverAdmissionDecisionCost`; `SparkAdmissionMergeDecision` is a pure move
of the emitted merge. CI: `test_dry_law` + full build; nothing calls the new
symbols yet.

**Phase B — generated admit calls the shared merge (generator output must stay
byte-equivalent).** Edit `runtime/pack/driver_compiler.c:721-748` to emit a call to
`SparkAdmissionMergeDecision` instead of the inlined 28 lines. The emitted
*condition* at `139` and the reject/accept behavior at `183-193` must be
byte-identical; `test_driver_compiler.c` is the guard. This is a coordinator-owned
shared edit (generator + generated source), done once, not per model.

**Phase C — serving-side wrappers collapse.** For each model in turn (order below),
replace `Spark<Model>ServingAdmit` + `SparkDsv4StageRunnerAdmit` with
`SparkAdmissionRequestFromSubmission/FromFrame` + `SparkAdmissionEvaluateAndApply`.
Pinned by `test_dsv4_stage_runner.c:36-40,220,230-232` (dsv4) and, for the others,
by the existing serving-adapter driver fixtures (`tests/fixtures/dsv4_serving_adapter_driver.c`,
`tests/fixtures/qwen36_serving_adapter_driver.c`, `tests/fixtures/model_serving_adapter_module.c`)
which assert the request fields the driver's `admit` sees. Each MODEL agent edits
its own `*_serving_adapter.c` / runner.

**Phase D — module-side admits collapse to a table.** Replace each
`Spark<Model>ResidentDecodeStageAdmit` body with the table + `SparkAdmissionEvaluateShape`
shown in §2.2. Pinned by the driver fixtures' `admit` assertions and by
`test_driver_compiler.c:183-193` (the merged decision must be identical). Model
agents own their tables; the scheduler agent supplies the table type + the
`SparkAdmissionEvaluateShape` reference implementation.

**Order:** (1) dsv4 — most complex (JIT-KV predicate + third call site), most test
coverage; (2) glm52 — has no JIT-KV, but its `ALLOW_DISPATCH_FLAG` quirk
(`1569`) exercises a flag the others don't; (3) qwen36 — deprecated-frozen
(`COORDINATION.md:147-149`), migrate only if it stays on the shared core, otherwise
leave the legacy admit in place; (4) qwen38 — its module admit is a stub
(`spark_qwen38_resident_decode_stage_module.c:1242-1251`), so it *starts* from the
core instead of migrating: fill the table, get a real admit for free.

### 3.3 Byte-identical pinning strategy

* **Don't rename, re-derive.** The request/decision structs, the rejection enum,
  the generated ABI (`SPARK_MODEL_DRIVER_ABI_VERSION`), and the emitted condition
  string stay byte-identical; only *call sites* and the *generator's emitted
  arithmetic* change, and the latter is guarded by `test_driver_compiler.c:139`.
* **One host test per promoted symbol.** `SparkAdmissionDecisionCost` is pinned by a
  new assertion mirroring `spark_orchestrator.c:507-523`; `SparkAdmissionMergeDecision`
  is pinned by a fixture that merges two synthetic decisions and compares against the
  current `driver_compiler.c:735-747` arithmetic.
* **No behavior change in Phase A/B.** Every merge has a no-op first step (add the
  symbol, then re-point callers), so each PR is green and reviewable alone.

---

## 4. Extension points

### 4.1 The B1-B1024 ladder (consolidation 3.2) hangs off the table

The ladder's model-neutral arithmetic is the ceiling walk
(`Spark<Model>BatchVariantBucketCeiling`, identical across
`spark_dsv4_batch_tuning.h:129-156`, `spark_glm52_batch_tuning.h:164-191`,
`spark_k3_batch_tuning.h:122-149`). The admission core does **not** consume the
ladder — the ladder sizes the *module's* pools, while admission sizes the *frame*.
The two meet at one field: `max_active_sequence_count`. Concretely, a model's
bucket ceiling (the compiled `-DSPARK_BATCH_BUCKET`) is the value the model fills
into `SparkAdmissionPolicyTable.max_active_sequence_count`, so the admission core
rejects any frame wider than the resident bucket. When §3.2 extracts the shared
ladder into `include/sparkpipe/`, the table's `max_active_sequence_count` becomes
the sole consumer of `SparkBatchVariantBucketCeiling(...)` — one integration
point, no core change.

### 4.2 Ring-window enforcement hook (consolidation 3.3) is a `predicate`

The hourly continuation rule (`COORDINATION.md:86-119`) needs a *fleet-level*
preemption signal, not a per-frame one. The clean extension is a
`SparkAdmissionPredicateFunction` plugged into the *orchestrator* (not a model):
a "window lease" predicate that rejects admission (or downgrades priority) when the
current holder's window has expired without a durable artifact. The
`deadline_time_ns` plumbing (§1.2) is exactly what a 60-minute window lease needs,
and `SparkAdmissionDecisionCost` gives the preemption hook a cost signal to feed
`tools/fleet_swap.sh`. This stays an extension point (no implementation here).

---

## 5. Joint contract with the kv-cache agent (marked, not decided)

The scheduler's admission core touches KV admission in exactly three places, all
already shared types:

1. `request->admission_flags` (`SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE/
   COMMIT/ABORT`, `spark_model_driver.h:45-51`).
2. `request->cache_lanes` / `cache_lane_count` (`SparkModelDriverCacheLane`,
   `spark_model_driver.h:94-108`).
3. The KV-page budgets the *scheduler* gates on before admission
   (`kv_logical_page_capacity` / `kv_physical_page_capacity`,
   `spark_model_serving_adapter.h:146-152`, consumed by
   `runtime/model_batch_engine.c:193-221, 1413-1441`).

**To reconcile jointly (do not decide unilaterally here):** whether the
prepare/commit/abort predicate that today lives inside
`SparkDsv4ResidentDecodeStageAdmit` (`spark_dsv4_resident_decode_stage_module.c:5910-5937`)
should become (a) a kv-cache-owned shared predicate that the table's
`predicate` slot calls, or (b) stay a per-model tail behind a stable
kv-cache interface. Either way the scheduler's core promises: it will not interpret
the flags, will not touch physical page layout, and will only carry the fields
through `SparkAdmissionRequestFromSubmission/FromFrame` into
`SparkAdmissionEvaluateShape`'s `predicate` slot. The `CacheLaneIsValid`
invariants (`spark_model_driver_support.h:73-107`) and the prepare idempotence
rule (`spark_model_driver.h:89-93`) are the shared contract both sides already
implement; the core's only addition is to enforce them in one place.

---

## 6. Verification note

All `file:line` citations verified with grep/read at `HEAD a947d65`. Net-new
surface is `include/sparkpipe/spark_admission.h` (+ optional
`src/spark_admission.c`); nothing else is created or edited in this proposal.
The DRY-law gate (`tests/test_dry_law.py:28-32`) forbids model tokens in the core;
per-model tables and predicates live in the model dirs and may name their model.

---

## Coordinator verification note (2026-08-17)

The rejection-mapping defect is CONFIRMED: the serving wrappers collapse
every non-BUSY admission rejection to SPARK_STATUS_CAPACITY_EXCEEDED
(e.g. spark_glm52_serving_adapter.c:913 - the ternary
REJECTED_BUSY ? BUSY : CAPACITY_EXCEEDED), dropping the DEADLINE and
UNSUPPORTED_SHAPE distinctions that SparkModelDriverAdmissionDecisionStatus
carries. The proposed SparkAdmissionMergeDecision fixes it by construction;
the per-adapter collapse is scheduled for migration phase C.
