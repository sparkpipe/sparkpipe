# Scheduler Subsystem Boundary

Archived 2026-09-28. This is the 2026-08-17 scheduler-agent proposal
(a947d65). Its consolidation plan landed as `src/spark_admission.c` (4cfa095)
and `include/sparkpipe/spark_batch_variant_tuning_common.h` (7ecf1f5), and
`scheduler/` also holds `continuous_batch.c` (c4e7fb4). The page and position
budgets of §2.4 moved to [driver acceptance](../DRIVER_ACCEPTANCE.md). The
fleet sections (§1.5, §1.8, §1.9, the fleet bullets of §2.1, §3.3) described
`tools/fleet_swap.sh` and the COORDINATION.md tier and ring-window rules. Those
are obsolete, so the sections were removed; `git show
9b283b6:docs/SCHEDULER_SUBSYSTEM_BOUNDARY.md` has the full text. Fleet
operations are in the [fleet release runbook](../FLEET_RELEASE_RUNBOOK.md).

Owner: SCHEDULER subsystem agent · Workspace: `.agents/scheduler` (clone of `unified`)
Scope: cross-model scheduling machinery. This document inventories what scheduling
exists on `unified`, draws the ownership boundary between the scheduler and its
four neighbours (model drivers, transport, KV, and the fleet/residentd lifecycle),
and states the consolidation plan. Proposal only — no commits, no pushes, no
shared/model edits. Every claim is cited `file:line` and was verified with
grep/read against this tree.

---

## 1. Inventory of scheduling on unified

### 1.1 The topology switch — `scheduler/topology_switch.c`

The only file under `scheduler/` (verified: `find scheduler/ -type f` returns
exactly `topology_switch.c`). It is a **recipe-switch state machine** that
re-shards KV residency between TP16 and PP16 in "about twenty seconds" with
requests in flight and KV warm where the NVMe tier still holds it
(`include/sparkpipe/spark_topology_switch.h:3-5`).

* States: `STEADY → QUIESCE → CHECKPOINT → SWAP → RESUME → STEADY`
  (`spark_topology_switch.h:113-121`). The protocol is crash-only with no CANCEL
  (`topology_switch.c:22-28`).
* **Admission gate**: `SparkTopologySwitchAdmissionsOpen` returns 1 only in
  `STEADY` (`topology_switch.c:195-198`). `SparkTopologySwitchBegin` closes
  admissions synchronously the moment it lands (`topology_switch.c:330-348`;
  contract `spark_topology_switch.h:297-303`).
* **Sequence lifecycle**: Track / SetSequenceKv / AtBoundary / Complete mirror
  admission and completion (`topology_switch.c:217-328`). `TrackSequence`
  refuses to register under any recipe but the current one (`topology_switch.c:226-231`).
* **Checkpoint/resume**: per-sequence manifest on the tier with pins; resume
  classifies each sequence WARM (skip prefill) vs RECOMPUTE (`topology_switch.c:366-534`).
* **Budget arithmetic**: `SparkTopologySwitchEstimateBudget` prices quiesce =
  one decode step, checkpoint = manifest bytes at write bandwidth, swap = pack
  bytes at read bandwidth, plus a configured fixed cost (`topology_switch.c:622-652`;
  struct `spark_topology_switch.h:195-204`).
* The swap device and the checkpoint write path are **vtables** so the schedule is
  host-verifiable (`spark_topology_switch.h:72-74, 135-167`). Tests drive mocks:
  `tests/test_topology_switch.c:324-363` (admission closes at Begin, reopens when
  resident).

**Where admission/priorities actually live.** The brief's phrase
"scheduler/ (topology_switch.c, admission/priorities)" is imprecise: `scheduler/`
holds no admission or priority files (grep for `*admission*`/`*priorit*` returns
nothing). The admission/priority core is spread across `include/sparkpipe/`,
`src/`, and `runtime/` (inventoried in §1.3 below).

### 1.2 `runtime/model_serving_adapter.c` — the adapter admission/validation paths

This file is the **model-neutral validation gate** every adapter flows through
before work reaches a driver. It is not the policy core; it is the shape/capacity/
ABI gate that a submission must pass to be admissible:

* `SparkModelServingAdapterValidateDescriptor` — capability-flag coherence, stage
  geometry, codecs, and ceilings (`model_serving_adapter.c:27-148`).
* `SparkModelServingAdapterValidateRuntimeLimits` — runtime limits vs descriptor,
  including the **page budgets the runtime/scheduler own** (`model_serving_adapter.c:150-181`;
  contract `spark_model_serving_adapter.h:146-152`).
* `SparkModelServingAdapterValidateInterface` — required capability flags and the
  full vtable (`model_serving_adapter.c:183-209`).
* `SparkModelServingAdapterValidateSubmission` — the per-submission admission gate
  (work-kind capability, `tokens_per_sequence` rules, row/lane coherence)
  (`model_serving_adapter.c:348-402`); `ValidateRuntimeSubmission` adds runtime
  ceilings (`model_serving_adapter.c:404-422`).
* `SparkModelServingAdapterPrepareSubmission` — validate + optional **prefetch**
  (`model_serving_adapter.c:424-440`), and `ResolvePrefetch` — the COMMIT/ABORT
  terminal resolution of a prepared cache admission (`model_serving_adapter.c:442-467`).
* Completion-side gates: `ValidateCompletion`, `ValidateCompletionResidency`,
  `ValidateStageCompletion` (`model_serving_adapter.c:519-604`).
* Shared-object load with required-capability enforcement:
  `LoadInterfaceFromSharedObject` (`model_serving_adapter.c:606-637`).

The prefetch COMMIT/ABORT contract ("a terminal resolution must not return BUSY or
PENDING") lives in `spark_model_serving_adapter.h:313-321`; quiesce "permanently
closes admission" for the instance (`spark_model_serving_adapter.h:325-333`).

### 1.3 The admission / priority core

* **Admission types** — `include/sparkpipe/spark_model_driver.h`:
  rejection enum (`184-192`, incl. `REJECTED_DEADLINE` at `189`); request with
  `deadline_time_ns` and `priority` (`194-215`); decision with cost/queue/
  service estimates (`217-234`); the `admit` vtable entry (`314`).
* **The orchestrator** — `src/spark_orchestrator.c` is the real policy core.
  `SparkBuildDriverAdmissionRequest` copies frame priority/deadline into the
  admission request (`488-505`); `SparkDriverAdmissionDecisionCost` folds
  endpoint cost + queue delay + service time + private-queue pressure + memcpy +
  host staging − residency score (`507-523`); `SparkReserveRouteEndpoint`
  round-robins the route and picks the **lowest-cost accepting endpoint**
  (`552-622`); `SparkOrchestratorSubmit` applies the decision and submits with
  retry on BUSY (`624-702`).
* **Shared decision helpers** — `runtime/stage_module_common.c:763-814`
  (`SparkStageModuleAdmissionDecisionInitialize/Accept/Reject`), consumed by the
  per-model stage modules.
* **Generated admit plumbing** — `runtime/pack/driver_compiler.c` emits per-driver
  `admit` + decision merge helpers (`692-729, 791-872, 1001`); the loader requires
  `admit` (`src/spark_driver_loader.c:145`).
* **Per-model duplication (the consolidation target)** — every model re-wraps the
  same admit call:
  * serving adapters: `SparkGlm52ServingAdmit` (`modules/glm52_resident_decode_stage/source/spark_glm52_serving_adapter.c:885-937`), `SparkDsv4ServingBuildCacheAdmission` + `SparkModelDriverEvaluateAdmission` (`modules/dsv4_resident_decode_stage/source/spark_dsv4_serving_adapter.c:1207-1310`), `SparkQwen38_27bServingAdmit` (`modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_serving_adapter.c:981-1028`), `SparkQwen38ServingAdmit` (`modules/qwen38_resident_decode_stage/source/spark_qwen38_serving_adapter.c:841-888`).
  * stage modules: `SparkGlm52ResidentDecodeStageAdmit` (`.../spark_glm52_resident_decode_stage_module.c:1554-1587`), `SparkDsv4ResidentDecodeStageAdmit` (`.../spark_dsv4_resident_decode_stage_module.c:5871-5943`), `SparkQwen38_27bResidentDecodeStageAdmit` (`.../spark_qwen38_27b_resident_decode_stage_module.c:2031-2112`), `SparkQwen38ResidentDecodeStageAdmit` (`.../spark_qwen38_resident_decode_stage_module.c:1242`).
  * K3 has **no** `admit` yet (grep of `modules/k3_resident_decode_stage/` for
    `admit|Admit|Admission` returns nothing) — consistent with K3 being land-locked
    per `COORDINATION.md:103-107`.

### 1.4 Batch formation — the model-neutral engine + the B1-B1024 ladder

**Engine** — `runtime/model_batch_engine.c` (2120 lines) is already
model-neutral: it drives everything through the adapter descriptor and a pipeline
client, never a named model.

* Request state machine and the three selection passes —
  `SPARK_MODEL_BATCH_SELECT_AGED/PRIORITY/FILL` (`10-21`); engine fields incl.
  `admission_open` (`116`) and per-request `priority` (`28`).
* **Pure scheduler functions** — `runtime/model_batch_scheduler.h:5-36`, implemented
  in `model_batch_engine.c`: group sizing (`160-178`), cache-bound lane count
  (`180-191`), cache demand fit (`193-203`), page-capacity fit (`205-221`),
  mixed-lane planning (`223-242`), work-kind round-robin with tail bypass
  (`244-289`).
* **Priority selection** — `SparkModelBatchSelectRequests` scans for the maximum
  priority among not-aged requests (`1550-1570`) then runs three passes
  (aged → priority → fill) (`1587-1594`); `SelectRequestPass` enforces the pass
  predicates (`1492-1535`).
* **Prefill row assignment** — `SparkModelBatchAssignPrefillCounts` distributes the
  group row budget across lanes (`1603-1641`).
* **Admission close/shutdown** — `SparkModelBatchEngineCloseAdmission`
  (`2021-2028`), `BeginShutdown` closes admission and cancels live requests
  (`2030-2054`). `Submit` is gated on `admission_open` (`1157-1191`, gate at
  `1169`). The CLI `node/model_batch.c` uses CloseAdmission after submitting all
  batch rows (`node/model_batch.c:546-548`).
* Public API — `include/sparkpipe/spark_model_batch_engine.h`; `priority` is a
  submit-time field (`71`); the single-owner event-loop contract (`111-120`).

**B1-B1024 ladder (batch tuning headers)** — eleven capacity-ceiling buckets,
"every power of two from B1 to B1024", `-DSPARK_BATCH_BUCKET=<n>` the only
difference between compiled variants:

* Reference: `modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h`
  — the "why" (`4-36`), the eleven-bucket `#error` guard (`48-55`), grouped-tile
  derivation (`150-156`), runtime bucket ceiling (`164-191`), module-id table
  (`193-223`).
* DSV4 variant: `modules/dsv4_resident_decode_stage/include/sparkpipe/spark_dsv4_batch_tuning.h`
  — same contract, `.b<n>.` convention (`11-19, 30-37`), `SEQUENCE_CEILING = bucket`
  (`105-110`), ceiling fn (`129-156`).
* K3 variant: `modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_batch_tuning.h`
  — same ladder (`19-22, 33-40`), top-16-of-896 tile (`108-117`), ceiling fn
  (`122-149`).

All three `Spark<Model>BatchVariantBucketCeiling` functions are **byte-identical
integer ladders**; only the module-id tables are model content (the headers say so:
`spark_dsv4_batch_tuning.h:126-128`, `spark_k3_batch_tuning.h:119-121`).

### 1.5 Gang scheduling for co-resident models

Removed: obsolete fleet procedure (see the note at the top).

### 1.6 Priority / deadline / quanta

* **Priority** — submit field `priority` (`spark_model_batch_engine.h:71`);
  admission field `priority` (`spark_model_driver.h:209`); serving submission
  field `priority` (`spark_model_serving_adapter.h:193`); engine maximum-priority
  selection (`model_batch_engine.c:1562-1570`). `REALTIME_PRIORITY_THRESHOLD` is a
  dispatch-policy constant being neutralized (`docs/DRY_CONSOLIDATION_PLAN.md:33-40`).
* **Deadline** — admission request `deadline_time_ns` (`spark_model_driver.h:206`);
  serving submission `deadline_time_ns` (`spark_model_serving_adapter.h:187`);
  rejection reason `REJECTED_DEADLINE` (`spark_model_driver.h:189`); residentd
  deadline expiry → IO_ERROR completion (`node/model_residentd.c:922-985`); adapter
  quiesce deadline (`spark_model_serving_adapter.h:325-333`).
* **Quanta** — "short bounded quanta" for high-priority interactive work
  (`README.md:49-53`); "bounded all-rank quanta" (`ARCHITECTURE.md:141-142`). The
  concrete unit of scheduling granularity is the **decode step**: the topology switch
  quiesce bound is "one decode step … atomic at the token boundary"
  (`spark_topology_switch.h:176`, `topology_switch.c:350-354`), and continuation
  position advances are tracked as `step_generation` in the continuation lease
  (`runtime/model_continuation_lease.h:7-13`).

### 1.7 The residentd lifecycle — `node/`

`node/README:1-13` states the role: "WHAT RUNS ON ONE" spark (`residentd.c` holds
model weights resident in GPU memory across processes; `memlink_tool.c` inspects
shared memory).

Lifecycle in `node/model_residentd.c` (2720 lines):

* Signal-driven stop: `SIGINT`/`SIGTERM` set `SparkModelResidentdStop`
  (`242-246`, registered at `2701-2702`).
* Boot: parse launch → load deployment JSON → build config → validate dirs →
  initialize (`2679-2703`). Init phases `adapter_load` → `transport_load` →
  resources (`1129-1302`, phase strings at `1201, 1236`). Deployment schema
  (`adapter`/`driver`/`transport`/`runtime_limits`/`nodes`) is parsed by
  `runtime/model_resident_deployment.c:10-38`; runtime-limit members at `27-32`.
* Ready banner prints rank/stage/inflight/active/rows/resident + adapter identity
  (`2707-2709`).
* Run loop: poll control/transport/wake fds, accept one client, progress routes
  (`2630-2677`); poll timeout drops to 10 ms while routes are active (`2618-2628`).
* Continuation lease: established on slot completion, decode position advanced by
  the *completed* token count (DSpark partial-accept safe) (`750-782`).
* Deadline expiry: `SparkModelResidentdExpireTransportRouteLocked` compares
  `deadline_time_ns` to `CLOCK_MONOTONIC` and queues an IO_ERROR completion
  (`966-985`).
* Shutdown: close client/listener → **quiesce the adapter** (progress until OK,
  bounded by `SPARK_MODEL_RESIDENTD_QUIESCE_TIMEOUT_NS`) → synchronize streams →
  close transports → destroy adapter (`1401-1470`). On quiesce failure it
  "preserv[es] live resources until process exit" (`1453-1457`).

### 1.8 Fleet swap / promotion (<60 s)

Removed: obsolete fleet procedure (see the note at the top).

### 1.9 COORDINATION.md directive — ring windows + hourly progress rule

Removed: obsolete fleet procedure (see the note at the top).

---

## 2. The boundary contract

Who owns what today, and the target line. "Owns" = may edit and is responsible for
correctness; everything else is review-only (charter: `AGENT_CHARTER.md:13-16`).

### 2.1 Scheduler owns

* **Admission policy + priority/deadline decision** — the orchestrator's
  cost-based endpoint selection (`src/spark_orchestrator.c:552-622`) and the
  batch engine's aged/priority/fill selection (`runtime/model_batch_engine.c:1492-1601`).
* **Batch formation + work-kind scheduling** — `runtime/model_batch_scheduler.h`
  (pure functions) and `runtime/model_batch_engine.c` (state machine). The B1-B1024
  ladder arithmetic is scheduler-owned; the per-family module-id tables are model
  content.
* **Topology-switch state machine + admissions gate** — `scheduler/topology_switch.c`
  and `include/sparkpipe/spark_topology_switch.h`.

### 2.2 Model drivers own

* The **per-module `admit` implementation** and its shape/capacity predicates —
  `Spark<Model>ResidentDecodeStageAdmit` in each
  `modules/*_resident_decode_stage/source/*_module.c` (cited §1.3). These decide
  `UNSUPPORTED_SHAPE` / `KV_CAPACITY` / `BUSY` from model geometry.
* The **per-model batch tuning constants** (experts, top_k, tile) and the module-id
  strings — the three `spark_*_batch_tuning.h` files. Scheduler may not name a
  model in shared code (DRY law, `AGENT_CHARTER.md:34-35`); model facts stay in
  tables.
* The per-model serving adapters (`modules/*/source/spark_*_serving_adapter.c`) and
  runners (`spark_dsv4_stage_runner.c`, `spark_k3_resident_decode_stage_runner.cu`).

### 2.3 Transport owns

* `ring/transport/` — `tp_collective.c`, `tp_device_collective.c`,
  `tp_device_collective_nccl.c`, `rdma_control.c`. The collectives carry their own
  admission gate (`atomic_uint admission_open` at `tp_device_collective.c:112`,
  `tp_device_collective_nccl.c:92`). The scheduler **queries** transport state; it
  does not own the fabric.
* `runtime/hidden_transport` glue and `runtime/model_resident_ipc.c` /
  `model_resident_client.c` / `model_pipeline_client.c` are the scheduler's
  **client surface** into residentd, but the wire protocol (`spark_model_resident_ipc.h`)
  is shared machinery — proposal-level changes flow through the coordinator.

### 2.4 KV owns

Moved to [driver acceptance](../DRIVER_ACCEPTANCE.md), "Page and position budgets".

### 2.5 Current blur (what consolidation fixes)

1. **Admission plumbing is re-implemented per model** — four serving adapters each
   wrap `driver->interface->admit` with identical boilerplate, and four stage modules
   each re-derive the accept/reject decision (§1.3). The shared decision struct and
   helpers already exist (`spark_model_driver.h:217-234`,
   `stage_module_common.c:770-814`); only the policy loop is duplicated.
2. **Batch ladder arithmetic is duplicated across three families** — the
   `Spark<Model>BatchVariantBucketCeiling` integer ladder is byte-identical
   (§1.4). Model content (module-id strings) and model-neutral arithmetic (the
   ceiling walk) are entangled.

---

## 3. Consolidation plan (with owners)

Aligns with `docs/DRY_CONSOLIDATION_PLAN.md` (especially the in-flight
dispatch-policy split at `33-40`). Landings go through the coordinator; the
scheduler agent proposes, does **not** edit shared/model dirs.

### 3.1 One admission/policy core consumed by all adapters

* **Goal.** A single model-neutral `admit` loop that (a) validates ABI/shape
  against the adapter descriptor, (b) applies capacity predicates via a small
  vtable of model callbacks, (c) produces the cost fields the orchestrator already
  consumes (`src/spark_orchestrator.c:507-523`). The per-model
  `Spark<Model>ServingAdmit` and `Spark<Model>ResidentDecodeStageAdmit` wrappers
  collapse into table-driven config + shared code.
* **Shape.** Extend the existing shared surface — keep `SparkModelDriverAdmissionRequest/Decision`
  (`spark_model_driver.h:194-234`) and `stage_module_common.c` helpers as the
  substrate. Introduce a neutral "admission policy" module under `runtime/` +
  `include/sparkpipe/` that owns: deadline check (`deadline_time_ns` vs
  `CLOCK_MONOTONIC`, mirroring `node/model_residentd.c:966-985`), KV-page fit
  (`model_batch_engine.c:193-221` logic lifted out), and priority ordering.
  Model residue = a table of shape/capacity predicates (DRY law: no model name in
  shared code).
* **Owners.** Scheduler agent proposes the core; coordinator reviews/lands under
  `include/sparkpipe/` + `runtime/` (shared). Each MODEL agent migrates its own
  `*_serving_adapter.c` / `*_module.c` to call it (their dirs, their edit).
* **Sequence.** Land after the in-flight dispatch-policy neutralization
  (`DRY_CONSOLIDATION_PLAN.md:33-40`), since both touch the same headers.

### 3.2 The model-neutral batch engine

* **Goal.** `runtime/model_batch_engine.c` is already model-neutral (§1.4); the
  remaining duplication is the **B1-B1024 ladder**. Extract one shared
  `SparkBatchVariantBucketCeiling` integer ladder (model-neutral) into
  `include/sparkpipe/`, leaving each family's `spark_*_batch_tuning.h` to supply
  only its geometry macros + module-id table. The per-family inline wrappers become
  a table lookup + one shared call, preserving the "duplicated per family because
  module IDs are model content" property (`spark_dsv4_batch_tuning.h:126-128`).
* **Owners.** Scheduler proposes the shared ladder + the table convention; each MODEL
  agent (dsv4, glm52, k3) adopts it in their tuning header. Coordinator lands the
  shared file and runs the variant build to prove the eleven IDs are byte-identical
  (the header's own invariant, `spark_glm52_batch_tuning.h:64-77`).
* **Guardrail.** Generated files must match their generator byte-exact
  (`AGENT_CHARTER.md:34-35`); the variant module IDs (`.b1.`…`.b1024.`) are
  content-addressed artifacts — do not rename, only re-derive (§1.4).

### 3.3 Ring-window enforcement hooks

Removed: obsolete fleet procedure (see the note at the top).

---

## 4. Verification note

Every `file:line` above was confirmed with grep/read against this tree. One
correction to the brief, made explicit rather than silently absorbed:

* `scheduler/` contains **only** `topology_switch.c`; "admission/priorities"
  live in `include/sparkpipe/` + `src/` + `runtime/` (§1.1).
