# SparkPipe full-program PERT and dispatch plan

Status: authoritative planning baseline, 2026-08-25. This is a dependency and
evidence plan, not a claim that planned functionality already exists.

The machine-readable source of truth is
`orchestration/program_pert.json`. It is generated and validated by
`tools/program_pert.py`; hand edits to the JSON are not authoritative. The
architecture and acceptance semantics are in `docs/PLATFORM_ARCHITECTURE.md`.

## 1. Scope decisions fixed by this graph

- The initial model matrix is Qwen 3.8 27B, DSV4 Flash, GLM 5.2, K3, DSV4
  Pro, Qwen 3.8 Max, and MiniMax H3.
- Full-precision compute means an accuracy-qualified BF16/FP16 spine with FP32
  accumulation and numerically sensitive reductions. Weight storage may use
  independently qualified 4, 5, 6, 7, 8, or 16-bit codecs.
- NVIDIA CUDA, AMD ROCm, Apple-Silicon Metal, and a deterministic CPU reference
  are first-class backends behind one hardware interface.
- Recipes describe model identity, precision, codec, topology, B/context
  envelope, KV format and ownership, target hardware, co-residency, and release
  policy. A mnemonic names the canonical recipe/build identity.
- The platform schedules requests globally to a qualified compute island, then
  schedules batches and ranks locally within that island.
- A provider fee is exactly 10% of capacity actually sold through SparkPipe,
  represented as qualified capacity-in-kind credits. It is not a cash fee and
  does not accrue on idle or merely advertised capacity.
- Customer, compute-provider, operator/SRE, and SDK surfaces are part of the
  product, not a post-platform afterthought.

## 2. Program size and meaning of the estimate

The generated baseline contains:

| Measure | Baseline |
| --- | ---: |
| Work packages and milestones | 423 |
| Workstreams | 25 |
| Expected engineering effort | 1,079.6 pair-days |
| Unconstrained critical path | 80.0 engineering days |
| Heuristic critical-path P90 | 84.7 engineering days |
| Resource/lock-constrained planning forecast | 272.4 engineering days |
| Root tasks | 4 |
| Production super-sinks | 1 |
| Dependency-free pairable planning roots | 3 |
| Pairable implementation/audit packages | 394 |
| Peak unconstrained parallel pairs | 33 near day 57.7 |

A `pair-day` is elapsed work for one implementation/audit pair under the PERT
estimate, not two independent implementation days. With redundancy factor two,
the implementer role may race two providers and the auditor role may race two
providers, so one logical pair can have four provider requests in flight. The
first valid response advances that role; local journals preserve identical
context across provider changes.

The seven model programs are not fed through one interchangeable model queue.
Each has a durable `model-driver:<code>` lane and one dedicated logical
implementation/audit pair whose provider-neutral context persists across that
model's 17 work packages. Each role is still raced across two providers, audit
still starts from the sealed patch in an independent context, and any
cross-model or shared-runtime edit returns to the coordinator for ownership and
write-lock review. The live dashboard must distinguish these planned lanes
from genuinely active agents.

The unconstrained number assumes unlimited independent pairs and no
scarce-hardware queue. Under that assumption, concurrency grows to the stated
peak after the contract fan-out. The separate resource-constrained forecast
applies the declared 32-pair worker pool, 128 simultaneous provider-request
slots, at least two independently qualified provider failure domains, typed
hardware quantities, conservative workstream/path locks, semantic
dependencies, and the OxAlpha dispatch gate. Both forecasts exclude external
legal/calendar lead time.

The O/M/P values are an initial coordinator heuristic, not historical
calibration. Their 84.7-day P90 is only mathematical propagation of that
heuristic. `FND-011` requires independent owner estimates, basis, variance,
hardware calendars, and lock scopes before any delivery commitment. These
numbers are planning bounds, not promises. Parallelism cannot compress a
serial correctness gate, unavailable AMD/Metal/fleet hardware, soak duration,
or legal approval.

## 3. Work breakdown

| Workstream | Tasks | Expected pair-days | Primary outcome |
| --- | ---: | ---: | --- |
| Architecture and foundation | 11 | 14.9 | Frozen identities, objects, evidence, ownership, calibrated estimates |
| OxAlpha development system | 16 | 20.2 | Resilient races, executable contracts, live dispatch state, provider discovery, audit/status |
| Main/unified reconciliation | 10 | 14.0 | Selective salvage onto clean current main |
| Recipes and compiler | 20 | 34.2 | Mnemonic recipe-to-build and topology legality |
| Artifacts, Ceph, and file agent | 20 | 40.9 | Rank-local packs, warm cache, storage sanity |
| Hardware-neutral interface | 10 | 16.5 | Common capabilities, execution, CPU oracle |
| NVIDIA CUDA | 10 | 21.7 | Zero-regression Spark backend |
| AMD ROCm | 15 | 48.1 | MI350/gfx950 DSV4 backend and qualification |
| Apple Silicon Metal | 15 | 52.1 | Metal v1 primitives, Qwen 3.8 27B, two-Mac, API, and release path |
| Topology and inventory | 8 | 14.8 | Signed heterogeneous resource graph |
| Collectives and transport | 10 | 20.9 | Portable local/fabric collective substrate |
| Inference runtime | 18 | 40.1 | Resident B1-B1024 prefill/decode engine |
| KV and prefix hierarchy | 20 | 48.8 | Selectable KV formats, sharded active KV, 2.5 TB parked-session backing |
| Compute islands and scheduling | 20 | 48.1 | Global broker plus island-local scheduler |
| Public API | 16 | 37.1 | Authenticated compatible streaming service |
| Metering and capacity credits | 12 | 26.1 | Reconciled usage, payout, and in-kind fee |
| Compute-provider marketplace | 8 | 24.2 | Enrollment, offers, leases, qualification |
| Customer/provider/operator UI and SDKs | 16 | 40.5 | Complete product surfaces |
| Observability and SOTA status | 6 | 12.8 | One correlated status and evidence view |
| Security and privacy | 5 | 15.2 | End-to-end threat and release gates |
| SRE, HA, CI, and release | 15 | 43.9 | Fenced state, DR, zero-drift promotion |
| Seven model programs | 119 | 414.9 | Seventeen contract/kernel/runtime/service/release packages per model |
| SOTA performance program | 10 | 21.1 | Daily comparable target and no-drop tuning loop |
| Commercial/external gates | 4 | 8.3 plus external lead | Early terms, payments, final approval, launch |
| Whole-program milestones | 9 | 0 | Evidence-backed integration gates |

## 4. Phase waves

Phases are dependency waves, not calendar promises.

| Phase | Tasks | What opens |
| --- | ---: | --- |
| 0: truth and development control | 31 | Current-state baseline, architecture, pair harness, live dispatch state, contract refinement, provider discovery, daily SOTA ledger |
| 1: portable contracts | 51 | Recipes/codecs, HAL, topology, API, selectable KV, security, CI, early commercial requirements |
| 2: implementation fan-out | 102 | Artifact factory, backends, runtime, gateway, scheduler, provider and UI foundations |
| 3: subsystem integration | 118 | Rank packs, per-shape model kernels, full runtime, collectives, KV pager, ledgers |
| 4: product qualification | 86 | Minimum legal/production model topologies, API load, UIs, performance, failover |
| 5: production qualification | 28 | Fresh SOTA/model releases, backend/provider/security/SRE zero-drift gates |
| 6: marketplace release | 7 | DR, payment integration, final approval, seven-model and whole-product gates |

Dependencies, write locks, hardware reservations, and the independent audit are
all admission conditions. A phase number never overrides them.

## 5. Dominant critical path

The current unlimited-resource critical path is:

```text
current truth + product/model identity
-> canonical objects and consistency
-> recipe/topology/capacity compiler
-> topology-sliced rank artifacts and target kernels
-> resident runtime and B1-B1024 engine
-> island scheduling, KV admission, and chaos
-> gateway/broker integration and API chaos
-> security gate and provider UI qualification
-> external compute-island beta
-> production marketplace gate
```

There are tied branches at model identity versus product requirements, at
prefill versus decode, and at the hierarchical-scheduler release milestone.
The generated JSON records every zero-slack task and edge as well as one
deterministic representative path.

This path explains two architectural priorities: finish canonical contracts
early so work can fan out, and build the runtime/scheduler/API spine before
optimizing isolated kernels that cannot yet serve traffic.

## 6. Long paths that start as soon as their inputs exist

### 6.1 Development factory

`OXA-001..016` closes strict response validation, exact race accounting,
provider-neutral context, native-process ownership, fail-closed audit parsing,
immutable fingerprints, recovery, provider statistics, real-time status, and
4/8/32-pair chaos. Ordinary logical pairs reserve four provider-request slots
for implementer 2-way and auditor 2-way races. Dispatch also requires fresh
evidence for at least two independent provider failure domains. The workstream
defines the executable-contract schema, refines WBS packages into bounded
coding/audit contracts, and continuously qualifies new independent provider
supply. Broad paired dispatch is blocked until `OXA-012` passes.

The current unintegrated controller is specifically not treated as ready: its
known audit work includes uncaught race-worker exceptions, dispatcher shutdown,
choice-index and fractional-config validation, durable child PID ownership,
fail-closed malformed auditor findings, final patch fingerprint stability, and
the exact resume-attempt policy.

### 6.2 Daily SOTA and optimization

`PERF-001..010` builds a recurring service that scans primary sources, decides
comparability over the full checkpoint/quality/hardware/power/workload/SLO/
statistics tuple, establishes mandatory parity and a 110% economic target,
profiles production-shaped end-to-end requests, emits disjoint work orders,
and retains only numerical and service-level wins. Production receipts expire
after 24 hours. Daily refresh is an operating policy outside the finite DAG;
"SOTA" is never frozen in this PERT baseline.

### 6.3 Storage and JIT KV

`ART-001..020` creates normalized, codec, topology, rank, target, build, and
release artifacts. From the verified checkpoint manifest and compiler envelope,
the global planner proves the approximately 1 TB per-Spark model budget before
any variant shard generation or fleet placement movement, then signs an exact
plan. A privileged
per-host executor—not the coordinator—performs only leased, in-root operations
on proved devices/mountpoints, with idempotent journaling, hashes, temporary
peaks, signed local receipts, and no implicit deletion. The 48 TB Ceph 14+2
tier has a redesign/safety gate because it is not assumed currently
operational.

`KV-001..020` defines BF16/FP16/FP8/INT8 key/value formats and scale/layout
identity, reference conversion and accuracy, backend kernels/capabilities,
1/N ownership, legal B/context/KV cells, active VRAM paging, parked-session
backing, prediction/prefetch, anti-thrash behavior, and failure recovery. The
2.5 TB rank-local parked-session gate requires at least 95% matched-control
throughput, no active-token backing reads, under 0.1% restore-deadline misses,
at most 10% wasted prefetch bytes, and the declared latency SLO. It does not
claim the Cartesian B1024 x 256K maximum.

### 6.4 Backends

- `CUDA-001..010` preserves and qualifies current Spark performance while
  extracting the shared interface.
- `AMD-001..015` goes from exact MI350/gfx950 toolchain and probe through HIP,
  GEMM, low-bit, attention/MoE, RCCL, DSV4 full-model, and performance gates.
  Real-device tasks remain hardware-blocked until qualified AMD access exists.
- `MET-001..015` is the Apple-Silicon Metal v1 program: SoC matrix, macOS/MSL CI,
  capabilities, buffers/queues/events, command plans, 16-bit/FP32 primitives,
  codecs, dense/attention/MoE, multi-Mac transport, Qwen 3.8 27B full model,
  compute-island API service, power/thermal/performance, and release evidence.
  It does not imply support for another model or SoC without that cell's own
  receipt.

### 6.5 Models

Each of the seven models has 17 work packages: exact source/oracle;
recipe/capacity plan; rank-local build; separate dense, attention/state, and
FFN/MoE production shapes; complete layer; explicit collective/topology plan;
minimum-legal-topology full model; production topology; KV-resolution,
prefill, decode, and speculation cells; API/soak; fresh comparable-SOTA parity
with a report of 110% status; and zero-drift release. Parity is the model
release threshold. `PERF-009` separately gates fee-neutral sold-capacity claims
at 110%; it is not a model-release prerequisite. Existing receipts seed
investigation but do not skip requalification from merged main.

### 6.6 Product and marketplace

The API path covers compatibility fixtures, asynchronous gateway, model
catalog, chat/completions/Responses, streaming, tools, structured output,
sampling, idempotency, cancellation, overload, tenant policy, broker routing,
load/chaos, and SDK compatibility.

The customer UI covers keys, model selection, playground, request history,
quota, spend, invoices, policy, and status. The provider UI covers enrollment,
node setup, qualification, islands, health, storage/KV budget, offers, leases,
maintenance, sold capacity, cash payout, 10% capacity credits, disputes, and
exports. The operator UI correlates fleet, releases, providers, incidents,
ledgers, and fenced overrides.

## 7. Dispatch policy

1. Codex owns the source-of-truth checkout, task graph, write locks, final
   review, integration, and merge decisions.
2. A PERT entry is a work package, not automatically an agent prompt. The
   refinement gate must compile it into one or more executable contracts with
   exact objective, non-goals, writes, fixtures, commands, immutable outputs,
   hardware, auditor checks, integration owner, and rollback.
3. After `OXA-012`, a scheduler may lease one admitted contract to an
   implementer role and later an independent auditor role. Initial redundancy
   is machine-fixed at two for each role, using independent failure domains and
   the same provider-neutral context. One logical pair reserves four provider
   request slots, and launch requires at least two independently qualified
   failure domains with supply evidence no older than 24 hours.
4. The first structurally valid response advances the role and triggers the
   next turn immediately against the winner plus a different backup.
5. Raw prompts, responses, tool results, patches, tests, receipts, timings, and
   provider outcomes live in bounded local journals rather than Codex context.
6. The auditor starts from a fresh workspace and an immutable patch hash. A
   malformed or incomplete audit is rejection, never approval.
7. Only an auditor-approved, hash-stable patch enters the coordinator queue.
   Codex rechecks scope, conflicts, tests, architecture, and secrets before
   integration.
8. Live dispatchability is computed from an execution overlay—not the static
   PERT—from integrated dependencies and dispatch gates, admitted contract,
   exclusive structured write locks, typed hardware quantity, four available
   provider-request slots, two fresh independent provider failure domains,
   role/auditor state, and heartbeat age. Provider errors cause retry/rerace;
   numerical or design failures return to implementation.
9. Dashboard state comes from the durable event journal and shows every role,
   task, attempt, provider race, retries, queue depth, hardware wait, audit
   result, integration state, and stale heartbeat age.

## 8. First launch sequence

The graph begins with four semantic roots: coordinator-owned current-state
evidence, qualification receipts, the provider-pool contract, and the daily
SOTA-ledger system. The three pairable roots are planning information, not a
live-ready queue. The first productive order is:

1. Close the eight known OxAlpha audit findings with tests, independently
   re-audit the controller, then run 4-pair chaos before scaling to 8 and 32.
2. In parallel, finish current-state/model identity, evidence schema, source
   reconciliation, and the daily SOTA baseline.
3. Freeze only the shared object/ABI/schema seams and executable-contract
   admission; immediately fan out recipe,
   HAL, API, topology, KV, security, CI, and UI contract packages.
4. Launch hardware-independent compiler, scheduler-simulator, artifact, API,
   and UI work while CUDA, AMD, Metal, storage, and fabric tasks wait only at
   their first true device gate.
5. Reserve scarce hardware for acceptance runs, not open-ended coding. Keep
   CUDA performance, AMD bring-up, Metal bring-up, storage/KV, and seven model
   programs as separate queues with explicit target receipts.

The first day can therefore complete and launch a large amount of contract,
harness, reconciliation, scanning, simulator, and scaffold work. It cannot
truthfully complete the 423-task production program in one day.

## 9. Gate vocabulary

`G0..G9` follow the architecture qualification lattice: deterministic contract,
host tests, target compile, primitive, complete layer, minimum-legal-topology
full model, production topology, API lifecycle, performance/capacity, and
merged-main zero-drift release. Cross-cutting gates are `GS` security, `GH` high
availability, `GF` finance/capacity-credit reconciliation, and `GX` product UX.

No consensus of agents substitutes for a gate receipt. Host syntax is not AMD
or Metal execution; a microbenchmark is not full-model speed; a branch run is
not a production release; and a provider assertion is not billable usage.

## 10. Regeneration and validation

```bash
python3 tools/program_pert.py --summary
python3 tools/program_pert.py --write orchestration/program_pert.json
python3 tests/test_program_pert.py
```

The validator rejects duplicate IDs, missing dependencies, cycles, backward
phases, malformed PERT estimates, untyped hardware quantities, missing write
locks, required tasks outside the production super-sink, stale generated JSON,
decision drift, missing product/backend/model/KV work, invalid critical paths,
ungated broad dispatch, and accidental completion claims.
