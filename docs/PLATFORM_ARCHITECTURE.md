# SparkPipe heterogeneous inference platform

Status: design baseline for implementation. This document extends the firmware
architecture in `SPEC.md`; it does not replace the exact module, artifact, or
validation contracts defined there.

## 1. Product objective

SparkPipe becomes one inference fabric across locally connected NVIDIA, AMD,
and Apple Silicon hardware, with an eventual option to admit third-party
capacity. A caller sees one versioned API and one model name. The platform
chooses an already resident deployment or promotes an exact compiled model
instance, forms batches, schedules collectives, accounts for KV residency, and
returns metered output.

The platform must support:

- dynamic batches from B1 through B1024;
- request contexts up to 256 Ki tokens, subject to an explicit capacity
  envelope for each compiled instance;
- weight storage at 4, 5, 6, 7, 8, or 16 bits while keeping quantization out of
  the mathematical model contract;
- TP, PP, EP, context/sequence parallelism, data-parallel replicas, and valid
  combinations of those axes;
- approximately `1/N` rank-local KV ownership when the model geometry permits
  it, rather than silently replicating KV across tensor ranks;
- an explicit KV representation and accuracy policy;
- immutable, content-addressed model builds selected through memorable names;
- sub-minute activation when all promotion prerequisites are warm;
- co-resident smaller models when resources and collective schedules permit;
- active-lane KV in accelerator memory and parked-lane KV in the 2.5 TB local
  backing tier;
- homogeneous execution groups within a model stage at first, with explicit
  stage boundaries for heterogeneous or disaggregated execution; and
- signed capacity leases and usage receipts for external providers.

This is a large program. The first 24-hour objective is a coherent control
plane, schemas, interfaces, test scaffolds, and independently audited patches.
Production CUDA, ROCm, Metal, public billing, and decentralized scheduling are
qualification programs, not one-day claims.

## 2. Non-negotiable invariants

1. Correctness precedes throughput. Every release claim names an exact
   checkpoint, target, topology, artifact hash, build receipt, and numerical
   receipt.
2. Quantized weight storage never implicitly changes compute semantics.
3. Active decode never demand-pages a lane's KV token by token. A lane is fully
   resident before its next decode quantum.
4. A topology that cannot shard a tensor, expert set, or KV head layout exactly
   is rejected or records the intentional replication. It is never described
   as sharded when it is not.
5. Runtime request submission performs no module search, validation, graph
   interpretation, or hardware probing.
6. A compiled deployment never mixes CUDA, ROCm, or Metal ranks inside one
   collective group until a separately qualified mixed-vendor protocol exists.
7. Model promotion is atomic: a deployment is either unavailable or fully
   ready under one immutable identity.
8. Provider capacity is admitted by signed lease and evidence, not by an
   unauthenticated heartbeat.
9. Measured, simulated, analytical, and unverified results remain distinct.
10. Only the coordinator-owned canonical checkout is integrated or pushed.

## 3. What is retained from the current codebase

The current codebase contains useful pieces of nearly every layer. The plan is
to preserve those contracts and remove Spark-specific assumptions at their
seams.

| Existing area | Keep | Required change |
| --- | --- | --- |
| `SPEC.md`, module library, generated direct-call drivers | Exact artifact and firmware model | Add a higher-level recipe compiler that emits the existing model description and never enters the hot path |
| `model_contracts/` and family adapters in `tools/generate_recipe.py` | Authoritative model geometry and deterministic placement logic | Generalize strategy axes, target capabilities, weight codecs, KV ownership, and capacity envelopes |
| `schema/model_description.schema.json` | Low-level compiled-stage language | Keep v1 valid; introduce a separate model-instance recipe schema rather than overloading firmware JSON |
| `cache/kv_cache.c`, page directory/store, NVMe and prefix-cache layers | Model-neutral allocation, identity, and tier mechanics | Converge duplicate implementations and make head/context ownership explicit |
| `runtime/spark_kv_backing.c` and `docs/JIT_KV_DESIGN.md` | Park/restore backing primitive and anti-thrash rule | Finish module transfer operations, scheduler backpressure, crash identity, and end-to-end tests |
| serving adapters and resident daemon | Model lifecycle and native execution | Extract neutral lifecycle/accounting skeletons without forcing model internals into the orchestrator |
| scheduler and admission code | Priority, deadline, gang, and batch concepts | Implement one fleet scheduler over capability inventory, placements, residency, and leases |
| release manager and content-addressed packages | Atomic publication and promotion foundation | Add recipe/build/deployment catalog records and multi-target artifacts |
| topology schemas and probes | Exact route and device evidence | Generalize from Spark/DGX profiles to capability graphs and locality groups |
| main-branch API work | OpenAI-compatible starting point | Replace subprocess/single-request assumptions with streaming, cancellation, idempotency, auth, quotas, and usage accounting |
| unified-branch hardware interface and AMD plan | Frozen DSV4-first seam and gfx950 work plan | Port in reviewed slices onto current main; do not merge the branch wholesale |

The `unified` branch contains valuable hardware-neutral and non-Qwen work, but
it also diverges across hundreds of files, predates current renames and API
repairs, and includes temporary artifacts. Current `main` is therefore the
reconciliation base. Each salvaged unit needs an explicit source commit,
destination paths, conflict review, and gates.

Initial selective salvage candidates are:

- the normative parts of `a03b801` (`hwiface_v1.md`) separated from its large
  stagepack fixture bundle;
- `f70c526` (MI350P/gfx950 plan);
- `3c52223` (Qwen 3.8 Max bring-up runbook);
- `3355cfd` and `5fab2a7` architecture/duplication audits as review input;
- shared stagepack and prefix-cache mechanics only with byte-parity tests; and
- individually proven model fixes, rebased after the main-branch rename.

`unified` and `unified-consolidated` are evidence sources, not merge targets.

## 4. Canonical object model

The control plane uses four immutable object types and two mutable pointers.

### 4.1 Model contract

A model contract records facts supplied by the checkpoint and architecture:
layer counts, tensor names and shapes, attention/KV geometry, expert geometry,
position encoding, tokenizer identity, reference precision, and checkpoint
hashes. It contains no fleet placement decision.

### 4.2 Model recipe

A recipe records an operator's requested instance:

- contract identity and exact checkpoint revision;
- weight storage codecs and quantization metadata;
- compute, accumulation, activation, collective, and output precision;
- topology policy and allowed degree sets;
- batch and context envelope;
- KV representation, sharding, paging, and accuracy policy;
- target capability constraints and placement locality;
- compiler/kernel optimization policy;
- promotion, co-residency, and exclusivity policy; and
- qualification bars.

Recipes are canonical JSON. Their SHA-256 covers every semantic field. Unknown
fields are rejected at publication so misspellings cannot mint unintended
instances.

### 4.3 Model build

A build is the deterministic resolution of a recipe for one target class and
one exact topology. It contains:

- rank/stage/expert/context placement;
- exact rank-local shard manifests;
- the emitted low-level model-description JSON for every stage;
- selected link units and fused-kernel artifacts;
- communicator and transport plans;
- memory and bandwidth estimates;
- the valid `(batch, context)` capacity envelope;
- build, numerical, and performance-validation identities; and
- a complete provenance DAG.

One recipe may produce several builds. A build is never retargeted in place.

### 4.4 Model deployment

A deployment binds one build to concrete devices, storage paths, endpoints,
capacity leases, and a readiness epoch. It records active/standby state and is
the only object that can accept requests.

### 4.5 Mnemonic alias and active set

A human mnemonic is a mutable alias such as `dsv4f-tp4-w6-kv8-prod`. It points
to one immutable recipe or build hash and includes a monotonically increasing
alias revision. Requests and receipts record both alias and immutable hash.

The active-model set is a declarative list of alias revisions plus placement
constraints. Changing it starts a reconciler; it does not directly kill or
copy a running model.

## 5. Precision and weight-storage contract

Storage precision and compute precision are independent fields.

```text
weight storage: packed_4, packed_5, packed_6, packed_7, int8, fp8, bf16
dequant output: bf16, fp16, fp32
activation:      bf16, fp16, fp32, model-defined mixed route
accumulator:     fp32 or an explicitly qualified model-defined route
collective:      bf16, fp16, fp32, int32, model-defined packed route
output/logits:   bf16, fp16, fp32
```

"Full precision calculations" means the quantized bytes are a storage
representation only: the compiled mathematical contract uses the model's
qualified floating-point route, normally BF16/FP16 operands with FP32
accumulation/reductions where required. It does not silently mean FP32 for
every operation. A recipe may request all-FP32, but capacity and throughput
then receive a separate build identity and qualification bar.

Every non-16-bit storage codec defines:

- bit packing and byte order;
- group and scale geometry;
- zero-point policy;
- exceptional value handling;
- deterministic dequantization;
- required target primitives;
- calibration/provenance identity; and
- tensor-, layer-, and end-to-end error bars.

Five-, six-, and seven-bit formats are real codecs, not aliases for eight-bit
containers. Offline packers emit content-addressed blocks that can be sliced
without decoding unrelated tensors. Fused kernels may dequantize while
loading, but their numerical output is compared with the same compute contract
as an ahead-of-time dequantized reference.

## 6. Parallelism and sharding

The placement language treats parallel axes independently:

- `DP`: complete deployment replicas for throughput and availability;
- `TP`: tensor dimensions and attention heads;
- `PP`: contiguous or compiler-approved layer stages;
- `EP`: routed-expert ownership;
- `ETP`: tensor parallelism inside an expert when one expert does not fit;
- `CP/SP`: context or sequence ownership for prefill/attention;
- `KVP`: explicit KV-head/latent ownership when different from TP;
- `PD`: disaggregated prefill and decode groups; and
- `INTERLEAVE`: virtual pipeline stages when bubbles justify it.

The compiler validates that the product and nesting of degrees match concrete
devices and that each tensor's split extent, head count, quantization group,
expert count, and collective has a legal partition. Valid replication is
recorded in the build; it reduces the claimed sharding factor.

The initial heterogeneous rule is stage-homogeneous:

- all ranks in a TP, EP, ETP, KVP, or device collective use one backend and a
  qualified fabric profile;
- PP or prefill/decode boundaries may cross hardware classes through an
  explicit transport contract; and
- cross-site execution is opt-in and must fit the latency/deadline envelope.

This preserves fused local collectives while still allowing an AMD prefill
stage, NVIDIA decode stage, or Apple development target in separate builds.

## 7. KV system

### 7.1 Ownership and representation

KV policy specifies key/value formats independently, scale granularity, page
tokens, block bytes, target error budget, KVP/CP degree, and whether recurrent
or compressed latent state accompanies each lane.

The build receipt reports:

```text
logical bytes per token
physical bytes per token per rank
replication factor
head/context ownership map
active VRAM capacity
local backing horizon
shared-prefix savings assumptions
```

The required `approximately 1/N` property is tested against physical bytes per
rank. If a topology has 16 TP ranks but only eight divisible KV heads, the
compiler must select KVP8 plus explicit replication, choose a different
parallel plan, or refuse it.

### 7.2 Active versus parked lanes

Attention decode reads the active lane's full usable context each token.
Consequently:

- active lanes have all required private KV resident in accelerator memory;
- parked lanes have no private accelerator KV and retain authenticated backing
  slots plus small identity/state metadata;
- shared prefixes may remain resident while referenced and spill by digest;
- restore completes before the first decode token of the next quantum; and
- admission queues instead of repeatedly evicting young active lanes.

The 2.5 TB tier makes accelerator memory appear larger at the request-lifecycle
level, not at the individual attention-load level. Prefetch predicts which
lane will run next and restores whole required block sets in the background.

### 7.3 Identity and crash behavior

Every backing object key includes model build hash, KV layout hash, rank/stage
fingerprint, sequence identity, token boundary, prefix digest, and mutation
epoch. A digest mismatch falls back to recomputation; it never attaches bytes
to a different build. Durable parked sessions require a journal and atomic slot
map. Until that journal is qualified, restart recovery is advisory and clearly
reported as such.

## 8. Artifact pipeline and warm cache

The 48 TB Ceph 14+2 tier stores immutable source checkpoints, normalized tensor
blocks, codec variants, sharded packages, compiled link units, and receipts.
Rank-local NVMe stores activation-ready shards. Accelerator memory stores only
the current resident set.

Artifacts form a DAG:

```text
checkpoint blocks
  -> normalized tensors
  -> codec blocks
  -> topology slices
  -> rank-local package
  -> target-specific kernels/link units
  -> compiled build
```

This avoids materializing every combinatorial variant. Frequently requested
builds can be pinned exhaustively; other builds are generated lazily from
shared content blocks. The catalog tracks reuse and garbage-collection
eligibility, but deletion is a separate authorized operation.

Promotion has an explicit state machine:

```text
ABSENT -> WARMING -> LOCAL_READY -> LOADING -> PREWARMING
       -> QUALIFYING -> READY -> ACTIVE -> DRAINING -> STANDBY
```

A promotion meets the 60-second goal only when all rank-local packages and
compiled target artifacts are already `LOCAL_READY`. The build records a byte
budget and a measured bandwidth floor that prove the goal is arithmetically
possible. Loading from Ceph is warming, not activation, unless a measured
profile proves otherwise.

## 9. Hardware interface

The backend boundary supplies target capabilities and seven execution-island
families from the frozen DSV4-first interface on `unified`. It uses static-link
resolution, opaque backend handles, and stream/queue-ordered completion. The
portable model core never branches on vendor in a request path.

Initial target keys are:

```text
cuda.sm121.gb10
rocm.gfx950.mi350p
metal.apple-silicon.<gpu-family>
cpu.reference.<isa>
```

The descriptor records memory capacity, usable bandwidth, wave/subgroup width,
supported operand/accumulator formats, matrix/tensor primitive shapes, graph or
command-buffer capabilities, peer access, collective backends, host-transfer
properties, and firmware ABI versions.

### 9.1 NVIDIA

Current CUDA/GB10 kernels remain the production baseline. CUDA-specific tile
selection stays below the interface. Extraction must show zero performance and
numerical regression for retained production shapes.

### 9.2 AMD

MI350P/gfx950 is the first ROCm target. HIP supplies memory, stream, event,
graph, and host-callback primitives; RCCL supplies same-target collectives.
Native MXFP4 layout decisions stay in the backend. Bring-up order is host
substrate, integer/exact islands, one TP1 layer, one full TP1 decode, then RCCL
TP4 and performance work.

### 9.3 Macintosh

The production Macintosh target is Apple Silicon. Metal command queues and
events implement the execution substrate; Metal kernels implement fused model
islands. MPSGraph or MLX may be used as bring-up/oracle tools but are not an
implicit production fallback. Unified memory changes copy policy but does not
remove residency accounting. Each SoC/GPU family receives its own capability
and performance receipt. Intel Macs are CPU-reference targets unless a
separate qualified backend is added.

## 10. Fleet inventory and placement

Each node publishes a signed capability record. Resources are separated into:

- fungible quantities: free memory, storage bytes, bandwidth, queue slots,
  measured throughput, power budget, and leased time; and
- non-fungible constraints: backend/ISA, exact device topology, link profile,
  trust tier, geography, data policy, artifact locality, and model approval.

### 10.1 Fleet compute islands

A **fleet compute island** is the request-routing and lease boundary: one
locally managed set of hardware on which a model deployment can execute with a
qualified latency/topology contract. It may be one Apple Silicon Mac, one AMD
server, a four-Spark TP group, or a larger provider-local cluster. This term is
distinct from the backend's fused kernel execution islands.

Each fleet compute island owns its detailed hardware inventory, gang
allocations, resident-model catalog, KV pressure, local admission, batching,
and device scheduling. It advertises short-lived signed service offers rather
than streaming raw device state to the public API tier. An offer binds:

```text
island and provider identity
exact model build and active allocation generation
supported feature, batch, and context envelope
available request, token, KV, prefill, and decode capacity
queue and service-time estimates
region, trust tier, retention and data policy
price/SLA card and qualification receipt
lease epoch, expiry and replay-resistant offer nonce
```

The global broker consumes these offers. The island-local scheduler remains the
only component that decides which accelerator, rank, KV slot, batch bucket, or
collective quantum executes a routed request.

The fleet placement controller solves a constrained deployment problem:

1. filter nodes by non-fungible requirements;
2. form locality groups that satisfy every collective and stage edge;
3. fit weights, workspace, active KV, parked-KV horizon, and promotion reserve;
4. choose an existing ready build or request compilation/warming;
5. score latency, throughput, reliability, energy, external cost, and
   fragmentation;
6. acquire signed, expiring capacity leases; and
7. publish an immutable deployment plan.

The first implementation is a single authoritative scheduler with an
append-only event log. Decentralization then separates placement authority
from provider execution using signed leases and receipts. A replicated
consensus control plane comes after state transitions and failure semantics are
stable; it is not required to prove the scheduling model.

## 11. Runtime scheduling

Scheduling is hierarchical. The gateway does not carry a dedicated per-device
hardware view once decentralized providers are admitted.

1. **fleet placement** operates over minutes and creates qualified deployments
   on compute islands;
2. **global request routing** resolves a mnemonic once, enforces tenant policy,
   and leases one matching compute-island offer per request;
3. **island residency scheduling** operates over seconds and controls local
   promotion, co-residency, lane parking, and restore futures;
4. **island request admission and batch formation** operates over milliseconds
   using the detailed hardware/KV view; and
5. **device gang scheduling** operates over execution quanta and collectives.

The global broker hard-filters on exact build, qualification, context/features,
region/data policy, trust, lease freshness, and capacity. It then scores SLO,
queue time, reliability, price, and affinity. A generation-fenced capacity
lease prevents the same provider capacity from being sold twice. The selected
island performs final local admission against current physical state.

A request may be rerouted only before its first output token. Every attempt has
a distinct identity; only the winning attempt is billable. After streaming
starts, the request remains pinned to its exact build, island, allocation, and
lease generation. Failure then terminates the stream or uses an explicitly
qualified continuation-transfer protocol; it never silently changes models.

Requests specify model, deadline, priority, maximum output, streaming policy,
and optional locality/data constraints. Callers never choose a B number.

The batcher maintains B1..B1024 buckets but may launch any nonempty width the
driver contract supports. It uses the compiled capacity envelope to cap
context by batch and to select prefill chunks, decode width, and speculation.
Fairness uses bounded quanta and deficit/deadline accounting. A collective
group is gang-scheduled; unrelated work cannot enter between committed ranks.

The scheduler treats restore, promotion, and compilation as futures. It can
admit a request into a bounded queue while those futures run, but it cannot
claim the deployment is active before readiness publication.

## 12. API, metering, and marketplace

The public gateway is a separate hardened control-plane service. The current
`node/model_api.c` is a private token-ID development shim and must not bind a
public interface or be advertised as full OpenAI compatibility.

The new gateway supports versioned OpenAI-compatible endpoints, including
model listing, chat completions, responses, streaming, tool-call payloads,
cancellation, and usage. Compatibility tests pin request validation, error
objects, finish reasons, stream ordering, idempotency, and token accounting.
It authenticates and reserves quota before asking the global broker to lease a
compute island; it does not select ranks or batch widths.

SparkPipe extensions live in an optional namespaced object for priority,
deadline, data locality, and reproducibility. Standard clients can omit it.

Public service requires:

- tenant keys, scoped credentials, quotas, and abuse controls;
- request idempotency and durable accounting;
- prompt/output token metering tied to deployment receipts;
- price snapshots and signed usage statements;
- provider payout and dispute records;
- per-model data-retention and trust policies; and
- explicit disclosure when an external provider can observe request content.

An external provider enrolls hardware, passes topology and numerical probes,
and offers a signed capacity schedule. The initial business rule interprets
the proposed ten percent as a reserved capacity tranche: ten percent of the
accepted capacity is available to the platform owner to use or resell under
the lease. The exact commercial contract, taxes, payments, and jurisdictional
rules require legal review and remain outside the firmware protocol.

Provider nodes receive only expiring deployment and request leases. They
cannot mint usage. The gateway and deployment agents produce independently
reconcilable receipts containing request, token, model-build, hardware,
timing, and lease identities.

## 13. Security and trust

- Secrets never enter model packages, prompts, logs, or agent patches.
- Every control-plane mutation is authenticated, authorized, and journaled.
- Tenant payload retention is policy-bound and defaults to the minimum needed
  for delivery/accounting.
- Artifact bytes are signed and verified at every tier transition.
- Provider trust tiers determine which models and data classes may run.
- Network peers are mutually authenticated; management links never satisfy a
  data-plane topology requirement.
- Resource limits cover request bytes, context, output, tools, concurrency,
  parked KV, storage, compilation, and promotion.
- Cancellation and lease expiry are tested under partial failure.
- Marketplace payout cannot rely only on provider-reported tokens or time.

## 14. Qualification lattice

Every build progresses through explicit gates:

```text
L0 schema and deterministic generation
L1 host syntax, unit, fuzz, and negative tests
L2 backend compile and target-capability checks
L3 single-island numerical oracle
L4 one-layer target numerical oracle
L5 one-rank full-model parity
L6 multi-rank collective and topology parity
L7 API lifecycle, cancellation, restart, and load
L8 performance and capacity envelope
L9 merged-main zero-drift deployment receipt
```

A target can be useful at an earlier level without being called production
ready. macOS simulation does not qualify Metal hardware; host HIP syntax does
not qualify gfx950; a branch-level Spark result does not qualify merged main.

Matrix dimensions include model/checkpoint, backend, topology, weight codec,
compute route, KV format, batch, context, speculation, prefix reuse,
park/restore, cancellation, failover, and API mode. Pairwise or covering-array
tests reduce combinatorics; named production cells remain exhaustive.

## 15. Branch reconciliation and delivery

1. Keep current `main` as the base.
2. Record every desired `unified` unit in a reconciliation manifest with source
   commit, paths, semantic owner, conflicting main changes, and gates.
3. Extract or cherry-pick one coherent unit at a time. Do not import `tmp/`,
   logs, binaries, stale names, or branch-only receipts as current truth.
4. Re-run host gates after every unit and target gates before any readiness
   claim.
5. Land schema and interfaces before parallel backend/model work.
6. Use short-lived implementation branches; only auditor-approved patches
   enter coordinator review.
7. Merge through PRs. Pull merged main on target nodes, rebuild/install,
   restart, and only then record production measurements.

## 16. Delivery phases

### Phase 0: first 24 hours

- freeze this architecture and PERT/task graph;
- stand up the resilient paired-agent controller and dashboard;
- create the recipe, target, topology, lease, and receipt schemas;
- establish branch-reconciliation and qualification manifests;
- implement host-only scheduler/resource-model skeletons;
- implement API compatibility contract tests and gateway skeleton;
- port the hardware-interface contract as reviewed source; and
- start CUDA/ROCm/Metal backend scaffolds behind compile gates.

### Phase 1: days 2-14

- one complete recipe-to-build path for Qwen 3.8 27B on CUDA;
- one-layer then full-model DSV4 ROCm bring-up on provisioned MI350P;
- Metal capability probe and one qualified kernel island on Apple Silicon;
- global placement, promotion, active-set reconciliation, and scheduler tests;
- JIT-KV park/restore integration and anti-thrash load tests; and
- production-grade authenticated streaming API on local hardware.

### Phase 2: weeks 3-8

- model-family convergence for DSV4, GLM 5.2, K3, Qwen Max, and remaining
  targets;
- topology and codec matrix expansion;
- Ceph artifact factory and sub-minute activation receipts;
- provider enrollment, leasing, metering, and payout sandbox; and
- multi-node failover and security qualification.

### Phase 3: production marketplace

- legal/commercial launch controls;
- trust tiers, data residency, provider attestations, dispute handling;
- replicated control plane and regional schedulers where justified; and
- audited billing, tax, and payout integrations.

## 17. Decisions intentionally deferred

- mixed-vendor collectives inside one TP/EP group;
- the final distributed-consensus mechanism;
- universal FP32 compute as a product tier;
- exact provider commercial terms;
- cross-site token-by-token pipeline execution; and
- deletion policy for generated shard variants.

Each deferred decision needs measured requirements. None is required to build
the first hardware-neutral, locally scheduled platform.
