# SparkPipe heterogeneous inference platform

Status: authoritative program architecture for implementation planning. This
document extends the firmware architecture in `SPEC.md`; it does not replace
the exact module, artifact, or validation contracts defined there. A statement
in this document is a requirement or design decision, not evidence that the
corresponding implementation exists.

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

The control plane separates content-addressed immutable definitions from
generation-fenced mutable state. No object mixes identity fields with observed
lifecycle state.

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

### 4.4 Deployment specification and state

An immutable `DeploymentSpec` binds one build to concrete device/resource
identities, rank packages, placement constraints, endpoints, and required
leases. A generation-fenced `DeploymentState` records observed paths, lease
identities, readiness, allocation generation, and active/standby lifecycle.
Only an `ACTIVE` state owned by the island reconciler can accept requests.

### 4.5 Model release, mnemonic alias, and active set

An immutable `ModelRelease` contains either one pinned build or a set of
target-specific builds proven equivalent for the public quality and feature
contract. Build-specific capacity envelopes and backend restrictions remain
visible; equivalence never means identical performance or topology.

A human mnemonic such as `dsv4f-w6-kv8-prod` is a mutable alias to one release
hash and includes a monotonically increasing alias revision. A deliberately
hardware-pinned product may use a release containing one build. The edge
records alias revision and release hash; the broker pins the exact member build
only when it commits the island lease. Requests and receipts retain all three
identities.

The active-model set is a declarative list of alias revisions plus placement
constraints. Changing it starts a reconciler; it does not directly kill or
copy a running model.

## 5. Precision and weight-storage contract

Source-weight identity, storage precision, effective dequantized weights, and
compute precision are independent fields.

```text
weight storage: packed_4, packed_5, packed_6, packed_7, int8, fp8, fp16, bf16
dequant output: bf16, fp16, fp32
activation:      bf16, fp16, fp32
accumulator:     fp32 for the full-precision class
collective:      bf16, fp16, fp32, int32, model-defined packed route
output/logits:   bf16, fp16, fp32
```

"Full precision calculations" means the quantized bytes are a storage
representation only: the compiled mathematical contract uses at least a
qualified 16-bit floating-point spine, normally BF16 or FP16 operands, with
FP32 accumulation and numerically sensitive reductions. It does not require
FP32 operands for every operation. Narrower accumulator, activation, router,
normalization, residual, or recurrent-state routes are a separately named
`reduced_precision` compute class with a distinct build identity; they cannot
satisfy the full-precision acceptance gate even if separately accurate enough
for another product tier. An all-FP32 route likewise receives its own build
identity and qualification bar.

The checkpoint contract hashes the original tensors and dtypes. A codec build
records the deterministic effective tensor after dequantization and its error
against that source. Low-bit storage is therefore never described as lossless
merely because later arithmetic follows the full-precision compute class.

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

The mandatory compiler-conformance matrix evaluates every `(TP, PP, EP)` tuple
whose degrees are in `{1, 2, 4, 8, 16}` and whose product fits the declared
reference-island rank count. This includes standalone TP2/4/8/16,
PP2/4/8/16, EP2/4/8/16, every fitting pairwise combination, and every fitting
three-axis combination. A cell must produce a byte-accounted legal build or a
deterministic `REJECTED_GEOMETRY`, `REJECTED_CAPACITY`, or
`REJECTED_TOPOLOGY` result. Model qualification then names the subset actually
deployed. ETP, CP, SP, KVP, DP, PD, and interleave add separate covering and
named production cells; they do not excuse a missing base matrix.

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

The production default performance budget is measured against a matched
all-resident control where that control fits: the JIT-backed workload must
retain at least 95 percent of control throughput, issue no backing read in an
active token's attention path, miss fewer than 0.1 percent of scheduled restore
deadlines, and waste no more than 10 percent of prefetched bytes. TTFT and
inter-token p50/p95/p99 remain inside the recipe's declared SLO. When the full
2.5 TB population cannot have an all-resident control, a scaled identical
working-set comparison is the gate and any larger-scale projection is labeled
analytical rather than measured. A recipe may tighten these limits; relaxing
them creates a separately disclosed service class.

### 7.3 Identity and crash behavior

Every backing object key includes model build hash, KV layout hash, rank/stage
fingerprint, sequence identity, token boundary, prefix digest, and mutation
epoch. A digest mismatch falls back to recomputation; it never attaches bytes
to a different build. Durable parked sessions require a journal and atomic slot
map. Until that journal is qualified, restart recovery is advisory and clearly
reported as such.

## 8. Artifact pipeline and warm cache

The planned 48 TB usable Ceph 14+2 warm tier will store immutable source
checkpoints, normalized tensor blocks, codec variants, sharded packages,
compiled link units, and receipts. It is not considered operational until its
physical devices, mountpoints, failure domains, erasure layout, rebuild path,
and sustained benchmark receipts pass the storage safety gate. Rank-local NVMe
stores activation-ready shards. Accelerator memory stores only the current
resident set.

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

The backend boundary supplies target capabilities and seven primitive families
from the frozen DSV4-first interface on `unified`: memory/queue lifecycle,
dense linear algebra, attention/KV, normalization/activation/residual,
routing/MoE, codec/packing, and sampling/output. These backend primitives are
not fleet compute islands. The interface uses static-link resolution, opaque
backend handles, and stream/queue-ordered completion. The portable model core
never branches on vendor in a request path.

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

Backend conformance and model support are separate claims. A backend can pass
the shared primitive suite without implying that every model is supported.
The initial production cells are:

| Backend release | Required v1 cell | What remains unclaimed |
| --- | --- | --- |
| CUDA/GB10 | shared primitive conformance plus requalified Qwen 3.8 27B and DSV4 Flash production paths | any model/topology cell without its own release receipt |
| ROCm/MI350P | shared primitive conformance plus DSV4 Flash TP1 and TP4 full-model/API/performance receipts | all other AMD model cells |
| Metal/Apple Silicon | shared dense, attention/KV, codec, routing/MoE, lifecycle, and collective conformance; Qwen 3.8 27B full-model/API/performance on a named Mac class; and a two-Mac transport/collective correctness cell | all other Metal model cells and unqualified SoC classes |

If Qwen 3.8 27B does not fit any declared supported Mac class, the Metal v1
production gate remains blocked. Lower-bit weight storage may make it fit, but
the confirmed full-precision compute class and numerical gate do not change.

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
and offers a signed capacity schedule. The initial business rule charges the
provider ten percent of capacity actually sold through the platform, paid in
capacity rather than cash. If a settlement window records `S` qualified
capacity units sold, `0.10 * S` capacity units become a platform-owned credit
that the platform may consume or resell. Unused advertised capacity does not
create the fee. The exact commercial contract, taxes, payments, and
jurisdictional rules require legal review and remain outside the firmware
protocol.

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
G0 schema and deterministic generation
G1 host syntax, unit, fuzz, and negative tests
G2 backend compile and target-capability checks
G3 single-island numerical oracle
G4 one-layer target numerical oracle
G5 minimum-legal-topology full-model parity
G6 multi-rank collective and topology parity
G7 API lifecycle, cancellation, restart, and load
G8 performance and capacity envelope
G9 merged-main zero-drift deployment receipt
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

- model-family convergence for DSV4, GLM 5.2, K3, Qwen 3.8 Max, and remaining
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
- provider commercial terms other than the fixed ten-percent sold-capacity
  in-kind/no-idle/no-cash fee rule;
- cross-site token-by-token pipeline execution; and
- deletion policy for generated shard variants.

Each deferred decision needs measured requirements. None is required to build
the first hardware-neutral, locally scheduled platform.

## 18. Present-state truth and migration premise

The program starts from a useful but inconsistent codebase, not from a clean
greenfield implementation. Status is therefore recorded in three independent
columns: operator assertion, repository evidence, and retained live receipt.
No column silently substitutes for another.

| Area | Present planning fact | Required evidence before a readiness claim |
| --- | --- | --- |
| Qwen 3.8 27B | Operator reports one-Spark inference works; parts of the tree and must-work target still carry the older Qwen 3.6 identity | Exact checkpoint migration, clean-main build, numerical receipt, API lifecycle receipt, and measured B/context cells |
| DSV4 Flash | Operator reports four-Spark inference works; retained branch-level TP4 receipts exist | Exact GA identity, merged-main zero-drift deployment, output parity, and current comparable-SOTA receipt |
| GLM 5.2 | Model code, packers, and validators exist, but full inference is not considered working | Real-pack one-layer oracle, full-model parity, multi-rank correctness, then performance |
| K3 | Driver, pack, and review artifacts exist; production inference is not established | Correct sharding, real-pack load, full-model parity, required fleet topology, then performance |
| DSV4 Pro | Review and bring-up artifacts exist | Exact contract, real pack, one-rank and multi-rank ladders, API and performance receipts |
| Qwen 3.8 Max | Review and single-stage artifacts exist | Exact contract, rank-local packs, authoritative oracle, full deployment and API receipts |
| MiniMax H3 | Product target only | Contract, packer, native module, full qualification lattice |
| CUDA | Only practical production backend today | Preserve current numerical and performance behavior while extracting the neutral seam |
| ROCm/AMD | Architecture and planning material exist; support is not presumed | gfx950 compile plus real MI350P numerical, collective, full-model, and performance receipts |
| Metal/macOS | No production backend is presumed | Apple-Silicon capability, kernel, full-model, API, and performance receipts per SoC class |
| Ceph warm tier | Proposed 48 TB usable 14+2 design conflicts with the currently stopped/masked operational state | Physical device/mount/failure-domain proof, reviewed enablement, rebuild test, sustained benchmark, and disaster-recovery receipt |
| Public API | Main contains useful API code but compatibility and robustness are incomplete | Protocol fixtures, auth/quota, streaming/cancel/idempotency, sustained load, and accounting receipts |
| JIT KV | Several cache/backing primitives exist | Common ownership model, durable journal, full-lane restore, pressure tests, and 2.5 TB tier receipt |
| Decentralized providers | Design only | Enrollment, attestation, signed offers/leases, independent usage receipts, settlement sandbox, and failure qualification |

`main` remains the integration base. `unified`, `unified-consolidated`, and
`dsh.sparkpipe` are evidence and salvage sources. They are never treated as a
second authority and are never merged wholesale. Every salvaged unit names its
source commit, destination, semantic conflict review, tests, and exact
qualification level.

## 19. Product boundary and actors

SparkPipe is five products sharing one set of immutable identities:

1. a native inference engine and model compiler;
2. a local and federated fleet control plane;
3. a versioned paid inference API;
4. a provider capacity marketplace; and
5. an internal model, artifact, qualification, and operations system.

The primary actors are:

- **API tenant**: buys requests or tokens under a model, feature, policy, and
  price contract;
- **platform operator**: controls aliases, active sets, qualification policy,
  routing, prices, and capacity-credit policy;
- **compute provider**: leases a qualified compute island and receives a
  reconciled payout without gaining authority to mint usage;
- **model engineer**: turns an exact checkpoint into recipes, builds, and
  qualification receipts;
- **fleet and storage operator**: proves physical resources, places immutable
  artifacts, and performs recoverable reconciliations; and
- **auditor**: independently verifies patches, builds, receipts, billing, and
  production claims.

The API tenant never addresses a rank, node, backend, shard, or batch bucket.
The provider never chooses tenant policy or billable token counts. The global
broker never schedules a CUDA stream. Those boundaries are deliberate.

## 20. Planes and ownership

The complete system is split into seven planes with versioned contracts:

```text
API edge plane
  auth, validation, quota reservation, streaming, cancellation, usage response
        |
global control plane
  aliases, policy, offers, routing, request leases, placement intents
        |
island control plane
  inventory, deployment reconciliation, residency, admission, batching, gangs
        |
execution data plane
  model stages, kernels, collectives, tokenizer/sampler, active KV

artifact plane
  checkpoints -> normalized blocks -> codecs -> shards -> builds -> releases

evidence and finance plane
  qualification, telemetry, usage, price snapshots, ledger, settlement

development plane
  task DAG, redundant OxAlpha calls, implementation/audit pairs, coordinator PRs
```

Every cross-plane call is generation-fenced and idempotent or has an explicit
transaction identity. A component may cache another plane's immutable object;
it may not become a second writer for that object's mutable pointer.

The initial production implementation may place several logical planes in one
process or database, but tests must preserve the boundaries so they can be
split without changing execution semantics.

## 21. End-to-end request lifecycle

A successful request follows one attributable path:

1. The edge assigns a request and idempotency identity and validates the
   versioned payload.
2. Tenant auth, feature policy, data policy, and a worst-case quota/price
   reservation succeed before provider selection.
3. The mnemonic resolves once to an alias revision and immutable release. A
   release supplies the set of qualified equivalent target builds.
4. The global broker filters signed, unexpired compute-island offers by release
   membership, exact member-build feature/capacity envelope, context, output,
   trust, region, retention, and SLA.
5. The broker scores remaining offers and acquires one generation-fenced
   request lease that pins the exact build and island. Optional redundant
   admission attempts have distinct attempt identities; only one may commit.
6. The selected island performs final admission against live weight,
   workspace, active-KV, backing-tier, batch, deadline, and gang capacity.
7. If necessary, residency scheduling restores a parked lane or promotes a
   `LOCAL_READY` deployment. The request remains queued with a bounded deadline;
   no false-ready response is emitted.
8. Prefill is chunked and scheduled under the compiled capacity envelope.
9. Decode joins a compatible dynamic batch. The driver sees an already
   validated fixed descriptor and performs no discovery or allocation.
10. Tokens are streamed in monotonically ordered frames tied to one build,
    island, allocation generation, and request attempt.
11. Cancellation is propagated to the island and every rank; resource release
    and the terminal event are idempotent.
12. Completion seals edge, island, and provider evidence. Prompt, cached,
    reasoning, accepted-speculative, and output token classes remain distinct.
13. The reserved quota is committed to actual usage and the remainder is
    released.
14. Independently produced usage evidence is reconciled into the integer
    double-entry ledger.
15. Retention policy decides whether request content, only hashes, or no
    payload-derived material survives delivery.

A request may switch islands only before its first output token unless a
separately qualified continuation-transfer protocol exists. A retry never
reuses a billable attempt identity.

## 22. Mutable state, events, and consistency

Immutable contracts, recipes, builds, packages, receipts, and ledger entries
are content-addressed. Mutable state is limited to named pointers and
state-machine instances:

- alias revision;
- desired active set;
- deployment state and allocation generation;
- provider enrollment and capacity-offer generation;
- request and attempt state;
- sequence/KV residency state;
- quota reservation state; and
- invoice, payout, and dispute state.

Every mutable transition appends an event before or atomically with publishing
the new generation. Consumers checkpoint an event cursor and tolerate replay.
Clock time is evidence, not identity; ordering comes from scoped sequence
numbers and generation fences.

| Mutable state | Sole transactional writer | Fence |
| --- | --- | --- |
| alias revision | catalog/alias service | alias name plus revision CAS |
| desired active set | active-set policy service | policy generation |
| deployment state/allocation | owning island reconciler | deployment and allocation generation |
| provider enrollment | provider registry | provider revision |
| signed capacity offer | one enrolled provider-island agent, accepted by the offer registry | island epoch, nonce, and expiry |
| request and attempt | gateway request coordinator | request id, attempt id, and terminal CAS |
| sequence/KV residency | owning island runtime | sequence epoch and slot generation |
| quota reservation | quota service | reservation id and ledger transaction |
| invoice, payout, credit, dispute | ledger/settlement service | immutable journal position and case revision |

Replicas and caches may serve reads or submit signed intents; none can publish a
second authoritative generation. Leader transfer changes the fenced writer,
not the ownership rule.

Required state machines include:

```text
request:
RECEIVED -> RESERVED -> ROUTING -> LEASED -> ISLAND_ADMITTED
         -> PREFILLING -> DECODING -> COMPLETED
terminal alternatives: REJECTED, CANCELLED, EXPIRED, FAILED

deployment:
ABSENT -> WARMING -> LOCAL_READY -> LOADING -> PREWARMING
       -> QUALIFYING -> READY -> ACTIVE -> DRAINING -> STANDBY
terminal/quarantine alternatives: FAILED, DEGRADED, QUARANTINED

provider:
APPLICANT -> PROBING -> QUALIFIED -> ACTIVE -> DEGRADED
          -> SUSPENDED -> RETIRED

artifact:
DECLARED -> BUILDING -> HASHED -> VERIFIED -> PUBLISHED -> PINNED
quarantine alternatives: CORRUPT, INCOMPLETE, REVOKED
```

No transient state is the only record of completed work. Restart tests stop the
writer between every adjacent pair of durable operations.

## 23. Storage, data placement, and the file agent

The storage design has explicit budgets rather than ad-hoc copies:

| Tier | Role | Planning constraint |
| --- | --- | --- |
| Planned Ceph 14+2, about 48 TB usable warm tier | canonical immutable checkpoints, normalized blocks, codec blocks, topology slices, builds, and receipts | unavailable to production until the physical/safety/benchmark gate passes; then content-addressed and demand-materialized |
| Spark rank-local NVMe | only artifacts needed by likely active builds plus local KV backing | target roughly 1 TB of model/build data per Spark so roughly 2-2.5 TB can remain available to KV backing, subject to proved physical capacity |
| Accelerator/unified memory | currently active weights, workspace, and active-lane KV | only compatible co-resident deployments; current allocation generation is authoritative |
| Mac cold/archive tier | recoverable cold source and receipts where configured | not confused with Spark warm/runtime storage and never inferred from a directory name |

The file system has two authorities. A global artifact catalog and placement
planner produces a content-addressed, signed plan; it cannot touch node files.
One privileged **per-host file executor** runs on each Spark or provider node
and is the only component allowed to execute that host's plan. Its credentials
are scoped to explicit physical device identities and configured mount roots.
Plans carry destination host, build/rank manifest, nonce, generation, lease,
byte ceiling, and expiry. The executor journals each step, is idempotent across
restart, refuses stale or out-of-root targets, and emits signed local receipts.
The coordinator, scheduler, and model agents may propose desired placement but
do not move arbitrary model files themselves. Before changing bytes on a
destination node, the planner and executor together:

1. proves physical block device, filesystem, and mountpoint;
2. snapshots source and destination manifests, free space, inode headroom, and
   active-process references;
3. resolves the exact recipe/build/rank manifest and expected byte budget;
4. emits a dry-run plan showing copies, hard-link/reflink reuse, generated
   shards, codec work, and temporary-space peaks;
5. stages resumably into an incomplete name and verifies per-file hashes;
6. atomically publishes only a complete rank-local manifest;
7. reports placement and remaining-capacity receipts; and
8. never deletes or overwrites source model data without a separate explicit
   authorization and recoverability plan.

There are two ordered budget gates. Before normalized, codec, topology, or rank
variants are generated, the verified checkpoint manifest and recipe capacity
envelope must prove the per-Spark model/KV budget and temporary peaks. After
the exact packages exist, the signed placement dry run verifies their actual
hashes and bytes against that ceiling before any fleet movement.

Never-changing reference data is represented once per physical device or
content block and referenced by rank manifests. A rank package contains only
the tensors, scales, tokenizer data, target artifacts, and metadata it needs.
The catalog detects full-checkpoint copies on every Spark and converts them
through a reviewed, no-delete migration before any reclaim proposal.

## 24. Model onboarding and support matrix

Every model follows the same lifecycle:

```text
source survey -> exact contract -> reference oracle -> normalized checkpoint
-> codec qualification -> topology legality -> rank packs -> one-layer gate
-> minimum-legal-topology full model -> production-topology parity
-> API lifecycle -> capacity matrix
-> comparable-SOTA loop -> merged-main zero-drift release
```

The first program matrix is:

| Program | CUDA objective | AMD objective | Macintosh objective | Initial topology focus |
| --- | --- | --- | --- | --- |
| Qwen 3.8 27B | preserve working one-Spark path, then B/context and API completion | first dense portability vehicle after HAL | required Metal v1 full-model cell on a named Mac class | TP1, then TP2/4 and DP replicas |
| DSV4 Flash | production TP4 correctness and at least comparable SOTA | first MI350P MoE/full-model target | contract/oracle first; full target capacity-dependent | TP4, TP4xPP4, EP variants |
| GLM 5.2 | establish real-pack correctness, then native MoE/attention speed | after DSV4 substrate | model contract and selected islands before full model | band deployment, TP/PP/EP compiler-selected |
| K3 | establish correctness and fleet-scale pack/runtime path | after common recurrent/MoE islands | deferred until memory and island coverage fit | likely exclusive large topology initially |
| DSV4 Pro | review-to-runtime program | reuse DSV4 Flash backend after exact contract delta | deferred | TP4xPP4 then alternatives |
| Qwen 3.8 Max | review-to-runtime, pack, oracle, and fleet program | dense/MoE islands as contract requires | deferred | compiler-selected large topology |
| MiniMax H3 | new-family onboarding through common contracts | after common backend coverage | deferred | decided by exact geometry and capacity |

The matrix is not a promise that every batch, context, topology, codec, and
backend Cartesian point will exist. The compiler publishes supported cells and
an explanation for rejected cells. New open-weight frontier models use the
same onboarding template and do not fork the API or scheduler.

## 25. API product contract

The public surface is versioned independently from firmware. The baseline
supports `/v1/models`, chat completions, responses, text completions where
useful, SSE streaming, tool calls, structured output, usage, and cancellation.
Each compatibility claim is tied to executable fixtures against named client
SDK versions.

Required semantics include:

- strict JSON and size validation with stable OpenAI-style error envelopes;
- exact model alias and immutable-build disclosure in response metadata;
- monotonic stream ordering, one terminal event, disconnect cancellation, and
  bounded backpressure;
- idempotency keys for non-streaming and streaming submission;
- tokenizer/chat-template identity in build provenance;
- explicit support/rejection of sampling, stop, seed, logprob, tools, response
  format, attachment, and reasoning fields;
- separate prompt, cached, reasoning, accepted-speculative, rejected-draft,
  and output usage counters;
- per-request deadline, priority, locality, and data-policy extensions under a
  SparkPipe namespace; and
- no direct exposure of provider secrets, ranks, private addresses, or raw
  scheduler state.

Admin, customer, and provider APIs are separate scopes. Public model listing
shows only builds the caller may route to, not every artifact in the catalog.

### 25.1 API-user console

The API-user UI is a product surface, not a demo page. It provides:

- account, organization, project, role, and scoped API-key management;
- model catalog with context/features, current qualification, price, region,
  data policy, and availability rather than internal topology details;
- an API playground with streaming, tools, structured output, usage, curl, and
  SDK examples generated from the exact request;
- request history and traces with content hidden or retained according to
  tenant policy;
- usage, quota, spend, invoices, credits, rate limits, and budget alerts;
- service status, incidents, SLO history, and per-model availability; and
- data-retention, external-provider, geography, and trust-tier controls.

Every displayed number comes from the same immutable usage, price, and status
objects used by the API. The UI cannot invent a second billing or model-status
calculation.

## 26. Provider and commercial lifecycle

Provider enrollment binds legal identity outside the protocol and technical
identity inside it. The technical path requires hardware inventory, topology,
bandwidth, numerical, sustained-load, failure, telemetry, artifact-integrity,
and data-policy probes. Qualification expires and can be revoked.

A provider advertises signed, short-lived offers. The platform acquires leases
and independently observes delivery. Settlement uses three evidence sources:
edge usage, island execution receipts, and provider telemetry. Disagreement
enters a dispute state; it never silently chooses the provider's number.

The ten-percent rule is modeled as a capacity-in-kind fee on sold capacity:

- the sold-capacity unit, settlement window, qualification state, and evidence
  are explicit;
- each provider owes the platform capacity credits equal to ten percent of the
  capacity actually sold through SparkPipe during that window;
- the fee is not cash and does not accrue on idle or merely advertised
  capacity;
- credits are generation-fenced and denominated in comparable capacity units
  so an unqualified slow unit cannot repay a qualified fast unit;
- the platform may schedule internal requests against credits or resell them,
  with the same usage and no-double-sale controls as ordinary capacity;
- provider pricing and cash payout for the sold service remain separate from
  the capacity-credit ledger; and
- sold usage, capacity credits, consumption, resale, expiry, reversals, taxes,
  and cash payouts use integer ledgers, never floating-point balance mutation.

Payments, tax handling, sanctions, consumer law, provider contracts, and data
processing agreements are external legal/financial gates. Engineering can
build a sandbox but cannot declare those gates complete.

### 26.1 Compute-provider console

The provider UI exposes only the provider's own islands and financial records.
It supports:

- organization, operator-role, credential, and payout-profile management;
- guided island enrollment and downloadable/installable provider-agent setup;
- hardware, topology, network, storage, software, trust, and qualification
  results with exact remediation steps;
- live capacity, offers, leases, active builds, maintenance/drain controls,
  health, telemetry, incidents, and qualification expiry;
- artifact warming progress and local storage/KV budgets without exposing
  tenant content;
- sold capacity, delivered usage, cash payout, disputes, and reconciliation;
- the ten-percent capacity-in-kind fee ledger, including earned obligation,
  available credits, platform consumption, resale, expiry, and reversals; and
- alerts for degraded delivery, stale offers, failed probes, policy changes,
  security action, and payout holds.

An operator/SRE console adds fleet-wide placement, aliases, active sets,
provider quarantine, capacity-credit scheduling, artifact provenance, incident
command, and ledger reconciliation. Its mutations require stronger roles and
append auditable control-plane events.

## 27. Observability, SOTA, and provider scorecards

Every request and deployment has correlated edge, broker, island, rank,
artifact, and ledger identities. Metrics have bounded cardinality; prompts,
secrets, API keys, raw tokens, private addresses, and unbounded request IDs do
not become metric labels.

The operational status surface includes:

- generation-fenced development Spark ownership, observed/desired assignment,
  current lease holder and deadline, next queued model, pending file-agent
  transition, and last qualifying throughput receipt;
- desired versus observed active models and promotion phase;
- node/island capacity, topology health, memory, storage, temperatures, power,
  and collective health;
- request queue, TTFT, inter-token latency, throughput, cancellation, and
  error class by model/build/batch/context envelope;
- KV active/parked bytes, restore queue, hit/miss, prefetch lead, churn, and
  backing-tier bandwidth;
- artifact availability and rank-local placement drift;
- provider offers, leases, sold-capacity credit obligation, delivered capacity,
  usage, and settlement state; and
- qualification/SOTA age and the current measured gap per supported cell.

API-provider racing and coding-agent quality are scored separately. Raw events
are retained and rollups are computed for 1-hour, 24-hour, 7-day, and lifetime
windows. Per provider/model/configuration/failure-domain scorecards include:

```text
attempts, valid responses, invalid responses, HTTP/transport errors, timeouts
race wins and win share
first-byte and completion p50/p95/p99
cancellation settlement lag and stranded-call count
reported input/output tokens and useful-task completion rate
auditor approval, rejection, blocked, and turn-limit rates
retries and coordinator integration acceptance
```

Transport speed never substitutes for useful agent completion. Provider
selection may optimize a bounded composite policy, but the raw dimensions and
sample counts remain visible.

The SOTA loop runs daily against primary sources. It freezes comparable cells,
profiles SparkPipe end to end, converts exposed wall time into narrow work
orders, and retains only quality-preserving measured wins. The economic target
is at least 110 percent of genuinely comparable SOTA throughput, not a claim
against incomparable hardware, model, quality, batch, or timing boundaries.

The canonical service-level comparison key is the exact checkpoint/revision,
quality and numerical gate, weight/compute/KV route, accelerator model and
count, topology/fabric, power and clock limits, batch and prompt/context/output
distribution, prefix/speculation mode, TTFT and inter-token SLO, metric unit and
direction, timing boundary, warmup, sample count, and statistical confidence
rule. A result missing any required key field is partial or incomparable and
cannot set the 110-percent economic target.

The development dashboard freezes a small operational slice of that matrix:
every supported model at B1, B8, and B64 with exactly 32,768 prompt tokens. It
lists SparkPipe's accepted best and the exact public SOTA prefill/output values,
the 110-percent targets, source/date/hardware, and the certified gap. An absent
or not-fully-comparable cell is `N/A`; a shorter prompt or different timing
boundary is never extrapolated into this view.

## 28. Reliability and failure semantics

The design is fail-closed at authority boundaries and fail-soft where retry is
safe.

| Failure | Required behavior |
| --- | --- |
| API disconnect | propagate cancellation; stop billable output; release reservation idempotently |
| Global broker restart | replay event log and leases; never double-route a committed attempt |
| Stale or duplicated offer | reject by provider/island generation, expiry, and nonce |
| Island admission race | only one generation-fenced commit wins; loser releases resources |
| Rank loss before first token | fail or reroute under a new attempt identity |
| Rank loss after first token | terminal failure unless a qualified continuation transfer exists |
| Provider partition | expire offers/leases; preserve independently observed usage; quarantine ambiguous settlement |
| Artifact hash mismatch | quarantine bytes and rebuild/restore from immutable provenance |
| Partial shard transfer | remain incomplete and unpublishable; resume by verified ranges |
| Backing-KV corruption | reject slot by digest and recompute; never attach to another build/sequence |
| KV tier saturation | backpressure or reject admission; do not evict in an unbounded loop |
| Promotion crash | recover from journal to old ACTIVE or new READY; never publish a half build |
| Metering disagreement | complete service delivery state separately; hold settlement for reconciliation |
| OxAlpha/API outage | retry with jitter/cooldown and race independent failure domains; preserve exact context |
| Agent crash or invalid final contract | retain artifacts/events and return task to the correct role/attempt |

Chaos tests cover every row, including process termination between durable
writes and simultaneous provider/rank/network failures.

## 29. Security, privacy, and abuse program

Security is a release workstream, not a final review. The program includes:

- threat models for tenant edge, operator control plane, provider island,
  artifact supply chain, model/plugin inputs, and billing;
- mutually authenticated control and data channels with scoped short-lived
  credentials;
- tenant, operator, provider, model, artifact, and service identities with
  least-privilege authorization;
- signed artifacts, provenance, dependency inventory, reproducible builds where
  practical, and revocation;
- provider isolation, workload/data-policy enforcement, log scrubbing, and
  retention/deletion attestations;
- rate, byte, token, context, tool, compilation, storage, and concurrency
  limits before expensive work;
- abuse monitoring and incident controls that do not require retaining prompt
  content by default;
- secret scanning and redaction in logs, receipts, agent sessions, patches, and
  support bundles; and
- independent review of public API, provider protocol, payout, and update
  mechanisms before external beta.

Untrusted provider hardware cannot be proved confidential by a heartbeat.
Sensitive tenants route only to a trust tier whose hardware, operator, region,
and retention controls satisfy policy.

## 30. Operations, release, and disaster recovery

Production changes use one evidence chain:

```text
auditor-approved patch -> coordinator review -> PR -> merged main
-> target pulls exact main -> clean rebuild/package -> atomic activation
-> restart -> live numerical/lifecycle/performance receipt
```

Hotpatches, copied binaries, dirty worktrees, unmerged branches, and simulator
results cannot satisfy the production gate. Rollback selects an earlier signed
release generation; it does not rebuild an approximation during an incident.

Operations require:

- declarative desired state and drift detection for nodes, networks, storage,
  releases, aliases, deployments, and provider offers;
- maintenance/drain, swap, rollback, and emergency-quarantine procedures;
- physical-device/mount receipts before storage operations;
- backup and restore for catalog, event log, leases, quota, ledger, aliases,
  provider identity, and signing metadata;
- regional/control-plane recovery objectives defined before external beta;
- synthetic canaries and real lifecycle probes for every active model;
- capacity, power, thermal, storage-wear, and network trend planning; and
- incident timelines that preserve evidence without exposing tenant content or
  secrets.

## 31. Development system and massively parallel execution

The development controller is separate from the inference control plane. Codex
is the coordinator and the repository is the source of truth. OxAlpha model
calls provide implementation and audit workers; no external harness owns
integration authority.

Each coding work package must have:

- frozen objective, non-goals, dependencies, write set, and protected paths;
- exact base commit and immutable candidate patch hash;
- deterministic host tests and explicitly labeled hardware gates;
- bounded tools, context, command output, workspace growth, and turn count;
- durable raw provider responses, tool artifacts, events, and final contract;
- at least one implementation result and an independent auditor result;
- retry/resume semantics for provider, process, and controller failure; and
- coordinator review before any canonical checkout change.

Target-hardware work additionally requires a durable model-lane binding and a
generation-fenced development lease. Big-model activation drains the current
small models as one atomic group and is executed by the files agent from a
coordinator plan. The initial lease is sixty minutes from proven activation.
Only a passed numerical benchmark at an exact B1/B8/B64, 32k-input cell can
renew it: a missing first baseline may be established once, and every later
renewal requires at least a 1.00-percent accepted-best gain in prefill or
output throughput. Code activity, tests, profiles, and heartbeats are not
progress. Expiry fences the old generation before the next queued model can
receive the Sparks; rollback restores the small-model group when appropriate.

The provider racer sends the same context to `R` providers in independent
failure domains, accepts the first structurally valid response, cancels losers,
and carries the exact conversation forward with the winner plus a fresh
backup. `R` begins at two and may rise only when enough independent providers
exist and scorecards show a net latency/reliability benefit.

Parallelism is bounded by semantic dependencies, target-hardware slots,
write-set locks, test capacity, and coordinator integration bandwidth. Dozens
of pairs are useful only after interfaces are frozen and tasks are truly
disjoint. Long-path model correctness, HAL, recipe, artifact, KV, scheduler,
and API-contract work starts first; cosmetic surfaces and speculative
optimizations do not consume scarce target windows.

## 32. Program requirements and final acceptance

The program is complete only when all of these requirements have direct,
retained evidence:

| Requirement | Final evidence class |
| --- | --- |
| One API routes to local and approved external compute islands | sustained multi-island lifecycle and failover receipts |
| B1 through B1024 offered load with contexts up to 256K where capacity permits | compiled capacity envelopes plus accepted/rejected matrix and load receipts |
| 4/5/6/7/8/16-bit weight storage with unchanged declared compute semantics | codec round-trip, tensor/layer/full-model error, memory, and performance receipts |
| TP/PP/EP degree matrix at 1/2/4/8/16 plus ETP/CP/SP/KVP/DP/PD legal combinations | exhaustive fitting base-matrix build-or-reject fixtures, covering arrays for added axes, and named multi-rank topology receipts |
| Approximately 1/N physical rank-local KV where geometry permits | byte-accounting and live ownership/replication receipts |
| Configurable KV representation | numerical, capacity, and throughput matrix |
| Recipe creates or recovers all shards and optimized kernels under a mnemonic | clean-cache rebuild/recovery and provenance equivalence receipt |
| Active mnemonic set swaps warm builds in at most one minute | measured all-rank promotion distribution, not an analytical estimate |
| Compatible small models coexist; exclusive models drain safely | contention, isolation, fairness, swap, and rollback receipts |
| About 2.5 TB rank-local KV backing behaves as parked-session capacity with minimal loss | matched control shows at least 95% throughput, no active-token backing read, under 0.1% restore-deadline misses, at most 10% wasted prefetch bytes, declared latency SLO, plus pressure/crash/long-context receipts |
| Rank-local model data remains within the configured storage budget | physical device/mount plus per-node manifest and free-space receipts |
| NVIDIA, AMD, and Apple-Silicon backends use one portable upper contract | shared primitive conformance plus named v1 model cells: requalified CUDA paths, DSV4 Flash TP1/TP4 on AMD, and Qwen 3.8 27B plus two-Mac collective correctness on Metal |
| Daily comparable-SOTA loop and at least parity, with 110 percent target | primary-source ledger keyed by the full service-level comparison tuple and paired production-shape A/B receipts |
| Paid tenant API has exact usage and no double billing | protocol, load, cancellation, quota, ledger, invoice, and reconciliation receipts |
| Providers pay a ten-percent capacity-in-kind fee on sold capacity | signed sold-capacity, credit mint, consumption/resale, no-double-sale, and settlement receipts |
| Customer, provider, and operator consoles are production surfaces | scoped-role, accessibility, responsive, stale/offline, browser E2E, and canonical-status/billing reconciliation receipts |
| Security, privacy, retention, trust tier, and artifact supply chain are enforced | independent threat, fuzz, authz, tenant-isolation, replay, tamper, dependency, retention, and incident-response gate |
| Customer payment collection and provider cash payout reconcile separately from the in-kind fee | payment/payout sandbox and production integration, duplicate webhook, hold, reversal, tax record, backup/restore, and ledger balance receipts |
| Control plane survives process, node, provider, and network failure | deterministic restart and chaos matrix |
| Development uses resilient provider racing and independent pair audits | scale run with raw scorecards, retries, rejection loops, and coordinator integrations |

The machine-readable PERT/WBS maps every row to implementation, qualification,
operations, and external-gate work packages. A task is not complete because its
code exists; its declared evidence must pass at the qualification level the
task claims.
