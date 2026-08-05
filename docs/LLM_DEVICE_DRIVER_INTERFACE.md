# LLM device-driver interface

This document defines the boundary between the neutral SparkPipe scheduler and a model-specific resident firmware driver. It describes the current implementation: model-driver ABI **7**, firmware-module ABI **4**, firmware-host-services ABI **2**, and immutable module-record schema **4**.

SparkPipe is not a universal tensor runtime. The neutral layer owns package loading, route selection, admission comparison, inflight accounting, completion retirement, CUDA-stream lifetime, hidden transport, prompt chunking, decode batching, and aggregate operational counters. A model driver owns model geometry, resident weights, KV layout, sequence lanes, CUDA graphs, MoE policy, speculation state, token selection, and all other model-specific execution details. It executes on the resident-owned stream and receives resident-owned boundary buffers; it does not create pipeline transport or private execution streams.

## Physical ownership

```text
include/sparkpipe/             neutral public ABI only
src/                          neutral core, compiler, loader, orchestrator
model-families/common/        reusable model-runtime facilities
model-families/<family>/      model-family host implementation
modules/<family>_*/           linkable model firmware
deployment/                   release and rollout services
```

The core/compiler/runtime compilation closure must not require any model-family include directory. `make architecture_audit` enforces that rule against source includes, archive members, and exported symbols.

## ABI objects

### Model-driver ABI 7

The generated or hand-authored driver exports `SparkModelDriverGetInterface` and a `SparkModelDriverInterface` whose `abi_version` is `SPARK_MODEL_DRIVER_ABI_VERSION`.

The interface provides:

```text
create       validate a sized request and bind an instance to a node context and execution stream
admit        report whether one exact frame shape can be accepted now
snapshot     expose neutral aggregate counters
submit       execute one published program through its program descriptor
destroy      release a quiescent instance
completion   return externally complete work when the program owns completion
```

Every descriptor carrying `descriptor_bytes` must be validated before fields added by the current ABI are read. The create request itself is versioned and exact-sized. Reserved fields must be zero. Unknown flags are rejected rather than ignored.

### Firmware-module ABI 4

Each operation is initialized with:

- `SparkFirmwareModuleConfiguration`, including ABI, descriptor size, operation index, model/stage/program/operation identities, and immutable configuration JSON;
- `SparkFirmwareModuleHostServices`, including completion and wake callbacks, node identity, node target, the opaque node context, and the resident-owned execution stream.

Every module initializer must call `SparkFirmwareModuleValidateInitialization`. The helper clears the returned module state, validates both descriptor sizes and ABI versions, and rejects nonzero reserved fields.

## Node and frame contexts

`SparkModelDriverCreateRequest.node_context` is model-specific and opaque to SparkPipe. It may bind resident allocations, model graph slots, and fixed capacities. Stream and hidden-transport ownership are separate neutral runtime resources and must not be hidden in this object.

`SparkModelDriverFrame.user_context` is model-specific. Persistent model identities—such as sequence lane, KV block table, prefill ranges, GDN snapshots, or MTP draft state—belong there or in model-specific views reachable from it.

The generic `driver_dispatch_slot` is an advisory execution-resource ticket. It is not a sequence lane, KV owner, or model-state identifier. A model module that does not implement dispatch-ticket validation must reject `SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID`. The DSV4, K3, MiMo 2.5, and Qwen 3.6 modules in this proposal do so and never read `frame->driver_dispatch_slot` as model state.

## Buffers

A model module must validate, before any transfer or kernel launch:

- exact buffer count for its stage position and frame mode;
- logical slot number;
- allowed and required read/write flags;
- non-null address;
- overflow-safe minimum byte size;
- whether the stage actually owns embedding input or final-head output.

Intermediate pipeline stages must not require artificial host token buffers. Hidden-state transport remains resident-owned; only the exact input and output boundary pointers are passed through the model-specific frame context.

## Shared model-runtime transport

Reusable transport implementations belong under `model-families/common/`, not in the neutral scheduler ABI. The shared TCP tensor-parallel reference collective uses its own ABI 2 and is intentionally narrower than a universal transport graph.

Every rank must receive one immutable configuration containing a nonzero collective identifier, degree, rank, numeric IPv4 peers, and explicit connection and operation deadlines. Reciprocal connection handshakes validate group identity and topology. Every all-reduce exchanges a protocol header that validates group, operation sequence, recursive-doubling step, sender rank, operation kind, and element count before any payload is transferred.

The collective supports one serial operation stream per instance. Peer loss, timeout, cross-group wiring, call-order disagreement, or shape disagreement permanently fails and closes the instance. The native-F32 reference wire format assumes homogeneous peers. These restrictions are explicit because silently reusing a damaged connection or interpreting mismatched payload lengths would corrupt resident model state.

## Program completion

A program declares one of two completion contracts.

### `submit_return`

The program is externally complete when its submit function returns. The generated driver synthesizes the completion record after all operations return successfully. DSV4, K3, MiMo 2.5, and Qwen 3.6 currently use this contract.

### External completion

The owning operation reports completion later through the host completion callback. Because completion ownership is otherwise ambiguous, the compiler rejects an external-completion program containing more than one operation.

No operation may claim asynchronous completion merely because it enqueues CUDA work. The published contract must match what the operation actually guarantees to its caller.

## Admission

Admission is advisory and must be fail-closed. A decision reports only neutral scheduling information:

- accepted or rejected;
- rejection reason;
- queue and service estimates;
- endpoint cost;
- residency match score;
- expected host-staging and device-copy bytes;
- private queue pressure;
- available execution slots;
- optional opaque dispatch ticket.

For multi-operation programs, generated drivers merge decisions conservatively:

- queue delay and pressure use the maximum;
- service time, endpoint cost, and transfer bytes use saturating sums;
- available capacity uses the minimum;
- residency score uses the minimum;
- conflicting non-invalid dispatch tickets are an ABI failure;
- an invalid acceptance/rejection enum or inconsistent accepted state is rejected.

Model modules must base acceptance on real slot and lane availability, not only configured capacity. Admission does not reserve a model lane. Execute must claim ownership atomically and return busy if the advisory result became stale.

## Lane ownership and continuity

A model module that owns persistent per-sequence state must:

1. validate all lane indices and reject duplicates in one frame;
2. atomically claim every referenced lane;
3. release only lanes successfully claimed by that request;
4. verify sequence identity and position continuity before reusing persistent state;
5. commit continuity only after successful execution;
6. invalidate continuity after a failed execution when partially updated state cannot be trusted.

Releasing an unclaimed lane is a concurrency violation because it can unlock state owned by another request. The shared stage helper and all four audited non-GLM drivers use explicit claim-success tracking.

## Snapshot

Snapshots expose aggregate neutral counters only. Generated multi-operation drivers merge them using semantics appropriate to each field:

- active counts and cumulative counters are summed with saturation;
- available slots use the minimum;
- queue pressure uses the maximum;
- capacities that represent a whole-program bottleneck use the minimum where appropriate;
- transfer costs are summed.

A snapshot must not expose model-specific page tables, expert queues, CUDA stream identities, graph nodes, or sequence-lane internals.

## Request-level execution

`SparkModelBatchEngine` is the neutral request-level owner above the pipeline
client. It receives token IDs, not model frames. It packs prefill rows and decode
lanes up to manifest limits, keeps each request on one persistent sequence slot,
and correlates every distributed completion before advancing request state. All
hot-path arrays are allocated during connect. Explicit-release adapters receive
a pipeline-wide release transaction before a slot is reused; position-zero
adapters receive a new request generation and may rebind only at position zero.

Tokenizers, chat templates, and HTTP compatibility parsers remain application or
family components. They are not device-driver entry points and cannot select an
alternate model implementation. The GLM service backend is one such application
and is not linked into the neutral runtime library.

## Destruction

The model-serving adapter ABI requires a `quiesce` entry point. The neutral resident closes admission, polls that hook to an absolute monotonic deadline, synchronizes its one execution stream, closes transport sessions, and only then destroys and unloads the adapter. The adapter must stop accepting submissions as soon as quiescence begins and return `OK` only when no callback or model work can reference adapter-owned state. Because the public driver destroy function remains `void`, a module that cannot prove quiescence preserves live state rather than causing use-after-free.

## Qualification and publication

Source presence, host compilation, and a CPU reference do not qualify GPU firmware.

The audited non-GLM package examples state:

```text
qualification.status       NOT_MEASURED
production_ready           false
fallback_allowed           false
runtime_backend_selection  forbidden
completion                 submit_return
```

Their module Makefiles default the controlled bring-up switch to zero. Initialization returns `SPARK_STATUS_MODULE_NOT_VALIDATED` unless the explicit family switch is set to one.

Publication additionally requires:

- the exact `sm_121a` CUDA target;
- a readable stage pack;
- an executable retained-receipt GPU validator;
- a validation recipe that includes the complete runtime-configuration hash;
- the configuration hash as a validator argument.

Module-record schema 4 binds the validator executable SHA-256, validator arguments, validation recipe, target, entry points, and exact link-unit bytes into the immutable identity. Replacing only the validator executable forces revalidation.

Current non-GLM configuration is still read from strict process environment variables. The configuration string is hashed for publication, but a production deployment should migrate these values into immutable configuration JSON passed through `SparkFirmwareModuleConfiguration` so runtime state cannot diverge from the qualified package.

## No fallback

A published program may not silently choose another backend, CPU reference, host-staged path, approximation, or unqualified shape. Unsupported input is rejected. Availability must be provided by routing to a separately identified and separately qualified package, not by changing implementation inside a package.

## Rule for extending the boundary

When a model does not fit, do not add model geometry or a universal tensor plan to the neutral layer. Either:

1. implement a different model-specific operation behind the existing ABI; or
2. extend the ABI with a neutral scheduling or lifecycle concept that remains meaningful across model families.

The acceptance test for a clean boundary is that a materially different second model can reach production without adding its geometry, caches, transport types, or execution policy to `include/sparkpipe/` or `src/`.
