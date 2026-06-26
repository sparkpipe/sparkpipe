# LLM device-driver interface

This document is the handoff contract between the SparkPipe scheduler and a model-specific LLM firmware driver.

SparkPipe is not an LLM runtime. SparkPipe does not own KV pages, attention metadata, CUDA graphs, expert queues, token-selection policy, quantization details, or model-stage topology. A model driver owns those details and exposes only the minimum neutral signals needed for scheduling across heterogeneous nodes.

## Boundary

```text
SparkPipe orchestrator
    owns routes, node targets, loaded driver instances, replica choice,
    inflight accounting, deadline-aware admission scoring, completion retirement.

Model-specific LLM driver
    owns resident weights, JIT KV cache, page layout, CUDA streams/events,
    CUDA graph instances, persistent kernels, MoE queues, MTP state,
    sparse token selection, transport handoff, and completion production.
```

The shared ABI is intentionally small:

```text
create       bind one driver instance to one node context
admit        answer whether this driver can accept one frame now
submit       run one exact externally callable program
snapshot     export neutral runtime counters
destroy      quiesce and release the driver instance
completion   report finished work to SparkPipe
```

SparkPipe passes an opaque `node_context` at creation time. For CUDA LLM firmware this is where the deployment binds streams, resident device buffers, CUDA graph slots, KV-cache storage, workspace pointers, table pointers, and fixed capacities. SparkPipe never interprets it.

## Driver ABI version 3

ABI v3 adds dispatch-ticket integrity and CUDA-shape counters without exposing LLM internals.

An admission decision may now return:

```text
opaque dispatch slot
dispatch generation
dispatch cookie 0
dispatch cookie 1
```

SparkPipe copies these fields into the submitted frame only when the driver returns a valid dispatch slot. The driver may use them to prevent stale advisory admission from reusing a slot that has since completed, been recycled, or changed ownership. SparkPipe treats the fields as bytes with scheduling meaning only to the driver.

The runtime snapshot can report:

```text
cuda_graph_capture_count
cuda_graph_replay_count
host_callback_completion_count
stale_admission_count
```

These are neutral performance counters. They do not describe graph topology, stream topology, or kernel structure. They exist to detect whether a supposedly fixed firmware path is still launching too many chains, failing to replay graphs, using host callback completion more often than expected, or racing stale admission tickets.

## Program scheduling profile

A program profile declares the contract SparkPipe may rely on when choosing routes:

```text
stream_ordered
external_completion
driver_owns_resident_state
driver_owns_kv_cache
jit_kv_cache
zero_copy_node_context
private_queue_pressure
no_host_staging
no_device_memcpy
fixed_firmware
captured_cuda_graph
stream_event_dependencies
residency_affinity_required
driver_private_expert_queues
batch_shape_fixed
validated_latency
```

These flags are not a generic LLM plan. They are promises about the compiled driver. If a driver declares `no_host_staging`, its completions and snapshots must keep host-staging bytes at zero. If it declares `jit_kv_cache`, SparkPipe may schedule by residency token and pressure, but it still may not inspect the page allocator. If it declares `driver_private_expert_queues`, SparkPipe may use private queue pressure, but it must not see individual expert queues.

## Admission

Admission answers one question: can this exact driver instance take this frame now, and at what neutral scheduling cost?

The driver may base its answer on internal details such as:

```text
free CUDA graph slot
free persistent-kernel lane
resident KV capacity
sequence residency affinity
MoE expert queue pressure
DSA/sparse-index work queue pressure
transport queue depth
stream ownership
batch-shape compatibility
```

The decision exposes only:

```text
accepted / rejected
rejection reason
estimated queue delay
estimated service time
endpoint cost
private queue pressure
residency match score
expected host staging bytes
expected device memcpy bytes
available dispatch slots
opaque dispatch ticket
```

SparkPipe chooses the lowest-cost accepted endpoint and submits the frame. If the advisory decision becomes stale and the driver returns busy, SparkPipe performs bounded retry across the route. Admission is not a heavyweight reservation protocol.

## Submit

The generated model driver calls exact module entry points directly. The LLM firmware module receives:

```text
request identity
sequence identity
sequence position
active sequence count
new-token count
priority/deadline
opaque dispatch ticket
residency token
driver completion function
```

For a firmware-owned resident decode stage, the frame must not contain host-staged tensor buffers. Dynamic submission data should be metadata, not payload. Payloads live in resident device memory that was bound at driver creation.

The hot path must not perform:

```text
module lookup
model graph traversal
kernel capability selection
fallback selection
heap allocation
runtime validation scan
manifest parsing
host tensor staging
avoidable device-to-device copy
device-wide synchronization
```

## Snapshot

Snapshot is the only sanctioned way for SparkPipe to observe driver internals. It reports aggregate neutral counters:

```text
active submissions
available dispatch slots
submitted/completed/rejected counts
resident sequence/token count
KV token capacity
host-staging bytes per submit
device-memcpy bytes per submit
CUDA graph capture/replay counts
stale admission count
private queue pressure
```

A snapshot must never expose KV-page tables, expert IDs, graph nodes, stream IDs, or model-specific operation state.

## Completion

The driver calls SparkPipe’s completion function after the submitted work is externally complete. Completion records include request identity, program identity, dispatch slot, accepted token count, residency token, status, and neutral memcpy/staging counters.

A CUDA driver may produce completion through a stream-ordered host function, a driver-owned event poller, a persistent-kernel doorbell, or transport completion. SparkPipe does not care which mechanism is used as long as ordering and counters are correct. A SOTA deployment should measure the completion mechanism and replace host callback completion if it becomes material latency.

## MoE and unavoidable pressure leakage

MoE cannot be scheduled well if SparkPipe sees no pressure at all, but exposing individual expert queues would contaminate the orchestrator. The compromise is:

```text
Driver owns expert queues and batching policy.
Driver exposes only private_queue_pressure, endpoint_cost, and service estimates.
SparkPipe routes by those neutral scalars.
```

If future expert scheduling needs more signal, add another neutral scalar to the driver ABI. Do not add expert-specific structures to SparkPipe.

## Artificial compatibility code rule

When a model does not fit the driver boundary, do not add a generic compatibility adapter that makes every model look like a universal graph. Either:

1. generate a different model-specific firmware module behind the same ABI; or
2. extend the ABI with a neutral scheduling concept that does not leak LLM internals.

SparkPipe remains the OS/orchestrator. Each model package remains a device driver.
