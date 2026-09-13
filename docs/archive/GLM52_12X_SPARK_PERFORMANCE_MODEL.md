# GLM 5.2 on 12 DGX Spark nodes: performance model and CUDA target shape

This is a planning document, not a performance claim. It exists to keep the 12-node design honest while CUDA bring-up continues on real Spark hardware.

## Hardware assumptions

Per Spark node:

```text
memory bandwidth:         273 GB/s
network:                  ConnectX-7 class 200 Gb/s
coherent memory capacity: 128 GB
```

For 12 nodes the aggregate memory bandwidth is approximately:

```text
12 * 273 GB/s = 3276 GB/s = 3.276 TB/s
```

That aggregate number only matters when the execution plan makes several nodes stream different resident weight slices at the same time. It does not automatically reduce single-sequence latency for a pure layer pipeline, because autoregressive decode still has a layer dependency chain.

## Current correctness-kernel projection

The latest reported GLM 5.2 layer-0/full-reference bring-up numbers are approximately:

```text
backend per checked layer:          4.6 ms
orchestrator path per checked layer: 5.8 ms
```

A simple projection over 78 layers is therefore:

```text
4.6 ms * 78 = 359 ms/token = 2.8 tok/s
5.8 ms * 78 = 452 ms/token = 2.2 tok/s
```

A 12-node layer pipeline with the same kernels can improve saturated multi-request throughput but not single-request latency:

```text
layers per node:        78 / 12 = 6.5
stage time at 5.8 ms:   37.7 ms
pipeline throughput:    26.5 tok/s after fill
single-sequence latency: about 452 ms/token plus network/control
```

So the current CUDA bring-up path is approximately:

```text
single stream:     ~2 tok/s
12-node pipeline:  ~20-30 aggregate tok/s after fill, assuming enough independent requests
```

This is correctness evidence, not a SOTA target.

## Bandwidth roofline for optimized firmware

The GLM 5.2 path must move active weights. Single-token decode on Spark is expected to be memory-bandwidth limited unless the driver has enough concurrent sequences to raise arithmetic intensity.

Approximate active-weight bands:

```text
BF16-heavy active path:       ~70 GB/token
FP8/NVFP4 intended path:      ~25-35 GB/token
```

Single-node memory roofline:

```text
70 GB / 273 GB/s    = 256 ms/token = 3.9 tok/s
35 GB / 273 GB/s    = 128 ms/token = 7.8 tok/s
25 GB / 273 GB/s    =  92 ms/token = 10.9 tok/s
```

Twelve-node aggregate roofline if resident weights are actually partitioned and streamed concurrently:

```text
70 GB / 3276 GB/s   = 21.4 ms/token = 46.7 aggregate tok/s
35 GB / 3276 GB/s   = 10.7 ms/token = 93.6 aggregate tok/s
25 GB / 3276 GB/s   =  7.6 ms/token = 131.0 aggregate tok/s
```

Real throughput will be below the roofline because of CUDA graph launch, attention/KV reads, routing, expert grouping, queueing, network hops, and imperfect memory utilization.

## Expected targets

### Near-term after replacing scalar CUDA math

Assuming cublasLt/CUTLASS projection plans, graph replay, no host staging, and FP8/BF16 attention projection paths:

```text
single Spark broad decode:       5-8 tok/s
12 Spark saturated throughput:   45-80 aggregate tok/s
```

### Strong firmware pass

Assuming production NVFP4 grouped experts, private expert queues, stable graph replay, restricted final-head programs, and measured scheduling costs:

```text
single Spark broad decode:       8-12 tok/s
12 Spark saturated throughput:   80-130 aggregate tok/s
```

### Stretch with favorable workload and MTP

Assuming high MTP acceptance, final restricted-output phases, enough resident concurrent sequences, and low overlap loss:

```text
single-request effective:        15-30 tok/s on favorable prompts
12 Spark aggregate effective:    120-220 tok/s on favorable batched/service workloads
```

A claim above these bands should come with the full command, exact model JSON, exact module artifact hashes, active-weight precision, graph replay counters, zero-copy trace, Nsight evidence, and end-to-end request/completion logs.

## Scheduling implication

There are two valid 12-node modes.

### Layer pipeline

```text
node 0: layers 0-6
node 1: layers 7-13
...
node 11: layers 72-77 + final head
```

This is simple and keeps activation handoff tiny. It improves aggregate throughput only when there are enough independent sequences to fill the pipeline. It does not make one sequence 12 times faster.

### Model/expert parallel firmware

```text
dense projections: tensor-parallel or replicated plan by profile
MoE experts:       expert-parallel placement across nodes
attention/KV:      resident owner per layer/window/profile
handoff:           fixed activation packets, not SparkPipe-visible tensors
```

This can improve single-sequence latency, but it is more delicate. The driver must hide expert IDs, KV pages, and tensor-parallel details behind opaque admission tickets and neutral pressure/cost counters.

## Boundary rule

SparkPipe must not learn GLM internals in order to make 12-node scheduling work. The model driver may expose only:

```text
admission accepted/rejected
opaque dispatch slot/generation/cookies
estimated service time
estimated queue delay
private queue pressure
residency match score
zero-copy/memcpy counters
completion records
runtime snapshots
```

The GLM firmware owns:

```text
layer partition
expert placement
KV ownership
stream/event topology
CUDA graph topology
DSA/sparse policy
MTP acceptance
restricted-vocab banks
transport handoff packet layout
```

## Iteration 081 note

Iteration 081 does not change the expected 12-node speed bands by itself because those bands require target-hardware compilation and profiling. It does, however, make the firmware stricter: a package can now demand the fast projection, attention, graph replay, MoE/logit, MTP, fixed-batch, and validated-latency resources and fail closed if a correctness-only kernel would otherwise run. This prevents optimistic 12-node projections from being backed by debug/reference CUDA paths.

## Iteration 082 note

The 12-node speed bands do not change until target hardware measures the new full-stage plans. The important change is the qualification rule: optimistic 12-node projections should be attached only to modules that require and satisfy the full-stage SOTA plan or an equivalent custom sparse-attention plan. Component-fast or diagnostic NVFP4 routes remain correctness/bring-up evidence unless their artifact record includes the full-stage or B12x MoE latency gate.

The measured top-8 routed NVFP4 diagnostic path remains the first performance target. The new route-slot cache removes one known artificial overhead, but the production solution is still grouped expert execution with driver-private route queues and tensor-core/FP4 kernels inside the GLM firmware module.
