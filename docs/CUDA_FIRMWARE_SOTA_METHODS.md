# CUDA firmware methods for SOTA-level handoff

This document describes the target CUDA shape for the current GLM 5.2 firmware modules. The code in this release is intended to be within debug distance for a CUDA/Codex pass: the firmware boundaries, resource ownership, scheduling ABI, graph slots, counters, fast-path contracts, and target hooks are in place. Remaining performance proof must come from SM121 compilation, Nsight traces, and model-stage latency gates.

## Iteration 081 CUDA fast-path tightening

Iteration 081 turns the previous SOTA-shaped hooks into a stricter production contract and removes several obvious debug/reference bottlenecks from the CUDA sources:

```text
strict SOTA execution flag bundles for decode-stage and sparse-MLA firmware
fail-closed validation for missing prebound projection/output/dense/logit plans
fail-closed validation for non-tiled attention, graph replay misses, debug synchronization, and unvalidated latency
fixed-active-batch checks before prebound plan dispatch
custom MTP draft hook for replacing the local vectorized MXFP4 reference kernel
warp-then-block reductions for sparse attention and decode reductions
raw KV split indexing fix for the real GLM 5.2 FP8/BF16 path
no SparkPipe runtime changes
```

The strict flags are designed for model JSON profiles that want to publish a SOTA artifact rather than a correctness artifact. If a profile sets the all-in fast-path flag, the module must have its exact resident fast resources prebound before submit. Otherwise validation fails before execution, instead of quietly falling back to scalar kernels.

The local vectorized MXFP4 MTP draft kernel remains a usable bring-up path. A production MTP implementation should bind a `SparkGlm52ResidentDecodeStageMtpDraftPlan` with a target-specific launch hook and workspace so draft logits, argmax, and acceptance counters can be fused or specialized without changing SparkPipe.

## Iteration 080 CUDA fast-path additions

The decode-stage firmware now has a concrete fast-path ABI for replacing correctness-first kernels without changing SparkPipe:

```text
prebound cublasLt row-major BF16/FP8 linear plans
prebound dense-MLP tensor-core path
driver-owned grouped-MoE launch hook
restricted-logits custom launch hook
tiled online sparse-MLA attention kernel
vectorized MXFP4 MTP draft-logit kernel
validated service-time fields in node_context/admission
```

The important shape is that these are module-local resources. The node context owns cublasLt descriptors, algorithms, workspaces, grouped-MoE state, and restricted-logits plans. SparkPipe only sees neutral admission and snapshot values.

For production profiles, the scalar kernels should remain only as validator/debug fallbacks. A SOTA-qualified GLM 5.2 decode package should set:

```text
projection_backend_mode = PREBOUND_CUBLASLT
mlp_execution_mode      = PREBOUND_TENSOR_CORE or DRIVER_GROUPED_MOE
attention_execution     = TILED_ONLINE_SOFTMAX after numerical validation
launch_check_mode       = NONE
phase_clock_mode        = DISABLED
enable_cuda_graph_replay= 1
```

Required target-hardware proof before admitting such a package:

```text
all cublasLt plans are created before submit
no cublasLt descriptor creation on the hot path
all required linear plans are present by exact plan index
MoE grouped launch owns expert queues privately
restricted logits do not materialize full vocab
MTP vectorized kernel matches scalar MXFP4 reference
graph replay dominates after warmup
zero host staging and zero avoidable device memcpy are trace-proven
estimated_service_time_ns is populated from the validated artifact
```


## Global rules for every CUDA firmware module

A CUDA firmware module is a self-contained static archive selected by exact model JSON. It should own all model-specific implementation detail below the SparkPipe driver ABI.

Required production properties:

```text
resident weights and workspaces bound once through node_context
caller-owned non-default streams
stream/event dependencies instead of broad synchronizations
no host tensor staging
zero avoidable device memcpy
JIT KV allocation and residency inside the driver
preallocated per-slot CUDA graph state
preallocated per-slot completion and workspace state
no hot-path allocation
no runtime kernel selection
no fallback path
no debug serial kernel on the qualified profile
full submission-to-completion validation on target hardware
```

The validator must prove more than kernel correctness. It must check cold-build completeness, numerical error, route-visible completion ordering, no forbidden staging, no avoidable memcpy, graph replay behavior, and maximum observed submission-to-completion latency.

## LLM driver admission and JIT KV

The driver admission function is the scheduler-facing front door. For a SOTA decode path it should evaluate:

```text
sequence residency token match
KV page availability
CUDA graph slot availability
persistent lane availability
private expert queue pressure
transport ring pressure
batch-shape compatibility
```

It returns only neutral cost and an opaque dispatch ticket. The dispatch generation added in ABI v3 prevents stale admission from reusing a slot after completion recycles it.

The KV allocator should be driver-owned and sequence-resident. SparkPipe may pass a residency token, but the driver owns page allocation, eviction, SWA/full-context policy, independent K/V precision, and page-layout details.

## CUDA graph and launch topology

The release adds per-slot graph state to both CUDA modules:

```text
cuda_graph_exec
graph_active_sequence_count
graph_capture_count
graph_replay_count
launch_chain_count
launch_error_count
```

The intended fast path is:

```text
admission chooses dispatch slot
submit finds matching captured graph
cudaGraphLaunch on the slot stream
stream-ordered completion enqueue
return
```

Graph capture is per slot and per fixed active-sequence shape. The first shape miss captures and instantiates the graph, then replays it. Production profiles should keep `batch_shape_fixed` where possible to avoid graph churn.

Launch checking is configurable:

```text
none            production path
peek            low-overhead debug
sync_on_error   bring-up only
```

The graph does not need to contain the host completion callback. The current code enqueues completion after graph launch so the arithmetic graph remains reusable and the completion mechanism can later be swapped for event polling or a doorbell without changing the kernel graph.

## Sparse index selection

The decode-stage firmware supports three sparse-index modes:

```text
preselected           SOTA path; sparse indices are produced by driver-owned DSA/persistent logic
copy_context_prefix   parallel correctness/bring-up path for dense-prefix sparse rows
debug_serial_topk     debug only; not a qualified production mode
```

The previous serial DSA top-k loop is intentionally quarantined behind `debug_serial_topk`. A real SOTA implementation should use one of these shapes:

```text
persistent DSA lane writes sparse_token_indices before decode graph replay
warp/block top-k over compacted candidate scores
cooperative-group selection with deterministic tie-breaking
layer/window-specific sparse policy fused with page-residency metadata
```

The important invariant is that SparkPipe never sees the sparse-token policy. It sees only admission cost and private pressure.

## RoPE + KV write

The current prepare kernel fuses adjacent-pair query RoPE, adjacent-pair key RoPE, and current-token latent KV write into the final paged MLA cache layout.

Target properties:

```text
write directly to final token-major page slot
no temporary K/V layout
support nonzero first-block offsets through block tables
write quantized scale metadata in the same path when using NVFP4/MXFP4 cache formats
support independent K and V precision inside the driver-owned KV contract
```

For a high-performance pass, specialize the prepare kernel for the exact active-sequence and head geometry. Use vectorized BF16 pairs or wider packed accesses where alignment allows.

## Sparse MLA attention

The sparse MLA module and the decode-stage attention kernel share the same logical layout:

```text
64 heads
512 latent elements
64 RoPE elements
576 BF16 cache elements per token
2048 selected context tokens
64-token physical KV blocks
```

The correctness-first kernel keeps the layout and block-table semantics explicit. The performance pass should focus on:

```text
tiling selected tokens to reduce shared-memory footprint
warp-specialized dot-product and softmax reductions
vectorized BF16 loads from token-major cache
avoiding per-head redundant query loads
using persistent blocks for stable decode shapes
measuring whether split softmax/value phases beat one fused kernel
```

Do not repack KV for attention. If the attention kernel wants a different memory order, the cache layout or fused write path must change at compile time for that model profile.

## Projection and RMSNorm bundle

The decode-stage firmware currently has correctness-first BF16 GEMV/GEMM-style kernels for:

```text
attention RMSNorm
hidden -> query latent
hidden -> query RoPE
hidden -> key RoPE
hidden -> current KV latent
attention output projection
final RMSNorm
restricted LM head
```

These kernels are intentionally not the final SOTA arithmetic implementation. The expected replacement choices are:

```text
cublasLt grouped matmul for projection batches
custom tensor-core kernels for fixed small batch decode
persistent projection kernels for fixed active-sequence slots
fused norm + projection where memory bandwidth dominates
weight-layout-specific kernels generated per model/profile
```

The firmware ABI is ready for this: these replacements remain inside the module archive and do not alter SparkPipe.

## Restricted logits and sampler

The current restricted-vocabulary path computes logits for 256 selected token IDs and performs device argmax. This is the correct production direction: do not compute full vocabulary logits when the model/profile only needs a restricted set.

Restricted logits do not remove the normal decode body. The hidden state still depends on attention, cached KV reads and writes, MoE or MLP execution, residuals, norms, routing, and MTP verify state. Restricted output is therefore a final-head optimization, not permission to skip transformer layers or experts.

Target improvements:

```text
store static banks as token_ids[K] plus lm_head_rows_packed[K][hidden]
load only the selected packed LM-head rows
compute selected logits only, with no full-vocabulary materialization
fuse selected-logit computation with argmax/top-k/sample for small K
support multiple resident restricted-vocab banks selected by driver-owned policy
cache dynamic repeated subsets inside the driver by opaque subset hash
record accepted token count without a host round trip
```

The production path must not:

```text
compute full vocabulary logits and then mask
allocate a full-vocabulary logits buffer
copy selected logits to host for sampling
expose token IDs or grammar state through SparkPipe
reuse a generic SparkPipe vocabulary-mask abstraction
```

SparkPipe should only receive neutral scheduling facts, committed token count, and completion status. It may know that a program is `final_restricted_decode_k256` or `classifier_decode`, but the token IDs, packed rows, grammar/trie state, sampler policy, and MTP acceptance policy stay in the model driver.

For thinking models, restricted-token acceleration is valid for the final answer phase, not the broad reasoning-token phase. A driver may run:

```text
full_decode
full_decode
full_decode
final_restricted_decode_k256
```

but it must not force natural-language reasoning through a tiny final-answer vocabulary.

Restricted-vocab MTP follows the same rule: useful for final answer emission or classifier-style programs, unsafe as a blanket replacement for broad-vocabulary reasoning. Pruning MoE experts because only a small set of output tokens is allowed is also unsafe unless published as a separate approximate firmware program with its own accuracy gate.

## MTP draft / verify / commit

The decode-stage MTP draft path is explicitly MXFP4:

```text
E2M1 packed payload
E8M0 scales
group size 32
no NVFP4 draft routing
```

That preserves the separation between MXFP4 draft experts and NVFP4 main/KV formats. The current MTP kernels provide the boundary and counters. The performance pass should:

```text
replace scalar decode with vectorized MXFP4 unpack/dequant
fuse draft logits and argmax for restricted vocab
support depth profiles selected by measured acceptance
emit accept/reject/commit/rollback counters in device memory
store negative depth results as qualification evidence
```

MTP depth is not assumed beneficial. It must pass net tokens/sec on target workloads.

## MoE expert queues

MoE belongs inside the LLM driver. The driver may maintain per-expert queues, batching/coalescing policy, and expert placement. SparkPipe only receives private pressure and service estimates.

SOTA target shape:

```text
driver-private expert queues per GPU/node
profile-level expert placement and split knobs
persistent or graph-captured expert batches
measured local batching/coalescing sweeps
neutral pressure exported through admission/snapshot
no expert structures in SparkPipe
```

The current release does not implement resident MoE expert execution. The ABI is designed so it can be added inside the decode-stage archive without changing the orchestrator.

## Transport handoff

For multi-node stages, transport must also be driver-owned or node-agent-owned behind the same neutral boundary. A driver should expose zero-copy/no-host-staging counters and completion ordering, not ring internals.

Target transport shape:

```text
stage output written directly to transport-visible resident buffer
CUDA event or memory semaphore orders send after compute
NIC/ring completion orders downstream admission
host bounce allowed only if the validator records it as a measured win
```

## Validation checklist for a publishable SOTA artifact

A module archive should not be published as SOTA unless the target-hardware validator records:

```text
cold build from empty directory
exact artifact hash
exact compiler/toolkit/architecture
numerical pass against oracle
no host tensor staging
zero avoidable device memcpy
no hot-path allocation
no broad device synchronization
non-default stream use
stream/event dependency trace
CUDA graph capture count bounded
graph replay count dominates steady state
full submission-to-completion latency maximum under ceiling
model-stage benchmark, not only microkernel benchmark
negative results for rejected MTP depth/profile choices
```

If a kernel is numerically correct but misses the full-stage ceiling, it remains an unvalidated candidate.

## Iteration 082 max-performance firmware shape

The preferred SOTA path is now a full-stage plan, not a chain of reusable correctness kernels. The component kernels remain useful for validation and bring-up, but an optimized GLM 5.2 package should bind a `SparkGlm52ResidentDecodeStageFullStagePlan` whose launch function owns the complete stage sequence behind one module ABI:

```text
resident input hidden state
        -> tensor-core q/kv/o projections
        -> fused RoPE + final-layout KV write
        -> tiled online MLA
        -> grouped NVFP4 routed experts / dense tensor-core MLP
        -> restricted logits or broad-head program variant
        -> MXFP4 MTP draft/verify/commit when enabled
        -> stream/event ordered completion
```

The full-stage plan must advertise all SOTA capabilities: stream ordering, CUDA graph replay, tensor-core projections, tiled online attention, fused RoPE/KV write, grouped MoE, fast restricted logits, fast MTP draft, zero host staging, and zero avoidable device memcpy. A profile that requires the full-stage plan must fail closed if the plan is missing or unvalidated.

The routed NVFP4 top-k diagnostic path now has a route-slot cache:

```text
router top-k ids
        -> resolve bound expert slot once per route
        -> hidden BF16 -> NVFP4 route quantization
        -> gate/up NVFP4 projections use cached bound slot
        -> fused SiLU*up -> NVFP4 quant uses cached bound slot
        -> down NVFP4 projection uses cached bound slot
        -> weighted combine
```

This removes repeated bound-expert scans from every output row. It is still not a substitute for production grouped expert GEMM, but it narrows the debug distance to the intended grouped-MoE implementation and makes the measured top-8 NVFP4 path less artificially bad.

For target CUDA/Codex work, the priority order is now:

```text
1. implement and bind full-stage plan for one GLM 5.2 layer/stage partition;
2. use cublasLt/CUTLASS or custom tensor-core plans for q/kv/o and dense MLP;
3. replace route-local scalar NVFP4 expert dots with grouped NVFP4 expert kernels;
4. keep route-slot cache for diagnostic kernels and as validation data for grouped queues;
5. capture/replay the full-stage graph per fixed slot/active-shape;
6. prove zero host staging and zero avoidable device memcpy by trace;
7. publish only if full submission-to-completion latency passes the stage ceiling.
```

## Iteration 083 grouped MoE fast path

The routed sparse layers must not run eight independent expert paths with repeated expert-slot scans. The target shape is now:

```text
post-attention normalized hidden
        ↓
prebound router-logits linear plan
        ↓
256-way top-k over router logits
        ↓
route → bound expert slot cache
        ↓
expert-major route grouping
        ↓
persistent grouped NVFP4 gate/up workers
        ↓
fused SiLU + NVFP4 route quant
        ↓
persistent grouped NVFP4 down workers
        ↓
weighted combine
```

`SparkGlm52ResidentDecodeStageLaunchPersistentGroupedNvfp4Moe` is the firmware-provided implementation of that shape. It is still a target-debug implementation: the final SM121 pass should replace the local E2M1/E4M3 decode inner loops with tuned FP4/Tensor Core kernels or CUTLASS-style grouped GEMMs, but the route grouping, plan ABI, and SparkPipe boundary should not change.

A SOTA sparse-layer package should set:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_MLP_EXECUTION_DRIVER_GROUPED_MOE
SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MLP
SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_FAST_MOE_ROUTER
SPARK_GLM52_RESIDENT_DECODE_STAGE_EXECUTION_REQUIRE_NVFP4_ROUTE_SLOT_CACHE
```

and bind:

```text
SPARK_GLM52_RESIDENT_DECODE_STAGE_LINEAR_PLAN_ROUTER_LOGITS
SparkGlm52ResidentDecodeStageGroupedMoePlan(plan_kind=PERSISTENT_NVFP4_TOPK)
```

The route-grouping buffers and compact work-item queue are driver-owned device workspaces. SparkPipe never sees expert IDs, route queues, or expert placement.
