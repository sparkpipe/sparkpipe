# CUDA firmware methods for SOTA-level handoff

This document describes the target CUDA shape for the current GLM 5.2 firmware modules. The code in this release is intended to be within debug distance for a CUDA/Codex pass: the firmware boundaries, resource ownership, scheduling ABI, graph slots, and counters are in place, while several arithmetic kernels are still correctness-first and should be replaced after target profiling.

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
