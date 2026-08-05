# Phase 3: DSV4 Exact Cache Allocation and Grouped-MoE Batch Execution

## Status

This phase implements two foundational corrections:

1. DeepSeek V4 Flash and Pro no longer need a worst-class cache reservation for every layer.
2. The GLM 5.2 batch-plane and queue contracts no longer charge or execute an artificial 128-token replay multiplier over the expert sweep.

The source tree is host-tested for these contracts. CUDA 13 `sm_121a` compilation and target-hardware execution remain unrun.

## DSV4 exact cache allocation

The old reservation policy treated every layer as if it simultaneously needed:

- the full sliding/CSA history footprint;
- the largest HCA compressor state;
- every compressor/indexer overlap buffer.

That is not the model's logical cache layout. Each layer has exactly one attention class:

- sliding attention, compression ratio 0;
- compressed sparse attention, ratio 4;
- heavily compressed attention, ratio 128.

The new planner builds a per-layer layout from the generated Flash or Pro schedule. It separately sizes:

- the sliding-history arena;
- the compressed-history arena;
- the compressor/indexer-state arena.

The new arena allocator allocates exactly those three planned arenas through the stage-module ledger. It does not allocate a worst-class slab per layer. Allocation failure rolls the ledger back to its entry checkpoint without releasing unrelated module allocations. Layer views are derived from bounds-checked offsets in the immutable plan.

### Representative reservation result

Configuration used by the retained report tool:

- one million aggregate retained tokens;
- eight active sequences;
- BF16 content, RoPE and compressor state;
- 8-bit indexer history;
- 512-entry compressed-history pages;
- backbone plus MTP layer.

| Variant | Exact reservation | Previous worst-class reservation | Saved | Reduction |
|---|---:|---:|---:|---:|
| DSV4 Flash | 6.148 GiB | 12.503 GiB | 6.355 GiB | 50.8% |
| DSV4 Pro | 8.806 GiB | 17.618 GiB | 8.812 GiB | 50.0% |

These numbers are a deterministic sizing example, not a claim about a final production cache policy. Different token capacity, precision, page size and stage placement produce different totals.

Relevant implementation:

- `model-families/dsv4/include/sparkpipe/spark_dsv4_cache_plan.h`
- `model-families/dsv4/src/spark_dsv4_cache_plan.c`
- `model-families/dsv4/include/sparkpipe/spark_dsv4_cache_arena.h`
- `model-families/dsv4/src/spark_dsv4_cache_arena.c`
- `tests/test_dsv4_cache_plan.c`
- `tests/studies/sparkpipe_dsv4_cache_plan_report.c`

## Grouped-MoE route and queue correction

### One route build per logical batch

K3, GLM 5.2, DSV4 and MiMo now share one device route-build contract:

1. produce top-k expert choices and mixture weights;
2. count routes per expert;
3. build the expert-major row offsets;
4. scatter source-token identities into expert-major order;
5. build separate W1 and W2 grouped-tile prefix tables;
6. run grouped expert computation;
7. scatter/fold route outputs back to token order.

The route planner now receives both token rows and packed route rows and fails unless:

```text
packed_rows == rows * top_k
```

Tile selection is priced from token rows, not packed routes. K3 previously passed packed routes as the token count, multiplying the estimated expert row load by top-k.

### Correct router semantics

The shared top-k kernel now has explicit score transformations and post-selection scaling:

| Family | Selection score | Renormalize selected gates | Routed scale |
|---|---|---:|---:|
| K3 | sigmoid | yes | 1.0 |
| GLM 5.2 | sigmoid | yes | 2.5 |
| DSV4 | sqrt(softplus) | yes | 1.5 |
| MiMo 2.5 | identity | no | 1.0 |

Selection bias influences which experts are chosen but is not leaked into the mixture weight.

### Sealed-batch expert queue

The GLM 5.2 host expert queue ABI now supports two explicit modes:

- `LOW_LATENCY`: threshold/deadline firing;
- `SEALED_BATCH`: collect a complete layer batch, seal it, and emit every active expert exactly once.

A sealed layer rejects additional enqueue operations until all active experts have fired. Seal fails if any one expert exceeds the retained firing capacity instead of silently splitting a purported one-sweep batch.

This removes the incorrect performance assumption that a GLM 5.2 expert sweep is replayed for every 128-token chunk. The grouped path has:

```text
expert replay/chunk multiplier = 1
```

For GLM 5.2 B1024 with 256 experts and top-k 8:

```text
8192 routes / 256 experts = 32 average rows per expert
planner tile M = 64
```

Thus the expected expert queue fits in one M tile. Routing skew may split an overloaded expert, but it does not multiply every expert sweep and must be measured as a tail effect rather than charged globally.

### Corrected source-only batch-plane model

The retained estimator now models:

```text
active experts in the batch
× complete FP8 expert bytes
× six MoE layers per rank
```

It does not divide expert bytes by a synthetic queue depth based on 78 layers and does not multiply the sweep by a 128-token replay wall.

At the retained 174 GB/s effective-bandwidth anchor and a 25% realtime reserve, the model reports the following source-only ceiling at 2K context:

| B | Expected active experts | Average routed rows/active expert | Planned tile M | Decoded rows/s | Expected committed tokens/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.9 | 1.01 | 16 | 72 | 51 |
| 64 | 221.5 | 2.31 | 16 | 163 | 116 |
| 256 | 255.9 | 8.00 | 16 | 542 | 384 |
| 512 | 256.0 | 16.00 | 32 | 1,021 | 724 |
| 1024 | 256.0 | 32.00 | 64 | 1,834 | 1,300 |

This is an analytical queue/roofline estimate, not a measured PP13 or PP16 service result. It omits pipeline bubbles, exact routing skew, non-expert kernels, transport stalls, CUDA launch overhead, cache behavior, and the current shared-GEMM blockers.

Relevant implementation:

- `inference/kernels/topk.cuh`
- `inference/kernels/route.cuh`
- `inference/llms/kimi_k3/layer.cuh`
- `modules/glm52_resident_decode_stage/source/cuda/layer.cuh`
- `inference/llms/deepseek_v4/layer.cuh`
- `inference/llms/mimo_2_5/layer.cuh`
- `model-families/glm52/include/sparkpipe/spark_glm52_expert_queue.h`
- `model-families/glm52/src/spark_glm52_expert_queue.c`
- `tests/studies/sparkpipe_glm52_batchplane_model.c`
- `tests/test_grouped_moe_source_contracts.py`

## Additional repairs retained in this phase

- DSV4 and MiMo router output storage is FP32, matching the router GEMM output contract.
- DSV4 sparse-selection threshold storage no longer aliases a float score buffer as `uint32_t`.
- GLM sparse-selection threshold storage uses a dedicated existing `uint32_t` workspace rather than a removed route-prefix field.
- DSV4 shared-W1 execution resets the GEMM descriptor before changing output type, preventing simultaneous stale F32 and BF16 outputs.
- The no-CUDA host build uses the repository's CUDA runtime stub instead of failing on a missing `/usr/local/cuda/include/cuda_runtime.h`.
- The Qwen 3.6 work-control implementation and K3 generated contract files omitted by the prior staging tree are restored.
- Ring-debug and one-switch topology code is included in the model-common host build and test inventory.

## Qualification boundary

Passed in this phase:

- exact DSV4 cache-plan and arena-allocation tests;
- DSV4 Flash/Pro generated-contract check;
- mandatory-model target check;
- grouped-MoE source-contract test;
- GLM expert queue low-latency and sealed-batch tests;
- GLM batch-plane estimator build and execution;
- K3 engine, geometry, layer, slice, pack, quantization and sharding host tests;
- Qwen 3.6 work-control test;
- ring/single-switch fabric-topology test;
- router and shared host-CUDA numerical harnesses.

Not established:

- CUDA 13 compilation;
- `compute_121a` PTX;
- `sm_121a` assembly or device linking;
- CUDA numerical correctness;
- race freedom;
- Blackwell register, spill, shared-memory or occupancy results;
- single-switch or ring hardware throughput;
- production readiness.
