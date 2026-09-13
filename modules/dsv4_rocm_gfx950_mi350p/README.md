# dsv4_rocm_gfx950_mi350p - HIP runtime backend for AMD MI350P

Target archive for target id `rocm.gfx950.mi350p` (hwiface_v1.md section 6).
Implements the neutral runtime primitive surface of `spark_hw_iface.h`
(contract sections 4.0-4.2) over the ROCm HIP API. gfx950 = CDNA4, MI350P.

## Status

- Runtime primitives (`spark_hw_rocm_*`): complete and compile-checked.
- gfx950 fail-closed guard + target descriptor: complete.
- Island entries, S7 bring-up order step 1 (plan section 5.2 "E0 + L5
  first"): landed as `.hip` kernels - `spark_dsv4_rocm_e0_prologue_embed`
  (E0 `prologue.embed`, C2) and `spark_dsv4_rocm_l5_moe_shared` (L5
  `layer.moe_shared`, C3). See "S7 island bring-up" below.
- Island entries, S7 step 3 L4 leg: landed - `spark_dsv4_rocm_l4_moe_routed`
  (L4 `layer.moe_routed`, route realization C2 + grouped-GEMM halves C3),
  including the routed FFN down projection (grouped W2) and the MXFP4_E2M1
  decode path. See "S7 island bring-up" below.
- Island entries, S7 step 4 L3 leg: landed - the cache_transition island as
  three lifecycle-disjoint symbols (`..._l3_cache_transition` frame path,
  `..._l3_initialize_pages`, `..._l3_update_page_table`) over five kernels:
  compressor ring advance with integer emission boundaries, RMS/RoPE/
  Hadamard/fp8-fp4 sim-quant emission post, verbatim ring scatter, page
  init, table commits. See "S7 island bring-up" below.
- Island entries for F1/L1/L2: NOT in this module yet (an F1 skeleton and
  an L1+L2 landing are reported by other commits but absent from these
  sources on disk - reconciliation owned by those workstreams). The archive
  stays additive: island symbols land beside these files. First L1 piece
  landed here 2026-08-25: the hcEnter half (`spark_dsv4_rocm_l1_hc_enter`,
  C3 with fixed trees - split-K mix dots + sum-of-squares partials, fixed-
  tree partial fold + flat rsqrt norm + inference Sinkhorn, pre-reduce with
  verbatim residual copy), closing the highest-impact S7 gap: the landed L1
  hcPost fold consumes post/comb that nothing else in the archive produced,
  and the normalized hidden input never existed without pre-reduce.

## Layout

| Path | Contents |
|---|---|
| `source/spark_hw_rocm_internal.h` | handle wrapper + frozen status mapping |
| `source/spark_hw_rocm_memory.c` | memory pool family, pinned host memory, copy/memset, read-ahead |
| `source/spark_hw_rocm_queue.c` | queue family incl. `hipLaunchHostFunc` callback |
| `source/spark_hw_rocm_event.c` | event family (timing disabled) |
| `source/spark_hw_rocm_graph.c` | graph capture/instantiate/launch/destroy |
| `source/spark_hw_rocm_target.c` | gfx950 guard + descriptor |
| `source/spark_dsv4_rocm_islands.h` | core-facing island entry surface (E0, L5, L4, L3), C linkage, geometry-free |
| `source/spark_dsv4_rocm_islands.hip` | all island kernels: E0 gather/expand; L5 gate/up-swiglu-down; L4 route-group, grouped-W13-swiglu, grouped-W2 down projection, weighted pair-reduce; L3 compressor ring advance, emission post (RMS/RoPE/Hadamard/sim-quant), verbatim scatter, page init, table commit; L1 hcEnter (split-K mix dots + sum-of-squares partials, partial fold + Sinkhorn finalize, pre-reduce + residual copy) |
| `tools/compile_check.sh` | compile check against ROCm headers |
| `tools/island_compile_check.sh` | island TU host-surface proof + header C proof + executed C2 host probe (+ hipcc mode on hardware) |
| `tools/l3_c2_host_probe.cpp` | host-executable C2 semantics tests for the island kernels (executor shims) |
| `tools/l1_hcenter_host_probe.cpp` | L1 hcEnter verification: executed pre-reduce kernel + deterministic replays of the fixed split-K/Sinkhorn trees vs naive fp64 ground truth and an independent finalize transcription |
| `tools/selftest.sh` | host-pure logic selftest (status mapping + SparkHwStatusToString), no HIP library at link |
| `tests/test_spark_hw_rocm_pure.c` | host-pure logic test driven by selftest.sh |
| `validation/l4_route_synthesis.h` | shared deterministic case synthesis + scalar reference model for the L4 island |
| `validation/l4_route_driver.c` | synthetic sealed-route batch driver: selfcheck invariants + SLR4 fixture generation |
| `validation/l4_route_device_check.hip` | on-hardware comparator: C2 bit-exact + C3 bounded vs fixtures through the real entry |
| `validation/run_l4_route_validation.sh` | driver orchestration (gen anywhere / check on gfx950) |
| `validation/fixtures/` | generated golden fixtures + SHA-256 manifest |
| `EVIDENCE_S7_STEP1_ISLAND_ENTRIES.md` | ran-where classification for the step-1 island work |
| `EVIDENCE_S7_STEP3_L4_GROUPED_MOE.md` | ran-where classification for the L4 island incl. delivered driver |
| `EVIDENCE_S7_STEP4_L3_CACHE_TRANSITION.md` | ran-where classification + defect note for the L3 island |
| `vendor/hip-headers/` | vendored ROCm 7.14 host-API subset (+PROVENANCE.md) |

The neutral header itself lives at `include/sparkpipe/spark_hw_iface.h` in the
repo root - owned by the interface (section 4), shared with the future CUDA
wrapper; it must never fork per target (rule R7 scope note).

## Primitive -> HIP mapping

| Interface primitive | HIP call | Notes |
|---|---|---|
| memory_pool_alloc/free | hipMalloc / hipFree | opaque control block; one device allocation per handle in v1 |
| host_pinned_alloc/free | hipHostMalloc / hipHostFree | SPARK_HW_PINNED_* bits are identity-mapped to hipHostMalloc* values |
| host_device_pointer | hipHostGetDevicePointer | mapped allocations only |
| copy_async | hipMemcpyAsync | SparkHwCopyKind translated to hipMemcpyKind |
| memset_async | hipMemsetAsync | value truncated to low byte like cudaMemsetAsync |
| read_ahead | hipLaunchHostFunc | three-clause contract of section 4.2, see below |
| queue_create | hipStreamCreateWithPriority | flags identity-mapped to hipStream* values |
| queue_query | hipStreamQuery | hipErrorNotReady -> SPARK_HW_NOT_READY (query only) |
| queue_synchronize | hipStreamSynchronize | failure paths only (section 4.2 note) |
| queue_enqueue_host_callback | hipLaunchHostFunc | the single stream-ordered frame callback |
| event_create | hipEventCreateWithFlags | hipEventDisableTiming forced on |
| event_record / queue_wait_event | hipEventRecord / hipStreamWaitEvent | zero wait bits |
| graph_capture_begin/end | hipStreamBeginCapture / hipStreamEndCapture | relaxed mode only in v1 |
| graph_instantiate | hipGraphInstantiate (ROCm 5-arg form) | resource refusal -> EXHAUSTED |
| graph_launch | hipGraphLaunch | |
| graph_destroy / graph_exec_destroy | hipGraphDestroy / hipGraphExecDestroy | may drain during teardown |

### read_ahead (section 4.2 weight read-ahead contract)

v1 enqueues a stream-ordered completion via `hipLaunchHostFunc` that writes
exactly one target-defined status word (0 = no occupancy data published)
into the pinned sink when capacity allows. The advisory bulk prefetch kick
is a no-op until island implementations register their context-to-region
maps; correctness never depends on this primitive, and any failed kick
degrades to no-op so it can never fail a frame. Bad handles return
SPARK_HW_INVALID.

## Status mapping (section 4.0 - no invented codes)

| hipError_t | SparkHwStatus |
|---|---|
| hipSuccess | SPARK_HW_OK |
| hipErrorNotReady | SPARK_HW_NOT_READY |
| hipErrorOutOfMemory (= hipErrorMemoryAllocation) | SPARK_HW_EXHAUSTED |
| hipErrorIllegalAddress, hipErrorContextIsDestroyed, hipErrorDeinitialized | SPARK_HW_LOST |
| hipErrorUnknown | SPARK_HW_LOST (ambiguous fatal fails closed) |
| hipErrorNotSupported | SPARK_HW_UNSUPPORTED |
| everything else | SPARK_HW_INVALID |

Allocation failures land as EXHAUSTED, never INVALID. There is no in-place
recovery from LOST in v1; the core tears the instance down.

`SparkHwStatusToString` is defined archive-side (`spark_hw_rocm_status_string.c`;
section 6 assigns interface symbols to the per-target archive) and falls back
to `SPARK_HW_UNKNOWN` for out-of-range values. The mapping table and strings
are unit-tested by `tools/selftest.sh`, including the alias traps
(OutOfMemory == MemoryAllocation == 2, NotInitialized == InitializationError
== 3) that force the if-chain instead of a switch.

## gfx950 fail-closed guard

`spark_hw_rocm_require_gfx950(int device)` mirrors
`SparkDsv4RequireNativeSm121` (resolution rule 3): it requires BOTH that
`hipDeviceProp_t.gcnArchName` begins with "gfx950" AND that the reported
compute capability is exactly (9,5). Any mismatch, query failure, or
out-of-range device refuses - arch name and capability must agree; nothing
downgrades. The verdict caches per device index (atomic byte), so repeated
checks cost one relaxed load; a failed probe records FAIL permanently -
fail-closed means we do not retry our way past a bad device.

`spark_hw_rocm_target_describe(SparkHwTarget *)` calls the guard first, then
fills the section 4.1 descriptor by querying CU count, the LDS limit
(MaxSharedMemoryPerBlock - queried, never hardcoded), capability major/minor,
and warp size (wavefront_lanes = 64 expected; informational, feeds no gate,
closes A12). Read once in ModuleInitialize; never consulted in the hot path.

## S7 island bring-up - steps 1, 3, and 4 (E0, L5, L4, L3)

The S7 order in `docs/coord/plan_amd_gfx950_mi350p.md` section 5.2 codes
E0 and L5 first because they are the two cheapest proof classes: E0 proves
integer exactness through the whole primitive stack, L5 is the simplest C3
accumulation island and doubles as the tolerance-plumbing shakeout.

Naming note for reviewers: in the frozen contract the embedding island is
E0 `prologue.embed` and the head island is F1 `head.final`; L1 is
`layer.boundary_norm_project` and L5 is `layer.moe_shared`. The step-1 pair
is therefore E0 + L5, and the entry symbols carry both the frozen id and
the frozen name (`spark_dsv4_rocm_e0_prologue_embed`,
`spark_dsv4_rocm_l5_moe_shared`).

### What the entries implement

| Island | Kernels | Determinism |
|---|---|---|
| E0 `prologue.embed` | one gather/expand kernel: token row -> hc_mult identical hyper-connection streams, verbatim bytes (16B vectors + word tail) | C2 by construction - no arithmetic touches payload bytes; fires the optional `spark_hw_read_ahead` kick |
| L5 `layer.moe_shared` | three uncaptured kernels: gate/up projections (one wavefront per neuron over w1/w3 halves), clamped swiglu (up two-sided, gate max-only, swish(gate)*up fp32), down projection added into the FFN accumulator (AccumAdd semantic) | C3 with ONE fixed reduction tree per shape bucket (B3): per-lane sequential fp32 k-stride, then log2(64)-step shuffle-down butterfly, then a single RNE bf16 store |
| L4 `layer.moe_routed` | four uncaptured kernels over the core-sealed logical route: single-CTA route grouping (histogram -> canonical exclusive-scan offsets -> counting-rank scatter; THE C2 observable is expert_offsets), grouped W13 gate/up + clamped swiglu over stacked experts, grouped W2 down projection (the routed FFN down projection, unweighted rows), weighted pair reduce into the FFN accumulator through the inverse map (weights fold exactly once) | split verdict: expert offsets + grouped assignment C2 integer-exact and run-invariant; GEMM/reduce halves C3 on the same fixed tree as L5, reduce sums k ascending in fp32 |
| L3 `layer.cache_transition` | five kernels across three lifecycle symbols: per-frame compressor ring advance (integer boundary = (position+1) % ratio == 0), emission post (RMS norm -> RoPE -> optional Hadamard -> fp8/fp4 block sim-quant) with ring scatter, verbatim direct scatter; allocation-time page init (parent copy + -inf spans); scattered page-table commits | split verdict: ring slots, page addresses, emitted flags and emit counters C2 integer-pure (counters are exact integer-atomic totals); direct-path payloads copied unrounded (C2-eligible); quantized emissions C3 on the fixed tree - the one deliberate difference is the wave64 RMS reduce, inside shared tolerances. Software E4M3 RNE encoder validated over 22M checks after a binade-carry defect was caught and fixed (EVIDENCE_S7_STEP4_L3_CACHE_TRANSITION.md) |
| L1 `layer.boundary_norm_project` (hcEnter half) | three kernels behind one entry mirroring the CUDA module's HcEnter sequence: split-K mix dots + sum-of-squares partials over the flattened packet (256-element splits, one thread per element), one-wave finalize folding partials on the fixed tree into mixes/pre/post/comb via rsqrt(mean-square) norm + inference Sinkhorn (sigmoid pre, doubled sigmoid post, row-softmax comb, alternating normalizations with +epsilon in every divisor), pre-reduce into normalized hidden with verbatim residual copy | C3 with ONE documented tree set: per-lane wave64 subsets -> butterfly -> ascending-split fold; Sinkhorn serial per row (order-free); residual copy byte-exact; deltas vs CUDA recorded at the kernels (wave64 subsets; expf where CUDA uses fast-math __expf). Verified by tools/l1_hcenter_host_probe.cpp (executed pre-reduce; replayed trees vs naive fp64 + independent finalize transcription) |

Weight formats for v1 islands: BF16 (scale-free), FP8_E4M3 with one E8M0
scale byte per scale_group elements per row (stagepack rule: fp8 block 128
columns), and - since the L4 leg - MXFP4_E2M1 with E8M0 scales (stagepack
rule: fp4 experts block 32; scale_group must be a multiple of 8 so an
eight-element decode run never crosses a block). Decode algorithms mirror the shared
documented formulas exactly; the landmine rule keeps
`spark_lm_kernels.cuh` out of every ROCm TU, so they are restated in the
.hip TU from their definitions rather than included.

L5 fork/join: the optional fork event is waited before any launch, the
optional join event is recorded after the last enqueue (both NULL-tolerant
for TP1 bring-up before L4 exists). Success paths never synchronize;
completion stays stream-ordered through the primitive family callback.

### Compile status (honest ran-where)

    tools/island_compile_check.sh vendor   # host-surface proofs, vendored headers
    tools/island_compile_check.sh system   # same vs /opt/rocm/include
    tools/island_compile_check.sh hipcc    # REAL device compile on hardware

Host proofs parse the entire .hip TU as plain C++ (device annotations
inert, implicit coordinates and wave shuffles shimmed) and consume the
entry header as C11 the way dsv4_core will. Beyond parsing, the
L3_C2_HOST_EXEC_PROBE step EXECUTES every C2 integer kernel on the host
under executor coordinate shims - route grouping canonicality, ring-slot
addressing with collision semantics, page init/commits, compressor
boundaries/counters/state - against hand-derived expectations (a blockDim-1
walk completes every strided loop; cross-lane-reduction kernels are
excluded by design and stay covered by formula probes plus hardware).
None of this proves device codegen: gfx950 verification happens when
MI350P lands, via the hipcc mode. Details in
EVIDENCE_S7_STEP1_ISLAND_ENTRIES.md, EVIDENCE_S7_STEP4_L3_CACHE_TRANSITION.md,
and EVIDENCE_HOST_EXECUTOR_PROBE.md.

## Assumptions

- One GPU per rank process (TP deployments); every call uses the calling
  thread's current HIP device, set once at initialize.
- Queues/events/graphs are HIP pointers carried behind incomplete-tag opaque
  typedefs (see the rendering note in spark_hw_iface.h).
- Unknown flag bits on pinned alloc, queue create, or event create return
  SPARK_HW_UNSUPPORTED instead of passing through silently.

## Compile check

All archive translation units compile clean (`-std=c11 -Wall -Wextra -Werror`)
against the vendored real ROCm 7.14 host-API headers:

    tools/compile_check.sh
    # COMPILE_CHECK_OK: all sources compiled against vendor HIP headers.

The host-pure logic (status mapping, status strings) additionally has a
unit selftest that runs on any machine - no GPU, no HIP library at link:

    tools/selftest.sh
    # test_spark_hw_rocm_pure PASS / SELFTEST_OK

On machines with a ROCm install, verify against the driver's own headers:

    tools/compile_check.sh system          # /opt/rocm/include
    HIP_HEADERS=/path/to/hip/include tools/compile_check.sh system

Evidence recorded 2026-08-24: Apple clang 16.0.0 (clang-1600.0.26.3),
vendored ROCm therock-7.14 headers (HIP_VERSION 7.14.60850); compile gate
re-recorded 2026-08-25 after adding spark_hw_rocm_status_string.c (six TUs
OK) and the identity-mapping _Static_assert block; selftest PASS same day.
