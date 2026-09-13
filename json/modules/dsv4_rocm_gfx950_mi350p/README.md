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
- Island entries for F1/L1 then L2/L3 (remaining steps of the S7 order):
  NOT in this module yet. The archive stays additive: island symbols land
  beside these files, nothing here changes.

## Layout

| Path | Contents |
|---|---|
| `source/spark_hw_rocm_internal.h` | handle wrapper + frozen status mapping |
| `source/spark_hw_rocm_memory.c` | memory pool family, pinned host memory, copy/memset, read-ahead |
| `source/spark_hw_rocm_queue.c` | queue family incl. `hipLaunchHostFunc` callback |
| `source/spark_hw_rocm_event.c` | event family (timing disabled) |
| `source/spark_hw_rocm_graph.c` | graph capture/instantiate/launch/destroy |
| `source/spark_hw_rocm_target.c` | gfx950 guard + descriptor |
| `source/spark_dsv4_rocm_islands.h` | core-facing island entry surface (E0 + L5), C linkage, geometry-free |
| `source/spark_dsv4_rocm_islands.hip` | E0 gather/expand, L5 gate/up, swiglu, down-accumulate, and the L4 route-group / grouped-W13-swiglu / grouped-W2 down projection / weighted pair-reduce kernels with their entries |
| `tools/compile_check.sh` | compile check against ROCm headers |
| `tools/island_compile_check.sh` | island TU host-surface proof + header C proof (+ hipcc mode on hardware) |
| `EVIDENCE_S7_STEP1_ISLAND_ENTRIES.md` | ran-where classification for the step-1 island work |
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

## S7 island bring-up - steps 1 and 3 (E0, L5, then the L4 leg)

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
entry header as C11 the way dsv4_core will. They prove the surface, not
the machine code: device verification for gfx950 happens when MI350P lands,
via the hipcc mode. Details in EVIDENCE_S7_STEP1_ISLAND_ENTRIES.md and
EVIDENCE_S7_STEP3_L4_GROUPED_MOE.md.

## Assumptions

- One GPU per rank process (TP deployments); every call uses the calling
  thread's current HIP device, set once at initialize.
- Queues/events/graphs are HIP pointers carried behind incomplete-tag opaque
  typedefs (see the rendering note in spark_hw_iface.h).
- Unknown flag bits on pinned alloc, queue create, or event create return
  SPARK_HW_UNSUPPORTED instead of passing through silently.

## Compile check

All five translation units compile clean (`-std=c11 -Wall -Wextra -Werror`)
against the vendored real ROCm 7.14 host-API headers:

    tools/compile_check.sh
    # COMPILE_CHECK_OK: all sources compiled against vendor HIP headers.

On machines with a ROCm install, verify against the driver's own headers:

    tools/compile_check.sh system          # /opt/rocm/include
    HIP_HEADERS=/path/to/hip/include tools/compile_check.sh system

Evidence recorded 2026-08-24: Apple clang 16.0.0 (clang-1600.0.26.3),
vendored ROCm therock-7.14 headers (HIP_VERSION 7.14.60850), all 5 TUs OK.
