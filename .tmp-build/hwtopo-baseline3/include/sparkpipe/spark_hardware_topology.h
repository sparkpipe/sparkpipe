#ifndef SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_H
#define SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_H

/*
 * Compile-time hardware topology, parameterized by target id.
 *
 * One profile per frozen target id (hwiface_v1.md section 6):
 *
 *     cuda.sm121.gb10      NVIDIA GB10 (Spark workstation GPU), SM 12.1
 *     rocm.gfx950.mi350p   AMD Instinct MI350P (gfx950 / CDNA4)
 *
 * Why this header exists: hwiface_v1.md binding rule 4 forbids a target
 * numeric constant in portable-core code, and section 5 lists the constants
 * that must move into a target-controlled home. This header is that home for
 * the values that are compile-time facts of the deployment. What a launch
 * consumes at runtime still comes from the SparkHwTarget descriptor filled
 * from the live device at ModuleInitialize (hwiface_v1.md section 4.1);
 * descriptor and header agree because both derive from this one table.
 *
 * Selection: pass -DSPARK_HW_TARGET_ID=SPARK_HW_TARGET_ROCM_GFX950_MI350P
 * (or the GB10 spelling) to pick the deployment explicitly. Without it the
 * header autodetects from the accelerator compiler (__HIP__, then
 * __CUDACC__) and otherwise falls back to cuda.sm121.gb10 - every
 * translation unit in today's tree is a GB10 deployment, so the fallback
 * only names existing reality. A ROCm build must set the macro or compile
 * under hipcc.
 *
 * Sentinel contract: a 0 value means "this profile does not pin the number
 * at compile time". Treat 0 as unpinned - size requests from the device
 * descriptor instead; never use 0 as a budget or bound.
 *
 * Provenance is cited per constant: "measured" values come from receipts or
 * docs in-tree; "advisory" values are the frozen advisor notes awaiting
 * on-hardware confirmation (docs/coord/plan_amd_gfx950_mi350p.md section
 * 3.3: measure, then pin).
 */

/* Frozen target ids (hwiface_v1.md section 6). */
#define SPARK_HW_TARGET_CUDA_SM121_GB10    1
#define SPARK_HW_TARGET_ROCM_GFX950_MI350P 2

#if !defined(SPARK_HW_TARGET_ID)
# if defined(__HIP__)
#  define SPARK_HW_TARGET_ID SPARK_HW_TARGET_ROCM_GFX950_MI350P
# elif defined(__CUDACC__)
#  define SPARK_HW_TARGET_ID SPARK_HW_TARGET_CUDA_SM121_GB10
# else
#  define SPARK_HW_TARGET_ID SPARK_HW_TARGET_CUDA_SM121_GB10
# endif
#endif

#if SPARK_HW_TARGET_ID == SPARK_HW_TARGET_CUDA_SM121_GB10

/*
 * NVIDIA GB10, compute capability 12.1.
 */

#define SPARK_HW_TARGET_NAME_STRING "cuda.sm121.gb10"

/* 48 SMs (docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:23, measured).
 * Launch geometry still takes its count from the runtime query into the
 * descriptor (multiprocessor_count); this pin exists for capacity math,
 * occupancy estimates and tests. */
#define SPARK_HW_MULTIPROCESSOR_COUNT 48u

/* Warp width. */
#define SPARK_HW_WAVEFRONT_LANES 32u

/* Fail-closed capability gate pins (hwiface_v1.md sections 4.1 and 5(a)). */
#define SPARK_HW_COMPUTE_CAPABILITY_MAJOR 12u
#define SPARK_HW_COMPUTE_CAPABILITY_MINOR 1u

/* Static __shared__ ceiling ptxas enforces per block (48 KB). */
#define SPARK_HW_STATIC_SHARED_LIMIT_BYTES 49152u

/* Dynamic shared-memory opt-in ceiling after
 * cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize):
 * 101376 B measured on GB10, frozen in hwiface_v1.md section 4.1. */
#define SPARK_HW_MAX_DYNAMIC_SHARED_BYTES 101376ull

/* L1/shared total per SM: 128 KB (GB10_CUDA_COST_MODEL_CALIBRATION.md:23;
 * formerly LM_SMEM_SM_TOTAL in inference/kernels/layout.cuh). */
#define SPARK_HW_SHARED_PER_SM_BYTES 131072u

/* L2 cache size: no in-tree measurement pins it yet; query
 * cudaDevAttrL2CacheSize into the descriptor instead. */
#define SPARK_HW_L2_CACHE_BYTES 0ull

/* Boost clock, kHz to match cudaDeviceProp::clockRate units:
 * up to 2.55 GHz (GB10_CUDA_COST_MODEL_CALIBRATION.md:23). */
#define SPARK_HW_MAX_CLOCK_KHZ 2550000u

/* Unified LPDDR5X memory bandwidth, GB/s (roadmap:29 via
 * tools/perf_estimate.py NODE_BW_GBPS). Informational/perf-model use. */
#define SPARK_HW_MEMORY_BANDWIDTH_GBPS 273.0

#elif SPARK_HW_TARGET_ID == SPARK_HW_TARGET_ROCM_GFX950_MI350P

/*
 * AMD Instinct MI350P (gfx950 / CDNA4). Numbers are advisor-provided until
 * measured on hardware (docs/coord/advisor_amd.md;
 * docs/coord/plan_amd_gfx950_mi350p.md section 3.3: measure, then pin).
 */

#define SPARK_HW_TARGET_NAME_STRING "rocm.gfx950.mi350p"

/* CU count: queried at init and pinned there (plan section 3.3), so the
 * compile-time value stays unpinned. */
#define SPARK_HW_MULTIPROCESSOR_COUNT 0u

/* 64-lane wavefronts (frozen descriptor note, hwiface_v1.md section 4.1;
 * informational per REV2 A12 - it must feed no gate). */
#define SPARK_HW_WAVEFRONT_LANES 64u

/* Capability pins deliberately unset until measured on gfx950 hardware
 * (plan section 3.3); the target's fail-closed guard fills them from the
 * device query at init. */
#define SPARK_HW_COMPUTE_CAPABILITY_MAJOR 0u
#define SPARK_HW_COMPUTE_CAPABILITY_MINOR 0u

/* LDS budgets are the gfx950 target's own, never imported from GB10
 * (hwiface_v1.md section 5(a)); pin them here only after measuring via
 * hipDeviceGetAttribute. */
#define SPARK_HW_STATIC_SHARED_LIMIT_BYTES 0u
#define SPARK_HW_MAX_DYNAMIC_SHARED_BYTES 0ull
#define SPARK_HW_SHARED_PER_SM_BYTES 0u

#define SPARK_HW_L2_CACHE_BYTES 0ull
#define SPARK_HW_MAX_CLOCK_KHZ 0u
#define SPARK_HW_MEMORY_BANDWIDTH_GBPS 0.0

#else
#error "SPARK_HW_TARGET_ID must be SPARK_HW_TARGET_CUDA_SM121_GB10 or SPARK_HW_TARGET_ROCM_GFX950_MI350P"
#endif

/* Explicit check for the sentinel contract at use sites. */
#define SPARK_HW_VALUE_IS_PINNED(value) ((value) != 0)

#endif /* SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_H */
