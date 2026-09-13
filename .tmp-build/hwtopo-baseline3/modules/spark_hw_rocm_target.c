// spark_hw_rocm_target.c - gfx950 fail-closed guard + target descriptor for
// rocm.gfx950.mi350p.
//
// Frozen contract:
//   - section 6 resolution rule 3: capability checks stay fail-closed inside
//     the target, mirroring SparkDsv4RequireNativeSm121 (major==12 && minor==1).
//     The gfx950 equivalent fails closed on anything that is not gfx950.
//   - section 4.1: the descriptor is read once in ModuleInitialize and never
//     consulted in the hot path; wavefront_lanes is informational and feeds no
//     gate (closes A12).
//
// Guard identity check: hipDeviceProp_t.gcnArchName must begin with "gfx950"
// (covers suffix variants such as "gfx950:sramecc+"), AND the reported compute
// capability must be (9,5). Both must agree; any mismatch or query failure is a
// refusal, never a downgrade. The verdict is cached per device index in atomic
// state so repeated calls cost one relaxed load (the per-executor-thread cache
// of the CUDA analog generalized to per-device).

#include <stdatomic.h>

#include "spark_hw_rocm_internal.h"

#define SPARK_HW_ROCM_MAX_DEVICES 64u
#define SPARK_HW_ROCM_GFX950_NAME "gfx950"
#define SPARK_HW_ROCM_TARGET_ID   "rocm.gfx950.mi350p" /* frozen, section 6 */

typedef enum SparkHwRocmGuardVerdict
{
    SPARK_HW_ROCM_GUARD_UNKNOWN = 0,
    SPARK_HW_ROCM_GUARD_PASS = 1,
    SPARK_HW_ROCM_GUARD_FAIL = 2
} SparkHwRocmGuardVerdict;

static _Atomic uint8_t s_guard_cache[SPARK_HW_ROCM_MAX_DEVICES];

static SparkHwStatus spark_hw_rocm_guard_probe(int device)
{
    if (device < 0 || (uint32_t)device >= SPARK_HW_ROCM_MAX_DEVICES)
    {
        return SPARK_HW_INVALID;
    }

    hipDeviceProp_t prop;
    memset(&prop, 0, sizeof(prop));
    hipError_t err = hipGetDeviceProperties(&prop, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }

    if (strncmp(prop.gcnArchName, SPARK_HW_ROCM_GFX950_NAME,
                sizeof(SPARK_HW_ROCM_GFX950_NAME) - 1u) != 0)
    {
        return SPARK_HW_UNSUPPORTED; /* not a gfx950 part: fail closed */
    }

    int major = 0;
    int minor = 0;
    err = hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    err = hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    if (major != 9 || minor != 5)
    {
        return SPARK_HW_UNSUPPORTED; /* arch name and capability disagree */
    }
    return SPARK_HW_OK;
}

SparkHwStatus spark_hw_rocm_require_gfx950(int device)
{
    if (device < 0 || (uint32_t)device >= SPARK_HW_ROCM_MAX_DEVICES)
    {
        return SPARK_HW_INVALID;
    }

    const uint8_t cached = atomic_load_explicit(&s_guard_cache[device],
                                                memory_order_relaxed);
    if (cached == (uint8_t)SPARK_HW_ROCM_GUARD_PASS)
    {
        return SPARK_HW_OK;
    }
    if (cached == (uint8_t)SPARK_HW_ROCM_GUARD_FAIL)
    {
        return SPARK_HW_UNSUPPORTED;
    }

    const SparkHwStatus status = spark_hw_rocm_guard_probe(device);
    const uint8_t verdict = status == SPARK_HW_OK
                                ? (uint8_t)SPARK_HW_ROCM_GUARD_PASS
                                : (uint8_t)SPARK_HW_ROCM_GUARD_FAIL;
    atomic_store_explicit(&s_guard_cache[device], verdict, memory_order_relaxed);
    /* A probe that failed for a transient reason (e.g. LOST) is recorded as
     * FAIL: fail-closed means we do not retry our way past a bad device. */
    return status;
}

SparkHwStatus spark_hw_rocm_target_describe(SparkHwTarget *out)
{
    if (out == NULL)
    {
        return SPARK_HW_INVALID;
    }

    int device = 0;
    hipError_t err = hipGetDevice(&device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }

    const SparkHwStatus guard = spark_hw_rocm_require_gfx950(device);
    if (guard != SPARK_HW_OK)
    {
        return guard; /* descriptor read fails closed off-target */
    }

    int cu_count = 0;
    int lds_limit = 0;
    int major = 0;
    int minor = 0;
    int warp_lanes = 0;

    err = hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    err = hipDeviceGetAttribute(&lds_limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    err = hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    err = hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    err = hipDeviceGetAttribute(&warp_lanes, hipDeviceAttributeWarpSize, device);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }

    out->target_id = SPARK_HW_ROCM_TARGET_ID;
    out->abi_version = 1u;
    out->multiprocessor_count = (uint32_t)cu_count;
    out->max_dynamic_shared_bytes = (uint64_t)lds_limit; /* queried LDS limit,
                                                            never hardcoded */
    out->capability_major = (uint32_t)major;
    out->capability_minor = (uint32_t)minor;
    out->wavefront_lanes = (uint32_t)warp_lanes; /* informational only (A12) */
    return SPARK_HW_OK;
}
