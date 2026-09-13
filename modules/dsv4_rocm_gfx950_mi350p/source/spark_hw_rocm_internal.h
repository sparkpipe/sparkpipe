#pragma once

// spark_hw_rocm_internal.h - shared plumbing for the rocm.gfx950.mi350p
// target archive (hwiface_v1.md section 6 link unit dsv4_rocm_gfx950_mi350p).
// Not part of the neutral interface; never included outside this module.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include <sparkpipe/spark_hw_iface.h>

// Identity-mapping enforcement ----------------------------------------------
// The interface flag encodings deliberately mirror the HIP constants so the
// wrappers translate by identity (see spark_hw_iface.h and the pass-through
// flag handling in queue/memory/event). That claim was prose-only; these
// asserts make it load-bearing: if a future ROCm bumps a constant, compilation
// fails here instead of silently mis-flagging streams, events, or pinned
// allocations. Deliberately NOT asserted: SPARK_HW_CAPTURE_RELAXED (the core-
// facing capture mode value is opaque; graph.c translates it explicitly to
// hipStreamCaptureModeRelaxed rather than relying on numeric identity).
_Static_assert(SPARK_HW_QUEUE_DEFAULT == hipStreamDefault,
    "SPARK_HW_QUEUE_DEFAULT must equal hipStreamDefault for pass-through");
_Static_assert(SPARK_HW_QUEUE_NONBLOCKING == hipStreamNonBlocking,
    "SPARK_HW_QUEUE_NONBLOCKING must equal hipStreamNonBlocking for "
    "pass-through");
_Static_assert(SPARK_HW_PINNED_DEFAULT == hipHostMallocDefault,
    "SPARK_HW_PINNED_DEFAULT must equal hipHostMallocDefault for pass-through");
_Static_assert(SPARK_HW_PINNED_PORTABLE == hipHostMallocPortable,
    "SPARK_HW_PINNED_PORTABLE must equal hipHostMallocPortable for pass-through");
_Static_assert(SPARK_HW_PINNED_MAPPED == hipHostMallocMapped,
    "SPARK_HW_PINNED_MAPPED must equal hipHostMallocMapped for pass-through");
_Static_assert(SPARK_HW_EVENT_DISABLE_TIMING == hipEventDisableTiming,
    "SPARK_HW_EVENT_DISABLE_TIMING must equal hipEventDisableTiming");

// Handle wrappers -----------------------------------------------------------
// HIP stream/event/graph handles are themselves C pointers, so the opaque
// void* contract handles carry them directly (rule R3: opaque to the core,
// inspected only here).

#define SPARK_HW_ROCM_MEMORY_MAGIC 0x5350524Du /* 'SPRM' */

typedef struct SparkHwRocmMemory
{
    uint32_t magic;
    uint32_t device;
    uint64_t bytes;
    void *dev_base;
} SparkHwRocmMemory;

// Status mapping (hwiface_v1.md section 4.0) ---------------------------------
// Frozen six-code surface; every hipError_t lands in exactly one bucket.
// Documented decisions:
//   - hipErrorOutOfMemory/hipErrorMemoryAllocation -> EXHAUSTED (allocation
//     failure is capacity refusal, never INVALID).
//   - hipErrorIllegalAddress (device fault), hipErrorContextIsDestroyed,
//     hipErrorDeinitialized -> LOST: fail-closed, instance teardown.
//   - hipErrorUnknown also maps LOST: an ambiguous fatal error must fail
//     closed rather than be misread as a recoverable argument bug.
//   - hipErrorNotSupported -> UNSUPPORTED (capability/descriptor refusals).
//   - everything else -> INVALID.
// If-chain, not switch: several hipError_t spellings share numeric values
// (e.g. OutOfMemory == MemoryAllocation == 2), which would trip duplicate
// case labels.

static inline SparkHwStatus spark_hw_rocm_map_status(hipError_t err)
{
    if (err == hipSuccess)
    {
        return SPARK_HW_OK;
    }
    if (err == hipErrorNotReady)
    {
        return SPARK_HW_NOT_READY;
    }
    if (err == hipErrorOutOfMemory || err == hipErrorMemoryAllocation)
    {
        return SPARK_HW_EXHAUSTED;
    }
    if (err == hipErrorIllegalAddress || err == hipErrorContextIsDestroyed ||
        err == hipErrorDeinitialized || err == hipErrorUnknown)
    {
        return SPARK_HW_LOST;
    }
    if (err == hipErrorNotSupported)
    {
        return SPARK_HW_UNSUPPORTED;
    }
    return SPARK_HW_INVALID;
}

// Handle validation helpers --------------------------------------------------

static inline SparkHwStatus spark_hw_rocm_check_memory(const SparkHwMemory *mem)
{
    if (mem == NULL)
    {
        return SPARK_HW_INVALID;
    }
    const SparkHwRocmMemory *m = (const SparkHwRocmMemory *)mem;
    if (m->magic != SPARK_HW_ROCM_MEMORY_MAGIC || m->dev_base == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return SPARK_HW_OK;
}
