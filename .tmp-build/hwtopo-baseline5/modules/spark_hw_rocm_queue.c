// spark_hw_rocm_queue.c - Queue primitive family for rocm.gfx950.mi350p.
//
// Frozen contract: hwiface_v1.md section 4.2. Mechanical HIP mapping:
//   queue_create            -> hipStreamCreateWithPriority; flag bits
//     identity-mapped (SPARK_HW_QUEUE_NONBLOCKING == hipStreamNonBlocking).
//   queue_destroy           -> hipStreamDestroy. May internally drain in-flight
//     frame work during teardown after the core quiesced the instance
//     (section 4.2 synchronization note).
//   queue_query             -> hipStreamQuery; hipErrorNotReady maps to
//     SPARK_HW_NOT_READY, which is legal for this primitive only.
//   queue_synchronize       -> hipStreamSynchronize; failure paths ONLY per the
//     section 4.2 note - a successful frame completes externally via
//     enqueue_host_callback.
//   queue_enqueue_host_callback -> hipLaunchHostFunc: the single stream-ordered
//     host callback a frame may rely on.

#include "spark_hw_rocm_internal.h"

SparkHwStatus spark_hw_queue_create(SparkHwQueue *q, uint32_t flags, int32_t priority)
{
    if (q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    if ((flags & ~SPARK_HW_QUEUE_NONBLOCKING) != 0u)
    {
        return SPARK_HW_UNSUPPORTED; /* unknown flag bits are refused */
    }
    hipStream_t stream = NULL;
    /* Pass-through guarded by _Static_assert in spark_hw_rocm_internal.h:
     * SPARK_HW_QUEUE_* equals the hipStream* values. */
    hipError_t err = hipStreamCreateWithPriority(&stream, flags, (int)priority);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    *q = (SparkHwQueue)stream;
    return SPARK_HW_OK;
}

SparkHwStatus spark_hw_queue_destroy(SparkHwQueue q)
{
    if (q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipStreamDestroy((hipStream_t)q));
}

SparkHwStatus spark_hw_queue_query(SparkHwQueue q)
{
    if (q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipStreamQuery((hipStream_t)q));
}

SparkHwStatus spark_hw_queue_synchronize(SparkHwQueue q)
{
    if (q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipStreamSynchronize((hipStream_t)q));
}

SparkHwStatus spark_hw_queue_enqueue_host_callback(SparkHwQueue q, void (*fn)(void*), void *arg)
{
    if (q == NULL || fn == NULL)
    {
        return SPARK_HW_INVALID;
    }
    /* hipHostFn_t is void (*)(void*) - identical to the frozen signature. */
    return spark_hw_rocm_map_status(
        hipLaunchHostFunc((hipStream_t)q, (hipHostFn_t)fn, arg));
}
