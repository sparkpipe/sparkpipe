// spark_hw_rocm_event.c - Event primitive family for rocm.gfx950.mi350p.
//
// Frozen contract: hwiface_v1.md section 4.2. Mechanical HIP mapping:
//   event_create      -> hipEventCreateWithFlags with hipEventDisableTiming
//     forced on: timing is NOT required by the interface (today's CUDA module
//     passes cudaEventDisableTiming). v1 accepts flags == 0 or exactly
//     SPARK_HW_EVENT_DISABLE_TIMING; anything else is UNSUPPORTED.
//   event_record      -> hipEventRecord on the given queue.
//   queue_wait_event  -> hipStreamWaitEvent with zero flag bits.

#include "spark_hw_rocm_internal.h"

/* The interface's timing flag is the HIP bit itself, so the forced-on OR
 * below sets exactly the one intended bit; pinned against header drift. */
_Static_assert(SPARK_HW_EVENT_DISABLE_TIMING ==
                   (uint32_t)hipEventDisableTiming,
               "SPARK_HW_EVENT_DISABLE_TIMING must equal hipEventDisableTiming");

SparkHwStatus spark_hw_event_create(SparkHwEvent *e, uint32_t flags)
{
    if (e == NULL)
    {
        return SPARK_HW_INVALID;
    }
    const uint32_t known = SPARK_HW_EVENT_DISABLE_TIMING;
    if ((flags & ~known) != 0u)
    {
        return SPARK_HW_UNSUPPORTED;
    }
    hipEvent_t event = NULL;
    hipError_t err = hipEventCreateWithFlags(&event,
                                             (unsigned)(flags | hipEventDisableTiming));
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err);
    }
    *e = (SparkHwEvent)event;
    return SPARK_HW_OK;
}

SparkHwStatus spark_hw_event_destroy(SparkHwEvent e)
{
    if (e == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipEventDestroy((hipEvent_t)e));
}

SparkHwStatus spark_hw_event_record(SparkHwEvent e, SparkHwQueue q)
{
    if (e == NULL || q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipEventRecord((hipEvent_t)e, (hipStream_t)q));
}

SparkHwStatus spark_hw_queue_wait_event(SparkHwQueue q, SparkHwEvent e)
{
    if (e == NULL || q == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(
        hipStreamWaitEvent((hipStream_t)q, (hipEvent_t)e, 0u));
}
