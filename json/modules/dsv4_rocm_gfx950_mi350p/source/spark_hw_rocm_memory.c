// spark_hw_rocm_memory.c - Memory primitive family for rocm.gfx950.mi350p.
//
// Frozen contract: hwiface_v1.md section 4.2. Mechanical HIP mapping:
//   memory_pool_alloc/free -> hipMalloc/hipFree behind an opaque control block.
//     v1 keeps one device allocation per handle; a slab-pool sub-allocator is a
//     target-internal optimization that may land later without touching the ABI
//     (the handle stays opaque, rule R3).
//   host_pinned_alloc/free  -> hipHostMalloc/hipHostFree; flag bits are the
//     identity-mapped SPARK_HW_PINNED_* encodings (see spark_hw_iface.h).
//   host_device_pointer     -> hipHostGetDevicePointer (mapped allocations only).
//   copy_async              -> hipMemcpyAsync with kind translated.
//   memset_async            -> hipMemsetAsync.
//   read_ahead              -> three-clause contract of section 4.2: stream-
//     ordered advisory kick + completion write into the pinned sink. The v1
//     backend owns no context-to-weight-region map yet (that arrives with the
//     island implementations), so the kick itself is a no-op and the completion
//     writes one target-defined status word (0 = no occupancy data). Fewer than
//     word_capacity words is allowed by the contract; correctness never depends
//     on this primitive completing, and a failed kick degrades to no-op - it is
//     never allowed to fail the frame.
//
// Device selection: every call operates on the current HIP device of the
// calling thread (one GPU per rank process in TP deployments; the deployment
// sets the device once at initialize).

#include "spark_hw_rocm_internal.h"

#define SPARK_HW_ROCM_READ_AHEAD_STATUS_WORDS 1u

typedef struct SparkHwRocmReadAheadCtx
{
    uint32_t *sink;
    uint32_t word_capacity;
} SparkHwRocmReadAheadCtx;

// Stream-ordered completion for read_ahead: writes at most word_capacity words
// of target-defined occupancy/status into the host-pinned sink.
static void spark_hw_rocm_read_ahead_complete(void *user_data)
{
    SparkHwRocmReadAheadCtx *ctx = (SparkHwRocmReadAheadCtx *)user_data;
    if (ctx == NULL)
    {
        return;
    }
    if (ctx->sink != NULL && ctx->word_capacity > 0u)
    {
        /* Target-defined status word: 0 = no occupancy data published in v1. */
        *ctx->sink = 0u;
    }
    free(ctx);
}

SparkHwStatus spark_hw_memory_pool_alloc(SparkHwMemory **mem, uint64_t bytes)
{
    if (mem == NULL || bytes == 0u)
    {
        return SPARK_HW_INVALID;
    }

    void *dev_base = NULL;
    hipError_t err = hipMalloc(&dev_base, (size_t)bytes);
    if (err != hipSuccess)
    {
        return spark_hw_rocm_map_status(err); /* OOM lands here as EXHAUSTED */
    }

    SparkHwRocmMemory *handle = (SparkHwRocmMemory *)malloc(sizeof(*handle));
    if (handle == NULL)
    {
        (void)hipFree(dev_base);
        return SPARK_HW_EXHAUSTED; /* host-side capacity refusal */
    }

    int device = 0;
    (void)hipGetDevice(&device); /* best-effort annotation only */
    handle->magic = SPARK_HW_ROCM_MEMORY_MAGIC;
    handle->device = (uint32_t)device;
    handle->bytes = bytes;
    handle->dev_base = dev_base;
    *mem = (SparkHwMemory *)handle;
    return SPARK_HW_OK;
}

SparkHwStatus spark_hw_memory_pool_free(SparkHwMemory *mem)
{
    /* Teardown path: hipFree below drains in-flight frame work on the backing
     * after the core has quiesced the instance (section 4.2 sync note allows
     * internal synchronization inside *_free during teardown). */
    SparkHwStatus status = spark_hw_rocm_check_memory(mem);
    if (status != SPARK_HW_OK)
    {
        return status;
    }

    SparkHwRocmMemory *handle = (SparkHwRocmMemory *)mem;
    hipError_t err = hipFree(handle->dev_base);
    handle->magic = 0u;
    handle->dev_base = NULL;
    free(handle);
    return spark_hw_rocm_map_status(err);
}

SparkHwStatus spark_hw_host_pinned_alloc(void **host, uint64_t bytes, uint32_t flags)
{
    if (host == NULL || bytes == 0u)
    {
        return SPARK_HW_INVALID;
    }
    if ((flags & ~(SPARK_HW_PINNED_PORTABLE | SPARK_HW_PINNED_MAPPED)) != 0u)
    {
        return SPARK_HW_UNSUPPORTED; /* unknown flag bits are refused */
    }
    /* Pass-through guarded by _Static_assert in spark_hw_rocm_internal.h:
     * SPARK_HW_PINNED_* equals the hipHostMalloc* values. */
    return spark_hw_rocm_map_status(hipHostMalloc(host, (size_t)bytes, flags));
}

SparkHwStatus spark_hw_host_pinned_free(void *host)
{
    if (host == NULL)
    {
        return SPARK_HW_INVALID;
    }
    return spark_hw_rocm_map_status(hipHostFree(host));
}

SparkHwStatus spark_hw_host_device_pointer(void *host, void **device)
{
    if (host == NULL || device == NULL)
    {
        return SPARK_HW_INVALID;
    }
    /* Valid only for allocations made with SPARK_HW_PINNED_MAPPED. */
    return spark_hw_rocm_map_status(hipHostGetDevicePointer(device, host, 0u));
}

SparkHwStatus spark_hw_copy_async(void *dst, const void *src, uint64_t bytes,
                                  SparkHwCopyKind kind, SparkHwQueue q)
{
    if (dst == NULL || src == NULL || q == NULL || bytes == 0u)
    {
        return SPARK_HW_INVALID;
    }

    hipMemcpyKind hip_kind;
    switch (kind)
    {
    case SPARK_HW_COPY_H2D:
        hip_kind = hipMemcpyHostToDevice;
        break;
    case SPARK_HW_COPY_D2D:
        hip_kind = hipMemcpyDeviceToDevice;
        break;
    case SPARK_HW_COPY_D2H:
        hip_kind = hipMemcpyDeviceToHost;
        break;
    default:
        return SPARK_HW_INVALID;
    }

    return spark_hw_rocm_map_status(
        hipMemcpyAsync(dst, src, (size_t)bytes, hip_kind, (hipStream_t)q));
}

SparkHwStatus spark_hw_memset_async(void *dst, uint32_t value, uint64_t bytes,
                                    SparkHwQueue q)
{
    if (dst == NULL || q == NULL || bytes == 0u)
    {
        return SPARK_HW_INVALID;
    }
    /* hipMemsetAsync takes the value as int; byte-wise fill semantics match
     * cudaMemsetAsync for uint8 truncation of the core's value. */
    return spark_hw_rocm_map_status(
        hipMemsetAsync(dst, (int)(value & 0xFFu), (size_t)bytes, (hipStream_t)q));
}

SparkHwStatus spark_hw_read_ahead(SparkHwQueue q, void *sink_u32,
                                  uint32_t word_capacity, void *context)
{
    (void)context; /* no context-to-region map exists until islands land; see
                    * file header comment for the v1 degradation policy. */
    if (q == NULL)
    {
        return SPARK_HW_INVALID; /* bad handles are refused (clause 3) */
    }

    SparkHwRocmReadAheadCtx *payload =
        (SparkHwRocmReadAheadCtx *)malloc(sizeof(*payload));
    if (payload == NULL)
    {
        /* Failed kick degrades to no-op and never fails the frame. */
        return SPARK_HW_OK;
    }
    payload->sink = (uint32_t *)sink_u32;
    payload->word_capacity =
        word_capacity < SPARK_HW_ROCM_READ_AHEAD_STATUS_WORDS ? word_capacity
                                                              : SPARK_HW_ROCM_READ_AHEAD_STATUS_WORDS;

    hipError_t err = hipLaunchHostFunc((hipStream_t)q,
                                       spark_hw_rocm_read_ahead_complete, payload);
    if (err != hipSuccess)
    {
        free(payload);
        return SPARK_HW_OK; /* degrade to no-op, never fail the frame */
    }
    return SPARK_HW_OK;
}
