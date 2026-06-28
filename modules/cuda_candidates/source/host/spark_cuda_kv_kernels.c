#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_common.h"
#include "sparkpipe/spark_cuda_kv_kernels.h"

static uint64_t SparkCudaKvTokenElementCount(const SparkCudaKvRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->token_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvCacheElementCount(const SparkCudaKvRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->block_count * (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvNvfp4SlotCount(const SparkCudaKvNvfp4Request *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->block_count * (uint64_t)request->block_size;
}

static uint64_t SparkCudaKvNvfp4ValuesPerSlot(const SparkCudaKvNvfp4Request *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvNvfp4PayloadRowBytes(const SparkCudaKvNvfp4Request *request)
{
    uint64_t values_per_slot;

    values_per_slot = SparkCudaKvNvfp4ValuesPerSlot(request);
    return (values_per_slot + 1u) >> 1u;
}

static uint64_t SparkCudaKvNvfp4ScaleRowBytes(const SparkCudaKvNvfp4Request *request)
{
    if (request == 0 || request->head_size == 0u)
    {
        return 0;
    }
    return (uint64_t)request->head_count * ((uint64_t)request->head_size / (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK);
}

static uint64_t SparkCudaKvNvfp4ScaleBlockCount(const SparkCudaKvNvfp4Request *request)
{
    return SparkCudaKvNvfp4SlotCount(request) * SparkCudaKvNvfp4ScaleRowBytes(request);
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static uint64_t SparkComputeCudaKvDmaRuntimeChecksum(const SparkCudaKvDmaRuntime *runtime)
{
    uint64_t checksum;

    if (runtime == 0)
    {
        return 0;
    }

    checksum = 0x4B56444D41525455ull;
    checksum = SparkMixU64(checksum, runtime->initialized ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->cuda_available ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->policy.sentinel);
    checksum = SparkMixU64(checksum, runtime->policy.stream_count);
    checksum = SparkMixU64(checksum, runtime->policy.event_count);
    checksum = SparkMixU64(checksum, runtime->policy.max_inflight_transfers);
    checksum = SparkMixU64(checksum, runtime->policy.pinned_staging_bytes);
    checksum = SparkMixU64(checksum, runtime->policy.prefer_pinned_host ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->policy.allow_async_streams ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->policy.allow_cuda_device_buffers ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->policy.require_cuda_device_buffers ? 1u : 0u);
    checksum = SparkMixU64(checksum, runtime->submitted_transfer_count);
    checksum = SparkMixU64(checksum, runtime->completed_transfer_count);
    checksum = SparkMixU64(checksum, runtime->bytes_transferred_h2d);
    checksum = SparkMixU64(checksum, runtime->bytes_transferred_d2h);
    checksum = SparkMixU64(checksum, runtime->bytes_transferred_d2d);
    checksum = SparkMixU64(checksum, runtime->stream_wait_count);
    checksum = SparkMixU64(checksum, runtime->event_record_count);
    checksum = SparkMixU64(checksum, runtime->event_sync_count);

    return checksum;
}
#endif

void SparkCudaKvDmaPolicyReset(SparkCudaKvDmaPolicy *policy)
{
    if (policy == 0)
    {
        return;
    }

    memset(policy, 0, sizeof(*policy));
    policy->sentinel = SPARKPIPE_CUDA_KV_DMA_SENTINEL;
    policy->stream_count = SPARKPIPE_CUDA_KV_DMA_STREAM_COUNT;
    policy->event_count = SPARKPIPE_CUDA_KV_DMA_EVENT_COUNT;
    policy->max_inflight_transfers = 3u;
    policy->pinned_staging_bytes = 64ull * 1024ull * 1024ull;
    policy->prefer_pinned_host = true;
    policy->allow_async_streams = true;
    policy->allow_cuda_device_buffers = true;
    policy->require_cuda_device_buffers = false;
}

SparkStatus SparkValidateCudaKvDmaPolicy(const SparkCudaKvDmaPolicy *policy)
{
    if (policy == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (policy->sentinel != SPARKPIPE_CUDA_KV_DMA_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (policy->stream_count == 0u || policy->stream_count > SPARKPIPE_CUDA_KV_DMA_STREAM_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (policy->event_count == 0u || policy->event_count > SPARKPIPE_CUDA_KV_DMA_EVENT_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (policy->max_inflight_transfers == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (policy->require_cuda_device_buffers && !policy->allow_cuda_device_buffers)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

uint64_t SparkComputeCudaKvPinnedHostBufferChecksum(const SparkCudaKvPinnedHostBuffer *buffer)
{
    uint64_t checksum;

    if (buffer == 0)
    {
        return 0;
    }

    checksum = 0x4B5650484F535442ull;
    checksum = SparkMixU64(checksum, buffer->allocated ? 1u : 0u);
    checksum = SparkMixU64(checksum, buffer->pinned_by_cuda ? 1u : 0u);
    checksum = SparkMixU64(checksum, buffer->capacity_bytes);
    checksum = SparkMixU64(checksum, buffer->host_pointer != 0 ? 1u : 0u);

    return checksum;
}

uint64_t SparkComputeCudaKvDeviceBufferChecksum(const SparkCudaKvDeviceBuffer *buffer)
{
    uint64_t checksum;

    if (buffer == 0)
    {
        return 0;
    }

    checksum = 0x4B56444556425546ull;
    checksum = SparkMixU64(checksum, buffer->allocated ? 1u : 0u);
    checksum = SparkMixU64(checksum, buffer->device_allocated_by_cuda ? 1u : 0u);
    checksum = SparkMixU64(checksum, buffer->capacity_bytes);
    checksum = SparkMixU64(checksum, buffer->device_pointer != 0 ? 1u : 0u);

    return checksum;
}

uint64_t SparkComputeCudaKvDmaCopyReportChecksum(const SparkCudaKvDmaCopyReport *report)
{
    uint64_t checksum;

    if (report == 0)
    {
        return 0;
    }

    checksum = 0x4B56444D41434F50ull;
    checksum = SparkMixU64(checksum, report->submitted ? 1u : 0u);
    checksum = SparkMixU64(checksum, report->completed ? 1u : 0u);
    checksum = SparkMixU64(checksum, report->used_cuda ? 1u : 0u);
    checksum = SparkMixU64(checksum, report->used_async_stream ? 1u : 0u);
    checksum = SparkMixU64(checksum, report->transfer_kind);
    checksum = SparkMixU64(checksum, report->byte_count);
    checksum = SparkMixU64(checksum, report->stream_wait_count);
    checksum = SparkMixU64(checksum, report->event_record_count);
    checksum = SparkMixU64(checksum, report->event_sync_count);

    return checksum;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkEnsureCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer, uint64_t required_bytes)
{
    void *new_host_pointer;

    if (buffer == 0 || required_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (buffer->allocated && buffer->host_pointer != 0 && buffer->capacity_bytes >= required_bytes)
    {
        return SPARK_STATUS_OK;
    }

    if (required_bytes > (uint64_t)((size_t)-1))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    if (buffer->host_pointer == 0)
    {
        new_host_pointer = calloc(1u, (size_t)required_bytes);
    }
    else
    {
        new_host_pointer = realloc(buffer->host_pointer, (size_t)required_bytes);
        if (new_host_pointer != 0 && required_bytes > buffer->capacity_bytes)
        {
            memset(&((uint8_t *)new_host_pointer)[buffer->capacity_bytes], 0, (size_t)(required_bytes - buffer->capacity_bytes));
        }
    }

    if (new_host_pointer == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    buffer->allocated = true;
    buffer->pinned_by_cuda = false;
    buffer->capacity_bytes = required_bytes;
    buffer->host_pointer = new_host_pointer;
    buffer->allocation_checksum = SparkComputeCudaKvPinnedHostBufferChecksum(buffer);

    return SPARK_STATUS_OK;
}

void SparkReleaseCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer)
{
    if (buffer == 0)
    {
        return;
    }

    free(buffer->host_pointer);
    memset(buffer, 0, sizeof(*buffer));
}

SparkStatus SparkEnsureCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer, uint64_t required_bytes)
{
    if (buffer == 0 || required_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(buffer, 0, sizeof(*buffer));
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

void SparkReleaseCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer)
{
    if (buffer == 0)
    {
        return;
    }

    memset(buffer, 0, sizeof(*buffer));
}

SparkStatus SparkInitializeCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaPolicy *policy)
{
    SparkStatus status;
    SparkCudaKvDmaPolicy default_policy;
    const SparkCudaKvDmaPolicy *effective_policy;

    if (runtime == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (policy == 0)
    {
        SparkCudaKvDmaPolicyReset(&default_policy);
        effective_policy = &default_policy;
    }
    else
    {
        effective_policy = policy;
    }

    status = SparkValidateCudaKvDmaPolicy(effective_policy);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(runtime, 0, sizeof(*runtime));
    runtime->policy = *effective_policy;
    runtime->runtime_checksum = SparkComputeCudaKvDmaRuntimeChecksum(runtime);

    if (effective_policy->require_cuda_device_buffers)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

void SparkDestroyCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime)
{
    if (runtime == 0)
    {
        return;
    }

    memset(runtime, 0, sizeof(*runtime));
}

SparkStatus SparkRunCudaKvDmaCopy(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaCopyRequest *request, SparkCudaKvDmaCopyReport *report)
{
    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    if (runtime == 0 || request == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_KV_DMA_SENTINEL || request->source_pointer == 0 || request->target_pointer == 0 || request->byte_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!runtime->initialized || !runtime->cuda_available)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif

SparkStatus SparkValidateCudaKvRequest(const SparkCudaKvRequest *request)
{
    uint64_t token_element_count;
    uint64_t cache_element_count;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->head_count == 0u || request->head_size == 0u || request->block_size == 0u || request->block_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_KV_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(request->fp8_scale) || request->fp8_scale <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    token_element_count = SparkCudaKvTokenElementCount(request);
    cache_element_count = SparkCudaKvCacheElementCount(request);
    if (token_element_count == 0u || cache_element_count == 0u || token_element_count > cache_element_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

uint64_t SparkCudaKvNvfp4PayloadBytes(const SparkCudaKvNvfp4Request *request)
{
    uint64_t slot_count;
    uint64_t row_bytes;

    slot_count = SparkCudaKvNvfp4SlotCount(request);
    row_bytes = SparkCudaKvNvfp4PayloadRowBytes(request);
    if (slot_count == 0u || row_bytes == 0u || request == 0)
    {
        return 0;
    }
    return ((slot_count - 1u) * request->payload_stride_bytes) + row_bytes;
}

uint64_t SparkCudaKvNvfp4ScaleBytes(const SparkCudaKvNvfp4Request *request)
{
    uint64_t slot_count;
    uint64_t row_bytes;

    slot_count = SparkCudaKvNvfp4SlotCount(request);
    row_bytes = SparkCudaKvNvfp4ScaleRowBytes(request);
    if (slot_count == 0u || row_bytes == 0u || request == 0)
    {
        return 0;
    }
    return ((slot_count - 1u) * request->scale_stride_bytes) + row_bytes;
}

SparkStatus SparkValidateCudaKvNvfp4Request(const SparkCudaKvNvfp4Request *request)
{
    uint64_t slot_count;
    uint64_t values_per_slot;
    uint64_t payload_row_bytes;
    uint64_t scale_row_bytes;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_KV_NVFP4_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->block_count == 0u || request->block_size == 0u || request->head_count == 0u || request->head_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->scale_block_size != SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK || (request->head_size % SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(request->global_scale) || request->global_scale <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    slot_count = SparkCudaKvNvfp4SlotCount(request);
    values_per_slot = SparkCudaKvNvfp4ValuesPerSlot(request);
    payload_row_bytes = SparkCudaKvNvfp4PayloadRowBytes(request);
    scale_row_bytes = SparkCudaKvNvfp4ScaleRowBytes(request);
    if (slot_count == 0u || values_per_slot == 0u || payload_row_bytes == 0u || scale_row_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->payload_stride_bytes < payload_row_bytes || request->scale_stride_bytes < scale_row_bytes)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->payload_stride_bytes & 15u) != 0u || (request->scale_stride_bytes & 15u) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaKvNvfp4PayloadBytes(request) == 0u || SparkCudaKvNvfp4ScaleBytes(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkFillCudaKvReportShape(const SparkCudaKvRequest *request, SparkCudaKvReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->token_element_count = SparkCudaKvTokenElementCount(request);
    report->cache_element_count = SparkCudaKvCacheElementCount(request);
    report->block_element_count = (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static void SparkFillCudaKvNvfp4ReportShape(const SparkCudaKvNvfp4Request *request, SparkCudaKvNvfp4Report *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->element_count = SparkCudaKvNvfp4SlotCount(request) * SparkCudaKvNvfp4ValuesPerSlot(request);
    report->payload_byte_count = SparkCudaKvNvfp4PayloadBytes(request);
    report->scale_byte_count = SparkCudaKvNvfp4ScaleBytes(request);
    report->payload_stride_bytes = request->payload_stride_bytes;
    report->scale_stride_bytes = request->scale_stride_bytes;
    report->slot_count = (uint32_t)SparkCudaKvNvfp4SlotCount(request);
    report->values_per_slot = (uint32_t)SparkCudaKvNvfp4ValuesPerSlot(request);
    report->scale_block_count = (uint32_t)SparkCudaKvNvfp4ScaleBlockCount(request);
    report->explicit_scale_stride_count = 1u;
}

SparkStatus SparkRunCudaKvPagedWriteBf16(const SparkCudaKvRequest *request, const void *device_key_input_bf16, const void *device_value_input_bf16, const uint32_t *device_slot_mapping, void *device_key_cache_bf16, void *device_value_cache_bf16, SparkCudaKvReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_key_input_bf16 == 0 || device_value_input_bf16 == 0 || device_slot_mapping == 0 || device_key_cache_bf16 == 0 || device_value_cache_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaKvCopyBlocksBf16(const SparkCudaKvRequest *request, const void *device_source_key_cache_bf16, const void *device_source_value_cache_bf16, const uint32_t *device_source_block_ids, const uint32_t *device_target_block_ids, void *device_target_key_cache_bf16, void *device_target_value_cache_bf16, SparkCudaKvReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_source_key_cache_bf16 == 0 || device_source_value_cache_bf16 == 0 || device_source_block_ids == 0 || device_target_block_ids == 0 || device_target_key_cache_bf16 == 0 || device_target_value_cache_bf16 == 0 || report == 0 || request->copy_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaKvConvertBf16ToFp8E4m3(const SparkCudaKvRequest *request, const void *device_input_bf16, void *device_output_fp8, SparkCudaKvReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_output_fp8 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaKvConvertFp8E4m3ToBf16(const SparkCudaKvRequest *request, const void *device_input_fp8, void *device_output_bf16, SparkCudaKvReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_fp8 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaKvConvertBf16ToNvfp4E2m1(const SparkCudaKvNvfp4Request *request, const void *device_input_bf16, void *device_output_payload_u8, void *device_output_scale_e4m3, SparkCudaKvNvfp4Report *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvNvfp4Request(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_output_payload_u8 == 0 || device_output_scale_e4m3 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvNvfp4ReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaKvConvertNvfp4E2m1ToBf16(const SparkCudaKvNvfp4Request *request, const void *device_input_payload_u8, const void *device_input_scale_e4m3, void *device_output_bf16, SparkCudaKvNvfp4Report *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaKvNvfp4Request(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_payload_u8 == 0 || device_input_scale_e4m3 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaKvNvfp4ReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
