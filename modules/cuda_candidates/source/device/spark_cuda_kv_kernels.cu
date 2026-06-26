#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_cuda_kv_kernels.h"

#define SPARK_CUDA_KV_THREADS 256u
#define SPARK_CUDA_KV_NVFP4_THREADS 32u
#define SPARK_CUDA_KV_VECTOR16_ELEMENTS 8u

static __device__ float SparkCudaKvBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaKvFloatToBf16(float value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static __device__ float SparkCudaKvFp8E4m3ToFloat(uint8_t value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    float result;

    sign = (uint32_t)(value >> 7u);
    exponent = (uint32_t)((value >> 3u) & 15u);
    mantissa = (uint32_t)(value & 7u);
    if ((value & 0x7fu) == 0u)
    {
        return 0.0f;
    }
    if (exponent == 0u)
    {
        result = ldexpf((float)mantissa / 8.0f, -6);
    }
    else
    {
        result = ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    }
    return sign != 0u ? -result : result;
}

static __device__ uint8_t SparkCudaKvBf16ToFp8E4m3Byte(uint16_t value, float inverse_scale)
{
    return (uint8_t)__nv_cvt_float_to_fp8(SparkCudaKvBf16ToFloat(value) * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

static __device__ uint16_t SparkCudaKvFp8E4m3ByteToBf16(uint8_t value, float scale)
{
    return SparkCudaKvFloatToBf16(SparkCudaKvFp8E4m3ToFloat(value) * scale);
}

static __device__ float SparkCudaKvWarpMax(float value)
{
    uint32_t offset;
    float other;

    for (offset = 16u; offset > 0u; offset >>= 1u)
    {
        other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}

static __device__ float SparkCudaKvNvfp4DecodeE2m1(uint8_t nibble)
{
    float value;

    switch (nibble & 7u)
    {
        case 1u:
        {
            value = 0.5f;
            break;
        }
        case 2u:
        {
            value = 1.0f;
            break;
        }
        case 3u:
        {
            value = 1.5f;
            break;
        }
        case 4u:
        {
            value = 2.0f;
            break;
        }
        case 5u:
        {
            value = 3.0f;
            break;
        }
        case 6u:
        {
            value = 4.0f;
            break;
        }
        case 7u:
        {
            value = 6.0f;
            break;
        }
        default:
        {
            value = 0.0f;
            break;
        }
    }
    return (nibble & 8u) != 0u ? -value : value;
}

static __device__ uint8_t SparkCudaKvNvfp4EncodeE2m1(float value)
{
    float abs_value;
    uint8_t sign;
    uint8_t encoded;

    if (!isfinite(value))
    {
        return 0u;
    }
    sign = value < 0.0f ? 8u : 0u;
    abs_value = fabsf(value);
    if (abs_value < 0.25f)
        encoded = 0u;
    else if (abs_value < 0.75f)
        encoded = 1u;
    else if (abs_value < 1.25f)
        encoded = 2u;
    else if (abs_value < 1.75f)
        encoded = 3u;
    else if (abs_value < 2.5f)
        encoded = 4u;
    else if (abs_value < 3.5f)
        encoded = 5u;
    else if (abs_value < 5.0f)
        encoded = 6u;
    else
        encoded = 7u;
    return (uint8_t)(sign | encoded);
}

static __device__ uint2 SparkCudaKvBf16x8ToFp8x8(uint4 values, float inverse_scale)
{
    uint2 packed;
    uint8_t b0;
    uint8_t b1;
    uint8_t b2;
    uint8_t b3;
    uint8_t b4;
    uint8_t b5;
    uint8_t b6;
    uint8_t b7;

    b0 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.x & 0xffffu), inverse_scale);
    b1 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.x >> 16u), inverse_scale);
    b2 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.y & 0xffffu), inverse_scale);
    b3 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.y >> 16u), inverse_scale);
    b4 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.z & 0xffffu), inverse_scale);
    b5 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.z >> 16u), inverse_scale);
    b6 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.w & 0xffffu), inverse_scale);
    b7 = SparkCudaKvBf16ToFp8E4m3Byte((uint16_t)(values.w >> 16u), inverse_scale);
    packed.x = ((uint32_t)b0) | (((uint32_t)b1) << 8u) | (((uint32_t)b2) << 16u) | (((uint32_t)b3) << 24u);
    packed.y = ((uint32_t)b4) | (((uint32_t)b5) << 8u) | (((uint32_t)b6) << 16u) | (((uint32_t)b7) << 24u);
    return packed;
}

static __device__ uint4 SparkCudaKvFp8x8ToBf16x8(uint2 values, float scale)
{
    uint4 packed;
    uint16_t b0;
    uint16_t b1;
    uint16_t b2;
    uint16_t b3;
    uint16_t b4;
    uint16_t b5;
    uint16_t b6;
    uint16_t b7;

    b0 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)(values.x & 0xffu), scale);
    b1 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)((values.x >> 8u) & 0xffu), scale);
    b2 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)((values.x >> 16u) & 0xffu), scale);
    b3 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)(values.x >> 24u), scale);
    b4 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)(values.y & 0xffu), scale);
    b5 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)((values.y >> 8u) & 0xffu), scale);
    b6 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)((values.y >> 16u) & 0xffu), scale);
    b7 = SparkCudaKvFp8E4m3ByteToBf16((uint8_t)(values.y >> 24u), scale);
    packed.x = ((uint32_t)b0) | (((uint32_t)b1) << 16u);
    packed.y = ((uint32_t)b2) | (((uint32_t)b3) << 16u);
    packed.z = ((uint32_t)b4) | (((uint32_t)b5) << 16u);
    packed.w = ((uint32_t)b6) | (((uint32_t)b7) << 16u);
    return packed;
}

static uint64_t SparkCudaKvTokenElementCountHost(const SparkCudaKvRequest *request)
{
    return (uint64_t)request->token_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvCacheElementCountHost(const SparkCudaKvRequest *request)
{
    return (uint64_t)request->block_count * (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvBlockElementCountHost(const SparkCudaKvRequest *request)
{
    return (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvNvfp4SlotCountHost(const SparkCudaKvNvfp4Request *request)
{
    return (uint64_t)request->block_count * (uint64_t)request->block_size;
}

static uint64_t SparkCudaKvNvfp4ValuesPerSlotHost(const SparkCudaKvNvfp4Request *request)
{
    return (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaKvNvfp4ScaleRowBytesHost(const SparkCudaKvNvfp4Request *request)
{
    return (uint64_t)request->head_count * ((uint64_t)request->head_size / (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK);
}

static uint64_t SparkCudaKvNvfp4ScaleBlockCountHost(const SparkCudaKvNvfp4Request *request)
{
    return SparkCudaKvNvfp4SlotCountHost(request) * SparkCudaKvNvfp4ScaleRowBytesHost(request);
}

static uint32_t SparkCudaKvBlockCountHost(uint64_t element_count)
{
    uint32_t block_count;

    block_count = (uint32_t)((element_count + (uint64_t)SPARK_CUDA_KV_THREADS - 1u) / (uint64_t)SPARK_CUDA_KV_THREADS);
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return block_count;
}

static int32_t SparkCudaKvIsVector16AlignedHost(const void *pointer)
{
    return ((((uintptr_t)pointer) & 15u) == 0u);
}

static int32_t SparkCudaKvCanVector16TokenRowsHost(const SparkCudaKvRequest *request, const void *key_input, const void *value_input, const void *key_cache, const void *value_cache)
{
    uint64_t inner_element_count;

    inner_element_count = (uint64_t)request->head_count * (uint64_t)request->head_size;
    if ((inner_element_count % SPARK_CUDA_KV_VECTOR16_ELEMENTS) != 0u)
    {
        return 0;
    }
    if (SparkCudaKvIsVector16AlignedHost(key_input) == 0 || SparkCudaKvIsVector16AlignedHost(value_input) == 0 || SparkCudaKvIsVector16AlignedHost(key_cache) == 0 || SparkCudaKvIsVector16AlignedHost(value_cache) == 0)
    {
        return 0;
    }
    return 1;
}

static int32_t SparkCudaKvCanGridBlocksHost(const SparkCudaKvRequest *request)
{
    if (request->copy_count > 65535u)
    {
        return 0;
    }
    return 1;
}

static int32_t SparkCudaKvCanVector16CacheHost(const SparkCudaKvRequest *request, const void *a, const void *b)
{
    if ((SparkCudaKvCacheElementCountHost(request) % SPARK_CUDA_KV_VECTOR16_ELEMENTS) != 0u)
    {
        return 0;
    }
    if (SparkCudaKvIsVector16AlignedHost(a) == 0 || SparkCudaKvIsVector16AlignedHost(b) == 0)
    {
        return 0;
    }
    return 1;
}

static int32_t SparkCudaKvCanVector16BlockCopyHost(const SparkCudaKvRequest *request, const void *source_key, const void *source_value, const void *target_key, const void *target_value)
{
    if ((SparkCudaKvBlockElementCountHost(request) % SPARK_CUDA_KV_VECTOR16_ELEMENTS) != 0u)
    {
        return 0;
    }
    if (SparkCudaKvIsVector16AlignedHost(source_key) == 0 || SparkCudaKvIsVector16AlignedHost(source_value) == 0 || SparkCudaKvIsVector16AlignedHost(target_key) == 0 || SparkCudaKvIsVector16AlignedHost(target_value) == 0)
    {
        return 0;
    }
    return 1;
}

static void SparkCudaKvFillReportShape(const SparkCudaKvRequest *request, SparkCudaKvReport *report)
{
    report->token_element_count = SparkCudaKvTokenElementCountHost(request);
    report->cache_element_count = SparkCudaKvCacheElementCountHost(request);
    report->block_element_count = SparkCudaKvBlockElementCountHost(request);
}

static void SparkCudaKvFillNvfp4ReportShapeHost(const SparkCudaKvNvfp4Request *request, SparkCudaKvNvfp4Report *report)
{
    report->element_count = SparkCudaKvNvfp4SlotCountHost(request) * SparkCudaKvNvfp4ValuesPerSlotHost(request);
    report->payload_byte_count = SparkCudaKvNvfp4PayloadBytes(request);
    report->scale_byte_count = SparkCudaKvNvfp4ScaleBytes(request);
    report->payload_stride_bytes = request->payload_stride_bytes;
    report->scale_stride_bytes = request->scale_stride_bytes;
    report->slot_count = (uint32_t)SparkCudaKvNvfp4SlotCountHost(request);
    report->values_per_slot = (uint32_t)SparkCudaKvNvfp4ValuesPerSlotHost(request);
    report->scale_block_count = (uint32_t)SparkCudaKvNvfp4ScaleBlockCountHost(request);
    report->explicit_scale_stride_count = 1u;
}

static uint64_t SparkCudaKvMixHost(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static uint64_t SparkCudaKvRuntimeChecksumHost(const SparkCudaKvDmaRuntime *runtime)
{
    uint64_t checksum;

    if (runtime == 0)
    {
        return 0u;
    }

    checksum = 0x4B564355444D4152ull;
    checksum = SparkCudaKvMixHost(checksum, runtime->initialized ? 1u : 0u);
    checksum = SparkCudaKvMixHost(checksum, runtime->cuda_available ? 1u : 0u);
    checksum = SparkCudaKvMixHost(checksum, runtime->policy.stream_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->policy.event_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->submitted_transfer_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->completed_transfer_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->bytes_transferred_h2d);
    checksum = SparkCudaKvMixHost(checksum, runtime->bytes_transferred_d2h);
    checksum = SparkCudaKvMixHost(checksum, runtime->bytes_transferred_d2d);
    checksum = SparkCudaKvMixHost(checksum, runtime->stream_wait_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->event_record_count);
    checksum = SparkCudaKvMixHost(checksum, runtime->event_sync_count);

    return checksum;
}

static void SparkCudaKvReleasePossiblyPinnedHostPointer(void *host_pointer, bool pinned_by_cuda)
{
    if (host_pointer == 0)
    {
        return;
    }

    if (pinned_by_cuda)
    {
        (void)cudaFreeHost(host_pointer);
    }
    else
    {
        free(host_pointer);
    }
}

extern "C" SparkStatus SparkEnsureCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer, uint64_t required_bytes)
{
    void *new_host_pointer;
    uint64_t bytes_to_preserve;
    cudaError_t cuda_status;

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

    new_host_pointer = 0;
    cuda_status = cudaHostAlloc(&new_host_pointer, (size_t)required_bytes, cudaHostAllocPortable);
    if (cuda_status != cudaSuccess || new_host_pointer == 0)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    memset(new_host_pointer, 0, (size_t)required_bytes);
    if (buffer->allocated && buffer->host_pointer != 0 && buffer->capacity_bytes != 0u)
    {
        bytes_to_preserve = buffer->capacity_bytes < required_bytes ? buffer->capacity_bytes : required_bytes;
        memcpy(new_host_pointer, buffer->host_pointer, (size_t)bytes_to_preserve);
    }

    SparkCudaKvReleasePossiblyPinnedHostPointer(buffer->host_pointer, buffer->pinned_by_cuda);
    buffer->allocated = true;
    buffer->pinned_by_cuda = true;
    buffer->capacity_bytes = required_bytes;
    buffer->host_pointer = new_host_pointer;
    buffer->allocation_checksum = SparkComputeCudaKvPinnedHostBufferChecksum(buffer);

    return SPARK_STATUS_OK;
}

extern "C" void SparkReleaseCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer)
{
    if (buffer == 0)
    {
        return;
    }

    SparkCudaKvReleasePossiblyPinnedHostPointer(buffer->host_pointer, buffer->pinned_by_cuda);
    memset(buffer, 0, sizeof(*buffer));
}

extern "C" SparkStatus SparkEnsureCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer, uint64_t required_bytes)
{
    void *new_device_pointer;
    uint64_t bytes_to_preserve;
    cudaError_t cuda_status;

    if (buffer == 0 || required_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->allocated && buffer->device_pointer != 0 && buffer->capacity_bytes >= required_bytes)
    {
        return SPARK_STATUS_OK;
    }
    if (required_bytes > (uint64_t)((size_t)-1))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    new_device_pointer = 0;
    cuda_status = cudaMalloc(&new_device_pointer, (size_t)required_bytes);
    if (cuda_status != cudaSuccess || new_device_pointer == 0)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }
    cuda_status = cudaMemset(new_device_pointer, 0, (size_t)required_bytes);
    if (cuda_status != cudaSuccess)
    {
        cudaFree(new_device_pointer);
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    if (buffer->allocated && buffer->device_pointer != 0 && buffer->capacity_bytes != 0u)
    {
        bytes_to_preserve = buffer->capacity_bytes < required_bytes ? buffer->capacity_bytes : required_bytes;
        cuda_status = cudaMemcpy(new_device_pointer, buffer->device_pointer, (size_t)bytes_to_preserve, cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess)
        {
            cudaFree(new_device_pointer);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (buffer->device_pointer != 0 && buffer->device_allocated_by_cuda)
    {
        cudaFree(buffer->device_pointer);
    }
    buffer->allocated = true;
    buffer->device_allocated_by_cuda = true;
    buffer->capacity_bytes = required_bytes;
    buffer->device_pointer = new_device_pointer;
    buffer->allocation_checksum = SparkComputeCudaKvDeviceBufferChecksum(buffer);

    return SPARK_STATUS_OK;
}

extern "C" void SparkReleaseCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer)
{
    if (buffer == 0)
    {
        return;
    }
    if (buffer->device_pointer != 0 && buffer->device_allocated_by_cuda)
    {
        cudaFree(buffer->device_pointer);
    }
    memset(buffer, 0, sizeof(*buffer));
}

extern "C" SparkStatus SparkInitializeCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaPolicy *policy)
{
    SparkCudaKvDmaPolicy default_policy;
    const SparkCudaKvDmaPolicy *effective_policy;
    SparkStatus status;
    cudaError_t cuda_status;
    int device_count;
    uint32_t stream_index;
    uint32_t event_index;
    cudaStream_t created_stream;
    cudaEvent_t created_event;

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

    if (!effective_policy->allow_cuda_device_buffers)
    {
        runtime->runtime_checksum = SparkCudaKvRuntimeChecksumHost(runtime);
        return effective_policy->require_cuda_device_buffers ? SPARK_STATUS_GRAPH_NOT_AVAILABLE : SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    device_count = 0;
    cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count <= 0)
    {
        runtime->runtime_checksum = SparkCudaKvRuntimeChecksumHost(runtime);
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }

    for (stream_index = 0u; stream_index < effective_policy->stream_count; ++stream_index)
    {
        created_stream = 0;
        cuda_status = cudaStreamCreateWithFlags(&created_stream, cudaStreamNonBlocking);
        if (cuda_status != cudaSuccess)
        {
            SparkDestroyCudaKvDmaRuntime(runtime);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        runtime->streams[stream_index] = (void *)created_stream;
    }

    for (event_index = 0u; event_index < effective_policy->event_count; ++event_index)
    {
        created_event = 0;
        cuda_status = cudaEventCreateWithFlags(&created_event, cudaEventDisableTiming);
        if (cuda_status != cudaSuccess)
        {
            SparkDestroyCudaKvDmaRuntime(runtime);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        runtime->events[event_index] = (void *)created_event;
    }

    runtime->initialized = true;
    runtime->cuda_available = true;
    runtime->runtime_checksum = SparkCudaKvRuntimeChecksumHost(runtime);
    return SPARK_STATUS_OK;
}

extern "C" void SparkDestroyCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime)
{
    uint32_t stream_index;
    uint32_t event_index;

    if (runtime == 0)
    {
        return;
    }

    for (event_index = 0u; event_index < SPARKPIPE_CUDA_KV_DMA_EVENT_COUNT; ++event_index)
    {
        if (runtime->events[event_index] != 0)
        {
            cudaEventDestroy((cudaEvent_t)runtime->events[event_index]);
        }
    }
    for (stream_index = 0u; stream_index < SPARKPIPE_CUDA_KV_DMA_STREAM_COUNT; ++stream_index)
    {
        if (runtime->streams[stream_index] != 0)
        {
            cudaStreamDestroy((cudaStream_t)runtime->streams[stream_index]);
        }
    }

    memset(runtime, 0, sizeof(*runtime));
}

static cudaMemcpyKind SparkCudaKvMemcpyKindHost(SparkCudaKvDmaTransferKind transfer_kind)
{
    switch (transfer_kind)
    {
        case SPARK_CUDA_KV_DMA_TRANSFER_HOST_TO_DEVICE:
        {
            return cudaMemcpyHostToDevice;
        }
        case SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_HOST:
        {
            return cudaMemcpyDeviceToHost;
        }
        case SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_DEVICE:
        {
            return cudaMemcpyDeviceToDevice;
        }
        default:
        {
            return cudaMemcpyDefault;
        }
    }
}

static uint32_t SparkCudaKvStreamIndexForTransferHost(const SparkCudaKvDmaRuntime *runtime, SparkCudaKvDmaTransferKind transfer_kind)
{
    if (runtime == 0 || runtime->policy.stream_count == 0u)
    {
        return 0u;
    }

    switch (transfer_kind)
    {
        case SPARK_CUDA_KV_DMA_TRANSFER_HOST_TO_DEVICE:
        {
            return 0u;
        }
        case SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_HOST:
        {
            return runtime->policy.stream_count > 1u ? 1u : 0u;
        }
        case SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_DEVICE:
        {
            return runtime->policy.stream_count > 2u ? 2u : 0u;
        }
        default:
        {
            return 0u;
        }
    }
}

extern "C" SparkStatus SparkRunCudaKvDmaCopy(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaCopyRequest *request, SparkCudaKvDmaCopyReport *report)
{
    const uint8_t *source_bytes;
    uint8_t *target_bytes;
    cudaMemcpyKind cuda_copy_kind;
    cudaStream_t cuda_stream;
    cudaEvent_t start_event;
    cudaEvent_t complete_event;
    cudaError_t cuda_status;
    uint32_t stream_index;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    if (runtime == 0 || request == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!runtime->initialized || !runtime->cuda_available)
    {
        return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
    }
    if (request->sentinel != SPARKPIPE_CUDA_KV_DMA_SENTINEL || request->source_pointer == 0 || request->target_pointer == 0 || request->byte_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->transfer_kind != SPARK_CUDA_KV_DMA_TRANSFER_HOST_TO_DEVICE && request->transfer_kind != SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_HOST && request->transfer_kind != SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_DEVICE)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    stream_index = SparkCudaKvStreamIndexForTransferHost(runtime, request->transfer_kind);
    cuda_stream = (cudaStream_t)runtime->streams[stream_index];
    start_event = (cudaEvent_t)runtime->events[0u];
    complete_event = (cudaEvent_t)runtime->events[runtime->policy.event_count > 1u ? 1u : 0u];
    cuda_copy_kind = SparkCudaKvMemcpyKindHost(request->transfer_kind);
    source_bytes = &((const uint8_t *)request->source_pointer)[request->source_offset_bytes];
    target_bytes = &((uint8_t *)request->target_pointer)[request->target_offset_bytes];

    cuda_status = cudaEventRecord(start_event, cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->event_record_count += 1u;

    cuda_status = cudaMemcpyAsync(target_bytes, source_bytes, (size_t)request->byte_count, cuda_copy_kind, cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    cuda_status = cudaEventRecord(complete_event, cuda_stream);
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    runtime->event_record_count += 1u;

    report->submitted = true;
    report->used_cuda = true;
    report->used_async_stream = runtime->policy.allow_async_streams;
    report->transfer_kind = request->transfer_kind;
    report->byte_count = request->byte_count;
    report->event_record_count = 2u;

    if (request->wait_for_completion)
    {
        cuda_status = cudaEventSynchronize(complete_event);
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        report->completed = true;
        report->event_sync_count = 1u;
        runtime->event_sync_count += 1u;
    }

    runtime->submitted_transfer_count += 1u;
    if (report->completed)
    {
        runtime->completed_transfer_count += 1u;
    }

    if (request->transfer_kind == SPARK_CUDA_KV_DMA_TRANSFER_HOST_TO_DEVICE)
    {
        runtime->bytes_transferred_h2d += request->byte_count;
    }
    else if (request->transfer_kind == SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_HOST)
    {
        runtime->bytes_transferred_d2h += request->byte_count;
    }
    else
    {
        runtime->bytes_transferred_d2d += request->byte_count;
    }

    report->transfer_checksum = SparkComputeCudaKvDmaCopyReportChecksum(report);
    runtime->runtime_checksum = SparkCudaKvRuntimeChecksumHost(runtime);
    return SPARK_STATUS_OK;
}

static __global__ void SparkCudaKvPagedWriteBf16Kernel(SparkCudaKvRequest request, const uint16_t *key_input, const uint16_t *value_input, const uint32_t *slot_mapping, uint16_t *key_cache, uint16_t *value_cache)
{
    uint64_t element_index;
    uint64_t token_element_count;
    uint64_t inner_element_count;
    uint64_t token_index;
    uint64_t inner_index;
    uint32_t slot_index;
    uint32_t max_slot_count;

    token_element_count = (uint64_t)request.token_count * (uint64_t)request.head_count * (uint64_t)request.head_size;
    inner_element_count = (uint64_t)request.head_count * (uint64_t)request.head_size;
    max_slot_count = request.block_count * request.block_size;
    element_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (element_index < token_element_count)
    {
        token_index = element_index / inner_element_count;
        inner_index = element_index - (token_index * inner_element_count);
        slot_index = slot_mapping[token_index];
        if (slot_index < max_slot_count)
        {
            key_cache[((uint64_t)slot_index * inner_element_count) + inner_index] = key_input[element_index];
            value_cache[((uint64_t)slot_index * inner_element_count) + inner_index] = value_input[element_index];
        }
        element_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvPagedWriteBf16Vector16Kernel(SparkCudaKvRequest request, const uint4 *key_input, const uint4 *value_input, const uint32_t *slot_mapping, uint4 *key_cache, uint4 *value_cache)
{
    uint64_t vector_index;
    uint64_t token_vector_count;
    uint64_t inner_vector_count;
    uint64_t token_index;
    uint64_t inner_vector_index;
    uint32_t slot_index;
    uint32_t max_slot_count;

    inner_vector_count = ((uint64_t)request.head_count * (uint64_t)request.head_size) / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
    token_vector_count = (uint64_t)request.token_count * inner_vector_count;
    max_slot_count = request.block_count * request.block_size;
    vector_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (vector_index < token_vector_count)
    {
        token_index = vector_index / inner_vector_count;
        inner_vector_index = vector_index - (token_index * inner_vector_count);
        slot_index = slot_mapping[token_index];
        if (slot_index < max_slot_count)
        {
            key_cache[((uint64_t)slot_index * inner_vector_count) + inner_vector_index] = key_input[vector_index];
            value_cache[((uint64_t)slot_index * inner_vector_count) + inner_vector_index] = value_input[vector_index];
        }
        vector_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvCopyBlocksBf16Kernel(SparkCudaKvRequest request, const uint16_t *source_key_cache, const uint16_t *source_value_cache, const uint32_t *source_block_ids, const uint32_t *target_block_ids, uint16_t *target_key_cache, uint16_t *target_value_cache)
{
    uint64_t element_index;
    uint64_t block_element_count;
    uint64_t total_element_count;
    uint64_t copy_index;
    uint64_t inner_index;
    uint32_t source_block_id;
    uint32_t target_block_id;

    block_element_count = (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size;
    total_element_count = (uint64_t)request.copy_count * block_element_count;
    element_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (element_index < total_element_count)
    {
        copy_index = element_index / block_element_count;
        inner_index = element_index - (copy_index * block_element_count);
        source_block_id = source_block_ids[copy_index];
        target_block_id = target_block_ids[copy_index];
        if (source_block_id < request.block_count && target_block_id < request.block_count)
        {
            target_key_cache[((uint64_t)target_block_id * block_element_count) + inner_index] = source_key_cache[((uint64_t)source_block_id * block_element_count) + inner_index];
            target_value_cache[((uint64_t)target_block_id * block_element_count) + inner_index] = source_value_cache[((uint64_t)source_block_id * block_element_count) + inner_index];
        }
        element_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvCopyBlocksBf16GridKernel(SparkCudaKvRequest request, const uint16_t *source_key_cache, const uint16_t *source_value_cache, const uint32_t *source_block_ids, const uint32_t *target_block_ids, uint16_t *target_key_cache, uint16_t *target_value_cache)
{
    uint64_t block_element_count;
    uint64_t inner_index;
    uint32_t source_block_id;
    uint32_t target_block_id;
    uint32_t copy_index;

    block_element_count = (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size;
    inner_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    copy_index = (uint32_t)blockIdx.y;
    if (inner_index < block_element_count && copy_index < request.copy_count)
    {
        source_block_id = source_block_ids[copy_index];
        target_block_id = target_block_ids[copy_index];
        if (source_block_id < request.block_count && target_block_id < request.block_count)
        {
            target_key_cache[((uint64_t)target_block_id * block_element_count) + inner_index] = source_key_cache[((uint64_t)source_block_id * block_element_count) + inner_index];
            target_value_cache[((uint64_t)target_block_id * block_element_count) + inner_index] = source_value_cache[((uint64_t)source_block_id * block_element_count) + inner_index];
        }
    }
}

static __global__ void SparkCudaKvCopyBlocksBf16Vector16GridKernel(SparkCudaKvRequest request, const uint4 *source_key_cache, const uint4 *source_value_cache, const uint32_t *source_block_ids, const uint32_t *target_block_ids, uint4 *target_key_cache, uint4 *target_value_cache)
{
    uint64_t block_vector_count;
    uint64_t inner_vector_index;
    uint32_t source_block_id;
    uint32_t target_block_id;
    uint32_t copy_index;

    block_vector_count = ((uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size) / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
    inner_vector_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    copy_index = (uint32_t)blockIdx.y;
    if (inner_vector_index < block_vector_count && copy_index < request.copy_count)
    {
        source_block_id = source_block_ids[copy_index];
        target_block_id = target_block_ids[copy_index];
        if (source_block_id < request.block_count && target_block_id < request.block_count)
        {
            target_key_cache[((uint64_t)target_block_id * block_vector_count) + inner_vector_index] = source_key_cache[((uint64_t)source_block_id * block_vector_count) + inner_vector_index];
            target_value_cache[((uint64_t)target_block_id * block_vector_count) + inner_vector_index] = source_value_cache[((uint64_t)source_block_id * block_vector_count) + inner_vector_index];
        }
    }
}

static __global__ void SparkCudaKvConvertBf16ToFp8E4m3Kernel(SparkCudaKvRequest request, const uint16_t *input_values, uint8_t *output_values)
{
    uint64_t element_index;
    uint64_t element_count;
    float value;

    element_count = (uint64_t)request.block_count * (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size;
    element_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (element_index < element_count)
    {
        value = SparkCudaKvBf16ToFloat(input_values[element_index]) / request.fp8_scale;
        output_values[element_index] = (uint8_t)__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        element_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvConvertBf16ToFp8E4m3Vector16Kernel(SparkCudaKvRequest request, const uint4 *input_values, uint2 *output_values)
{
    uint64_t vector_index;
    uint64_t vector_count;
    float inverse_scale;

    vector_count = ((uint64_t)request.block_count * (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size) / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
    inverse_scale = 1.0f / request.fp8_scale;
    vector_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (vector_index < vector_count)
    {
        output_values[vector_index] = SparkCudaKvBf16x8ToFp8x8(input_values[vector_index], inverse_scale);
        vector_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvConvertFp8E4m3ToBf16Kernel(SparkCudaKvRequest request, const uint8_t *input_values, uint16_t *output_values)
{
    uint64_t element_index;
    uint64_t element_count;
    float value;

    element_count = (uint64_t)request.block_count * (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size;
    element_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (element_index < element_count)
    {
        value = SparkCudaKvFp8E4m3ToFloat(input_values[element_index]) * request.fp8_scale;
        output_values[element_index] = SparkCudaKvFloatToBf16(value);
        element_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvConvertFp8E4m3ToBf16Vector16Kernel(SparkCudaKvRequest request, const uint2 *input_values, uint4 *output_values)
{
    uint64_t vector_index;
    uint64_t vector_count;

    vector_count = ((uint64_t)request.block_count * (uint64_t)request.block_size * (uint64_t)request.head_count * (uint64_t)request.head_size) / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
    vector_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (vector_index < vector_count)
    {
        output_values[vector_index] = SparkCudaKvFp8x8ToBf16x8(input_values[vector_index], request.fp8_scale);
        vector_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvConvertBf16ToNvfp4E2m1Kernel(SparkCudaKvNvfp4Request request, const uint16_t *input_values, uint8_t *output_payload, uint8_t *output_scales)
{
    __shared__ float decoded_scale;
    __shared__ uint8_t scale_byte;
    uint64_t scale_blocks_per_head;
    uint64_t scale_blocks_per_slot;
    uint64_t total_scale_blocks;
    uint64_t scale_block_index;

    scale_blocks_per_head = (uint64_t)request.head_size / (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK;
    scale_blocks_per_slot = (uint64_t)request.head_count * scale_blocks_per_head;
    total_scale_blocks = (uint64_t)request.block_count * (uint64_t)request.block_size * scale_blocks_per_slot;
    scale_block_index = (uint64_t)blockIdx.x;
    while (scale_block_index < total_scale_blocks)
    {
        uint64_t slot_index;
        uint64_t scale_inner_index;
        uint64_t head_index;
        uint64_t head_block_index;
        uint64_t slot_inner_offset;
        uint64_t input_offset;
        uint64_t payload_offset;
        uint64_t scale_offset;
        float local_max;
        float max_value;

        slot_index = scale_block_index / scale_blocks_per_slot;
        scale_inner_index = scale_block_index - (slot_index * scale_blocks_per_slot);
        head_index = scale_inner_index / scale_blocks_per_head;
        head_block_index = scale_inner_index - (head_index * scale_blocks_per_head);
        slot_inner_offset = (head_index * (uint64_t)request.head_size) + (head_block_index * (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK);
        input_offset = (slot_index * (uint64_t)request.head_count * (uint64_t)request.head_size) + slot_inner_offset;
        payload_offset = (slot_index * request.payload_stride_bytes) + (slot_inner_offset >> 1u);
        scale_offset = (slot_index * request.scale_stride_bytes) + scale_inner_index;
        local_max = 0.0f;
        if (threadIdx.x < SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK)
        {
            local_max = fabsf(SparkCudaKvBf16ToFloat(input_values[input_offset + threadIdx.x]));
        }
        max_value = SparkCudaKvWarpMax(local_max);
        if (threadIdx.x == 0u)
        {
            float scale_value;

            scale_value = max_value == 0.0f ? 0.0f : ((max_value * request.global_scale) / SPARKPIPE_CUDA_KV_NVFP4_FP4_MAX);
            if (scale_value > SPARKPIPE_CUDA_KV_NVFP4_E4M3_MAX)
            {
                scale_value = SPARKPIPE_CUDA_KV_NVFP4_E4M3_MAX;
            }
            scale_byte = (uint8_t)__nv_cvt_float_to_fp8(scale_value, __NV_SATFINITE, __NV_E4M3);
            decoded_scale = SparkCudaKvFp8E4m3ToFloat(scale_byte);
            output_scales[scale_offset] = scale_byte;
        }
        __syncthreads();
        if (threadIdx.x < 8u)
        {
            uint32_t pair_offset;
            float low_value;
            float high_value;
            uint8_t low_nibble;
            uint8_t high_nibble;

            pair_offset = threadIdx.x << 1u;
            low_value = SparkCudaKvBf16ToFloat(input_values[input_offset + pair_offset]);
            high_value = SparkCudaKvBf16ToFloat(input_values[input_offset + pair_offset + 1u]);
            low_nibble = decoded_scale == 0.0f ? 0u : SparkCudaKvNvfp4EncodeE2m1((low_value * request.global_scale) / decoded_scale);
            high_nibble = decoded_scale == 0.0f ? 0u : SparkCudaKvNvfp4EncodeE2m1((high_value * request.global_scale) / decoded_scale);
            output_payload[payload_offset + threadIdx.x] = (uint8_t)(low_nibble | (high_nibble << 4u));
        }
        __syncthreads();
        scale_block_index += (uint64_t)gridDim.x;
    }
}

static __global__ void SparkCudaKvConvertNvfp4E2m1ToBf16Kernel(SparkCudaKvNvfp4Request request, const uint8_t *input_payload, const uint8_t *input_scales, uint16_t *output_values)
{
    uint64_t scale_blocks_per_head;
    uint64_t scale_blocks_per_slot;
    uint64_t total_scale_blocks;
    uint64_t scale_block_index;

    scale_blocks_per_head = (uint64_t)request.head_size / (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK;
    scale_blocks_per_slot = (uint64_t)request.head_count * scale_blocks_per_head;
    total_scale_blocks = (uint64_t)request.block_count * (uint64_t)request.block_size * scale_blocks_per_slot;
    scale_block_index = (uint64_t)blockIdx.x;
    while (scale_block_index < total_scale_blocks)
    {
        uint64_t slot_index;
        uint64_t scale_inner_index;
        uint64_t head_index;
        uint64_t head_block_index;
        uint64_t slot_inner_offset;
        uint64_t output_offset;
        uint64_t payload_offset;
        uint64_t scale_offset;
        float scale_value;

        slot_index = scale_block_index / scale_blocks_per_slot;
        scale_inner_index = scale_block_index - (slot_index * scale_blocks_per_slot);
        head_index = scale_inner_index / scale_blocks_per_head;
        head_block_index = scale_inner_index - (head_index * scale_blocks_per_head);
        slot_inner_offset = (head_index * (uint64_t)request.head_size) + (head_block_index * (uint64_t)SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK);
        output_offset = (slot_index * (uint64_t)request.head_count * (uint64_t)request.head_size) + slot_inner_offset;
        payload_offset = (slot_index * request.payload_stride_bytes) + (slot_inner_offset >> 1u);
        scale_offset = (slot_index * request.scale_stride_bytes) + scale_inner_index;
        scale_value = SparkCudaKvFp8E4m3ToFloat(input_scales[scale_offset]) / request.global_scale;
        if (threadIdx.x < SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK)
        {
            uint32_t payload_pair_index;
            uint8_t packed_value;
            uint8_t nibble;
            float decoded_value;

            payload_pair_index = threadIdx.x >> 1u;
            packed_value = input_payload[payload_offset + payload_pair_index];
            nibble = (threadIdx.x & 1u) == 0u ? (packed_value & 0x0fu) : (packed_value >> 4u);
            decoded_value = SparkCudaKvNvfp4DecodeE2m1(nibble) * scale_value;
            output_values[output_offset + threadIdx.x] = SparkCudaKvFloatToBf16(decoded_value);
        }
        scale_block_index += (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaKvPagedWriteBf16(const SparkCudaKvRequest *request, const void *device_key_input_bf16, const void *device_value_input_bf16, const uint32_t *device_slot_mapping, void *device_key_cache_bf16, void *device_value_cache_bf16, SparkCudaKvReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t element_count;

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
    SparkCudaKvFillReportShape(request, report);
    if (SparkCudaKvCanVector16TokenRowsHost(request, device_key_input_bf16, device_value_input_bf16, device_key_cache_bf16, device_value_cache_bf16) != 0)
    {
        element_count = report->token_element_count / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
        SparkCudaKvPagedWriteBf16Vector16Kernel<<<SparkCudaKvBlockCountHost(element_count), SPARK_CUDA_KV_THREADS>>>(*request, (const uint4 *)device_key_input_bf16, (const uint4 *)device_value_input_bf16, device_slot_mapping, (uint4 *)device_key_cache_bf16, (uint4 *)device_value_cache_bf16);
    }
    else
    {
        element_count = report->token_element_count;
        SparkCudaKvPagedWriteBf16Kernel<<<SparkCudaKvBlockCountHost(element_count), SPARK_CUDA_KV_THREADS>>>(*request, (const uint16_t *)device_key_input_bf16, (const uint16_t *)device_value_input_bf16, device_slot_mapping, (uint16_t *)device_key_cache_bf16, (uint16_t *)device_value_cache_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->paged_write_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaKvCopyBlocksBf16(const SparkCudaKvRequest *request, const void *device_source_key_cache_bf16, const void *device_source_value_cache_bf16, const uint32_t *device_source_block_ids, const uint32_t *device_target_block_ids, void *device_target_key_cache_bf16, void *device_target_value_cache_bf16, SparkCudaKvReport *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t element_count;
    dim3 copy_grid;

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
    SparkCudaKvFillReportShape(request, report);
    if (SparkCudaKvCanGridBlocksHost(request) != 0 && SparkCudaKvCanVector16BlockCopyHost(request, device_source_key_cache_bf16, device_source_value_cache_bf16, device_target_key_cache_bf16, device_target_value_cache_bf16) != 0)
    {
        element_count = report->block_element_count / SPARK_CUDA_KV_VECTOR16_ELEMENTS;
        copy_grid = dim3(SparkCudaKvBlockCountHost(element_count), request->copy_count, 1u);
        SparkCudaKvCopyBlocksBf16Vector16GridKernel<<<copy_grid, SPARK_CUDA_KV_THREADS>>>(*request, (const uint4 *)device_source_key_cache_bf16, (const uint4 *)device_source_value_cache_bf16, device_source_block_ids, device_target_block_ids, (uint4 *)device_target_key_cache_bf16, (uint4 *)device_target_value_cache_bf16);
    }
    else if (SparkCudaKvCanGridBlocksHost(request) != 0)
    {
        copy_grid = dim3(SparkCudaKvBlockCountHost(report->block_element_count), request->copy_count, 1u);
        SparkCudaKvCopyBlocksBf16GridKernel<<<copy_grid, SPARK_CUDA_KV_THREADS>>>(*request, (const uint16_t *)device_source_key_cache_bf16, (const uint16_t *)device_source_value_cache_bf16, device_source_block_ids, device_target_block_ids, (uint16_t *)device_target_key_cache_bf16, (uint16_t *)device_target_value_cache_bf16);
    }
    else
    {
        element_count = (uint64_t)request->copy_count * report->block_element_count;
        SparkCudaKvCopyBlocksBf16Kernel<<<SparkCudaKvBlockCountHost(element_count), SPARK_CUDA_KV_THREADS>>>(*request, (const uint16_t *)device_source_key_cache_bf16, (const uint16_t *)device_source_value_cache_bf16, device_source_block_ids, device_target_block_ids, (uint16_t *)device_target_key_cache_bf16, (uint16_t *)device_target_value_cache_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->block_copy_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaKvConvertBf16ToFp8E4m3(const SparkCudaKvRequest *request, const void *device_input_bf16, void *device_output_fp8, SparkCudaKvReport *report)
{
    cudaError_t cuda_status;
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
    SparkCudaKvFillReportShape(request, report);
    if (SparkCudaKvCanVector16CacheHost(request, device_input_bf16, device_output_fp8) != 0)
    {
        SparkCudaKvConvertBf16ToFp8E4m3Vector16Kernel<<<SparkCudaKvBlockCountHost(report->cache_element_count / SPARK_CUDA_KV_VECTOR16_ELEMENTS), SPARK_CUDA_KV_THREADS>>>(*request, (const uint4 *)device_input_bf16, (uint2 *)device_output_fp8);
    }
    else
    {
        SparkCudaKvConvertBf16ToFp8E4m3Kernel<<<SparkCudaKvBlockCountHost(report->cache_element_count), SPARK_CUDA_KV_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (uint8_t *)device_output_fp8);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->bf16_to_fp8_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaKvConvertFp8E4m3ToBf16(const SparkCudaKvRequest *request, const void *device_input_fp8, void *device_output_bf16, SparkCudaKvReport *report)
{
    cudaError_t cuda_status;
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
    SparkCudaKvFillReportShape(request, report);
    if (SparkCudaKvCanVector16CacheHost(request, device_input_fp8, device_output_bf16) != 0)
    {
        SparkCudaKvConvertFp8E4m3ToBf16Vector16Kernel<<<SparkCudaKvBlockCountHost(report->cache_element_count / SPARK_CUDA_KV_VECTOR16_ELEMENTS), SPARK_CUDA_KV_THREADS>>>(*request, (const uint2 *)device_input_fp8, (uint4 *)device_output_bf16);
    }
    else
    {
        SparkCudaKvConvertFp8E4m3ToBf16Kernel<<<SparkCudaKvBlockCountHost(report->cache_element_count), SPARK_CUDA_KV_THREADS>>>(*request, (const uint8_t *)device_input_fp8, (uint16_t *)device_output_bf16);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->fp8_to_bf16_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaKvConvertBf16ToNvfp4E2m1(const SparkCudaKvNvfp4Request *request, const void *device_input_bf16, void *device_output_payload_u8, void *device_output_scale_e4m3, SparkCudaKvNvfp4Report *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t scale_block_count;
    uint32_t grid_block_count;

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
    SparkCudaKvFillNvfp4ReportShapeHost(request, report);
    scale_block_count = SparkCudaKvNvfp4ScaleBlockCountHost(request);
    grid_block_count = SparkCudaKvBlockCountHost(scale_block_count);
    SparkCudaKvConvertBf16ToNvfp4E2m1Kernel<<<grid_block_count, SPARK_CUDA_KV_NVFP4_THREADS>>>(*request, (const uint16_t *)device_input_bf16, (uint8_t *)device_output_payload_u8, (uint8_t *)device_output_scale_e4m3);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->bf16_to_nvfp4_kernel_count = 1u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaKvConvertNvfp4E2m1ToBf16(const SparkCudaKvNvfp4Request *request, const void *device_input_payload_u8, const void *device_input_scale_e4m3, void *device_output_bf16, SparkCudaKvNvfp4Report *report)
{
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t scale_block_count;
    uint32_t grid_block_count;

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
    SparkCudaKvFillNvfp4ReportShapeHost(request, report);
    scale_block_count = SparkCudaKvNvfp4ScaleBlockCountHost(request);
    grid_block_count = SparkCudaKvBlockCountHost(scale_block_count);
    SparkCudaKvConvertNvfp4E2m1ToBf16Kernel<<<grid_block_count, SPARK_CUDA_KV_NVFP4_THREADS>>>(*request, (const uint8_t *)device_input_payload_u8, (const uint8_t *)device_input_scale_e4m3, (uint16_t *)device_output_bf16);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    report->nvfp4_to_bf16_kernel_count = 1u;
    return SPARK_STATUS_OK;
}
