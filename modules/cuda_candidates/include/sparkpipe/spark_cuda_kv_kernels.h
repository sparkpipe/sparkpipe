#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_KV_SENTINEL 0x535043554B564331ull
#define SPARKPIPE_CUDA_KV_DMA_SENTINEL 0x535043554B564432ull
#define SPARKPIPE_CUDA_KV_NVFP4_SENTINEL 0x535043554B564634ull
#define SPARKPIPE_CUDA_KV_DMA_STREAM_COUNT 3u
#define SPARKPIPE_CUDA_KV_DMA_EVENT_COUNT 4u
#define SPARKPIPE_CUDA_KV_NVFP4_SCALE_BLOCK 16u
#define SPARKPIPE_CUDA_KV_NVFP4_FP4_MAX 6.0f
#define SPARKPIPE_CUDA_KV_NVFP4_E4M3_MAX 448.0f

typedef enum SparkCudaKvDmaTransferKind
{
    SPARK_CUDA_KV_DMA_TRANSFER_HOST_TO_DEVICE = 1,
    SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_HOST = 2,
    SPARK_CUDA_KV_DMA_TRANSFER_DEVICE_TO_DEVICE = 3
} SparkCudaKvDmaTransferKind;

typedef struct SparkCudaKvRequest
{
    uint32_t token_count;
    uint32_t head_count;
    uint32_t head_size;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t copy_count;
    float fp8_scale;
    uint64_t sentinel;
} SparkCudaKvRequest;

typedef struct SparkCudaKvReport
{
    uint64_t token_element_count;
    uint64_t cache_element_count;
    uint64_t block_element_count;
    uint32_t paged_write_kernel_count;
    uint32_t block_copy_kernel_count;
    uint32_t bf16_to_fp8_kernel_count;
    uint32_t fp8_to_bf16_kernel_count;
    uint32_t invalid_slot_count;
    uint32_t invalid_block_count;
    uint32_t sentinel_violation_count;
} SparkCudaKvReport;

typedef struct SparkCudaKvNvfp4Request
{
    uint32_t block_count;
    uint32_t block_size;
    uint32_t head_count;
    uint32_t head_size;
    uint32_t scale_block_size;
    uint32_t reserved;
    uint64_t payload_stride_bytes;
    uint64_t scale_stride_bytes;
    float global_scale;
    uint64_t sentinel;
} SparkCudaKvNvfp4Request;

typedef struct SparkCudaKvNvfp4Report
{
    uint64_t element_count;
    uint64_t payload_byte_count;
    uint64_t scale_byte_count;
    uint64_t payload_stride_bytes;
    uint64_t scale_stride_bytes;
    uint32_t slot_count;
    uint32_t values_per_slot;
    uint32_t scale_block_count;
    uint32_t bf16_to_nvfp4_kernel_count;
    uint32_t nvfp4_to_bf16_kernel_count;
    uint32_t explicit_scale_stride_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
    uint32_t saturated_scale_count;
} SparkCudaKvNvfp4Report;

typedef struct SparkCudaKvDmaPolicy
{
    uint64_t sentinel;
    uint32_t stream_count;
    uint32_t event_count;
    uint32_t max_inflight_transfers;
    uint64_t pinned_staging_bytes;
    bool prefer_pinned_host;
    bool allow_async_streams;
    bool allow_cuda_device_buffers;
    bool require_cuda_device_buffers;
} SparkCudaKvDmaPolicy;

typedef struct SparkCudaKvPinnedHostBuffer
{
    bool allocated;
    bool pinned_by_cuda;
    uint64_t capacity_bytes;
    void *host_pointer;
    uint64_t allocation_checksum;
} SparkCudaKvPinnedHostBuffer;

typedef struct SparkCudaKvDeviceBuffer
{
    bool allocated;
    bool device_allocated_by_cuda;
    uint64_t capacity_bytes;
    void *device_pointer;
    uint64_t allocation_checksum;
} SparkCudaKvDeviceBuffer;

typedef struct SparkCudaKvDmaRuntime
{
    bool initialized;
    bool cuda_available;
    SparkCudaKvDmaPolicy policy;
    void *streams[SPARKPIPE_CUDA_KV_DMA_STREAM_COUNT];
    void *events[SPARKPIPE_CUDA_KV_DMA_EVENT_COUNT];
    uint64_t submitted_transfer_count;
    uint64_t completed_transfer_count;
    uint64_t bytes_transferred_h2d;
    uint64_t bytes_transferred_d2h;
    uint64_t bytes_transferred_d2d;
    uint64_t stream_wait_count;
    uint64_t event_record_count;
    uint64_t event_sync_count;
    uint64_t runtime_checksum;
} SparkCudaKvDmaRuntime;

typedef struct SparkCudaKvDmaCopyRequest
{
    uint64_t sentinel;
    SparkCudaKvDmaTransferKind transfer_kind;
    const void *source_pointer;
    void *target_pointer;
    uint64_t source_offset_bytes;
    uint64_t target_offset_bytes;
    uint64_t byte_count;
    bool wait_for_completion;
} SparkCudaKvDmaCopyRequest;

typedef struct SparkCudaKvDmaCopyReport
{
    bool submitted;
    bool completed;
    bool used_cuda;
    bool used_async_stream;
    SparkCudaKvDmaTransferKind transfer_kind;
    uint64_t byte_count;
    uint64_t stream_wait_count;
    uint64_t event_record_count;
    uint64_t event_sync_count;
    uint64_t transfer_checksum;
} SparkCudaKvDmaCopyReport;

void SparkCudaKvDmaPolicyReset(SparkCudaKvDmaPolicy *policy);
SparkStatus SparkValidateCudaKvDmaPolicy(const SparkCudaKvDmaPolicy *policy);
uint64_t SparkComputeCudaKvPinnedHostBufferChecksum(const SparkCudaKvPinnedHostBuffer *buffer);
uint64_t SparkComputeCudaKvDeviceBufferChecksum(const SparkCudaKvDeviceBuffer *buffer);
uint64_t SparkComputeCudaKvDmaCopyReportChecksum(const SparkCudaKvDmaCopyReport *report);
SparkStatus SparkEnsureCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer, uint64_t required_bytes);
void SparkReleaseCudaKvPinnedHostBuffer(SparkCudaKvPinnedHostBuffer *buffer);
SparkStatus SparkEnsureCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer, uint64_t required_bytes);
void SparkReleaseCudaKvDeviceBuffer(SparkCudaKvDeviceBuffer *buffer);
SparkStatus SparkInitializeCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaPolicy *policy);
void SparkDestroyCudaKvDmaRuntime(SparkCudaKvDmaRuntime *runtime);
SparkStatus SparkRunCudaKvDmaCopy(SparkCudaKvDmaRuntime *runtime, const SparkCudaKvDmaCopyRequest *request, SparkCudaKvDmaCopyReport *report);

SparkStatus SparkValidateCudaKvRequest(const SparkCudaKvRequest *request);
SparkStatus SparkValidateCudaKvNvfp4Request(const SparkCudaKvNvfp4Request *request);
uint64_t SparkCudaKvNvfp4PayloadBytes(const SparkCudaKvNvfp4Request *request);
uint64_t SparkCudaKvNvfp4ScaleBytes(const SparkCudaKvNvfp4Request *request);
SparkStatus SparkRunCudaKvPagedWriteBf16(const SparkCudaKvRequest *request, const void *device_key_input_bf16, const void *device_value_input_bf16, const uint32_t *device_slot_mapping, void *device_key_cache_bf16, void *device_value_cache_bf16, SparkCudaKvReport *report);
SparkStatus SparkRunCudaKvCopyBlocksBf16(const SparkCudaKvRequest *request, const void *device_source_key_cache_bf16, const void *device_source_value_cache_bf16, const uint32_t *device_source_block_ids, const uint32_t *device_target_block_ids, void *device_target_key_cache_bf16, void *device_target_value_cache_bf16, SparkCudaKvReport *report);
SparkStatus SparkRunCudaKvConvertBf16ToFp8E4m3(const SparkCudaKvRequest *request, const void *device_input_bf16, void *device_output_fp8, SparkCudaKvReport *report);
SparkStatus SparkRunCudaKvConvertFp8E4m3ToBf16(const SparkCudaKvRequest *request, const void *device_input_fp8, void *device_output_bf16, SparkCudaKvReport *report);
SparkStatus SparkRunCudaKvConvertBf16ToNvfp4E2m1(const SparkCudaKvNvfp4Request *request, const void *device_input_bf16, void *device_output_payload_u8, void *device_output_scale_e4m3, SparkCudaKvNvfp4Report *report);
SparkStatus SparkRunCudaKvConvertNvfp4E2m1ToBf16(const SparkCudaKvNvfp4Request *request, const void *device_input_payload_u8, const void *device_input_scale_e4m3, void *device_output_bf16, SparkCudaKvNvfp4Report *report);

#ifdef __cplusplus
}
#endif
