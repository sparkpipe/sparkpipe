#ifndef SPARKPIPE_SPARK_KV_MOVER_H
#define SPARKPIPE_SPARK_KV_MOVER_H

#include "sparkpipe/spark_cuda_kv_kernels.h"
#include "sparkpipe/spark_kv_page_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkKvPayloadRegionKind
{
    SPARK_KV_PAYLOAD_REGION_HOT = 1,
    SPARK_KV_PAYLOAD_REGION_BACKING = 2
} SparkKvPayloadRegionKind;

typedef struct SparkKvPagePayloadHandle
{
    bool valid;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint64_t page_table_fingerprint;
    uint64_t hot_byte_offset;
    uint64_t hot_byte_count;
    uint64_t backing_byte_offset;
    uint64_t backing_byte_count;
    uint64_t payload_checksum;
} SparkKvPagePayloadHandle;

typedef struct SparkKvPayloadStageBuffer
{
    bool initialized;
    bool dma_runtime_ready;
    uint32_t stage_id;
    uint64_t hot_payload_bytes;
    uint64_t backing_payload_bytes;
    uint64_t hot_payload_checksum;
    uint64_t backing_payload_checksum;
    SparkCudaKvPinnedHostBuffer hot_host_buffer;
    SparkCudaKvPinnedHostBuffer backing_host_buffer;
    SparkCudaKvDeviceBuffer hot_device_buffer;
    SparkCudaKvDeviceBuffer backing_device_buffer;
} SparkKvPayloadStageBuffer;

typedef struct SparkKvMovementStageState
{
    uint32_t stage_id;
    uint64_t generation;
    uint64_t last_page_table_fingerprint;
    uint64_t materialized_hot_pages;
    uint64_t represented_warm_pages;
    uint64_t represented_cold_pages;
    uint64_t materialized_hot_bytes;
    uint64_t represented_backing_bytes;
    uint64_t pinned_hot_capacity_bytes;
    uint64_t pinned_backing_capacity_bytes;
    uint64_t device_hot_capacity_bytes;
    uint64_t device_backing_capacity_bytes;
    uint64_t hot_payload_checksum;
    uint64_t backing_payload_checksum;
    uint64_t cumulative_loaded_bytes;
    uint64_t cumulative_promoted_bytes;
    uint64_t cumulative_evicted_bytes;
    uint64_t cumulative_released_bytes;
    uint64_t cumulative_copy_operations;
    uint64_t cumulative_wait_count;
    uint64_t cumulative_capacity_violation_count;
    uint64_t cumulative_missing_hot_page_violation_count;
    uint64_t cumulative_payload_write_bytes;
    uint64_t cumulative_payload_read_bytes;
    uint64_t cumulative_cuda_h2d_bytes;
    uint64_t cumulative_cuda_d2h_bytes;
    uint64_t cumulative_dma_copy_operations;
    SparkKvPagePayloadHandle last_payload_handle;
} SparkKvMovementStageState;

typedef struct SparkKvByteMover
{
    uint32_t stage_count;
    bool transfer_runtime_ready;
    SparkCudaKvDmaPolicy transfer_policy;
    SparkCudaKvDmaRuntime transfer_runtime;
    SparkKvMovementStageState stages[SPARKPIPE_MAX_STAGES];
    SparkKvPayloadStageBuffer payload_buffers[SPARKPIPE_MAX_STAGES];
} SparkKvByteMover;

typedef struct SparkKvMaterializationRequest
{
    uint64_t fabric_tick;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t slot_generation;
    uint64_t hot_capacity_pages;
    uint64_t backing_capacity_pages;
    const SparkKvPageTable *page_table;
} SparkKvMaterializationRequest;

typedef struct SparkKvMaterializationReport
{
    bool materialized;
    bool pinned_host_ready;
    bool cuda_device_ready;
    bool dma_runtime_ready;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint64_t page_table_fingerprint;
    uint64_t required_hot_pages;
    uint64_t represented_warm_pages;
    uint64_t represented_cold_pages;
    uint64_t bytes_loaded_to_hot;
    uint64_t bytes_promoted_to_hot;
    uint64_t bytes_copied_from_warm;
    uint64_t bytes_copied_from_cold;
    uint64_t bytes_evicted_from_hot;
    uint64_t bytes_released_from_hot;
    uint64_t hot_bytes_materialized;
    uint64_t backing_bytes_represented;
    uint64_t pinned_hot_capacity_bytes;
    uint64_t pinned_backing_capacity_bytes;
    uint64_t device_hot_capacity_bytes;
    uint64_t device_backing_capacity_bytes;
    uint64_t copy_operation_count;
    uint64_t wait_count;
    uint64_t capacity_violation_count;
    uint64_t missing_hot_page_violation_count;
    uint64_t materialization_checksum;
    SparkKvPagePayloadHandle payload_handle;
} SparkKvMaterializationReport;

typedef struct SparkKvPayloadIoRequest
{
    uint32_t stage_id;
    SparkKvPayloadRegionKind region_kind;
    uint64_t byte_offset;
    uint64_t byte_count;
    const uint8_t *source_bytes;
    uint8_t *target_bytes;
    bool use_cuda_dma_if_ready;
} SparkKvPayloadIoRequest;

typedef struct SparkKvPayloadIoReport
{
    bool completed;
    bool used_pinned_host_buffer;
    bool used_cuda_dma;
    SparkKvPayloadRegionKind region_kind;
    uint32_t stage_id;
    uint64_t byte_offset;
    uint64_t byte_count;
    uint64_t payload_checksum;
    uint64_t cuda_h2d_bytes;
    uint64_t cuda_d2h_bytes;
    uint64_t dma_copy_operation_count;
    uint64_t event_record_count;
    uint64_t event_sync_count;
} SparkKvPayloadIoReport;

typedef struct SparkKvPayloadRegionView
{
    bool valid;
    bool pinned_host_ready;
    bool cuda_device_ready;
    bool dma_runtime_ready;
    uint32_t stage_id;
    SparkKvPayloadRegionKind region_kind;
    uint64_t region_bytes;
    uint64_t host_capacity_bytes;
    uint64_t device_capacity_bytes;
    uint64_t payload_checksum;
    uint64_t view_checksum;
    uint8_t *host_bytes;
    void *device_bytes;
} SparkKvPayloadRegionView;

SparkStatus SparkInitializeKvByteMover(SparkKvByteMover *kv_mover, uint32_t stage_count);
SparkStatus SparkInitializeKvByteMoverWithTransferPolicy(SparkKvByteMover *kv_mover, uint32_t stage_count, const SparkCudaKvDmaPolicy *transfer_policy);
void SparkDestroyKvByteMover(SparkKvByteMover *kv_mover);
uint64_t SparkComputeKvPageTableFingerprint(const SparkKvPageTable *page_table);
uint64_t SparkComputeKvPagePayloadHandleChecksum(const SparkKvPagePayloadHandle *payload_handle);
uint64_t SparkComputeKvMaterializationReportChecksum(const SparkKvMaterializationReport *materialization_report);
uint64_t SparkComputeKvPayloadBytesChecksum(const uint8_t *payload_bytes, uint64_t payload_byte_count);
SparkStatus SparkMaterializeKvByteRegions(SparkKvByteMover *kv_mover, const SparkKvMaterializationRequest *materialization_request, SparkKvMaterializationReport *materialization_report);
SparkStatus SparkValidateKvMaterializationReport(const SparkKvMaterializationReport *materialization_report, const SparkKvPageTable *page_table);
SparkStatus SparkWriteKvPayloadBytes(SparkKvByteMover *kv_mover, const SparkKvPayloadIoRequest *payload_write_request, SparkKvPayloadIoReport *payload_io_report);
SparkStatus SparkReadKvPayloadBytes(SparkKvByteMover *kv_mover, const SparkKvPayloadIoRequest *payload_read_request, SparkKvPayloadIoReport *payload_io_report);
SparkStatus SparkGetKvPayloadRegionView(SparkKvByteMover *kv_mover, uint32_t stage_id, SparkKvPayloadRegionKind region_kind, SparkKvPayloadRegionView *region_view);
const SparkKvMovementStageState *SparkGetKvMovementStageStateConst(const SparkKvByteMover *kv_mover, uint32_t stage_id);

#ifdef __cplusplus
}
#endif

#endif
