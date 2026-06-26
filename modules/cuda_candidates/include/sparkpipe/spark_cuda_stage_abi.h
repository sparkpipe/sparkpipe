#ifndef SPARKPIPE_SPARK_CUDA_STAGE_ABI_H
#define SPARKPIPE_SPARK_CUDA_STAGE_ABI_H

#include "sparkpipe/spark_activation_layout.h"
#include "sparkpipe/spark_kv_page_table.h"
#include "sparkpipe/spark_kv_mover.h"
#include "sparkpipe/spark_event_plan.h"
#include "sparkpipe/spark_memory_region.h"
#include "sparkpipe/spark_stage_pool.h"
#include "sparkpipe/spark_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkCudaStageBackendKind
{
    SPARK_CUDA_STAGE_BACKEND_C_DUMMY = SPARKPIPE_CUDA_STAGE_BACKEND_C_DUMMY,
    SPARK_CUDA_STAGE_BACKEND_OPTIONAL_CUDA_DUMMY = SPARKPIPE_CUDA_STAGE_BACKEND_OPTIONAL_CUDA_DUMMY,
    SPARK_CUDA_STAGE_BACKEND_PRODUCTION_ADAPTER = SPARKPIPE_CUDA_STAGE_BACKEND_PRODUCTION_ADAPTER
} SparkCudaStageBackendKind;

typedef struct SparkCudaKvLaunchViewDescriptor
{
    uint32_t stage_id;
    uint32_t projected_pages;
    uint32_t represented_hot_pages;
    uint32_t represented_warm_pages;
    uint32_t represented_cold_pages;
    uint32_t truncated_pages;
    uint32_t page_table_capacity;
    uint64_t kv_page_bytes;
    uint64_t required_by_tick_min;
    uint64_t safe_release_after_tick_max;
    uint64_t page_table_fingerprint;
} SparkCudaKvLaunchViewDescriptor;

typedef struct SparkCudaStageLaunchDescriptor
{
    uint32_t descriptor_version;
    uint32_t descriptor_bytes;
    uint64_t fabric_tick;
    uint32_t stage_id;
    uint32_t model_lane;
    uint32_t profile_id;
    uint64_t slot_generation;
    uint32_t physical_slot_count;
    uint32_t active_slot_count;
    uint64_t active_mask_checksum;
    uint32_t mapped_slot_count;
    uint64_t slot_mapping_checksum;
    uint32_t activation_element_size_bytes;
    uint32_t activation_hidden_size;
    uint64_t activation_row_stride_bytes;
    uint64_t activation_payload_bytes;
    uint64_t input_buffer_id;
    uint64_t input_activation_checksum;
    uint64_t workspace_total_bytes;
    uint64_t workspace_alignment_bytes;
    SparkCudaKvLaunchViewDescriptor kv_view;
    SparkKvPagePayloadHandle kv_payload_handle;
    uint64_t kv_materialization_checksum;
    uint64_t kv_materialized_hot_bytes;
    uint64_t kv_materialized_backing_bytes;
    uint64_t kv_bytes_loaded_to_hot;
    uint64_t kv_bytes_promoted_to_hot;
    uint64_t kv_bytes_copied_from_warm;
    uint64_t kv_bytes_copied_from_cold;
    uint64_t kv_bytes_evicted_from_hot;
    uint64_t kv_bytes_released_from_hot;
    uint64_t kv_copy_operation_count;
    uint64_t kv_materialization_wait_count;
    uint64_t kv_materialization_ready_count;
    uint64_t kv_materialization_capacity_violation_count;
    uint64_t kv_materialization_missing_hot_page_violation_count;
    SparkStageMemoryRegionSet region_set;
    SparkStageStreamDependencyPlan stream_dependency_plan;
    SparkStageEventDependencyReport event_dependency_report;
    uint64_t region_set_checksum;
    uint64_t region_allocation_checksum;
    uint32_t region_allocator_arena_count;
    uint32_t region_allocator_allocation_count;
    uint32_t region_allocator_failure_count;
    uint32_t region_ownership_violation_count;
    uint32_t region_bounds_violation_count;
    uint32_t region_overlap_violation_count;
    uint64_t stream_dependency_plan_checksum;
    uint64_t event_dependency_checksum;
    uint32_t event_dependency_count;
    uint32_t event_count;
    uint32_t event_missing_recv_ready_count;
    uint32_t event_missing_kv_ready_count;
    uint32_t event_stale_violation_count;
    uint32_t event_dependency_violation_count;
    uint64_t event_ring_checksum_before;
    uint64_t event_ring_checksum_after;
    uint64_t persistent_pool_checksum;
    uint64_t persistent_pool_reuse_count;
    uint64_t persistent_pool_high_water_bytes;
    uint32_t persistent_pool_stale_handle_violation_count;
    uint64_t descriptor_checksum;
} SparkCudaStageLaunchDescriptor;

typedef struct SparkCudaStageLaunchResult
{
    uint32_t descriptor_version;
    uint32_t backend_kind;
    bool descriptor_validated;
    uint64_t descriptor_checksum;
    uint64_t output_activation_checksum;
    uint64_t activation_bytes_observed;
    uint64_t workspace_bytes_observed;
    uint64_t activation_region_bytes_observed;
    uint64_t workspace_region_bytes_observed;
    uint64_t kv_hot_region_bytes_observed;
    uint64_t kv_backing_region_bytes_observed;
    uint64_t kv_materialization_checksum;
    uint64_t kv_materialized_hot_bytes;
    uint64_t kv_materialized_backing_bytes;
    uint64_t kv_bytes_loaded_to_hot;
    uint64_t kv_bytes_promoted_to_hot;
    uint64_t kv_bytes_copied_from_warm;
    uint64_t kv_bytes_copied_from_cold;
    uint64_t kv_bytes_evicted_from_hot;
    uint64_t kv_bytes_released_from_hot;
    uint64_t kv_copy_operation_count;
    uint64_t kv_materialization_wait_count;
    uint64_t kv_materialization_ready_count;
    uint64_t kv_materialization_capacity_violation_count;
    uint64_t kv_materialization_missing_hot_page_violation_count;
    uint64_t region_set_checksum;
    uint64_t region_allocation_checksum;
    uint32_t region_allocator_arena_count;
    uint32_t region_allocator_allocation_count;
    uint32_t region_allocator_failure_count;
    uint32_t region_ownership_violation_count;
    uint32_t region_bounds_violation_count;
    uint32_t region_overlap_violation_count;
    uint64_t stream_dependency_plan_checksum;
    uint64_t event_dependency_checksum;
    uint32_t event_dependency_count;
    uint32_t event_count;
    uint32_t event_missing_recv_ready_count;
    uint32_t event_missing_kv_ready_count;
    uint32_t event_stale_violation_count;
    uint32_t event_dependency_violation_count;
    uint64_t event_ring_checksum_before;
    uint64_t event_ring_checksum_after;
    uint64_t persistent_pool_checksum;
    uint64_t persistent_pool_reuse_count;
    uint64_t persistent_pool_high_water_bytes;
    uint32_t persistent_pool_stale_handle_violation_count;
    uint32_t active_slots_observed;
    uint32_t hot_kv_pages_observed;
    uint64_t backend_validation_checksum;
} SparkCudaStageLaunchResult;

typedef struct SparkCudaStageExecutionInput
{
    uint64_t fabric_tick;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t slot_generation;
    const SparkActivationBuffer *input_activation;
    SparkActivationBuffer *output_activation;
    const SparkActiveSlotMask *active_slot_mask;
    const SparkPhysicalSlotMapping *slot_mapping;
    const SparkKvPageTable *kv_page_table;
    const SparkKvMaterializationReport *kv_materialization_report;
    const SparkStageWorkspaceReservation *workspace_reservation;
    SparkStagePersistentPool *persistent_pool;
    SparkStageEventRing *event_ring;
} SparkCudaStageExecutionInput;

typedef struct SparkCudaStageExecutionOutput
{
    uint64_t output_activation_checksum;
    uint64_t activation_bytes_observed;
    uint64_t workspace_bytes_observed;
    uint64_t activation_region_bytes_observed;
    uint64_t workspace_region_bytes_observed;
    uint64_t kv_hot_region_bytes_observed;
    uint64_t kv_backing_region_bytes_observed;
    uint64_t kv_materialization_checksum;
    uint64_t kv_materialized_hot_bytes;
    uint64_t kv_materialized_backing_bytes;
    uint64_t kv_bytes_loaded_to_hot;
    uint64_t kv_bytes_promoted_to_hot;
    uint64_t kv_bytes_copied_from_warm;
    uint64_t kv_bytes_copied_from_cold;
    uint64_t kv_bytes_evicted_from_hot;
    uint64_t kv_bytes_released_from_hot;
    uint64_t kv_copy_operation_count;
    uint64_t kv_materialization_wait_count;
    uint64_t kv_materialization_ready_count;
    uint64_t kv_materialization_capacity_violation_count;
    uint64_t kv_materialization_missing_hot_page_violation_count;
    uint64_t region_set_checksum;
    uint64_t region_allocation_checksum;
    uint32_t region_allocator_arena_count;
    uint32_t region_allocator_allocation_count;
    uint32_t region_allocator_failure_count;
    uint32_t region_ownership_violation_count;
    uint32_t region_bounds_violation_count;
    uint32_t region_overlap_violation_count;
    uint64_t stream_dependency_plan_checksum;
    uint64_t event_dependency_checksum;
    uint32_t event_dependency_count;
    uint32_t event_count;
    uint32_t event_missing_recv_ready_count;
    uint32_t event_missing_kv_ready_count;
    uint32_t event_stale_violation_count;
    uint32_t event_dependency_violation_count;
    uint64_t event_ring_checksum_before;
    uint64_t event_ring_checksum_after;
    uint64_t persistent_pool_checksum;
    uint64_t persistent_pool_reuse_count;
    uint64_t persistent_pool_high_water_bytes;
    uint32_t persistent_pool_stale_handle_violation_count;
    uint32_t active_slots_observed;
    uint32_t hot_kv_pages_observed;
    uint64_t descriptor_checksum;
    uint64_t backend_validation_checksum;
    SparkCudaStageBackendKind backend_kind;
    bool descriptor_validated;
} SparkCudaStageExecutionOutput;

SparkStatus SparkBuildCudaKvLaunchViewDescriptor(const SparkKvPageTable *page_table, SparkCudaKvLaunchViewDescriptor *kv_view_descriptor);
uint64_t SparkComputeCudaKvLaunchViewChecksum(const SparkCudaKvLaunchViewDescriptor *kv_view_descriptor);
uint64_t SparkComputeCudaStageLaunchDescriptorChecksum(const SparkCudaStageLaunchDescriptor *launch_descriptor);
SparkStatus SparkBuildCudaStageLaunchDescriptor(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageLaunchDescriptor *launch_descriptor);
SparkStatus SparkValidateCudaStageLaunchDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor);
SparkStatus SparkExecuteDummyCudaStageFromDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor, SparkCudaStageLaunchResult *launch_result);
SparkStatus SparkExecuteCudaStageWithBackend(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageBackendKind backend_kind, SparkCudaStageExecutionOutput *execution_output);
SparkStatus SparkExecuteDummyCudaStage(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageExecutionOutput *execution_output);

#ifdef __cplusplus
}
#endif

#endif
