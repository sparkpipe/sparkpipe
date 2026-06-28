#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_stage_abi.h"

static __device__ uint64_t SparkCudaMixU64Device(uint64_t checksum, uint64_t value)
{
    checksum ^= value + 0x9e3779b97f4a7c15ull + (checksum << 6u) + (checksum >> 2u);
    return checksum;
}

static __global__ void SparkCudaDummyStageKernel(SparkCudaStageLaunchDescriptor launch_descriptor, SparkCudaStageLaunchResult *launch_result)
{
    uint64_t checksum;

    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

    checksum = launch_descriptor.input_activation_checksum;
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.descriptor_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.fabric_tick);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.stage_id);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.model_lane);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.profile_id);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.slot_generation);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.kv_view.represented_hot_pages);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.kv_materialization_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.workspace_total_bytes);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_set_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_allocation_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_allocator_arena_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_allocator_allocation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_allocator_failure_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_ownership_violation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_bounds_violation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.region_overlap_violation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.stream_dependency_plan_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_dependency_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_dependency_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_missing_recv_ready_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_missing_kv_ready_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_stale_violation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_dependency_violation_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_ring_checksum_before);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.event_ring_checksum_after);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.persistent_pool_checksum);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.persistent_pool_reuse_count);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.persistent_pool_high_water_bytes);
    checksum = SparkCudaMixU64Device(checksum, launch_descriptor.persistent_pool_stale_handle_violation_count);

    memset(launch_result, 0, sizeof(*launch_result));
    launch_result->descriptor_version = launch_descriptor.descriptor_version;
    launch_result->backend_kind = SPARK_CUDA_STAGE_BACKEND_OPTIONAL_CUDA_DUMMY;
    launch_result->descriptor_validated = true;
    launch_result->descriptor_checksum = launch_descriptor.descriptor_checksum;
    launch_result->output_activation_checksum = checksum;
    launch_result->activation_bytes_observed = launch_descriptor.activation_payload_bytes;
    launch_result->workspace_bytes_observed = launch_descriptor.workspace_total_bytes;
    launch_result->activation_region_bytes_observed = launch_descriptor.region_set.total_activation_region_bytes;
    launch_result->workspace_region_bytes_observed = launch_descriptor.region_set.total_workspace_region_bytes;
    launch_result->kv_hot_region_bytes_observed = launch_descriptor.region_set.total_kv_hot_region_bytes;
    launch_result->kv_backing_region_bytes_observed = launch_descriptor.region_set.total_kv_backing_region_bytes;
    launch_result->kv_materialization_checksum = launch_descriptor.kv_materialization_checksum;
    launch_result->kv_materialized_hot_bytes = launch_descriptor.kv_materialized_hot_bytes;
    launch_result->kv_materialized_backing_bytes = launch_descriptor.kv_materialized_backing_bytes;
    launch_result->kv_bytes_loaded_to_hot = launch_descriptor.kv_bytes_loaded_to_hot;
    launch_result->kv_bytes_promoted_to_hot = launch_descriptor.kv_bytes_promoted_to_hot;
    launch_result->kv_bytes_copied_from_warm = launch_descriptor.kv_bytes_copied_from_warm;
    launch_result->kv_bytes_copied_from_cold = launch_descriptor.kv_bytes_copied_from_cold;
    launch_result->kv_bytes_evicted_from_hot = launch_descriptor.kv_bytes_evicted_from_hot;
    launch_result->kv_bytes_released_from_hot = launch_descriptor.kv_bytes_released_from_hot;
    launch_result->kv_copy_operation_count = launch_descriptor.kv_copy_operation_count;
    launch_result->kv_materialization_wait_count = launch_descriptor.kv_materialization_wait_count;
    launch_result->kv_materialization_ready_count = launch_descriptor.kv_materialization_ready_count;
    launch_result->kv_materialization_capacity_violation_count = launch_descriptor.kv_materialization_capacity_violation_count;
    launch_result->kv_materialization_missing_hot_page_violation_count = launch_descriptor.kv_materialization_missing_hot_page_violation_count;
    launch_result->region_set_checksum = launch_descriptor.region_set_checksum;
    launch_result->region_allocation_checksum = launch_descriptor.region_allocation_checksum;
    launch_result->region_allocator_arena_count = launch_descriptor.region_allocator_arena_count;
    launch_result->region_allocator_allocation_count = launch_descriptor.region_allocator_allocation_count;
    launch_result->region_allocator_failure_count = launch_descriptor.region_allocator_failure_count;
    launch_result->region_ownership_violation_count = launch_descriptor.region_ownership_violation_count;
    launch_result->region_bounds_violation_count = launch_descriptor.region_bounds_violation_count;
    launch_result->region_overlap_violation_count = launch_descriptor.region_overlap_violation_count;
    launch_result->stream_dependency_plan_checksum = launch_descriptor.stream_dependency_plan_checksum;
    launch_result->event_dependency_checksum = launch_descriptor.event_dependency_checksum;
    launch_result->event_dependency_count = launch_descriptor.event_dependency_count;
    launch_result->event_count = launch_descriptor.event_count;
    launch_result->event_missing_recv_ready_count = launch_descriptor.event_missing_recv_ready_count;
    launch_result->event_missing_kv_ready_count = launch_descriptor.event_missing_kv_ready_count;
    launch_result->event_stale_violation_count = launch_descriptor.event_stale_violation_count;
    launch_result->event_dependency_violation_count = launch_descriptor.event_dependency_violation_count;
    launch_result->event_ring_checksum_before = launch_descriptor.event_ring_checksum_before;
    launch_result->event_ring_checksum_after = launch_descriptor.event_ring_checksum_after;
    launch_result->persistent_pool_checksum = launch_descriptor.persistent_pool_checksum;
    launch_result->persistent_pool_reuse_count = launch_descriptor.persistent_pool_reuse_count;
    launch_result->persistent_pool_high_water_bytes = launch_descriptor.persistent_pool_high_water_bytes;
    launch_result->persistent_pool_stale_handle_violation_count = launch_descriptor.persistent_pool_stale_handle_violation_count;
    launch_result->active_slots_observed = launch_descriptor.active_slot_count;
    launch_result->hot_kv_pages_observed = launch_descriptor.kv_view.represented_hot_pages;
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(checksum, launch_result->backend_kind);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->kv_materialization_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->kv_materialized_hot_bytes);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->kv_materialized_backing_bytes);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_set_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_allocation_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_allocator_arena_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_allocator_allocation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_allocator_failure_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_ownership_violation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_bounds_violation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->region_overlap_violation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->stream_dependency_plan_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_dependency_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_dependency_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_missing_recv_ready_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_missing_kv_ready_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_stale_violation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_dependency_violation_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_ring_checksum_before);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->event_ring_checksum_after);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->persistent_pool_checksum);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->persistent_pool_reuse_count);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->persistent_pool_high_water_bytes);
    launch_result->backend_validation_checksum = SparkCudaMixU64Device(launch_result->backend_validation_checksum, launch_result->persistent_pool_stale_handle_violation_count);
}

extern "C" SparkStatus SparkRunOptionalCudaDummyKernelDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor, SparkCudaStageLaunchResult *launch_result)
{
    SparkCudaStageLaunchResult *device_launch_result;
    cudaError_t cuda_status;
    SparkStatus status;

    if (launch_descriptor == 0 || launch_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkValidateCudaStageLaunchDescriptor(launch_descriptor);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    device_launch_result = 0;
    cuda_status = cudaMalloc((void **)&device_launch_result, sizeof(*device_launch_result));
    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    SparkCudaDummyStageKernel<<<1, 32>>>(*launch_descriptor, device_launch_result);
    cuda_status = cudaGetLastError();
    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaDeviceSynchronize();
    }

    if (cuda_status == cudaSuccess)
    {
        cuda_status = cudaMemcpy(launch_result, device_launch_result, sizeof(*launch_result), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_launch_result);

    if (cuda_status != cudaSuccess)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }

    return SPARK_STATUS_OK;
}
