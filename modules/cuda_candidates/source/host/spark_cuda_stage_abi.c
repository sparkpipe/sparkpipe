#include <string.h>

#include "sparkpipe/spark_cuda_stage_abi.h"
#include "sparkpipe/spark_slot_table.h"

#ifdef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunOptionalCudaDummyKernelDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor, SparkCudaStageLaunchResult *launch_result);
#endif

static uint64_t SparkComputeKvPageTableEntryChecksum(const SparkKvPageTableEntry *entry)
{
    uint64_t checksum;

    if (entry == 0 || !entry->allocated)
    {
        return 0;
    }

    checksum = 0x4B5650454E545259ull;
    checksum = SparkMixU64(checksum, entry->stage_id);
    checksum = SparkMixU64(checksum, entry->slot_id);
    checksum = SparkMixU64(checksum, entry->logical_page_index);
    checksum = SparkMixU64(checksum, (uint64_t)entry->model_lane);
    checksum = SparkMixU64(checksum, entry->request_id);
    checksum = SparkMixU64(checksum, entry->kv_handle);
    checksum = SparkMixU64(checksum, entry->required_by_tick);
    checksum = SparkMixU64(checksum, entry->safe_release_after_tick);
    checksum = SparkMixU64(checksum, entry->last_touched_tick);
    checksum = SparkMixU64(checksum, (uint64_t)entry->residency);

    return checksum;
}

SparkStatus SparkBuildCudaKvLaunchViewDescriptor(const SparkKvPageTable *page_table, SparkCudaKvLaunchViewDescriptor *kv_view_descriptor)
{
    uint32_t entry_index;
    uint64_t fingerprint;
    uint64_t first_required_by_tick;
    uint64_t last_safe_release_after_tick;

    if (page_table == 0 || kv_view_descriptor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(kv_view_descriptor, 0, sizeof(*kv_view_descriptor));
    kv_view_descriptor->stage_id = page_table->stage_id;
    kv_view_descriptor->represented_hot_pages = (uint32_t)SparkCountRepresentedKvPagesByResidency(page_table, SPARK_KV_PAGE_RESIDENCY_HOT);
    kv_view_descriptor->represented_warm_pages = (uint32_t)SparkCountRepresentedKvPagesByResidency(page_table, SPARK_KV_PAGE_RESIDENCY_WARM);
    kv_view_descriptor->represented_cold_pages = (uint32_t)SparkCountRepresentedKvPagesByResidency(page_table, SPARK_KV_PAGE_RESIDENCY_COLD);
    kv_view_descriptor->projected_pages = kv_view_descriptor->represented_hot_pages + kv_view_descriptor->represented_warm_pages + kv_view_descriptor->represented_cold_pages;
    kv_view_descriptor->page_table_capacity = page_table->capacity;
    kv_view_descriptor->kv_page_bytes = SPARKPIPE_KV_PAGE_BYTES;

    if (kv_view_descriptor->projected_pages > page_table->capacity)
    {
        kv_view_descriptor->truncated_pages = kv_view_descriptor->projected_pages - page_table->capacity;
    }

    fingerprint = 0x4B564C41554E4348ull;
    first_required_by_tick = 0;
    last_safe_release_after_tick = 0;

    for (entry_index = 0; entry_index < page_table->capacity && entry_index < SPARKPIPE_MAX_KV_PAGE_TABLE_ENTRIES_PER_STAGE; ++entry_index)
    {
        const SparkKvPageTableEntry *entry;

        entry = &page_table->entries[entry_index];
        if (!entry->allocated)
        {
            continue;
        }

        if (first_required_by_tick == 0 || entry->required_by_tick < first_required_by_tick)
        {
            first_required_by_tick = entry->required_by_tick;
        }

        if (entry->safe_release_after_tick > last_safe_release_after_tick)
        {
            last_safe_release_after_tick = entry->safe_release_after_tick;
        }

        fingerprint = SparkMixU64(fingerprint, SparkComputeKvPageTableEntryChecksum(entry));
    }

    kv_view_descriptor->required_by_tick_min = first_required_by_tick;
    kv_view_descriptor->safe_release_after_tick_max = last_safe_release_after_tick;
    kv_view_descriptor->page_table_fingerprint = fingerprint;

    return SPARK_STATUS_OK;
}

uint64_t SparkComputeCudaKvLaunchViewChecksum(const SparkCudaKvLaunchViewDescriptor *kv_view_descriptor)
{
    uint64_t checksum;

    if (kv_view_descriptor == 0)
    {
        return 0;
    }

    checksum = 0x4B56444553435249ull;
    checksum = SparkMixU64(checksum, kv_view_descriptor->stage_id);
    checksum = SparkMixU64(checksum, kv_view_descriptor->projected_pages);
    checksum = SparkMixU64(checksum, kv_view_descriptor->represented_hot_pages);
    checksum = SparkMixU64(checksum, kv_view_descriptor->represented_warm_pages);
    checksum = SparkMixU64(checksum, kv_view_descriptor->represented_cold_pages);
    checksum = SparkMixU64(checksum, kv_view_descriptor->truncated_pages);
    checksum = SparkMixU64(checksum, kv_view_descriptor->page_table_capacity);
    checksum = SparkMixU64(checksum, kv_view_descriptor->kv_page_bytes);
    checksum = SparkMixU64(checksum, kv_view_descriptor->required_by_tick_min);
    checksum = SparkMixU64(checksum, kv_view_descriptor->safe_release_after_tick_max);
    checksum = SparkMixU64(checksum, kv_view_descriptor->page_table_fingerprint);

    return checksum;
}

uint64_t SparkComputeCudaStageLaunchDescriptorChecksum(const SparkCudaStageLaunchDescriptor *launch_descriptor)
{
    uint64_t checksum;

    if (launch_descriptor == 0)
    {
        return 0;
    }

    checksum = 0x435544414C41554Eull;
    checksum = SparkMixU64(checksum, launch_descriptor->descriptor_version);
    checksum = SparkMixU64(checksum, launch_descriptor->descriptor_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->fabric_tick);
    checksum = SparkMixU64(checksum, launch_descriptor->stage_id);
    checksum = SparkMixU64(checksum, launch_descriptor->model_lane);
    checksum = SparkMixU64(checksum, launch_descriptor->profile_id);
    checksum = SparkMixU64(checksum, launch_descriptor->slot_generation);
    checksum = SparkMixU64(checksum, launch_descriptor->physical_slot_count);
    checksum = SparkMixU64(checksum, launch_descriptor->active_slot_count);
    checksum = SparkMixU64(checksum, launch_descriptor->active_mask_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->mapped_slot_count);
    checksum = SparkMixU64(checksum, launch_descriptor->slot_mapping_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->activation_element_size_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->activation_hidden_size);
    checksum = SparkMixU64(checksum, launch_descriptor->activation_row_stride_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->activation_payload_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->input_buffer_id);
    checksum = SparkMixU64(checksum, launch_descriptor->input_activation_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->workspace_total_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->workspace_alignment_bytes);
    checksum = SparkMixU64(checksum, SparkComputeCudaKvLaunchViewChecksum(&launch_descriptor->kv_view));
    checksum = SparkMixU64(checksum, SparkComputeKvPagePayloadHandleChecksum(&launch_descriptor->kv_payload_handle));
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialized_hot_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialized_backing_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_loaded_to_hot);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_promoted_to_hot);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_copied_from_warm);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_copied_from_cold);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_evicted_from_hot);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_released_from_hot);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_copy_operation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_wait_count);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_ready_count);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_capacity_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_missing_hot_page_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_set_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->region_allocation_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->region_allocator_arena_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_allocator_allocation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_allocator_failure_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_ownership_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_bounds_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->region_overlap_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->stream_dependency_plan_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->event_dependency_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->event_dependency_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_missing_recv_ready_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_missing_kv_ready_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_stale_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_dependency_violation_count);
    checksum = SparkMixU64(checksum, launch_descriptor->event_ring_checksum_before);
    checksum = SparkMixU64(checksum, launch_descriptor->event_ring_checksum_after);
    checksum = SparkMixU64(checksum, launch_descriptor->persistent_pool_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->persistent_pool_reuse_count);
    checksum = SparkMixU64(checksum, launch_descriptor->persistent_pool_high_water_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->persistent_pool_stale_handle_violation_count);

    return checksum;
}

static SparkStatus SparkValidateCudaStageExecutionInput(const SparkCudaStageExecutionInput *execution_input)
{
    uint32_t active_slots;

    if (execution_input == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->input_activation == 0 || execution_input->output_activation == 0 || execution_input->active_slot_mask == 0 || execution_input->slot_mapping == 0 || execution_input->kv_page_table == 0 || execution_input->kv_materialization_report == 0 || execution_input->workspace_reservation == 0 || execution_input->event_ring == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (!execution_input->input_activation->initialized)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->stage_id != execution_input->workspace_reservation->stage_id || execution_input->stage_id != execution_input->kv_page_table->stage_id || execution_input->event_ring->stage_id != execution_input->stage_id)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkValidateKvMaterializationReport(execution_input->kv_materialization_report, execution_input->kv_page_table) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_KV_LEASE_NOT_READY;
    }

    if (execution_input->kv_materialization_report->stage_id != execution_input->stage_id || execution_input->kv_materialization_report->model_lane != execution_input->model_lane || execution_input->kv_materialization_report->profile_id != execution_input->profile_id || execution_input->kv_materialization_report->fabric_tick > execution_input->fabric_tick)
    {
        return SPARK_STATUS_KV_LEASE_NOT_READY;
    }

    if (execution_input->model_lane != execution_input->workspace_reservation->model_lane)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->profile_id != execution_input->workspace_reservation->profile_id)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->active_slot_mask->generation != execution_input->slot_generation || execution_input->slot_mapping->generation != execution_input->slot_generation)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    active_slots = SparkCountActiveSlotsInMask(execution_input->active_slot_mask);
    if (active_slots != execution_input->slot_mapping->mapped_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (execution_input->input_activation->active_slot_mask.generation != execution_input->slot_generation)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkBuildCudaStageLaunchDescriptor(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageLaunchDescriptor *launch_descriptor)
{
    SparkStageRegionArenaSet arena_set;
    SparkStageRegionAllocationReport allocation_report;
    SparkStagePersistentPoolReport pool_report;
    SparkStatus status;

    if (execution_input == 0 || launch_descriptor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkValidateCudaStageExecutionInput(execution_input);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(launch_descriptor, 0, sizeof(*launch_descriptor));
    launch_descriptor->descriptor_version = SPARKPIPE_CUDA_STAGE_LAUNCH_DESCRIPTOR_VERSION;
    launch_descriptor->descriptor_bytes = (uint32_t)sizeof(*launch_descriptor);
    launch_descriptor->fabric_tick = execution_input->fabric_tick;
    launch_descriptor->stage_id = execution_input->stage_id;
    launch_descriptor->model_lane = (uint32_t)execution_input->model_lane;
    launch_descriptor->profile_id = (uint32_t)execution_input->profile_id;
    launch_descriptor->slot_generation = execution_input->slot_generation;
    launch_descriptor->physical_slot_count = execution_input->active_slot_mask->physical_slot_count;
    launch_descriptor->active_slot_count = SparkCountActiveSlotsInMask(execution_input->active_slot_mask);
    launch_descriptor->active_mask_checksum = SparkComputeActiveSlotMaskChecksum(execution_input->active_slot_mask);
    launch_descriptor->mapped_slot_count = execution_input->slot_mapping->mapped_slot_count;
    launch_descriptor->slot_mapping_checksum = SparkComputePhysicalSlotMappingChecksum(execution_input->slot_mapping);
    launch_descriptor->activation_element_size_bytes = execution_input->input_activation->layout.element_size_bytes;
    launch_descriptor->activation_hidden_size = execution_input->input_activation->layout.hidden_size;
    launch_descriptor->activation_row_stride_bytes = execution_input->input_activation->layout.row_stride_bytes;
    launch_descriptor->activation_payload_bytes = execution_input->input_activation->layout.aligned_payload_bytes;
    launch_descriptor->input_buffer_id = execution_input->input_activation->buffer_id;
    launch_descriptor->input_activation_checksum = execution_input->input_activation->payload_checksum;
    launch_descriptor->workspace_total_bytes = execution_input->workspace_reservation->total_bytes;
    launch_descriptor->workspace_alignment_bytes = SPARKPIPE_WORKSPACE_ALIGNMENT_BYTES;

    status = SparkBuildCudaKvLaunchViewDescriptor(execution_input->kv_page_table, &launch_descriptor->kv_view);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    launch_descriptor->kv_payload_handle = execution_input->kv_materialization_report->payload_handle;
    launch_descriptor->kv_materialization_checksum = execution_input->kv_materialization_report->materialization_checksum;
    launch_descriptor->kv_materialized_hot_bytes = execution_input->kv_materialization_report->hot_bytes_materialized;
    launch_descriptor->kv_materialized_backing_bytes = execution_input->kv_materialization_report->backing_bytes_represented;
    launch_descriptor->kv_bytes_loaded_to_hot = execution_input->kv_materialization_report->bytes_loaded_to_hot;
    launch_descriptor->kv_bytes_promoted_to_hot = execution_input->kv_materialization_report->bytes_promoted_to_hot;
    launch_descriptor->kv_bytes_copied_from_warm = execution_input->kv_materialization_report->bytes_copied_from_warm;
    launch_descriptor->kv_bytes_copied_from_cold = execution_input->kv_materialization_report->bytes_copied_from_cold;
    launch_descriptor->kv_bytes_evicted_from_hot = execution_input->kv_materialization_report->bytes_evicted_from_hot;
    launch_descriptor->kv_bytes_released_from_hot = execution_input->kv_materialization_report->bytes_released_from_hot;
    launch_descriptor->kv_copy_operation_count = execution_input->kv_materialization_report->copy_operation_count;
    launch_descriptor->kv_materialization_wait_count = execution_input->kv_materialization_report->wait_count;
    launch_descriptor->kv_materialization_ready_count = execution_input->kv_materialization_report->materialized ? 1u : 0u;
    launch_descriptor->kv_materialization_capacity_violation_count = execution_input->kv_materialization_report->capacity_violation_count;
    launch_descriptor->kv_materialization_missing_hot_page_violation_count = execution_input->kv_materialization_report->missing_hot_page_violation_count;

    memset(&arena_set, 0, sizeof(arena_set));
    memset(&allocation_report, 0, sizeof(allocation_report));
    memset(&pool_report, 0, sizeof(pool_report));

    if (execution_input->persistent_pool != 0)
    {
        status = SparkBuildPersistentPoolStageMemoryRegionSet(&launch_descriptor->region_set,
                                                              execution_input->persistent_pool,
                                                              &pool_report,
                                                              execution_input->fabric_tick,
                                                              execution_input->slot_generation,
                                                              execution_input->input_activation,
                                                              execution_input->output_activation,
                                                              execution_input->workspace_reservation,
                                                              execution_input->kv_page_table);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }

        allocation_report.arena_count = pool_report.arena_count;
        allocation_report.allocation_count = pool_report.region_count;
        allocation_report.allocation_failure_count = pool_report.allocation_failure_count;
        allocation_report.ownership_violation_count = pool_report.owner_violation_count;
        allocation_report.bounds_violation_count = pool_report.bounds_violation_count;
        allocation_report.overlap_violation_count = pool_report.overlap_violation_count;
        allocation_report.allocated_region_bytes = pool_report.high_water_bytes;
        allocation_report.activation_input_arena_id = pool_report.activation_input_arena_id;
        allocation_report.activation_output_arena_id = pool_report.activation_output_arena_id;
        allocation_report.workspace_arena_id = pool_report.workspace_arena_id;
        allocation_report.kv_hot_arena_id = pool_report.kv_hot_arena_id;
        allocation_report.kv_backing_arena_id = pool_report.kv_backing_arena_id;
        allocation_report.arena_set_checksum = pool_report.pool_checksum;
        allocation_report.allocation_checksum = pool_report.region_allocation_checksum;
    }
    else
    {
        status = SparkBuildAllocatorBackedStageMemoryRegionSet(&launch_descriptor->region_set,
                                                               &arena_set,
                                                               &allocation_report,
                                                               execution_input->fabric_tick,
                                                               execution_input->stage_id,
                                                               execution_input->model_lane,
                                                               execution_input->profile_id,
                                                               execution_input->slot_generation,
                                                               execution_input->input_activation,
                                                               execution_input->output_activation,
                                                               execution_input->workspace_reservation,
                                                               execution_input->kv_page_table);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    status = SparkBuildStageStreamDependencyPlan(&launch_descriptor->stream_dependency_plan,
                                                 execution_input->fabric_tick,
                                                 execution_input->stage_id,
                                                 execution_input->model_lane,
                                                 execution_input->profile_id,
                                                 execution_input->slot_generation,
                                                 &launch_descriptor->region_set);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkBuildStageEventDependencyReport(&launch_descriptor->event_dependency_report,
                                                  execution_input->event_ring,
                                                  execution_input->fabric_tick,
                                                  execution_input->stage_id,
                                                  execution_input->model_lane,
                                                  execution_input->profile_id,
                                                  execution_input->slot_generation,
                                                  true,
                                                  launch_descriptor->kv_view.represented_hot_pages > 0u,
                                                  execution_input->input_activation->payload_checksum,
                                                  launch_descriptor->kv_materialization_checksum);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    launch_descriptor->region_set_checksum = launch_descriptor->region_set.region_set_checksum;
    launch_descriptor->region_allocation_checksum = allocation_report.allocation_checksum;
    launch_descriptor->region_allocator_arena_count = allocation_report.arena_count;
    launch_descriptor->region_allocator_allocation_count = allocation_report.allocation_count;
    launch_descriptor->region_allocator_failure_count = allocation_report.allocation_failure_count;
    launch_descriptor->region_ownership_violation_count = allocation_report.ownership_violation_count;
    launch_descriptor->region_bounds_violation_count = allocation_report.bounds_violation_count;
    launch_descriptor->region_overlap_violation_count = allocation_report.overlap_violation_count;
    launch_descriptor->stream_dependency_plan_checksum = launch_descriptor->stream_dependency_plan.dependency_plan_checksum;
    launch_descriptor->event_dependency_checksum = launch_descriptor->event_dependency_report.event_dependency_checksum;
    launch_descriptor->event_dependency_count = launch_descriptor->event_dependency_report.dependency_count;
    launch_descriptor->event_count = launch_descriptor->event_dependency_report.event_count;
    launch_descriptor->event_missing_recv_ready_count = launch_descriptor->event_dependency_report.missing_recv_ready_count;
    launch_descriptor->event_missing_kv_ready_count = launch_descriptor->event_dependency_report.missing_kv_ready_count;
    launch_descriptor->event_stale_violation_count = launch_descriptor->event_dependency_report.stale_event_violation_count;
    launch_descriptor->event_dependency_violation_count = launch_descriptor->event_dependency_report.dependency_violation_count;
    launch_descriptor->event_ring_checksum_before = launch_descriptor->event_dependency_report.event_ring_checksum_before;
    launch_descriptor->event_ring_checksum_after = launch_descriptor->event_dependency_report.event_ring_checksum_after;
    launch_descriptor->persistent_pool_checksum = pool_report.pool_checksum;
    launch_descriptor->persistent_pool_reuse_count = pool_report.reuse_count;
    launch_descriptor->persistent_pool_high_water_bytes = pool_report.high_water_bytes;
    launch_descriptor->persistent_pool_stale_handle_violation_count = pool_report.stale_handle_violation_count;
    launch_descriptor->descriptor_checksum = SparkComputeCudaStageLaunchDescriptorChecksum(launch_descriptor);

    return SparkValidateCudaStageLaunchDescriptor(launch_descriptor);
}

SparkStatus SparkValidateCudaStageLaunchDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor)
{
    if (launch_descriptor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->descriptor_version != SPARKPIPE_CUDA_STAGE_LAUNCH_DESCRIPTOR_VERSION || launch_descriptor->descriptor_bytes != sizeof(*launch_descriptor))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->stage_id >= SPARKPIPE_MAX_STAGES || launch_descriptor->physical_slot_count == 0 || launch_descriptor->physical_slot_count > SPARKPIPE_MAX_PHYSICAL_SLOTS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (!SparkIsValidModelLane((SparkModelLaneKind)launch_descriptor->model_lane) || SparkGetPhysicalProfile((SparkPhysicalProfileId)launch_descriptor->profile_id) == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->active_slot_count != launch_descriptor->mapped_slot_count || launch_descriptor->active_slot_count > launch_descriptor->physical_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->activation_element_size_bytes == 0 || launch_descriptor->activation_hidden_size == 0 || launch_descriptor->activation_row_stride_bytes == 0 || launch_descriptor->activation_payload_bytes == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->workspace_total_bytes == 0 || launch_descriptor->workspace_alignment_bytes == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->kv_view.stage_id != launch_descriptor->stage_id)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->kv_materialization_capacity_violation_count != 0 || launch_descriptor->kv_materialization_missing_hot_page_violation_count != 0)
    {
        return SPARK_STATUS_KV_LEASE_NOT_READY;
    }

    if (launch_descriptor->kv_materialization_checksum == 0 || launch_descriptor->kv_materialization_ready_count == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->kv_materialized_hot_bytes != (uint64_t)launch_descriptor->kv_view.represented_hot_pages * (uint64_t)SPARKPIPE_KV_PAGE_BYTES)
    {
        return SPARK_STATUS_KV_LEASE_NOT_READY;
    }

    if (launch_descriptor->kv_materialized_backing_bytes != ((uint64_t)launch_descriptor->kv_view.represented_warm_pages + (uint64_t)launch_descriptor->kv_view.represented_cold_pages) * (uint64_t)SPARKPIPE_KV_PAGE_BYTES)
    {
        return SPARK_STATUS_KV_LEASE_NOT_READY;
    }

    if (launch_descriptor->kv_view.represented_hot_pages > 0u)
    {
        if (!launch_descriptor->kv_payload_handle.valid || launch_descriptor->kv_payload_handle.stage_id != launch_descriptor->stage_id || launch_descriptor->kv_payload_handle.model_lane != (SparkModelLaneKind)launch_descriptor->model_lane || launch_descriptor->kv_payload_handle.profile_id != (SparkPhysicalProfileId)launch_descriptor->profile_id)
        {
            return SPARK_STATUS_KV_LEASE_NOT_READY;
        }

        if (launch_descriptor->kv_payload_handle.hot_byte_count < launch_descriptor->kv_materialized_hot_bytes)
        {
            return SPARK_STATUS_KV_LEASE_NOT_READY;
        }

        if (launch_descriptor->kv_payload_handle.payload_checksum == 0 || launch_descriptor->kv_payload_handle.payload_checksum != SparkComputeKvPagePayloadHandleChecksum(&launch_descriptor->kv_payload_handle))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }

    if (launch_descriptor->region_set_checksum == 0 || launch_descriptor->region_set_checksum != SparkComputeStageMemoryRegionSetChecksum(&launch_descriptor->region_set))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkValidateStageMemoryRegionSet(&launch_descriptor->region_set) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->region_allocation_checksum == 0 || launch_descriptor->region_allocation_checksum != launch_descriptor->region_set.region_allocation_checksum)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->region_allocator_arena_count == 0 || launch_descriptor->region_allocator_allocation_count == 0 || launch_descriptor->region_allocator_failure_count != 0 || launch_descriptor->region_ownership_violation_count != 0 || launch_descriptor->region_bounds_violation_count != 0 || launch_descriptor->region_overlap_violation_count != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->persistent_pool_stale_handle_violation_count != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->stream_dependency_plan_checksum == 0 || launch_descriptor->stream_dependency_plan_checksum != SparkComputeStageStreamDependencyPlanChecksum(&launch_descriptor->stream_dependency_plan))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkValidateStageStreamDependencyPlan(&launch_descriptor->stream_dependency_plan) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->event_dependency_checksum == 0 || launch_descriptor->event_dependency_checksum != SparkComputeStageEventDependencyReportChecksum(&launch_descriptor->event_dependency_report))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (SparkValidateStageEventDependencyReport(&launch_descriptor->event_dependency_report) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->event_missing_recv_ready_count != 0 || launch_descriptor->event_missing_kv_ready_count != 0 || launch_descriptor->event_stale_violation_count != 0 || launch_descriptor->event_dependency_violation_count != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (launch_descriptor->descriptor_checksum != SparkComputeCudaStageLaunchDescriptorChecksum(launch_descriptor))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    return SPARK_STATUS_OK;
}

SparkStatus SparkExecuteDummyCudaStageFromDescriptor(const SparkCudaStageLaunchDescriptor *launch_descriptor, SparkCudaStageLaunchResult *launch_result)
{
    uint64_t checksum;
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

    checksum = launch_descriptor->input_activation_checksum;
    checksum = SparkMixU64(checksum, launch_descriptor->descriptor_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->fabric_tick);
    checksum = SparkMixU64(checksum, launch_descriptor->stage_id);
    checksum = SparkMixU64(checksum, launch_descriptor->model_lane);
    checksum = SparkMixU64(checksum, launch_descriptor->profile_id);
    checksum = SparkMixU64(checksum, launch_descriptor->slot_generation);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_view.represented_hot_pages);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialization_checksum);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialized_hot_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_materialized_backing_bytes);
    checksum = SparkMixU64(checksum, launch_descriptor->kv_bytes_loaded_to_hot);
    checksum = SparkMixU64(checksum, launch_descriptor->workspace_total_bytes);

    memset(launch_result, 0, sizeof(*launch_result));
    launch_result->descriptor_version = launch_descriptor->descriptor_version;
    launch_result->backend_kind = SPARK_CUDA_STAGE_BACKEND_C_DUMMY;
    launch_result->descriptor_validated = true;
    launch_result->descriptor_checksum = launch_descriptor->descriptor_checksum;
    launch_result->output_activation_checksum = checksum;
    launch_result->activation_bytes_observed = launch_descriptor->activation_payload_bytes;
    launch_result->workspace_bytes_observed = launch_descriptor->workspace_total_bytes;
    launch_result->activation_region_bytes_observed = launch_descriptor->region_set.total_activation_region_bytes;
    launch_result->workspace_region_bytes_observed = launch_descriptor->region_set.total_workspace_region_bytes;
    launch_result->kv_hot_region_bytes_observed = launch_descriptor->region_set.total_kv_hot_region_bytes;
    launch_result->kv_backing_region_bytes_observed = launch_descriptor->region_set.total_kv_backing_region_bytes;
    launch_result->kv_materialization_checksum = launch_descriptor->kv_materialization_checksum;
    launch_result->kv_materialized_hot_bytes = launch_descriptor->kv_materialized_hot_bytes;
    launch_result->kv_materialized_backing_bytes = launch_descriptor->kv_materialized_backing_bytes;
    launch_result->kv_bytes_loaded_to_hot = launch_descriptor->kv_bytes_loaded_to_hot;
    launch_result->kv_bytes_promoted_to_hot = launch_descriptor->kv_bytes_promoted_to_hot;
    launch_result->kv_bytes_copied_from_warm = launch_descriptor->kv_bytes_copied_from_warm;
    launch_result->kv_bytes_copied_from_cold = launch_descriptor->kv_bytes_copied_from_cold;
    launch_result->kv_bytes_evicted_from_hot = launch_descriptor->kv_bytes_evicted_from_hot;
    launch_result->kv_bytes_released_from_hot = launch_descriptor->kv_bytes_released_from_hot;
    launch_result->kv_copy_operation_count = launch_descriptor->kv_copy_operation_count;
    launch_result->kv_materialization_wait_count = launch_descriptor->kv_materialization_wait_count;
    launch_result->kv_materialization_ready_count = launch_descriptor->kv_materialization_ready_count;
    launch_result->kv_materialization_capacity_violation_count = launch_descriptor->kv_materialization_capacity_violation_count;
    launch_result->kv_materialization_missing_hot_page_violation_count = launch_descriptor->kv_materialization_missing_hot_page_violation_count;
    launch_result->region_set_checksum = launch_descriptor->region_set_checksum;
    launch_result->region_allocation_checksum = launch_descriptor->region_allocation_checksum;
    launch_result->region_allocator_arena_count = launch_descriptor->region_allocator_arena_count;
    launch_result->region_allocator_allocation_count = launch_descriptor->region_allocator_allocation_count;
    launch_result->region_allocator_failure_count = launch_descriptor->region_allocator_failure_count;
    launch_result->region_ownership_violation_count = launch_descriptor->region_ownership_violation_count;
    launch_result->region_bounds_violation_count = launch_descriptor->region_bounds_violation_count;
    launch_result->region_overlap_violation_count = launch_descriptor->region_overlap_violation_count;
    launch_result->stream_dependency_plan_checksum = launch_descriptor->stream_dependency_plan_checksum;
    launch_result->event_dependency_checksum = launch_descriptor->event_dependency_checksum;
    launch_result->event_dependency_count = launch_descriptor->event_dependency_count;
    launch_result->event_count = launch_descriptor->event_count;
    launch_result->event_missing_recv_ready_count = launch_descriptor->event_missing_recv_ready_count;
    launch_result->event_missing_kv_ready_count = launch_descriptor->event_missing_kv_ready_count;
    launch_result->event_stale_violation_count = launch_descriptor->event_stale_violation_count;
    launch_result->event_dependency_violation_count = launch_descriptor->event_dependency_violation_count;
    launch_result->event_ring_checksum_before = launch_descriptor->event_ring_checksum_before;
    launch_result->event_ring_checksum_after = launch_descriptor->event_ring_checksum_after;
    launch_result->persistent_pool_checksum = launch_descriptor->persistent_pool_checksum;
    launch_result->persistent_pool_reuse_count = launch_descriptor->persistent_pool_reuse_count;
    launch_result->persistent_pool_high_water_bytes = launch_descriptor->persistent_pool_high_water_bytes;
    launch_result->persistent_pool_stale_handle_violation_count = launch_descriptor->persistent_pool_stale_handle_violation_count;
    launch_result->active_slots_observed = launch_descriptor->active_slot_count;
    launch_result->hot_kv_pages_observed = launch_descriptor->kv_view.represented_hot_pages;
    launch_result->backend_validation_checksum = SparkMixU64(checksum, launch_result->backend_kind);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_materialization_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_materialized_hot_bytes);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_materialized_backing_bytes);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_loaded_to_hot);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_promoted_to_hot);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_copied_from_warm);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_copied_from_cold);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_evicted_from_hot);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_bytes_released_from_hot);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_copy_operation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_materialization_wait_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->kv_materialization_ready_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_set_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_allocation_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_allocator_arena_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_allocator_allocation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_allocator_failure_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_ownership_violation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_bounds_violation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->region_overlap_violation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->stream_dependency_plan_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_dependency_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_dependency_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_missing_recv_ready_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_missing_kv_ready_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_stale_violation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_dependency_violation_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_ring_checksum_before);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->event_ring_checksum_after);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->persistent_pool_checksum);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->persistent_pool_reuse_count);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->persistent_pool_high_water_bytes);
    launch_result->backend_validation_checksum = SparkMixU64(launch_result->backend_validation_checksum, launch_result->persistent_pool_stale_handle_violation_count);

    return SPARK_STATUS_OK;
}

static SparkStatus SparkRunLaunchDescriptorOnBackend(const SparkCudaStageLaunchDescriptor *launch_descriptor, SparkCudaStageBackendKind backend_kind, SparkCudaStageLaunchResult *launch_result)
{
    if (launch_descriptor == 0 || launch_result == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    switch (backend_kind)
    {
        case SPARK_CUDA_STAGE_BACKEND_C_DUMMY:
        {
            return SparkExecuteDummyCudaStageFromDescriptor(launch_descriptor, launch_result);
        }
        case SPARK_CUDA_STAGE_BACKEND_OPTIONAL_CUDA_DUMMY:
        {
#ifdef SPARKPIPE_ENABLE_CUDA_DUMMY
            return SparkRunOptionalCudaDummyKernelDescriptor(launch_descriptor, launch_result);
#else
            (void)launch_descriptor;
            (void)launch_result;
            return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
#endif
        }
        default:
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
}

SparkStatus SparkExecuteCudaStageWithBackend(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageBackendKind backend_kind, SparkCudaStageExecutionOutput *execution_output)
{
    SparkCudaStageLaunchDescriptor launch_descriptor;
    SparkCudaStageLaunchResult launch_result;
    SparkStatus status;

    if (execution_input == 0 || execution_output == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkBuildCudaStageLaunchDescriptor(execution_input, &launch_descriptor);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkRunLaunchDescriptorOnBackend(&launch_descriptor, backend_kind, &launch_result);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkInitializeActivationBuffer(execution_input->output_activation,
                                             execution_input->input_activation->buffer_id + 1u,
                                             &execution_input->input_activation->layout,
                                             execution_input->active_slot_mask,
                                             launch_result.output_activation_checksum);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    execution_input->output_activation->payload_checksum = launch_result.output_activation_checksum;

    memset(execution_output, 0, sizeof(*execution_output));
    execution_output->output_activation_checksum = launch_result.output_activation_checksum;
    execution_output->activation_bytes_observed = launch_result.activation_bytes_observed;
    execution_output->workspace_bytes_observed = launch_result.workspace_bytes_observed;
    execution_output->activation_region_bytes_observed = launch_result.activation_region_bytes_observed;
    execution_output->workspace_region_bytes_observed = launch_result.workspace_region_bytes_observed;
    execution_output->kv_hot_region_bytes_observed = launch_result.kv_hot_region_bytes_observed;
    execution_output->kv_backing_region_bytes_observed = launch_result.kv_backing_region_bytes_observed;
    execution_output->kv_materialization_checksum = launch_result.kv_materialization_checksum;
    execution_output->kv_materialized_hot_bytes = launch_result.kv_materialized_hot_bytes;
    execution_output->kv_materialized_backing_bytes = launch_result.kv_materialized_backing_bytes;
    execution_output->kv_bytes_loaded_to_hot = launch_result.kv_bytes_loaded_to_hot;
    execution_output->kv_bytes_promoted_to_hot = launch_result.kv_bytes_promoted_to_hot;
    execution_output->kv_bytes_copied_from_warm = launch_result.kv_bytes_copied_from_warm;
    execution_output->kv_bytes_copied_from_cold = launch_result.kv_bytes_copied_from_cold;
    execution_output->kv_bytes_evicted_from_hot = launch_result.kv_bytes_evicted_from_hot;
    execution_output->kv_bytes_released_from_hot = launch_result.kv_bytes_released_from_hot;
    execution_output->kv_copy_operation_count = launch_result.kv_copy_operation_count;
    execution_output->kv_materialization_wait_count = launch_result.kv_materialization_wait_count;
    execution_output->kv_materialization_ready_count = launch_result.kv_materialization_ready_count;
    execution_output->kv_materialization_capacity_violation_count = launch_result.kv_materialization_capacity_violation_count;
    execution_output->kv_materialization_missing_hot_page_violation_count = launch_result.kv_materialization_missing_hot_page_violation_count;
    execution_output->region_set_checksum = launch_result.region_set_checksum;
    execution_output->region_allocation_checksum = launch_result.region_allocation_checksum;
    execution_output->region_allocator_arena_count = launch_result.region_allocator_arena_count;
    execution_output->region_allocator_allocation_count = launch_result.region_allocator_allocation_count;
    execution_output->region_allocator_failure_count = launch_result.region_allocator_failure_count;
    execution_output->region_ownership_violation_count = launch_result.region_ownership_violation_count;
    execution_output->region_bounds_violation_count = launch_result.region_bounds_violation_count;
    execution_output->region_overlap_violation_count = launch_result.region_overlap_violation_count;
    execution_output->stream_dependency_plan_checksum = launch_result.stream_dependency_plan_checksum;
    execution_output->event_dependency_checksum = launch_result.event_dependency_checksum;
    execution_output->event_dependency_count = launch_result.event_dependency_count;
    execution_output->event_count = launch_result.event_count;
    execution_output->event_missing_recv_ready_count = launch_result.event_missing_recv_ready_count;
    execution_output->event_missing_kv_ready_count = launch_result.event_missing_kv_ready_count;
    execution_output->event_stale_violation_count = launch_result.event_stale_violation_count;
    execution_output->event_dependency_violation_count = launch_result.event_dependency_violation_count;
    execution_output->event_ring_checksum_before = launch_result.event_ring_checksum_before;
    execution_output->event_ring_checksum_after = launch_result.event_ring_checksum_after;
    execution_output->persistent_pool_checksum = launch_result.persistent_pool_checksum;
    execution_output->persistent_pool_reuse_count = launch_result.persistent_pool_reuse_count;
    execution_output->persistent_pool_high_water_bytes = launch_result.persistent_pool_high_water_bytes;
    execution_output->persistent_pool_stale_handle_violation_count = launch_result.persistent_pool_stale_handle_violation_count;
    execution_output->active_slots_observed = launch_result.active_slots_observed;
    execution_output->hot_kv_pages_observed = launch_result.hot_kv_pages_observed;
    execution_output->descriptor_checksum = launch_result.descriptor_checksum;
    execution_output->backend_validation_checksum = launch_result.backend_validation_checksum;
    execution_output->backend_kind = backend_kind;
    execution_output->descriptor_validated = launch_result.descriptor_validated;

    return SPARK_STATUS_OK;
}

SparkStatus SparkExecuteDummyCudaStage(const SparkCudaStageExecutionInput *execution_input, SparkCudaStageExecutionOutput *execution_output)
{
    return SparkExecuteCudaStageWithBackend(execution_input, SPARK_CUDA_STAGE_BACKEND_C_DUMMY, execution_output);
}
