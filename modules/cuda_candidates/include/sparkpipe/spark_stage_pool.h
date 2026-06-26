#ifndef SPARKPIPE_SPARK_STAGE_POOL_H
#define SPARKPIPE_SPARK_STAGE_POOL_H

#include "sparkpipe/spark_region_allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkStagePersistentPool
{
    bool initialized;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t pool_generation;
    uint32_t activation_buffer_count;
    uint64_t activation_input_stride_bytes;
    uint64_t activation_output_stride_bytes;
    uint64_t kv_hot_capacity_pages;
    uint64_t kv_backing_capacity_pages;
    uint64_t launch_count;
    uint64_t reuse_count;
    uint64_t high_water_bytes;
    uint32_t stale_handle_violation_count;
    uint32_t owner_violation_count;
    uint32_t bounds_violation_count;
    uint32_t overlap_violation_count;
    SparkMemoryArena activation_input_arena;
    SparkMemoryArena activation_output_arena;
    SparkMemoryArena workspace_arena;
    SparkMemoryArena kv_hot_arena;
    SparkMemoryArena kv_backing_arena;
    uint64_t pool_checksum;
} SparkStagePersistentPool;

typedef struct SparkStagePersistentPoolReport
{
    uint32_t arena_count;
    uint32_t region_count;
    uint32_t allocation_failure_count;
    uint32_t stale_handle_violation_count;
    uint32_t owner_violation_count;
    uint32_t bounds_violation_count;
    uint32_t overlap_violation_count;
    uint64_t pool_generation;
    uint64_t launch_count;
    uint64_t reuse_count;
    uint64_t high_water_bytes;
    uint64_t activation_input_arena_id;
    uint64_t activation_output_arena_id;
    uint64_t workspace_arena_id;
    uint64_t kv_hot_arena_id;
    uint64_t kv_backing_arena_id;
    uint64_t pool_checksum;
    uint64_t region_allocation_checksum;
} SparkStagePersistentPoolReport;

uint64_t SparkComputeStagePersistentPoolChecksum(const SparkStagePersistentPool *stage_pool);
SparkStatus SparkInitializeStagePersistentPool(SparkStagePersistentPool *stage_pool, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t pool_generation, const SparkActivationLayout *activation_layout, const SparkStageWorkspaceReservation *workspace_reservation, uint64_t kv_hot_capacity_pages, uint64_t kv_backing_capacity_pages);
SparkStatus SparkValidateStagePersistentPool(const SparkStagePersistentPool *stage_pool);
SparkStatus SparkEnsureStagePersistentPool(SparkStagePersistentPool *stage_pool, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t requested_generation, const SparkActivationLayout *activation_layout, const SparkStageWorkspaceReservation *workspace_reservation, const SparkKvPageTable *kv_page_table);
SparkStatus SparkBuildPersistentPoolStageMemoryRegionSet(SparkStageMemoryRegionSet *region_set, SparkStagePersistentPool *stage_pool, SparkStagePersistentPoolReport *pool_report, uint64_t fabric_tick, uint64_t launch_generation, const SparkActivationBuffer *input_activation, const SparkActivationBuffer *output_activation, const SparkStageWorkspaceReservation *workspace_reservation, const SparkKvPageTable *kv_page_table);
SparkStatus SparkValidateStagePersistentPoolReport(const SparkStagePersistentPoolReport *pool_report);

#ifdef __cplusplus
}
#endif

#endif
