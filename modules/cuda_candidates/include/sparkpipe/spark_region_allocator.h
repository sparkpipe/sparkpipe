#ifndef SPARKPIPE_SPARK_REGION_ALLOCATOR_H
#define SPARKPIPE_SPARK_REGION_ALLOCATOR_H

#include "sparkpipe/spark_memory_region.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkMemoryArena
{
    bool initialized;
    SparkMemoryArenaKind arena_kind;
    uint64_t arena_id;
    uint32_t owner_stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t generation;
    uint64_t capacity_bytes;
    uint64_t alignment_bytes;
    uint64_t next_offset_bytes;
    uint64_t allocated_bytes;
    uint32_t allocation_count;
    uint32_t allocation_failure_count;
    uint64_t arena_checksum;
} SparkMemoryArena;

typedef struct SparkMemoryArenaAllocation
{
    bool allocated;
    SparkMemoryArenaKind arena_kind;
    uint64_t arena_id;
    uint64_t allocation_generation;
    uint64_t base_offset_bytes;
    uint64_t byte_count;
    uint64_t range_end_bytes;
    uint64_t alignment_bytes;
    uint64_t allocation_checksum;
} SparkMemoryArenaAllocation;

typedef struct SparkStageRegionArenaSet
{
    bool initialized;
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    SparkMemoryArena activation_input_arena;
    SparkMemoryArena activation_output_arena;
    SparkMemoryArena workspace_arena;
    SparkMemoryArena kv_hot_arena;
    SparkMemoryArena kv_backing_arena;
    uint32_t arena_count;
    uint32_t allocation_count;
    uint32_t allocation_failure_count;
    uint64_t allocated_region_bytes;
    uint64_t arena_set_checksum;
} SparkStageRegionArenaSet;

typedef struct SparkStageRegionAllocationReport
{
    uint32_t arena_count;
    uint32_t allocation_count;
    uint32_t allocation_failure_count;
    uint32_t ownership_violation_count;
    uint32_t bounds_violation_count;
    uint32_t overlap_violation_count;
    uint64_t allocated_region_bytes;
    uint64_t activation_input_arena_id;
    uint64_t activation_output_arena_id;
    uint64_t workspace_arena_id;
    uint64_t kv_hot_arena_id;
    uint64_t kv_backing_arena_id;
    uint64_t arena_set_checksum;
    uint64_t allocation_checksum;
} SparkStageRegionAllocationReport;

uint64_t SparkBuildMemoryArenaId(SparkMemoryArenaKind arena_kind, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation);
uint64_t SparkComputeMemoryArenaChecksum(const SparkMemoryArena *memory_arena);
SparkStatus SparkInitializeMemoryArena(SparkMemoryArena *memory_arena, SparkMemoryArenaKind arena_kind, uint32_t owner_stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, uint64_t capacity_bytes, uint64_t alignment_bytes);
SparkStatus SparkAllocateMemoryArenaRegion(SparkMemoryArena *memory_arena, uint64_t byte_count, uint64_t alignment_bytes, SparkMemoryArenaAllocation *allocation);
SparkStatus SparkBuildStageRegionArenaSet(SparkStageRegionArenaSet *arena_set, uint64_t fabric_tick, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, const SparkActivationBuffer *input_activation, const SparkStageWorkspaceReservation *workspace_reservation, const SparkKvPageTable *kv_page_table);
uint64_t SparkComputeStageRegionArenaSetChecksum(const SparkStageRegionArenaSet *arena_set);
SparkStatus SparkValidateStageRegionArenaSet(const SparkStageRegionArenaSet *arena_set);
SparkStatus SparkBuildAllocatorBackedStageMemoryRegionSet(SparkStageMemoryRegionSet *region_set, SparkStageRegionArenaSet *arena_set, SparkStageRegionAllocationReport *allocation_report, uint64_t fabric_tick, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, const SparkActivationBuffer *input_activation, const SparkActivationBuffer *output_activation, const SparkStageWorkspaceReservation *workspace_reservation, const SparkKvPageTable *kv_page_table);
SparkStatus SparkValidateStageRegionAllocationReport(const SparkStageRegionAllocationReport *allocation_report);

#ifdef __cplusplus
}
#endif

#endif
