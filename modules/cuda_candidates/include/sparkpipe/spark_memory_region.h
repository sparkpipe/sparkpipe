#ifndef SPARKPIPE_SPARK_MEMORY_REGION_H
#define SPARKPIPE_SPARK_MEMORY_REGION_H

#include "sparkpipe/spark_activation_layout.h"
#include "sparkpipe/spark_kv_page_table.h"
#include "sparkpipe/spark_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SparkMemoryRegionKind
{
    SPARK_MEMORY_REGION_ACTIVATION_INPUT = 1,
    SPARK_MEMORY_REGION_ACTIVATION_OUTPUT = 2,
    SPARK_MEMORY_REGION_WORKSPACE = 3,
    SPARK_MEMORY_REGION_KV_HOT = 4,
    SPARK_MEMORY_REGION_KV_WARM = 5,
    SPARK_MEMORY_REGION_KV_COLD = 6
} SparkMemoryRegionKind;

typedef enum SparkMemoryRegionResidencyKind
{
    SPARK_MEMORY_REGION_RESIDENCY_HOST_SIMULATED = 1,
    SPARK_MEMORY_REGION_RESIDENCY_DEVICE_FUTURE = 2,
    SPARK_MEMORY_REGION_RESIDENCY_PINNED_HOST_FUTURE = 3,
    SPARK_MEMORY_REGION_RESIDENCY_NVME_FUTURE = 4
} SparkMemoryRegionResidencyKind;

typedef enum SparkMemoryRegionAccessFlags
{
    SPARK_MEMORY_REGION_ACCESS_READ = 1u,
    SPARK_MEMORY_REGION_ACCESS_WRITE = 2u,
    SPARK_MEMORY_REGION_ACCESS_READ_WRITE = 3u
} SparkMemoryRegionAccessFlags;

typedef enum SparkMemoryArenaKind
{
    SPARK_MEMORY_ARENA_ACTIVATION_INPUT = 1,
    SPARK_MEMORY_ARENA_ACTIVATION_OUTPUT = 2,
    SPARK_MEMORY_ARENA_WORKSPACE = 3,
    SPARK_MEMORY_ARENA_KV_HOT = 4,
    SPARK_MEMORY_ARENA_KV_BACKING = 5
} SparkMemoryArenaKind;

typedef struct SparkMemoryRegionDescriptor
{
    bool initialized;
    SparkMemoryRegionKind region_kind;
    SparkMemoryRegionResidencyKind residency_kind;
    uint32_t owner_stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint64_t region_id;
    SparkMemoryArenaKind arena_kind;
    uint64_t arena_id;
    uint64_t arena_capacity_bytes;
    uint64_t allocation_generation;
    uint64_t base_offset_bytes;
    uint64_t byte_count;
    uint64_t range_end_bytes;
    uint64_t alignment_bytes;
    uint32_t access_flags;
    uint64_t content_checksum;
    uint64_t descriptor_checksum;
} SparkMemoryRegionDescriptor;

typedef struct SparkStageMemoryRegionSet
{
    SparkMemoryRegionDescriptor activation_input_region;
    SparkMemoryRegionDescriptor activation_output_region;
    SparkMemoryRegionDescriptor workspace_region;
    SparkMemoryRegionDescriptor kv_hot_region;
    SparkMemoryRegionDescriptor kv_warm_region;
    SparkMemoryRegionDescriptor kv_cold_region;
    uint64_t total_activation_region_bytes;
    uint64_t total_workspace_region_bytes;
    uint64_t total_kv_hot_region_bytes;
    uint64_t total_kv_backing_region_bytes;
    uint64_t region_allocation_checksum;
    uint32_t ownership_violation_count;
    uint32_t bounds_violation_count;
    uint32_t overlap_violation_count;
    uint64_t region_set_checksum;
} SparkStageMemoryRegionSet;

typedef struct SparkStageStreamDependencyPlan
{
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t fabric_tick;
    uint64_t generation;
    uint64_t recv_ready_event_id;
    uint64_t kv_ready_event_id;
    uint64_t compute_done_event_id;
    uint64_t send_ready_event_id;
    uint32_t stream_count;
    uint32_t dependency_count;
    bool requires_recv_ready;
    bool requires_kv_ready;
    bool emits_send_ready;
    uint64_t dependency_plan_checksum;
} SparkStageStreamDependencyPlan;

uint64_t SparkBuildMemoryRegionId(SparkMemoryRegionKind region_kind, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, uint32_t local_index);
uint64_t SparkComputeMemoryRegionDescriptorChecksum(const SparkMemoryRegionDescriptor *region_descriptor);
SparkMemoryArenaKind SparkGetDefaultArenaKindForMemoryRegion(SparkMemoryRegionKind region_kind);
SparkStatus SparkBuildMemoryRegionDescriptor(SparkMemoryRegionDescriptor *region_descriptor, SparkMemoryRegionKind region_kind, SparkMemoryRegionResidencyKind residency_kind, uint32_t owner_stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t fabric_tick, uint64_t generation, uint32_t local_index, uint64_t base_offset_bytes, uint64_t byte_count, uint64_t alignment_bytes, uint32_t access_flags, uint64_t content_checksum);
SparkStatus SparkBuildMemoryRegionDescriptorWithArena(SparkMemoryRegionDescriptor *region_descriptor, SparkMemoryRegionKind region_kind, SparkMemoryRegionResidencyKind residency_kind, SparkMemoryArenaKind arena_kind, uint64_t arena_id, uint64_t arena_capacity_bytes, uint64_t allocation_generation, uint32_t owner_stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t fabric_tick, uint64_t generation, uint32_t local_index, uint64_t base_offset_bytes, uint64_t byte_count, uint64_t alignment_bytes, uint32_t access_flags, uint64_t content_checksum);
bool SparkMemoryRegionDescriptorsOverlap(const SparkMemoryRegionDescriptor *left_region, const SparkMemoryRegionDescriptor *right_region);
SparkStatus SparkValidateMemoryRegionDescriptor(const SparkMemoryRegionDescriptor *region_descriptor);
SparkStatus SparkBuildStageMemoryRegionSet(SparkStageMemoryRegionSet *region_set, uint64_t fabric_tick, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, const SparkActivationBuffer *input_activation, const SparkActivationBuffer *output_activation, const SparkStageWorkspaceReservation *workspace_reservation, const SparkKvPageTable *kv_page_table);
uint64_t SparkComputeStageMemoryRegionSetChecksum(const SparkStageMemoryRegionSet *region_set);
SparkStatus SparkValidateStageMemoryRegionSet(const SparkStageMemoryRegionSet *region_set);
SparkStatus SparkBuildStageStreamDependencyPlan(SparkStageStreamDependencyPlan *dependency_plan, uint64_t fabric_tick, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint64_t generation, const SparkStageMemoryRegionSet *region_set);
uint64_t SparkComputeStageStreamDependencyPlanChecksum(const SparkStageStreamDependencyPlan *dependency_plan);
SparkStatus SparkValidateStageStreamDependencyPlan(const SparkStageStreamDependencyPlan *dependency_plan);

#ifdef __cplusplus
}
#endif

#endif
