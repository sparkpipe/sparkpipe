#ifndef SPARKPIPE_SPARK_WORKSPACE_H
#define SPARKPIPE_SPARK_WORKSPACE_H

#include "sparkpipe/spark_activation_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkStageWorkspaceReservation
{
    uint32_t stage_id;
    SparkModelLaneKind model_lane;
    SparkPhysicalProfileId profile_id;
    uint64_t activation_bytes;
    uint64_t kv_scratch_bytes;
    uint64_t graph_scratch_bytes;
    uint64_t moe_scratch_bytes;
    uint64_t alignment_bytes;
    uint64_t total_bytes;
} SparkStageWorkspaceReservation;

typedef struct SparkWorkspaceReservationTable
{
    uint32_t stage_count;
    uint64_t total_reserved_bytes;
    uint64_t maximum_stage_reserved_bytes;
    SparkStageWorkspaceReservation reservations[SPARKPIPE_MAX_STAGES];
} SparkWorkspaceReservationTable;

SparkStatus SparkBuildStageWorkspaceReservation(SparkStageWorkspaceReservation *reservation, uint32_t stage_id, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint32_t hidden_size, uint32_t local_layer_count, bool moe_enabled);
SparkStatus SparkBuildWorkspaceReservationTable(SparkWorkspaceReservationTable *reservation_table, uint32_t stage_count, SparkModelLaneKind model_lane, SparkPhysicalProfileId profile_id, uint32_t hidden_size, uint32_t local_layer_count, bool moe_enabled);

#ifdef __cplusplus
}
#endif

#endif
