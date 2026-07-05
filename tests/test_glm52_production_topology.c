#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_production_topology.h"

static void SparkTestInitializePp13StagePlan(SparkGlm52StagePlan *stage_plan)
{
    uint32_t stage_index;

    memset(stage_plan, 0, sizeof(*stage_plan));
    stage_plan->abi_version = SPARK_GLM52_STAGE_PLAN_ABI_VERSION;
    stage_plan->descriptor_bytes = SPARK_GLM52_STAGE_PLAN_DESCRIPTOR_BYTES;
    stage_plan->stage_count = 13u;
    for (stage_index = 0u; stage_index < 13u; ++stage_index)
    {
        stage_plan->stages[stage_index].first_layer_index = stage_index * 6u;
        stage_plan->stages[stage_index].layer_count = 6u;
        stage_plan->stages[stage_index].flags =
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN |
            SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_OUTPUT_HIDDEN;
    }
    stage_plan->stages[0u].flags |=
        SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_DENSE_PREFIX;
    stage_plan->stages[12u].flags =
        SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_INPUT_HIDDEN |
        SPARK_GLM52_STAGE_PLAN_STAGE_FLAG_FINAL_TOKEN;
}

static const SparkGlm52ProductionTopologyIndexShareSideBand *
SparkTestFindSidebandForSourceLayer(
    const SparkGlm52ProductionTopology *topology,
    uint32_t source_layer_index)
{
    uint32_t sideband_index;

    for (sideband_index = 0u;
         sideband_index < topology->indexshare_sideband_count;
         ++sideband_index)
    {
        if (topology->indexshare_sidebands[sideband_index].source_layer_index ==
            source_layer_index)
        {
            return &topology->indexshare_sidebands[sideband_index];
        }
    }
    return 0;
}

static void SparkTestGlm52IndexShareSchedule(void)
{
    uint32_t group_end_layer_exclusive;

    assert(SparkGlm52DsaIndexShareSourceLayer(0u) == 0u);
    assert(SparkGlm52DsaIndexShareSourceLayer(1u) == 0u);
    assert(SparkGlm52DsaIndexShareSourceLayer(3u) == 0u);
    assert(SparkGlm52DsaIndexShareSourceLayer(4u) == 4u);
    assert(SparkGlm52DsaIndexShareSourceLayer(5u) == 4u);
    assert(SparkGlm52DsaIndexShareSourceLayer(6u) == 4u);
    assert(SparkGlm52DsaIndexShareSourceLayer(7u) == 4u);
    assert(SparkGlm52DsaIndexShareSourceLayer(78u) == UINT32_MAX);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        4u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 8u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        5u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 8u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        77u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 78u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        78u,
        &group_end_layer_exclusive) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52ProductionTopologyPp13Sideband(void)
{
    SparkGlm52StagePlan stage_plan;
    SparkGlm52ProductionTopology topology;
    const SparkGlm52ProductionTopologyIndexShareSideBand *sideband;
    char error_buffer[256];

    SparkTestInitializePp13StagePlan(&stage_plan);
    assert(SparkGlm52ProductionTopologyBuild(
        &stage_plan,
        64u,
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT,
        16u,
        576u,
        &topology,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(topology.stage_count == 13u);
    assert(topology.active_sequence_capacity == 64u);
    assert(topology.selected_token_count ==
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT);
    assert((topology.topology_flags &
        SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE) != 0u);
    assert(topology.mla_cache_element_count == 576u);
    sideband = SparkTestFindSidebandForSourceLayer(&topology, 4u);
    assert(sideband != 0);
    assert(sideband->source_layer_index == 4u);
    assert(sideband->group_end_layer_exclusive == 8u);
    assert(sideband->export_stage_index == 0u);
    assert(sideband->import_stage_index == 1u);
    assert(sideband->first_imported_consumer_layer_index == 6u);
    assert(sideband->imported_consumer_layer_count == 2u);
    assert(sideband->payload_bytes == 524288u);
    assert(topology.stages[0u].exported_sideband_count != 0u);
    assert(topology.stages[1u].imported_sideband_count != 0u);
    assert(topology.stages[0u].first_layer_index == 0u);
    assert(topology.stages[0u].layer_count == 6u);
    assert(topology.stages[1u].first_layer_index == 6u);
    assert(topology.stages[1u].layer_count == 6u);
    topology.topology_flags &=
        ~SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE;
    assert(SparkGlm52ProductionTopologyValidate(
        &topology,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52ProductionTopologyRejectsBadDimensions(void)
{
    SparkGlm52StagePlan stage_plan;
    SparkGlm52ProductionTopology topology;
    char error_buffer[256];

    SparkTestInitializePp13StagePlan(&stage_plan);
    assert(SparkGlm52ProductionTopologyBuild(
        &stage_plan,
        0u,
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT,
        16u,
        576u,
        &topology,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestGlm52IndexShareSchedule();
    SparkTestGlm52ProductionTopologyPp13Sideband();
    SparkTestGlm52ProductionTopologyRejectsBadDimensions();
    return 0;
}
