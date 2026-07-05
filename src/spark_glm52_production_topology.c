#include "sparkpipe/spark_glm52_production_topology.h"

#include <stdio.h>
#include <string.h>

static SparkStatus SparkGlm52ProductionTopologyReport(
    char *error_buffer,
    uint32_t error_buffer_bytes,
    SparkStatus status,
    const char *message)
{
    if (error_buffer != 0 && error_buffer_bytes != 0u)
    {
        if (message == 0)
        {
            error_buffer[0] = '\0';
        }
        else
        {
            (void)snprintf(error_buffer, error_buffer_bytes, "%s", message);
        }
    }
    return status;
}

static uint32_t SparkGlm52ProductionTopologyMinimumU32(uint32_t left,uint32_t right)
{
    return left < right ? left : right;
}

static uint32_t SparkGlm52ProductionTopologyStageEndLayer(
    const SparkGlm52StagePlanStage *stage)
{
    return stage->first_layer_index + stage->layer_count;
}

static uint32_t SparkGlm52ProductionTopologyStageContainsLayer(
    const SparkGlm52StagePlanStage *stage,
    uint32_t layer_index)
{
    return layer_index >= stage->first_layer_index &&
        layer_index < SparkGlm52ProductionTopologyStageEndLayer(stage);
}

static SparkStatus SparkGlm52ProductionTopologyFindStageForLayer(
    const SparkGlm52StagePlan *stage_plan,
    uint32_t layer_index,
    uint32_t *stage_index_out)
{
    uint32_t stage_index;

    if (stage_plan == 0 || stage_index_out == 0 ||
        layer_index >= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (stage_index = 0u; stage_index < stage_plan->stage_count; ++stage_index)
    {
        if (SparkGlm52ProductionTopologyStageContainsLayer(
                &stage_plan->stages[stage_index],
                layer_index))
        {
            *stage_index_out = stage_index;
            return SPARK_STATUS_OK;
        }
    }
    return SPARK_STATUS_NOT_FOUND;
}

static SparkStatus SparkGlm52ProductionTopologyInitializeStages(
    const SparkGlm52StagePlan *stage_plan,
    SparkGlm52ProductionTopology *topology)
{
    uint32_t stage_index;

    if (stage_plan == 0 || topology == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    topology->stage_count = stage_plan->stage_count;
    for (stage_index = 0u; stage_index < stage_plan->stage_count; ++stage_index)
    {
        topology->stages[stage_index].first_layer_index =
            stage_plan->stages[stage_index].first_layer_index;
        topology->stages[stage_index].layer_count =
            stage_plan->stages[stage_index].layer_count;
        topology->stages[stage_index].stage_plan_flags =
            stage_plan->stages[stage_index].flags;
        topology->stages[stage_index].first_exported_sideband_index = UINT32_MAX;
        topology->stages[stage_index].first_imported_sideband_index = UINT32_MAX;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52ProductionTopologyAppendSideband(
    SparkGlm52ProductionTopology *topology,
    uint32_t source_layer_index,
    uint32_t group_end_layer_exclusive,
    uint32_t export_stage_index,
    uint32_t import_stage_index,
    uint32_t first_imported_consumer_layer_index)
{
    SparkGlm52ProductionTopologyIndexShareSideBand *sideband;
    SparkGlm52ProductionTopologyStage *export_stage;
    SparkGlm52ProductionTopologyStage *import_stage;
    uint32_t sideband_index;

    if (topology == 0 || export_stage_index >= topology->stage_count ||
        import_stage_index >= topology->stage_count ||
        topology->indexshare_sideband_count >=
            SPARK_GLM52_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    sideband_index = topology->indexshare_sideband_count;
    sideband = &topology->indexshare_sidebands[sideband_index];
    memset(sideband, 0, sizeof(*sideband));
    sideband->source_layer_index = source_layer_index;
    sideband->group_end_layer_exclusive = group_end_layer_exclusive;
    sideband->export_stage_index = export_stage_index;
    sideband->import_stage_index = import_stage_index;
    sideband->first_imported_consumer_layer_index =
        first_imported_consumer_layer_index;
    sideband->imported_consumer_layer_count =
        group_end_layer_exclusive - first_imported_consumer_layer_index;
    sideband->selected_token_count = topology->selected_token_count;
    sideband->active_sequence_capacity = topology->active_sequence_capacity;
    sideband->payload_bytes =
        (uint64_t)topology->active_sequence_capacity *
        (uint64_t)topology->selected_token_count * sizeof(uint32_t);
    sideband->flags =
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_SELECTED_TOKEN_INDICES |
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_DEVICE_TO_DEVICE;
    export_stage = &topology->stages[export_stage_index];
    import_stage = &topology->stages[import_stage_index];
    if (export_stage->exported_sideband_count == 0u)
    {
        export_stage->first_exported_sideband_index = sideband_index;
    }
    if (import_stage->imported_sideband_count == 0u)
    {
        import_stage->first_imported_sideband_index = sideband_index;
    }
    export_stage->exported_sideband_count += 1u;
    import_stage->imported_sideband_count += 1u;
    topology->indexshare_sideband_count += 1u;
    return SPARK_STATUS_OK;
}

uint32_t SparkGlm52DsaIndexShareSourceLayer(uint32_t layer_index)
{
    if (layer_index >= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT)
    {
        return UINT32_MAX;
    }
    return layer_index -
        (layer_index % SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT);
}

SparkStatus SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
    uint32_t layer_index,
    uint32_t *group_end_layer_exclusive_out)
{
    uint32_t source_layer_index;

    if (group_end_layer_exclusive_out == 0 ||
        layer_index >= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    source_layer_index = SparkGlm52DsaIndexShareSourceLayer(layer_index);
    *group_end_layer_exclusive_out = SparkGlm52ProductionTopologyMinimumU32(
        source_layer_index +
            SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT,
        SPARK_GLM52_STAGE_PLAN_LAYER_COUNT);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52ProductionTopologyBuild(
    const SparkGlm52StagePlan *stage_plan,
    uint32_t active_sequence_capacity,
    uint32_t selected_token_count,
    uint32_t kv_block_token_count,
    uint32_t mla_cache_element_count,
    SparkGlm52ProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    SparkStatus status;
    uint32_t source_layer_index;
    uint32_t group_end_layer_exclusive;
    uint32_t boundary_layer_index;
    uint32_t boundary_stage_index;
    uint32_t export_stage_index;

    if (topology == 0 || active_sequence_capacity == 0u ||
        selected_token_count == 0u || kv_block_token_count == 0u ||
        mla_cache_element_count == 0u)
    {
        return SparkGlm52ProductionTopologyReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "production topology arguments are invalid");
    }
    status = SparkGlm52StagePlanValidate(
        stage_plan,
        error_buffer,
        error_buffer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    memset(topology, 0, sizeof(*topology));
    topology->abi_version = SPARK_GLM52_PRODUCTION_TOPOLOGY_ABI_VERSION;
    topology->descriptor_bytes = SPARK_GLM52_PRODUCTION_TOPOLOGY_DESCRIPTOR_BYTES;
    topology->topology_flags =
        SPARK_GLM52_PRODUCTION_TOPOLOGY_PRODUCTION_REQUIRED_FLAGS;
    topology->selected_token_count = selected_token_count;
    topology->active_sequence_capacity = active_sequence_capacity;
    topology->kv_block_token_count = kv_block_token_count;
    topology->mla_cache_element_count = mla_cache_element_count;
    status = SparkGlm52ProductionTopologyInitializeStages(stage_plan, topology);
    if (status != SPARK_STATUS_OK)
    {
        return SparkGlm52ProductionTopologyReport(
            error_buffer,
            error_buffer_bytes,
            status,
            "failed to initialize topology stages");
    }
    for (source_layer_index = 0u;
         source_layer_index < SPARK_GLM52_STAGE_PLAN_LAYER_COUNT;
         source_layer_index +=
            SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT)
    {
        group_end_layer_exclusive = SparkGlm52ProductionTopologyMinimumU32(
            source_layer_index +
                SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT,
            SPARK_GLM52_STAGE_PLAN_LAYER_COUNT);
        status = SparkGlm52ProductionTopologyFindStageForLayer(
            stage_plan,
            source_layer_index,
            &export_stage_index);
        if (status != SPARK_STATUS_OK)
        {
            return SparkGlm52ProductionTopologyReport(
                error_buffer,
                error_buffer_bytes,
                status,
                "IndexShare source layer is outside stage plan");
        }
        for (boundary_stage_index = 1u;
             boundary_stage_index < stage_plan->stage_count;
             ++boundary_stage_index)
        {
            boundary_layer_index =
                stage_plan->stages[boundary_stage_index].first_layer_index;
            if (source_layer_index < boundary_layer_index &&
                boundary_layer_index < group_end_layer_exclusive)
            {
                status = SparkGlm52ProductionTopologyAppendSideband(
                    topology,
                    source_layer_index,
                    group_end_layer_exclusive,
                    export_stage_index,
                    boundary_stage_index,
                    boundary_layer_index);
                if (status != SPARK_STATUS_OK)
                {
                    return SparkGlm52ProductionTopologyReport(
                        error_buffer,
                        error_buffer_bytes,
                        status,
                        "too many IndexShare stage-boundary sidebands");
                }
            }
        }
    }
    return SparkGlm52ProductionTopologyValidate(
        topology,
        error_buffer,
        error_buffer_bytes);
}

SparkStatus SparkGlm52ProductionTopologyValidate(
    const SparkGlm52ProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes)
{
    const SparkGlm52ProductionTopologyIndexShareSideBand *sideband;
    uint32_t sideband_index;

    if (topology == 0 ||
        topology->abi_version != SPARK_GLM52_PRODUCTION_TOPOLOGY_ABI_VERSION ||
        topology->descriptor_bytes !=
            SPARK_GLM52_PRODUCTION_TOPOLOGY_DESCRIPTOR_BYTES)
    {
        return SparkGlm52ProductionTopologyReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "production topology ABI mismatch");
    }
    if ((topology->topology_flags &
            SPARK_GLM52_PRODUCTION_TOPOLOGY_PRODUCTION_REQUIRED_FLAGS) !=
        SPARK_GLM52_PRODUCTION_TOPOLOGY_PRODUCTION_REQUIRED_FLAGS)
    {
        return SparkGlm52ProductionTopologyReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "production topology is missing required GLM52 DSA/MLA flags");
    }
    if (topology->stage_count == 0u ||
        topology->stage_count > SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT ||
        topology->active_sequence_capacity == 0u ||
        topology->selected_token_count == 0u ||
        topology->kv_block_token_count == 0u ||
        topology->mla_cache_element_count == 0u ||
        topology->indexshare_sideband_count >
            SPARK_GLM52_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS)
    {
        return SparkGlm52ProductionTopologyReport(
            error_buffer,
            error_buffer_bytes,
            SPARK_STATUS_INVALID_ARGUMENT,
            "production topology dimensions are invalid");
    }
    for (sideband_index = 0u;
         sideband_index < topology->indexshare_sideband_count;
         ++sideband_index)
    {
        sideband = &topology->indexshare_sidebands[sideband_index];
        if (sideband->source_layer_index >= SPARK_GLM52_STAGE_PLAN_LAYER_COUNT ||
            sideband->group_end_layer_exclusive >
                SPARK_GLM52_STAGE_PLAN_LAYER_COUNT ||
            sideband->source_layer_index >=
                sideband->first_imported_consumer_layer_index ||
            sideband->first_imported_consumer_layer_index >=
                sideband->group_end_layer_exclusive ||
            sideband->export_stage_index >= topology->stage_count ||
            sideband->import_stage_index >= topology->stage_count ||
            sideband->export_stage_index >= sideband->import_stage_index ||
            sideband->selected_token_count != topology->selected_token_count ||
            sideband->active_sequence_capacity !=
                topology->active_sequence_capacity ||
            sideband->payload_bytes !=
                (uint64_t)topology->active_sequence_capacity *
                (uint64_t)topology->selected_token_count * sizeof(uint32_t) ||
            (sideband->flags &
                SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_SELECTED_TOKEN_INDICES) == 0u)
        {
            return SparkGlm52ProductionTopologyReport(
                error_buffer,
                error_buffer_bytes,
                SPARK_STATUS_INVALID_ARGUMENT,
                "production topology sideband is invalid");
        }
    }
    return SPARK_STATUS_OK;
}
