#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_production_topology.h"

#define SPARK_TEST_GLM52_TOPOLOGY_ACTIVE_SEQUENCE_CAPACITY 64u

static void SparkTestInitializeRingStagePlan(SparkGlm52StagePlan *stage_plan)
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
    assert(SparkGlm52DsaIndexShareSourceLayer(1u) == 1u);
    assert(SparkGlm52DsaIndexShareSourceLayer(2u) == 2u);
    assert(SparkGlm52DsaIndexShareSourceLayer(3u) == 2u);
    assert(SparkGlm52DsaIndexShareSourceLayer(5u) == 2u);
    assert(SparkGlm52DsaIndexShareSourceLayer(6u) == 6u);
    assert(SparkGlm52DsaIndexShareSourceLayer(9u) == 6u);
    assert(SparkGlm52DsaIndexShareSourceLayer(10u) == 10u);
    assert(SparkGlm52DsaIndexShareSourceLayer(77u) == 74u);
    assert(SparkGlm52DsaIndexShareSourceLayer(78u) == UINT32_MAX);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        0u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 1u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        3u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 6u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        77u,
        &group_end_layer_exclusive) == SPARK_STATUS_OK);
    assert(group_end_layer_exclusive == 78u);
    assert(SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
        78u,
        &group_end_layer_exclusive) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52ProductionTopologyRingSideband(void)
{
    SparkGlm52StagePlan stage_plan;
    SparkGlm52ProductionTopology topology;
    const SparkGlm52ProductionTopologyIndexShareSideBand *sideband;
    char error_buffer[256];

    SparkTestInitializeRingStagePlan(&stage_plan);
    assert(SparkGlm52ProductionTopologyBuild(
        &stage_plan,
        SPARK_TEST_GLM52_TOPOLOGY_ACTIVE_SEQUENCE_CAPACITY,
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT,
        16u,
        SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,
        &topology,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_OK);
    assert(topology.stage_count == 13u);
    assert(topology.active_sequence_capacity ==
        SPARK_TEST_GLM52_TOPOLOGY_ACTIVE_SEQUENCE_CAPACITY);
    assert(topology.selected_token_count ==
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT);
    assert((topology.topology_flags &
        SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE) != 0u);
    assert(topology.mla_cache_element_count ==
        SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS);
    {
        uint32_t tap_count, sideband_index;
        uint32_t expected_export_stages[5] = { 1u, 3u, 6u, 9u, 11u };
        uint32_t expected_capture_layers[5] = { 7u, 22u, 38u, 54u, 69u };

        tap_count = 0u;
        for (sideband_index = 0u;
             sideband_index < topology.indexshare_sideband_count;
             ++sideband_index)
        {
            const SparkGlm52ProductionTopologyIndexShareSideBand *tap =
                &topology.indexshare_sidebands[sideband_index];
            if ((tap->flags &
                    SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_DSPARK_HIDDEN_TAP) == 0u)
            {
                continue;
            }
            assert(tap_count < 5u);
            assert(tap->source_layer_index == expected_capture_layers[tap_count]);
            assert(tap->group_end_layer_exclusive ==
                expected_capture_layers[tap_count] + 1u);
            assert(tap->export_stage_index == expected_export_stages[tap_count]);
            assert(tap->import_stage_index == topology.stage_count - 1u);
            assert(tap->payload_bytes ==
                (uint64_t)SPARK_TEST_GLM52_TOPOLOGY_ACTIVE_SEQUENCE_CAPACITY *
                SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES);
            tap_count += 1u;
        }
        assert(tap_count == 5u);
    }
    {
        uint32_t hop_index, kind_bits, bytes_per_sequence;
        uint32_t expected_taps[13] =
            { 0u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 3u, 4u, 4u, 5u, 0u };

        for (hop_index = 0u; hop_index + 1u < topology.stage_count; ++hop_index)
        {
            assert(SparkGlm52ProductionTopologyHopSidebandLayout(
                &topology,
                hop_index,
                &kind_bits,
                &bytes_per_sequence) == SPARK_STATUS_OK);
            assert((kind_bits &
                SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_DSPARK_HIDDEN_TAP) ==
                (expected_taps[hop_index] != 0u
                    ? SPARK_HIDDEN_TRANSPORT_SIDEBAND_KIND_DSPARK_HIDDEN_TAP
                    : 0u));
            assert(bytes_per_sequence >=
                expected_taps[hop_index] *
                    SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES);
            assert((bytes_per_sequence -
                expected_taps[hop_index] *
                    SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES) %
                sizeof(uint32_t) == 0u);
        }
    }
    sideband = SparkTestFindSidebandForSourceLayer(&topology, 10u);
    assert(sideband != 0);
    assert(sideband->source_layer_index == 10u);
    assert(sideband->group_end_layer_exclusive == 14u);
    assert(sideband->export_stage_index == 1u);
    assert(sideband->import_stage_index == 2u);
    assert(sideband->first_imported_consumer_layer_index == 12u);
    assert(sideband->imported_consumer_layer_count == 2u);
    assert(sideband->payload_bytes ==
        (uint64_t)SPARK_TEST_GLM52_TOPOLOGY_ACTIVE_SEQUENCE_CAPACITY *
        SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES);
    assert(topology.stages[1u].exported_sideband_count != 0u);
    assert(topology.stages[2u].imported_sideband_count != 0u);
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

    SparkTestInitializeRingStagePlan(&stage_plan);
    assert(SparkGlm52ProductionTopologyBuild(
        &stage_plan,
        0u,
        SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT,
        16u,
        SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,
        &topology,
        error_buffer,
        sizeof(error_buffer)) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestGlm52IndexShareSchedule();
    SparkTestGlm52ProductionTopologyRingSideband();
    SparkTestGlm52ProductionTopologyRejectsBadDimensions();
    return 0;
}
