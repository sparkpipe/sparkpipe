#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PRODUCTION_TOPOLOGY_ABI_VERSION 1u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52ProductionTopology))
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_SELECTED_TOKEN_COUNT 2048u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_INDEXSHARE_GROUP_LAYER_COUNT 4u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS \
    (SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT - 1u)

#define SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_OFFICIAL_DSA_INDEXSHARE \
    0x00000001u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION \
    0x00000002u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE \
    0x00000004u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE \
    0x00000008u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_PRODUCTION_REQUIRED_FLAGS \
    (SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_OFFICIAL_DSA_INDEXSHARE | \
     SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_BOUNDED_LONG_CONTEXT_ATTENTION | \
     SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_INDEXSHARE_STAGE_BOUNDARY_STATE | \
     SPARK_GLM52_PRODUCTION_TOPOLOGY_FLAG_MLA_COMPRESSED_KV_CACHE)

#define SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_SELECTED_TOKEN_INDICES \
    0x00000001u
#define SPARK_GLM52_PRODUCTION_TOPOLOGY_SIDEBAND_FLAG_DEVICE_TO_DEVICE \
    0x00000002u

typedef struct SparkGlm52ProductionTopologyIndexShareSideBand
{
    uint32_t source_layer_index;
    uint32_t group_end_layer_exclusive;
    uint32_t export_stage_index;
    uint32_t import_stage_index;
    uint32_t first_imported_consumer_layer_index;
    uint32_t imported_consumer_layer_count;
    uint32_t selected_token_count;
    uint32_t active_sequence_capacity;
    uint64_t payload_bytes;
    uint32_t flags;
    uint32_t reserved0;
} SparkGlm52ProductionTopologyIndexShareSideBand;

typedef struct SparkGlm52ProductionTopologyStage
{
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t stage_plan_flags;
    uint32_t exported_sideband_count;
    uint32_t imported_sideband_count;
    uint32_t first_exported_sideband_index;
    uint32_t first_imported_sideband_index;
    uint32_t reserved0;
} SparkGlm52ProductionTopologyStage;

typedef struct SparkGlm52ProductionTopology
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t topology_flags;
    uint32_t stage_count;
    uint32_t selected_token_count;
    uint32_t active_sequence_capacity;
    uint32_t kv_block_token_count;
    uint32_t mla_cache_element_count;
    uint32_t indexshare_sideband_count;
    uint32_t reserved0;
    SparkGlm52ProductionTopologyStage stages[
        SPARK_GLM52_STAGE_PLAN_MAX_STAGE_COUNT];
    SparkGlm52ProductionTopologyIndexShareSideBand indexshare_sidebands[
        SPARK_GLM52_PRODUCTION_TOPOLOGY_MAX_INDEXSHARE_SIDEBANDS];
} SparkGlm52ProductionTopology;

uint32_t SparkGlm52DsaIndexShareSourceLayer(uint32_t layer_index);

SparkStatus SparkGlm52DsaIndexShareFindGroupEndLayerExclusive(
    uint32_t layer_index,
    uint32_t *group_end_layer_exclusive_out);

SparkStatus SparkGlm52ProductionTopologyBuild(
    const SparkGlm52StagePlan *stage_plan,
    uint32_t active_sequence_capacity,
    uint32_t selected_token_count,
    uint32_t kv_block_token_count,
    uint32_t mla_cache_element_count,
    SparkGlm52ProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52ProductionTopologyValidate(
    const SparkGlm52ProductionTopology *topology,
    char *error_buffer,
    uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif
