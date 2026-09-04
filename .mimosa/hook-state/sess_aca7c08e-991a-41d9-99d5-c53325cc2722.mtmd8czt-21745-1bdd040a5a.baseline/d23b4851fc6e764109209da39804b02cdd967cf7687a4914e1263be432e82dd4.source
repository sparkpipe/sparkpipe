#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DRAFT_BRIDGE_ABI_VERSION 1u
#define SPARK_DRAFT_BRIDGE_CONFIG_BYTES \
    ((uint32_t)sizeof(SparkDraftBridgeConfig))
#define SPARK_DRAFT_BRIDGE_NODE_BYTES \
    ((uint32_t)sizeof(SparkDraftBridgeNode))
#define SPARK_DRAFT_BRIDGE_PROPOSAL_INFO_BYTES \
    ((uint32_t)sizeof(SparkDraftBridgeProposalInfo))

#define SPARK_DRAFT_BRIDGE_PROTOCOL_VERSION 1u
#define SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES 32u
#define SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT 8u
#define SPARK_DRAFT_BRIDGE_REQUEST_HEADER_BYTES 96u
#define SPARK_DRAFT_BRIDGE_RESPONSE_HEADER_BYTES 28u
#define SPARK_DRAFT_BRIDGE_NODE_RECORD_BYTES 20u
#define SPARK_DRAFT_BRIDGE_RESPONSE_FOOTER_BYTES 48u
#define SPARK_DRAFT_BRIDGE_MAX_DEPTH 8u
#define SPARK_DRAFT_BRIDGE_MAX_TREE_DEPTH 32u
#define SPARK_DRAFT_BRIDGE_MAX_TREE_NODES 4096u
#define SPARK_DRAFT_BRIDGE_MAX_TIME_BUDGET_MS 60000u
#define SPARK_DRAFT_BRIDGE_MAX_PORT 65535u
#define SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX 0xFFFFFFFFu

#define SPARK_DRAFT_BRIDGE_SERVER_STATUS_OK 0u
#define SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_REQUEST 1u
#define SPARK_DRAFT_BRIDGE_SERVER_STATUS_BAD_SEQUENCE 2u
#define SPARK_DRAFT_BRIDGE_SERVER_STATUS_INTERNAL 3u
#define SPARK_DRAFT_BRIDGE_SERVER_STATUS_UNAVAILABLE 4u

typedef struct SparkDraftBridge SparkDraftBridge;

typedef struct SparkDraftBridgeConfig
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    const char *host;
    uint32_t port;
    char target_model[SPARK_DRAFT_BRIDGE_TARGET_MODEL_BYTES];
    uint32_t tap_row_bytes;
    uint32_t max_committed_tokens;
    uint32_t max_tap_rows;
    uint32_t max_nodes;
    uint32_t connect_timeout_ms;
    uint32_t io_timeout_ms;
} SparkDraftBridgeConfig;

typedef struct SparkDraftBridgeNode
{
    uint32_t token_id;
    uint32_t parent_index;
    uint32_t depth;
    uint32_t source_bit;
    float score;
} SparkDraftBridgeNode;

typedef struct SparkDraftBridgeProposalInfo
{
    uint32_t server_status;
    uint32_t drafter_us[SPARK_DRAFT_BRIDGE_DRAFTER_STAGE_COUNT];
    uint32_t graft_us;
    uint32_t expansions;
    uint32_t elapsed_us;
} SparkDraftBridgeProposalInfo;

SparkStatus SparkDraftBridgeValidateConfig(
    const SparkDraftBridgeConfig *config);
SparkStatus SparkDraftBridgeInitialize(
    const SparkDraftBridgeConfig *config,
    SparkDraftBridge **bridge_out);
void SparkDraftBridgeDestroy(
    SparkDraftBridge *bridge);
SparkStatus SparkDraftBridgeProposeTree(
    SparkDraftBridge *bridge,
    uint32_t speculator_mask,
    uint64_t sequence_id,
    uint64_t generation,
    uint64_t position,
    uint32_t anchor_token_id,
    const uint32_t *committed_token_ids,
    uint32_t committed_token_count,
    const void *tap_rows,
    uint32_t tap_row_count,
    uint32_t depth,
    uint32_t time_budget_ms,
    uint32_t max_depth,
    SparkDraftBridgeNode *nodes_out,
    uint32_t node_capacity,
    uint32_t *node_count_out,
    SparkDraftBridgeProposalInfo *info_out);

#ifdef __cplusplus
}
#endif
