#pragma once

#include <stdint.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_sampling.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DRAFT_SERVER_ABI_VERSION 1u
#define SPARK_DRAFT_SERVER_CONFIG_BYTES ((uint32_t)sizeof(SparkDraftServerConfig))
#define SPARK_DRAFT_SERVER_POLL_FDS 2u

typedef struct SparkDraftServer SparkDraftServer;

typedef struct SparkDraftServerConfig
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
	uint32_t io_timeout_ms;
} SparkDraftServerConfig;

typedef struct SparkDraftServerRequest
{
	uint32_t speculator_mask;
	uint64_t sequence_id;
	uint64_t generation;
	uint64_t position;
	uint32_t anchor_token_id;
	const uint32_t *committed_token_ids;
	uint32_t committed_token_count;
	const void *tap_rows;
	uint32_t tap_row_count;
	uint32_t depth;
	uint32_t time_budget_ms;
	uint32_t max_depth;
	uint32_t node_capacity;
	SparkRowSampling sampling;
	uint64_t received_ns;
} SparkDraftServerRequest;

typedef struct SparkDraftServerStats
{
	uint64_t connections;
	uint64_t requests;
	uint64_t bad_requests;
	uint64_t answered;
	uint64_t dropped;
} SparkDraftServerStats;

SparkStatus SparkDraftServerCreate(const SparkDraftServerConfig *config, SparkDraftServer **server_out);
void SparkDraftServerDestroy(SparkDraftServer *server);
uint32_t SparkDraftServerPort(const SparkDraftServer *server);
uint32_t SparkDraftServerPollFds(const SparkDraftServer *server, int *fds, uint32_t *events, uint64_t *deadline_ns);
SparkStatus SparkDraftServerAdvance(SparkDraftServer *server, SparkDraftServerRequest *request_out);
SparkStatus SparkDraftServerAnswer(SparkDraftServer *server, uint32_t server_status, const SparkDraftBridgeNode *nodes, uint32_t node_count,
	const SparkDraftBridgeProposalInfo *info);
void SparkDraftServerGetStats(const SparkDraftServer *server, SparkDraftServerStats *stats_out);

#ifdef __cplusplus
}
#endif
