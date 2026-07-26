#ifndef SPARKPIPE_SPARK_STAGE_KV_CLIENT_H
#define SPARKPIPE_SPARK_STAGE_KV_CLIENT_H

#include <stdint.h>

#include "sparkpipe/spark_kv_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stage-module client for the family KV backing tier (spark_kv_store, the
 * Mooncake provider). One instance per stage: it loads the provider shared
 * object, initializes it with the stage's rank, layer slice and fingerprints,
 * and gives modules fingerprinted key construction plus batch submit/poll
 * under the corrected contract (poll returns OK on delivery; the batch
 * outcome travels only in completion.status).
 *
 * Enablement is EXPLICIT: the module passes either the literal string "none"
 * (tier disabled, every call after Open is a refused no-op returning
 * NOT_FOUND/INVALID) or a provider .so path. There is no inferred default.
 *
 * Keys bind model fingerprint, cache-layout fingerprint, rank, sequence and
 * logical block, so a model or layout change can never consume stale KV.
 * The residency DECISION - which packets prefetch when, against which work
 * queue - belongs to the runtime work-control layer exactly as it does for
 * glm52; this client is the provider plumbing that layer drives.
 */

typedef struct SparkStageKvClient
{
	const char *module_tag;
	SparkKvStoreDynamicLibrary library;
	void *store_state;
	uint64_t next_batch_id;
	uint32_t enabled;
} SparkStageKvClient;

SparkStatus SparkStageKvClientOpen(SparkStageKvClient *client, const char *module_tag, const char *provider_path_or_none, uint32_t rank_index, uint32_t first_layer_index, uint32_t layer_count, uint64_t model_fingerprint, uint64_t cache_layout_fingerprint, const char *service_address, const char *ipc_socket_path, uint64_t client_memory_pool_bytes, uint32_t worker_count);
int32_t SparkStageKvClientFormatKey(char *key, uint32_t key_capacity, uint64_t model_fingerprint, uint64_t cache_layout_fingerprint, uint32_t rank_index, uint64_t sequence_id, uint32_t logical_block);
SparkStatus SparkStageKvClientSubmit(SparkStageKvClient *client, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority, uint64_t *batch_id);
SparkStatus SparkStageKvClientPoll(SparkStageKvClient *client, uint64_t batch_id, SparkKvStoreCompletion *completion);
SparkStatus SparkStageKvClientAllocateBuffer(SparkStageKvClient *client, uint64_t buffer_bytes, void **buffer);
void SparkStageKvClientClose(SparkStageKvClient *client);

#ifdef __cplusplus
}
#endif

#endif
