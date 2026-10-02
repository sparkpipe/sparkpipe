#pragma once

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_status.h"

#define SPARK_STAGE_KV_REGION_PAGE_MAJOR 1u
#define SPARK_STAGE_KV_REGION_LAYER_MAJOR 2u
#define SPARK_STAGE_KV_MAX_REGIONS 2u

typedef struct SparkStageKvRegion
{
	uint32_t layout;
	uint32_t layer_count;
	uint64_t layer_page_bytes;
} SparkStageKvRegion;

typedef struct SparkStageKvConfiguration
{
	SparkStageModuleLedger *ledger;
	const char *module_tag;
	uint32_t block_token_count;
	uint32_t region_count;
	SparkStageKvRegion regions[SPARK_STAGE_KV_MAX_REGIONS];
	uint32_t arena_kv_head_count;
	uint32_t arena_head_dim;
	uint32_t arena_bytes_per_scalar;
	SparkKvCacheCapacityRequest capacity_request;
	const char *model_id;
	const char *model_revision;
	const char *layout_fingerprint;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t max_input_row_count;
	uint32_t logical_page_count;
	uint32_t physical_page_count;
	uint32_t pipeline_slot_count;
	uint32_t owner_rank;
	uint32_t owner_count;
	const char *backing_directory;
	uint64_t backing_maximum_bytes;
} SparkStageKvConfiguration;

typedef struct SparkStageKvBinding
{
	const char *module_tag;
	uint32_t block_token_count;
	uint32_t region_count;
	uint32_t pages_per_sequence;
	uint32_t logical_page_count;
	uint32_t physical_page_count;
	uint32_t resident_sequence_capacity;
	uint32_t max_sequence_positions;
	uint32_t pipeline_slot_count;
	uint32_t owner_rank;
	uint32_t owner_count;
	SparkModelDriverCacheLane *owned_lanes;
	uint32_t *owned_slots;
	SparkStageKvRegion regions[SPARK_STAGE_KV_MAX_REGIONS];
	uint8_t *region_base[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t region_packed_page_bytes[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t region_layer_stride_bytes[SPARK_STAGE_KV_MAX_REGIONS];
	uint64_t page_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
	SparkKvCacheArena arena;
	SparkKvPageCache page_cache;
	SparkKvPageStore page_store;
	SparkKvLaneTransactions transactions;
	SparkKvLaneTransaction *lanes;
	uint32_t *logical_pages;
	uint32_t *physical_pages;
	SparkKvCacheBlock *blocks;
	uint32_t *resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
	uint8_t *staging;
	atomic_uchar *lane_bound;
	atomic_ullong *lane_sequence_ids;
	atomic_ullong *lane_next_positions;
	pthread_mutex_t mutex;
	uint32_t mutex_initialized;
	uint64_t control_generation;
	uint64_t reset_generation;
	char backing_default[256];
} SparkStageKvBinding;

SparkStatus SparkStageKvBindingInitialize(SparkStageKvBinding *binding,const SparkStageKvConfiguration *configuration);
void SparkStageKvBindingDestroy(SparkStageKvBinding *binding);
SparkStatus SparkStageKvBindingAdmit(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision);
SparkStatus SparkStageKvBindingReset(SparkStageKvBinding *binding,uint64_t generation);
SparkStatus SparkStageKvBindingAdmitReset(SparkStageKvBinding *binding,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision,atomic_uint *slot_states,atomic_uint *lane_states,void *stream);
SparkStatus SparkStageKvBindingContinuity(SparkStageKvBinding *binding,const atomic_uint *lane_states,const SparkModelDriverFrame *frame,uint32_t row_count,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions);
SparkStatus SparkStageKvBindingClaim(SparkStageKvBinding *binding,const SparkModelDriverFrame *frame,uint32_t active_count,const uint32_t *row_resident_slots,const uint64_t *row_sequence_ids,const uint64_t *row_positions,const uint64_t *next_positions);
SparkStatus SparkStageKvBindingUploadPageTables(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,void *stream);
SparkStatus SparkStageKvBindingFinish(SparkStageKvBinding *binding,const uint32_t *resident_slots,uint32_t lane_count,SparkStatus status,uint32_t extra_tokens,const uint8_t *bound,const uint64_t *sequence_ids,const uint64_t *next_positions);
uint32_t SparkStageKvBindingResidentCount(const SparkStageKvBinding *binding);

static inline uint32_t SparkStageKvBindingOwns(const SparkStageKvBinding *binding,uint32_t resident_slot)
{
	return(binding->owner_count <= 1u || resident_slot % binding->owner_count == binding->owner_rank ? 1u : 0u);
}
