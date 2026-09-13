#pragma once

 





















#include <stdint.h>

#include "runtime/prefix_cache.h"
#include "runtime/paged_kv_common.h"
#include "sparkpipe/spark_status.h"

#define SPARK_QWEN36_PAGED_KV_NO_BLOCK SPARK_PAGED_KV_NO_BLOCK
#define SPARK_QWEN36_PAGED_KV_NO_SLOT SPARK_PAGED_KV_NO_SLOT
#define SPARK_QWEN36_PAGED_KV_NO_LANE SPARK_PAGED_KV_NO_LANE
 


#define SPARK_QWEN36_PAGED_KV_MAX_BLOCKS_PER_LANE 128u

 
#define SPARK_QWEN36_PAGED_KV_SEQUENCE_BASE UINT64_C(0x5133600000000000)

typedef SparkPagedKvConfiguration SparkQwen36PagedKvConfiguration;
typedef SparkPagedKvMatch SparkQwen36PagedKvMatch;
typedef SparkPagedKvCheckpoint SparkQwen36PagedKvCheckpoint;
typedef SparkPagedKv SparkQwen36PagedKv;

SparkStatus SparkQwen36PagedKvInitialize(
	SparkQwen36PagedKv *cache,
	const SparkQwen36PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkQwen36PagedKvDestroy(SparkQwen36PagedKv *cache);
 


void SparkQwen36PagedKvLaneReset(SparkQwen36PagedKv *cache, uint32_t lane);
 



SparkStatus SparkQwen36PagedKvAdmit(
	SparkQwen36PagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkQwen36PagedKvMatch *match_out);
 




SparkStatus SparkQwen36PagedKvCover(
	SparkQwen36PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
 
uint64_t SparkQwen36PagedKvCommittedTokens(
	const SparkQwen36PagedKv *cache,
	uint32_t lane);
 
uint32_t SparkQwen36PagedKvFreeBlocks(const SparkQwen36PagedKv *cache);
 


uint32_t SparkQwen36PagedKvCheckpointOffer(
	SparkQwen36PagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
 


void SparkQwen36PagedKvCheckpointCommit(
	SparkQwen36PagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
 
void SparkQwen36PagedKvCheckpointAbort(
	SparkQwen36PagedKv *cache,
	uint32_t lane,
	uint32_t slot);
