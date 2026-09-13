#pragma once

 


























































#include <stdint.h>
#include <stdlib.h>

#include "runtime/prefix_cache.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PAGED_KV_NO_BLOCK UINT32_MAX
#define SPARK_PAGED_KV_NO_SLOT UINT32_MAX
#define SPARK_PAGED_KV_NO_LANE UINT32_MAX

 
struct SparkPagedKvConfiguration;

 





typedef struct SparkPagedKvGeometryCallbacks
{
	 



	uint64_t sequence_id_base;
	 



	uint32_t max_blocks_per_lane;
	 



	SparkStatus (*validate_configuration)(
		const struct SparkPagedKvConfiguration *configuration);
}
SparkPagedKvGeometryCallbacks;

typedef struct SparkPagedKvConfiguration
{
	 
	uint32_t block_token_count;
	 
	uint32_t lane_count;
	 
	uint32_t blocks_per_lane;
	 

	uint32_t physical_page_capacity;
	 

	uint32_t logical_page_capacity;
	 


	uint32_t checkpoint_slot_count;
	 



	uint64_t block_stride_bytes;
}
SparkPagedKvConfiguration;

 

typedef struct SparkPagedKvMatch
{
	uint32_t block_count;
	uint32_t checkpoint_slot;
}
SparkPagedKvMatch;

typedef struct SparkPagedKvCheckpoint
{
	uint32_t live;
	uint32_t lane;
	 
	uint32_t boundary_blocks;
	 


	uint32_t witness_block;
	 
	uint64_t last_use;
}
SparkPagedKvCheckpoint;

typedef struct SparkPagedKv
{
	SparkPagedKvConfiguration configuration;
	 
	const SparkPagedKvGeometryCallbacks *geometry;
	 
	SparkPrefixCacheCore core;
	 
	uint32_t reuse_enabled;
	 
	uint32_t *blocks_by_lane;
	uint32_t *counts_by_lane;
	 


	uint32_t *lane_core_blocks;
	uint32_t *lane_live;
	SparkPagedKvCheckpoint *checkpoints;
	 

	uint32_t reserved_slot;
	uint32_t reserved_lane;
	uint64_t lru_clock;
	 


	uint32_t *admit_scratch;
}
SparkPagedKv;

SparkStatus SparkPagedKvInitialize(
	SparkPagedKv *cache,
	const SparkPagedKvConfiguration *configuration,
	const SparkPagedKvGeometryCallbacks *geometry,
	uint32_t *blocks_by_lane,
	uint32_t *counts_by_lane);
void SparkPagedKvDestroy(SparkPagedKv *cache);
 


void SparkPagedKvLaneReset(SparkPagedKv *cache, uint32_t lane);
 



SparkStatus SparkPagedKvAdmit(
	SparkPagedKv *cache,
	uint32_t lane,
	const uint32_t *tokens,
	uint32_t token_count,
	SparkPagedKvMatch *match_out);
 













SparkStatus SparkPagedKvCover(
	SparkPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	const uint32_t *tokens,
	uint32_t token_count);
 
uint64_t SparkPagedKvCommittedTokens(
	const SparkPagedKv *cache,
	uint32_t lane);
 
uint32_t SparkPagedKvFreeBlocks(const SparkPagedKv *cache);
 


uint32_t SparkPagedKvCheckpointOffer(
	SparkPagedKv *cache,
	uint32_t lane,
	uint64_t end_position,
	uint32_t *slot_out);
 


void SparkPagedKvCheckpointCommit(
	SparkPagedKv *cache,
	uint32_t lane,
	uint32_t slot,
	uint64_t end_position);
 
void SparkPagedKvCheckpointAbort(
	SparkPagedKv *cache,
	uint32_t lane,
	uint32_t slot);

 





 

static inline uint32_t SparkPagedKvBlocksForPositions(
	uint64_t positions, uint32_t block_token_count)
{
	return (uint32_t)((positions + block_token_count - 1u) /
		block_token_count);
}

 


static inline void *SparkPagedKvCheckedCalloc(uint64_t count, uint64_t bytes)
{
	if ( count == 0u || bytes == 0u || count > SIZE_MAX / bytes )
		return(0);
	return(calloc((size_t)count,(size_t)bytes));
}

 



static inline uint32_t SparkPagedKvPoolGeometryIsValid(
	uint32_t logical_page_capacity,
	uint32_t physical_page_capacity,
	uint64_t maximum_positions,
	uint32_t block_token_count)
{
	uint32_t lane_page_capacity;
	if ( logical_page_capacity == 0u || physical_page_capacity == 0u ||
		block_token_count == 0u )
		return(0u);
	lane_page_capacity =
		SparkPagedKvBlocksForPositions(maximum_positions,
			block_token_count);
	return(logical_page_capacity >= physical_page_capacity &&
		logical_page_capacity >= lane_page_capacity ? 1u : 0u);
}

#ifdef __cplusplus
}
#endif
