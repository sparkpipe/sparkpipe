#include "runtime/paged_kv_common.h"

#include <stdlib.h>
#include <string.h>

 

static uint64_t SparkPagedKvSequenceId(const SparkPagedKv *cache,
	uint32_t lane)
{
	return cache->geometry->sequence_id_base + (uint64_t)lane;
}

static uint32_t SparkPagedKvRowBase(const SparkPagedKv *cache,
	uint32_t lane)
{
	return (uint32_t)((uint64_t)lane *
		cache->configuration.blocks_per_lane);
}

 







static SparkStatus SparkPagedKvScratchBorrow(SparkPagedKv *cache,
	uint32_t *block_out)
{
	uint32_t block,evicted;
	if ( cache->core.free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
		(void)SparkPrefixCacheCoreTrim(&cache->core,1u,&evicted);
	if ( cache->core.free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	block = cache->core.free_block_head;
	cache->core.free_block_head =
		cache->core.blocks[block].free_next;
	 







	cache->core.blocks[block].state = SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE;
	cache->core.blocks[block].reference_count = 0u;
	cache->core.blocks[block].token_count = 0u;
	*block_out = block;
	return(SPARK_STATUS_OK);
}

 



static void SparkPagedKvScratchReturn(SparkPagedKv *cache, uint32_t block)
{
	cache->core.blocks[block].state = SPARK_PREFIX_CACHE_CORE_BLOCK_FREE;
	cache->core.blocks[block].reference_count = 0u;
	cache->core.blocks[block].token_count = 0u;
	cache->core.blocks[block].free_next = cache->core.free_block_head;
	cache->core.free_block_head = block;
}

 









static void SparkPagedKvSyncRow(SparkPagedKv *cache, uint32_t lane)
{
	uint32_t row_base = SparkPagedKvRowBase(cache,lane);
	uint32_t committed_blocks = 0u,outstanding;
	if ( cache->lane_live[lane] != 0u )
	{
		uint32_t tokens,blocks;
		tokens = (uint32_t)SparkPrefixCacheCoreSequenceTokenCount(
			&cache->core,SparkPagedKvSequenceId(cache,lane));
		blocks = SparkPagedKvBlocksForPositions(tokens,
			cache->configuration.block_token_count);
		if ( blocks > cache->configuration.blocks_per_lane )
			blocks = cache->configuration.blocks_per_lane;
		if ( blocks != 0u )
			(void)SparkPrefixCacheCoreBuildBlockTable(&cache->core,
				SparkPagedKvSequenceId(cache,lane),tokens,
				cache->blocks_by_lane + row_base,blocks,
				&committed_blocks);
	}
	outstanding = cache->counts_by_lane[lane];
	cache->lane_core_blocks[lane] = committed_blocks;
	 

	if ( outstanding > committed_blocks )
		return;
	cache->counts_by_lane[lane] = committed_blocks;
}

SparkStatus SparkPagedKvInitialize(SparkPagedKv *cache,
	const SparkPagedKvConfiguration *configuration,
	const SparkPagedKvGeometryCallbacks *geometry,
	uint32_t *blocks_by_lane, uint32_t *counts_by_lane)
{
	SparkPrefixCacheCoreConfiguration core_configuration;
	uint32_t lane,index,buckets;
	SparkStatus status;
	if ( cache == 0 || configuration == 0 || geometry == 0 ||
		blocks_by_lane == 0 || counts_by_lane == 0 ||
		configuration->block_token_count == 0u ||
		configuration->block_token_count >
			SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS ||
		configuration->lane_count == 0u ||
		configuration->blocks_per_lane == 0u ||
		configuration->blocks_per_lane >
			geometry->max_blocks_per_lane ||
		configuration->physical_page_capacity == 0u ||
		configuration->logical_page_capacity <
			configuration->physical_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( geometry->validate_configuration != 0 )
	{
		status = geometry->validate_configuration(configuration);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	memset(cache,0,sizeof(*cache));
	cache->configuration = *configuration;
	cache->geometry = geometry;
	cache->blocks_by_lane = blocks_by_lane;
	cache->counts_by_lane = counts_by_lane;
	cache->reuse_enabled = configuration->checkpoint_slot_count != 0u ? 1u : 0u;
	cache->reserved_slot = SPARK_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_PAGED_KV_NO_LANE;
	memset(&core_configuration,0,sizeof(core_configuration));
	core_configuration.abi_version = SPARK_PREFIX_CACHE_CORE_ABI_VERSION;
	core_configuration.descriptor_bytes =
		SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES;
	core_configuration.block_token_count =
		configuration->block_token_count;
	core_configuration.block_stride_bytes =
		configuration->block_stride_bytes != 0u ?
		configuration->block_stride_bytes :
		configuration->block_token_count;
	core_configuration.block_count = configuration->physical_page_capacity;
	core_configuration.max_sequence_count = configuration->lane_count;
	core_configuration.sequence_block_capacity =
		configuration->blocks_per_lane;
	 
	buckets = 16u;
	while ( buckets < configuration->physical_page_capacity * 2u &&
		buckets < (uint32_t)1u << 31u )
		buckets <<= 1u;
	core_configuration.hash_bucket_count = buckets;
	status = SparkPrefixCacheCoreInitialize(&cache->core,
		&core_configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkPagedKvDestroy(cache);
		return(status);
	}
	if ( cache->reuse_enabled != 0u )
	{
		cache->checkpoints = (SparkPagedKvCheckpoint *)calloc(
			configuration->checkpoint_slot_count,
			sizeof(cache->checkpoints[0]));
		if ( cache->checkpoints == 0 )
		{
			SparkPagedKvDestroy(cache);
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		for (index=0u; index<configuration->checkpoint_slot_count; index++)
			cache->checkpoints[index].witness_block =
				SPARK_PAGED_KV_NO_BLOCK;
	}
	cache->lane_core_blocks = (uint32_t *)calloc(
		configuration->lane_count,sizeof(uint32_t));
	cache->lane_live = (uint32_t *)calloc(
		configuration->lane_count,sizeof(uint32_t));
	cache->admit_scratch = (uint32_t *)calloc(
		configuration->blocks_per_lane,sizeof(uint32_t));
	if ( cache->lane_core_blocks == 0 || cache->lane_live == 0 ||
		cache->admit_scratch == 0 )
	{
		SparkPagedKvDestroy(cache);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (lane=0u; lane<configuration->lane_count; lane++)
	{
		for (index=0u; index<configuration->blocks_per_lane; index++)
			blocks_by_lane[((uint64_t)lane *
				configuration->blocks_per_lane) + index] =
				SPARK_PAGED_KV_NO_BLOCK;
		counts_by_lane[lane] = 0u;
	}
	return(SPARK_STATUS_OK);
}

void SparkPagedKvDestroy(SparkPagedKv *cache)
{
	if ( cache == 0 )
		return;
	SparkPrefixCacheCoreDestroy(&cache->core);
	free(cache->checkpoints);
	free(cache->lane_core_blocks);
	free(cache->lane_live);
	free(cache->admit_scratch);
	memset(cache,0,sizeof(*cache));
}

void SparkPagedKvLaneReset(SparkPagedKv *cache, uint32_t lane)
{
	uint32_t slot,row_base,index;
	if ( cache == 0 || lane >= cache->configuration.lane_count )
		return;
	row_base = SparkPagedKvRowBase(cache,lane);
	if ( cache->lane_live[lane] != 0u )
	{
		(void)SparkPrefixCacheCoreReleaseSequence(&cache->core,
			SparkPagedKvSequenceId(cache,lane));
		cache->lane_live[lane] = 0u;
	}
	 
	for (index=cache->lane_core_blocks[lane];
		index<cache->counts_by_lane[lane]; index++)
	{
		uint32_t block = cache->blocks_by_lane[row_base + index];
		if ( block != SPARK_PAGED_KV_NO_BLOCK )
			SparkPagedKvScratchReturn(cache,block);
	}
	for (index=0u; index<cache->configuration.blocks_per_lane; index++)
		cache->blocks_by_lane[row_base + index] =
			SPARK_PAGED_KV_NO_BLOCK;
	cache->counts_by_lane[lane] = 0u;
	cache->lane_core_blocks[lane] = 0u;
	for (slot=0u; slot<cache->configuration.checkpoint_slot_count; slot++)
		if ( cache->checkpoints[slot].live != 0u &&
			cache->checkpoints[slot].lane == lane )
		{
			cache->checkpoints[slot].live = 0u;
			cache->checkpoints[slot].witness_block =
				SPARK_PAGED_KV_NO_BLOCK;
		}
	if ( cache->reserved_lane == lane )
	{
		cache->reserved_slot = SPARK_PAGED_KV_NO_SLOT;
		cache->reserved_lane = SPARK_PAGED_KV_NO_LANE;
	}
}

 

static uint32_t SparkPagedKvWitnessedDepth(const SparkPagedKv *cache,
	const uint32_t *attached_blocks, uint32_t limit_blocks,
	uint32_t *slot_out)
{
	uint32_t boundary,slot;
	for (boundary=limit_blocks; boundary>=1u; boundary--)
	{
		for (slot=0u; slot<cache->configuration.checkpoint_slot_count;
			slot++)
		{
			const SparkPagedKvCheckpoint *checkpoint =
				&cache->checkpoints[slot];
			if ( checkpoint->live == 0u ||
				checkpoint->boundary_blocks != boundary ||
				checkpoint->witness_block !=
					attached_blocks[boundary - 1u] )
				continue;
			*slot_out = slot;
			return(boundary);
		}
	}
	return(0u);
}

SparkStatus SparkPagedKvAdmit(SparkPagedKv *cache,
	uint32_t lane, const uint32_t *tokens, uint32_t token_count,
	SparkPagedKvMatch *match_out)
{
	SparkPagedKvMatch match;
	uint32_t attached_count,lcp_tokens,resumed_blocks,readmit_tokens,slot;
	uint32_t block_tokens;
	SparkStatus status;
	match.block_count = 0u;
	match.checkpoint_slot = SPARK_PAGED_KV_NO_SLOT;
	if ( match_out != 0 )
	{
		match_out->block_count = 0u;
		match_out->checkpoint_slot = SPARK_PAGED_KV_NO_SLOT;
	}
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		tokens == 0 || token_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_tokens = cache->configuration.block_token_count;
	 
	SparkPagedKvLaneReset(cache,lane);
	if ( cache->reuse_enabled == 0u )
		return(SPARK_STATUS_OK);
	status = SparkPrefixCacheCoreAdmitSequence(&cache->core,
		SparkPagedKvSequenceId(cache,lane),tokens,token_count,
		&lcp_tokens);
	if ( status != SPARK_STATUS_OK )
	{
		SparkPagedKvLaneReset(cache,lane);
		return(status);
	}
	cache->lane_live[lane] = 1u;
	 
	resumed_blocks = 0u;
	slot = SPARK_PAGED_KV_NO_SLOT;
	if ( lcp_tokens != 0u )
	{
		uint32_t lcp_block_count;
		lcp_block_count = lcp_tokens / block_tokens;
		status = SparkPrefixCacheCoreBuildBlockTable(&cache->core,
			SparkPagedKvSequenceId(cache,lane),lcp_tokens,
			cache->admit_scratch,lcp_block_count,&attached_count);
		if ( status == SPARK_STATUS_OK &&
			attached_count == lcp_block_count &&
			lcp_block_count <= cache->configuration.blocks_per_lane )
			resumed_blocks = SparkPagedKvWitnessedDepth(cache,
				cache->admit_scratch,lcp_block_count,&slot);
		else
			resumed_blocks = 0u;
		if ( resumed_blocks < lcp_block_count )
		{
			 








			(void)SparkPrefixCacheCoreReleaseSequence(&cache->core,
				SparkPagedKvSequenceId(cache,lane));
			readmit_tokens = resumed_blocks != 0u ?
				resumed_blocks * block_tokens : block_tokens - 1u;
			if ( readmit_tokens > token_count )
				readmit_tokens = token_count;
			status = SparkPrefixCacheCoreAdmitSequence(&cache->core,
				SparkPagedKvSequenceId(cache,lane),tokens,
				readmit_tokens,&lcp_tokens);
			if ( status != SPARK_STATUS_OK ||
				lcp_tokens != (resumed_blocks != 0u ?
					readmit_tokens : 0u) )
			{
				SparkPagedKvLaneReset(cache,lane);
				return(status != SPARK_STATUS_OK ? status :
					SPARK_STATUS_INTERNAL_ERROR);
			}
		}
	}
	match.block_count = resumed_blocks;
	match.checkpoint_slot = slot;
	SparkPagedKvSyncRow(cache,lane);
	if ( match_out != 0 )
		*match_out = match;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkPagedKvCover(SparkPagedKv *cache,
	uint32_t lane, uint64_t end_position, const uint32_t *tokens,
	uint32_t token_count)
{
	uint32_t block_tokens,required,count,row_base,ordinal,appended,block;
	SparkStatus status;
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		end_position == 0u || (tokens == 0 && token_count != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_tokens = cache->configuration.block_token_count;
	required = SparkPagedKvBlocksForPositions(end_position,block_tokens);
	if ( required > cache->configuration.blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	 




	if ( token_count != 0u && cache->lane_live[lane] != 0u &&
		cache->counts_by_lane[lane] == cache->lane_core_blocks[lane] )
	{
		status = SparkPrefixCacheCoreAppendTokens(&cache->core,
			SparkPagedKvSequenceId(cache,lane),tokens,token_count);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	SparkPagedKvSyncRow(cache,lane);
	row_base = SparkPagedKvRowBase(cache,lane);
	count = cache->counts_by_lane[lane];
	appended = 0u;
	for (ordinal=count; ordinal<required; ordinal++)
	{
		status = SparkPagedKvScratchBorrow(cache,&block);
		if ( status != SPARK_STATUS_OK )
		{
			 




			cache->counts_by_lane[lane] = count + appended;
			return(status);
		}
		cache->blocks_by_lane[row_base + ordinal] = block;
		appended++;
	}
	cache->counts_by_lane[lane] = count + appended;
	return(SPARK_STATUS_OK);
}

uint64_t SparkPagedKvCommittedTokens(const SparkPagedKv *cache,
	uint32_t lane)
{
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		cache->lane_live[lane] == 0u )
		return(0u);
	return SparkPrefixCacheCoreSequenceTokenCount(&cache->core,
		SparkPagedKvSequenceId(cache,lane));
}

uint32_t SparkPagedKvFreeBlocks(const SparkPagedKv *cache)
{
	SparkPrefixCacheCoreStats stats;
	if ( cache == 0 )
		return(0u);
	SparkPrefixCacheCoreQueryStats(&cache->core,&stats);
	return(stats.free_block_count);
}

uint32_t SparkPagedKvCheckpointOffer(SparkPagedKv *cache,
	uint32_t lane, uint64_t end_position, uint32_t *slot_out)
{
	uint32_t boundary,slot,victim_slot;
	uint64_t victim_use;
	if ( cache == 0 || slot_out == 0 ||
		lane >= cache->configuration.lane_count )
		return(0u);
	*slot_out = SPARK_PAGED_KV_NO_SLOT;
	if ( cache->reuse_enabled == 0u ||
		cache->reserved_slot != SPARK_PAGED_KV_NO_SLOT ||
		cache->lane_live[lane] == 0u )
		return(0u);
	if ( end_position == 0u ||
		end_position % cache->configuration.block_token_count != 0u )
		return(0u);
	boundary = (uint32_t)(end_position /
		cache->configuration.block_token_count);
	victim_slot = SPARK_PAGED_KV_NO_SLOT;
	victim_use = UINT64_MAX;
	for (slot=0u; slot<cache->configuration.checkpoint_slot_count; slot++)
	{
		const SparkPagedKvCheckpoint *checkpoint =
			&cache->checkpoints[slot];
		if ( checkpoint->live == 0u )
		{
			victim_slot = slot;
			victim_use = 0u;
			break;
		}
		if ( checkpoint->lane == lane &&
			checkpoint->boundary_blocks < boundary &&
			checkpoint->last_use < victim_use )
		{
			victim_slot = slot;
			victim_use = checkpoint->last_use;
		}
	}
	if ( victim_slot == SPARK_PAGED_KV_NO_SLOT )
	{
		 


		for (slot=0u; slot<cache->configuration.checkpoint_slot_count;
			slot++)
		{
			if ( cache->checkpoints[slot].last_use < victim_use )
			{
				victim_use = cache->checkpoints[slot].last_use;
				victim_slot = slot;
			}
		}
	}
	if ( victim_slot == SPARK_PAGED_KV_NO_SLOT )
		return(0u);
	cache->reserved_slot = victim_slot;
	cache->reserved_lane = lane;
	*slot_out = victim_slot;
	return(1u);
}

void SparkPagedKvCheckpointCommit(SparkPagedKv *cache,
	uint32_t lane, uint32_t slot, uint64_t end_position)
{
	SparkPagedKvCheckpoint *checkpoint;
	uint32_t boundary,row_base,witness;
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		slot >= cache->configuration.checkpoint_slot_count ||
		cache->reserved_slot != slot || cache->reserved_lane != lane )
		return;
	cache->reserved_slot = SPARK_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_PAGED_KV_NO_LANE;
	boundary = (uint32_t)(end_position /
		cache->configuration.block_token_count);
	if ( boundary == 0u || boundary > cache->lane_core_blocks[lane] )
		return;
	 






	if ( end_position > SparkPagedKvCommittedTokens(cache,lane) )
		return;
	row_base = SparkPagedKvRowBase(cache,lane);
	witness = cache->blocks_by_lane[row_base + boundary - 1u];
	if ( witness == SPARK_PAGED_KV_NO_BLOCK )
		return;
	checkpoint = &cache->checkpoints[slot];
	checkpoint->live = 1u;
	checkpoint->lane = lane;
	checkpoint->boundary_blocks = boundary;
	checkpoint->witness_block = witness;
	checkpoint->last_use = ++cache->lru_clock;
}

void SparkPagedKvCheckpointAbort(SparkPagedKv *cache,
	uint32_t lane, uint32_t slot)
{
	if ( cache == 0 || cache->reserved_slot != slot ||
		cache->reserved_lane != lane )
		return;
	cache->reserved_slot = SPARK_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_PAGED_KV_NO_LANE;
}
