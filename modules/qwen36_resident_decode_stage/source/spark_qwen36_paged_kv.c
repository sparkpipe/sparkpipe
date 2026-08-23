#include "spark_qwen36_paged_kv.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Core sequence ids are derived from the lane index inside a private
 * range: one live residency per lane, so id collisions cannot happen. */
#define SPARK_QWEN36_PAGED_KV_SEQUENCE_BASE UINT64_C(0x5133600000000000)

static uint64_t SparkQwen36PagedKvSequenceId(uint32_t lane)
{
	return SPARK_QWEN36_PAGED_KV_SEQUENCE_BASE + (uint64_t)lane;
}

static uint32_t SparkQwen36PagedKvRowBase(const SparkQwen36PagedKv *cache,
	uint32_t lane)
{
	return (uint32_t)((uint64_t)lane *
		cache->configuration.blocks_per_lane);
}

/* Borrow one physical block from the core's free list for private
 * scratch. Evicts least-recently-used published blocks when the pool is
 * exhausted, exactly like the core's own open-block allocation.
 * Seam decision (audit DRY note): this is DELIBERATELY module-local -
 * the core's dead ReservePrivateTail/CopyBlockList reserved-tail API is
 * deleted and a shared borrow helper was rejected (core surface vs the
 * Solutions/(codesize x 2) ceiling, zero callers once PRIVATE marks
 * scratch), so this is the ONE borrow path any later port inherits. */
static SparkStatus SparkQwen36PagedKvScratchBorrow(
	SparkQwen36PagedKv *cache, uint32_t *block_out)
{
	uint32_t block,evicted;
	if ( cache->core.free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
		(void)SparkPrefixCacheCoreTrim(&cache->core,1u,&evicted);
	if ( cache->core.free_block_head == SPARK_PREFIX_CACHE_CORE_NO_BLOCK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	block = cache->core.free_block_head;
	cache->core.free_block_head =
		cache->core.blocks[block].free_next;
	/* Mark the block PRIVATE (the core's own open-block allocation
	 * does exactly this): leaving it BLOCK_FREE made QueryStats count
	 * in-use scratch as free, so SparkQwen36PagedKvFreeBlocks() and
	 * the spec-coverage feasibility check over-reported pool room and
	 * borrows failed mid-coverage instead of being refused up front.
	 * PRIVATE keeps scratch out of free_block_count; it is still
	 * invisible to Trim (never on the LRU) and to sequences (never in
	 * a sequence's block list). */
	cache->core.blocks[block].state = SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE;
	cache->core.blocks[block].reference_count = 0u;
	cache->core.blocks[block].token_count = 0u;
	*block_out = block;
	return(SPARK_STATUS_OK);
}

/* Return borrowed scratch to the core free list: PRIVATE -> FREE and
 * back onto the same linkage the core's single free path maintains
 * (scratch never sits on the LRU or content index, so nothing else
 * needs clearing). */
static void SparkQwen36PagedKvScratchReturn(SparkQwen36PagedKv *cache,
	uint32_t block)
{
	cache->core.blocks[block].state = SPARK_PREFIX_CACHE_CORE_BLOCK_FREE;
	cache->core.blocks[block].reference_count = 0u;
	cache->core.blocks[block].token_count = 0u;
	cache->core.blocks[block].free_next = cache->core.free_block_head;
	cache->core.free_block_head = block;
}

/* Sync the lane's table row from the cache state: the core-committed
 * prefix comes out of the sequence's block table, everything beyond it
 * is borrowed scratch. The committed frontier can only advance while NO
 * scratch is outstanding (Cover refuses appends otherwise), so a row
 * already holding scratch keeps it: resetting counts to the core-owned
 * count here would discard the record of outstanding scratch and the
 * next Cover would re-borrow straight over slots still naming last
 * round's blocks - unreachable by Trim (never LRU/indexed) AND by
 * LaneReset (now past counts_by_lane). Permanent leak per decode step.
 * BuildBlockTable refreshes only the core-owned prefix slice in place. */
static void SparkQwen36PagedKvSyncRow(SparkQwen36PagedKv *cache,
	uint32_t lane)
{
	uint32_t row_base = SparkQwen36PagedKvRowBase(cache,lane);
	uint32_t committed_blocks = 0u,outstanding;
	if ( cache->lane_live[lane] != 0u )
	{
		uint32_t tokens,blocks;
		tokens = (uint32_t)SparkPrefixCacheCoreSequenceTokenCount(
			&cache->core,SparkQwen36PagedKvSequenceId(lane));
		blocks = (tokens + cache->configuration.block_token_count - 1u) /
			cache->configuration.block_token_count;
		if ( blocks > cache->configuration.blocks_per_lane )
			blocks = cache->configuration.blocks_per_lane;
		if ( blocks != 0u )
			(void)SparkPrefixCacheCoreBuildBlockTable(&cache->core,
				SparkQwen36PagedKvSequenceId(lane),tokens,
				cache->blocks_by_lane + row_base,blocks,
				&committed_blocks);
	}
	outstanding = cache->counts_by_lane[lane];
	cache->lane_core_blocks[lane] = committed_blocks;
	/* Outstanding scratch survives the sync; the frontier can never be
	 * below it while the record is valid (LaneReset zeroes both). */
	if ( outstanding > committed_blocks )
		return;
	cache->counts_by_lane[lane] = committed_blocks;
}

SparkStatus SparkQwen36PagedKvInitialize(SparkQwen36PagedKv *cache,
	const SparkQwen36PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane, uint32_t *counts_by_lane)
{
	SparkPrefixCacheCoreConfiguration core_configuration;
	uint32_t lane,index,buckets;
	SparkStatus status;
	if ( cache == 0 || configuration == 0 || blocks_by_lane == 0 ||
		counts_by_lane == 0 || configuration->block_token_count == 0u ||
		configuration->block_token_count >
			SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS ||
		configuration->lane_count == 0u ||
		configuration->blocks_per_lane == 0u ||
		configuration->blocks_per_lane >
			SPARK_QWEN36_PAGED_KV_MAX_BLOCKS_PER_LANE ||
		configuration->physical_page_capacity == 0u ||
		configuration->logical_page_capacity <
			configuration->physical_page_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(cache,0,sizeof(*cache));
	cache->configuration = *configuration;
	cache->blocks_by_lane = blocks_by_lane;
	cache->counts_by_lane = counts_by_lane;
	cache->reuse_enabled = configuration->checkpoint_slot_count != 0u ? 1u : 0u;
	cache->reserved_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_QWEN36_PAGED_KV_NO_LANE;
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
	/* Content-index buckets: power of two, ~2x the pool. */
	buckets = 16u;
	while ( buckets < configuration->physical_page_capacity * 2u &&
		buckets < (uint32_t)1u << 31u )
		buckets <<= 1u;
	core_configuration.hash_bucket_count = buckets;
	status = SparkPrefixCacheCoreInitialize(&cache->core,
		&core_configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen36PagedKvDestroy(cache);
		return(status);
	}
	if ( cache->reuse_enabled != 0u )
	{
		cache->checkpoints = (SparkQwen36PagedKvCheckpoint *)calloc(
			configuration->checkpoint_slot_count,
			sizeof(cache->checkpoints[0]));
		if ( cache->checkpoints == 0 )
		{
			SparkQwen36PagedKvDestroy(cache);
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		for (index=0u; index<configuration->checkpoint_slot_count; index++)
			cache->checkpoints[index].witness_block =
				SPARK_QWEN36_PAGED_KV_NO_BLOCK;
	}
	cache->lane_core_blocks = (uint32_t *)calloc(
		configuration->lane_count,sizeof(uint32_t));
	cache->lane_live = (uint32_t *)calloc(
		configuration->lane_count,sizeof(uint32_t));
	if ( cache->lane_core_blocks == 0 || cache->lane_live == 0 )
	{
		SparkQwen36PagedKvDestroy(cache);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (lane=0u; lane<configuration->lane_count; lane++)
	{
		for (index=0u; index<configuration->blocks_per_lane; index++)
			blocks_by_lane[((uint64_t)lane *
				configuration->blocks_per_lane) + index] =
				SPARK_QWEN36_PAGED_KV_NO_BLOCK;
		counts_by_lane[lane] = 0u;
	}
	return(SPARK_STATUS_OK);
}

void SparkQwen36PagedKvDestroy(SparkQwen36PagedKv *cache)
{
	if ( cache == 0 )
		return;
	SparkPrefixCacheCoreDestroy(&cache->core);
	free(cache->checkpoints);
	free(cache->lane_core_blocks);
	free(cache->lane_live);
	memset(cache,0,sizeof(*cache));
}

void SparkQwen36PagedKvLaneReset(SparkQwen36PagedKv *cache, uint32_t lane)
{
	uint32_t slot,row_base,index;
	if ( cache == 0 || lane >= cache->configuration.lane_count )
		return;
	row_base = SparkQwen36PagedKvRowBase(cache,lane);
	if ( cache->lane_live[lane] != 0u )
	{
		(void)SparkPrefixCacheCoreReleaseSequence(&cache->core,
			SparkQwen36PagedKvSequenceId(lane));
		cache->lane_live[lane] = 0u;
	}
	/* Ordinals at and beyond the core prefix are borrowed scratch. */
	for (index=cache->lane_core_blocks[lane];
		index<cache->counts_by_lane[lane]; index++)
	{
		uint32_t block = cache->blocks_by_lane[row_base + index];
		if ( block != SPARK_QWEN36_PAGED_KV_NO_BLOCK )
			SparkQwen36PagedKvScratchReturn(cache,block);
	}
	for (index=0u; index<cache->configuration.blocks_per_lane; index++)
		cache->blocks_by_lane[row_base + index] =
			SPARK_QWEN36_PAGED_KV_NO_BLOCK;
	cache->counts_by_lane[lane] = 0u;
	cache->lane_core_blocks[lane] = 0u;
	for (slot=0u; slot<cache->configuration.checkpoint_slot_count; slot++)
		if ( cache->checkpoints[slot].live != 0u &&
			cache->checkpoints[slot].lane == lane )
		{
			cache->checkpoints[slot].live = 0u;
			cache->checkpoints[slot].witness_block =
				SPARK_QWEN36_PAGED_KV_NO_BLOCK;
		}
	if ( cache->reserved_lane == lane )
	{
		cache->reserved_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
		cache->reserved_lane = SPARK_QWEN36_PAGED_KV_NO_LANE;
	}
}

/* Deepest boundary <= limit_blocks whose checkpoint is bound, alive, and
 * witnessed by exactly the attached block at that ordinal. */
static uint32_t SparkQwen36PagedKvWitnessedDepth(
	const SparkQwen36PagedKv *cache,
	const uint32_t *attached_blocks,
	uint32_t limit_blocks,
	uint32_t *slot_out)
{
	uint32_t boundary,slot;
	for (boundary=limit_blocks; boundary>=1u; boundary--)
	{
		for (slot=0u; slot<cache->configuration.checkpoint_slot_count;
			slot++)
		{
			const SparkQwen36PagedKvCheckpoint *checkpoint =
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

SparkStatus SparkQwen36PagedKvAdmit(SparkQwen36PagedKv *cache,
	uint32_t lane, const uint32_t *tokens, uint32_t token_count,
	SparkQwen36PagedKvMatch *match_out)
{
	SparkQwen36PagedKvMatch match;
	uint32_t attached[SPARK_QWEN36_PAGED_KV_MAX_BLOCKS_PER_LANE];
	uint32_t attached_count,lcp_tokens,resumed_blocks,readmit_tokens,slot;
	uint32_t block_tokens;
	SparkStatus status;
	match.block_count = 0u;
	match.checkpoint_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	if ( match_out != 0 )
	{
		match_out->block_count = 0u;
		match_out->checkpoint_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	}
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		tokens == 0 || token_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_tokens = cache->configuration.block_token_count;
	/* A fresh prompt owns its lane from zero. */
	SparkQwen36PagedKvLaneReset(cache,lane);
	if ( cache->reuse_enabled == 0u )
		return(SPARK_STATUS_OK);
	status = SparkPrefixCacheCoreAdmitSequence(&cache->core,
		SparkQwen36PagedKvSequenceId(lane),tokens,token_count,
		&lcp_tokens);
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen36PagedKvLaneReset(cache,lane);
		return(status);
	}
	cache->lane_live[lane] = 1u;
	/* AdmitSequence reports matched TOKENS (a block-granular multiple). */
	resumed_blocks = 0u;
	slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	if ( lcp_tokens != 0u )
	{
		uint32_t lcp_block_count;
		lcp_block_count = lcp_tokens / block_tokens;
		status = SparkPrefixCacheCoreBuildBlockTable(&cache->core,
			SparkQwen36PagedKvSequenceId(lane),lcp_tokens,attached,
			lcp_block_count,&attached_count);
		if ( status == SPARK_STATUS_OK &&
			attached_count == lcp_block_count &&
			lcp_block_count <= cache->configuration.blocks_per_lane )
			resumed_blocks = SparkQwen36PagedKvWitnessedDepth(cache,
				attached,lcp_block_count,&slot);
		else
			resumed_blocks = 0u;
		if ( resumed_blocks < lcp_block_count )
		{
			/* The GDN clamp bites: reuse is only sound at boundaries
			 * whose donor checkpoint is still bound, so the sequence
			 * must hold EXACTLY [0,resumed_blocks) of donor blocks and
			 * take fresh private continuation past it - never write
			 * inside an immutable published block. With NO witnessed
			 * boundary the walk restarts at zero, which must not keep
			 * ANY attachment: re-admit fewer than one block so the
			 * sequence opens private and every later publish stays
			 * canonically aligned with the core's chain hashing. */
			(void)SparkPrefixCacheCoreReleaseSequence(&cache->core,
				SparkQwen36PagedKvSequenceId(lane));
			readmit_tokens = resumed_blocks != 0u ?
				resumed_blocks * block_tokens : block_tokens - 1u;
			if ( readmit_tokens > token_count )
				readmit_tokens = token_count;
			status = SparkPrefixCacheCoreAdmitSequence(&cache->core,
				SparkQwen36PagedKvSequenceId(lane),tokens,
				readmit_tokens,&lcp_tokens);
			if ( status != SPARK_STATUS_OK ||
				lcp_tokens != (resumed_blocks != 0u ?
					readmit_tokens : 0u) )
			{
				SparkQwen36PagedKvLaneReset(cache,lane);
				return(status != SPARK_STATUS_OK ? status :
					SPARK_STATUS_INTERNAL_ERROR);
			}
		}
	}
	match.block_count = resumed_blocks;
	match.checkpoint_slot = slot;
	SparkQwen36PagedKvSyncRow(cache,lane);
	if ( match_out != 0 )
		*match_out = match;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen36PagedKvCover(SparkQwen36PagedKv *cache,
	uint32_t lane, uint64_t end_position, const uint32_t *tokens,
	uint32_t token_count)
{
	uint32_t block_tokens,required,count,row_base,ordinal,appended,block;
	SparkStatus status;
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		end_position == 0u || (tokens == 0 && token_count != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_tokens = cache->configuration.block_token_count;
	required = (uint32_t)((end_position + block_tokens - 1u) / block_tokens);
	if ( required > cache->configuration.blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	/* Appending is only sound while the row is purely core-owned: once
	 * scratch covers positions past the committed frontier (speculative
	 * rounds, post-scratch decode growth), or on a reuse-disabled lane
	 * with no sequence at all, the device truth lives in borrowed blocks
	 * and the tokens are ignored - coverage growth is borrow-only. */
	if ( token_count != 0u && cache->lane_live[lane] != 0u &&
		cache->counts_by_lane[lane] == cache->lane_core_blocks[lane] )
	{
		status = SparkPrefixCacheCoreAppendTokens(&cache->core,
			SparkQwen36PagedKvSequenceId(lane),tokens,token_count);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	SparkQwen36PagedKvSyncRow(cache,lane);
	row_base = SparkQwen36PagedKvRowBase(cache,lane);
	count = cache->counts_by_lane[lane];
	appended = 0u;
	for (ordinal=count; ordinal<required; ordinal++)
	{
		status = SparkQwen36PagedKvScratchBorrow(cache,&block);
		if ( status != SPARK_STATUS_OK )
		{
			/* Leave the partial growth visible: the caller drops the
			 * whole submission on coverage failure. The count moves
			 * WITH each borrow so LaneReset can still return every
			 * attached block - a count left behind here would orphan
			 * the just-borrowed blocks exactly like the F1 bug. */
			cache->counts_by_lane[lane] = count + appended;
			return(status);
		}
		cache->blocks_by_lane[row_base + ordinal] = block;
		appended++;
	}
	cache->counts_by_lane[lane] = count + appended;
	return(SPARK_STATUS_OK);
}

uint64_t SparkQwen36PagedKvCommittedTokens(
	const SparkQwen36PagedKv *cache, uint32_t lane)
{
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		cache->lane_live[lane] == 0u )
		return(0u);
	return SparkPrefixCacheCoreSequenceTokenCount(&cache->core,
		SparkQwen36PagedKvSequenceId(lane));
}

uint32_t SparkQwen36PagedKvFreeBlocks(const SparkQwen36PagedKv *cache)
{
	SparkPrefixCacheCoreStats stats;
	if ( cache == 0 )
		return(0u);
	SparkPrefixCacheCoreQueryStats(&cache->core,&stats);
	return(stats.free_block_count);
}

uint32_t SparkQwen36PagedKvCheckpointOffer(SparkQwen36PagedKv *cache,
	uint32_t lane, uint64_t end_position, uint32_t *slot_out)
{
	uint32_t boundary,slot,victim_slot;
	uint64_t victim_use;
	if ( cache == 0 || slot_out == 0 ||
		lane >= cache->configuration.lane_count )
		return(0u);
	*slot_out = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	if ( cache->reuse_enabled == 0u ||
		cache->reserved_slot != SPARK_QWEN36_PAGED_KV_NO_SLOT ||
		cache->lane_live[lane] == 0u )
		return(0u);
	if ( end_position == 0u ||
		end_position % cache->configuration.block_token_count != 0u )
		return(0u);
	boundary = (uint32_t)(end_position /
		cache->configuration.block_token_count);
	victim_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	victim_use = UINT64_MAX;
	for (slot=0u; slot<cache->configuration.checkpoint_slot_count; slot++)
	{
		const SparkQwen36PagedKvCheckpoint *checkpoint =
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
	if ( victim_slot == SPARK_QWEN36_PAGED_KV_NO_SLOT )
	{
		/* All slots hold other lanes: steal the global LRU. A stolen
		 * checkpoint shortens later matches but never turns them
		 * wrong - its witness leaves with it. */
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
	if ( victim_slot == SPARK_QWEN36_PAGED_KV_NO_SLOT )
		return(0u);
	cache->reserved_slot = victim_slot;
	cache->reserved_lane = lane;
	*slot_out = victim_slot;
	return(1u);
}

void SparkQwen36PagedKvCheckpointCommit(SparkQwen36PagedKv *cache,
	uint32_t lane, uint32_t slot, uint64_t end_position)
{
	SparkQwen36PagedKvCheckpoint *checkpoint;
	uint32_t boundary,row_base,witness;
	if ( cache == 0 || lane >= cache->configuration.lane_count ||
		slot >= cache->configuration.checkpoint_slot_count ||
		cache->reserved_slot != slot || cache->reserved_lane != lane )
		return;
	cache->reserved_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_QWEN36_PAGED_KV_NO_LANE;
	boundary = (uint32_t)(end_position /
		cache->configuration.block_token_count);
	if ( boundary == 0u || boundary > cache->lane_core_blocks[lane] )
		return;
	/* lane_core_blocks counts ceil(committed/BLOCK) blocks, so on a lane
	 * whose committed token count is not block-aligned the top counted
	 * block is still OPEN - later appends keep writing into it. A commit
	 * whose end_position runs past the committed frontier rounds down
	 * into that open block and binds the GDN witness to a mutable block
	 * (audit F4): the witness must sit inside the committed tokens. */
	if ( end_position >
		SparkQwen36PagedKvCommittedTokens(cache,lane) )
		return;
	row_base = SparkQwen36PagedKvRowBase(cache,lane);
	witness = cache->blocks_by_lane[row_base + boundary - 1u];
	if ( witness == SPARK_QWEN36_PAGED_KV_NO_BLOCK )
		return;
	checkpoint = &cache->checkpoints[slot];
	checkpoint->live = 1u;
	checkpoint->lane = lane;
	checkpoint->boundary_blocks = boundary;
	checkpoint->witness_block = witness;
	checkpoint->last_use = ++cache->lru_clock;
}

void SparkQwen36PagedKvCheckpointAbort(SparkQwen36PagedKv *cache,
	uint32_t lane, uint32_t slot)
{
	if ( cache == 0 || cache->reserved_slot != slot ||
		cache->reserved_lane != lane )
		return;
	cache->reserved_slot = SPARK_QWEN36_PAGED_KV_NO_SLOT;
	cache->reserved_lane = SPARK_QWEN36_PAGED_KV_NO_LANE;
}
