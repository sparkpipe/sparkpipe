#include "spark_k3_paged_kv.h"

/*
 * Thin port shim: every call forwards to the shared paged-KV engine
 * (runtime/paged_kv_common.c) with this stage's geometry callback row
 * bound. No allocator logic lives here - the engine is shared verbatim
 * with the other ports of the family.
 */

static const SparkPagedKvGeometryCallbacks spark_k3_paged_kv_geometry =
{
	/* .sequence_id_base */ SPARK_K3_PAGED_KV_SEQUENCE_BASE,
	/* .max_blocks_per_lane */ SPARK_K3_PAGED_KV_MAX_BLOCKS_PER_LANE,
	/* .validate_configuration */ 0
};

SparkStatus SparkK3PagedKvInitialize(SparkK3PagedKv *cache,
	const SparkK3PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane, uint32_t *counts_by_lane)
{
	return SparkPagedKvInitialize(cache,configuration,
		&spark_k3_paged_kv_geometry,blocks_by_lane,counts_by_lane);
}

void SparkK3PagedKvDestroy(SparkK3PagedKv *cache)
{
	SparkPagedKvDestroy(cache);
}

void SparkK3PagedKvLaneReset(SparkK3PagedKv *cache, uint32_t lane)
{
	SparkPagedKvLaneReset(cache,lane);
}

SparkStatus SparkK3PagedKvAdmit(SparkK3PagedKv *cache,
	uint32_t lane, const uint32_t *tokens, uint32_t token_count,
	SparkK3PagedKvMatch *match_out)
{
	return SparkPagedKvAdmit(cache,lane,tokens,token_count,match_out);
}

SparkStatus SparkK3PagedKvCover(SparkK3PagedKv *cache,
	uint32_t lane, uint64_t end_position, const uint32_t *tokens,
	uint32_t token_count)
{
	return SparkPagedKvCover(cache,lane,end_position,tokens,token_count);
}

uint64_t SparkK3PagedKvCommittedTokens(
	const SparkK3PagedKv *cache, uint32_t lane)
{
	return SparkPagedKvCommittedTokens(cache,lane);
}

uint32_t SparkK3PagedKvFreeBlocks(const SparkK3PagedKv *cache)
{
	return SparkPagedKvFreeBlocks(cache);
}

uint32_t SparkK3PagedKvCheckpointOffer(SparkK3PagedKv *cache,
	uint32_t lane, uint64_t end_position, uint32_t *slot_out)
{
	return SparkPagedKvCheckpointOffer(cache,lane,end_position,slot_out);
}

void SparkK3PagedKvCheckpointCommit(SparkK3PagedKv *cache,
	uint32_t lane, uint32_t slot, uint64_t end_position)
{
	SparkPagedKvCheckpointCommit(cache,lane,slot,end_position);
}

void SparkK3PagedKvCheckpointAbort(SparkK3PagedKv *cache,
	uint32_t lane, uint32_t slot)
{
	SparkPagedKvCheckpointAbort(cache,lane,slot);
}
