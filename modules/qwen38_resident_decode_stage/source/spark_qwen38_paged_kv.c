#include "spark_qwen38_paged_kv.h"

/*
 * Thin port shim: every call forwards to the shared paged-KV engine
 * (runtime/paged_kv_common.c) with this stage's geometry callback row
 * bound. No allocator logic lives here - the engine is shared verbatim
 * with the other ports of the family.
 */

static const SparkPagedKvGeometryCallbacks spark_qwen38_paged_kv_geometry =
{
	/* .sequence_id_base */ SPARK_QWEN38_PAGED_KV_SEQUENCE_BASE,
	/* .max_blocks_per_lane */ SPARK_QWEN38_PAGED_KV_MAX_BLOCKS_PER_LANE,
	/* .validate_configuration */ 0
};

SparkStatus SparkQwen38PagedKvInitialize(SparkQwen38PagedKv *cache,
	const SparkQwen38PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane, uint32_t *counts_by_lane)
{
	return SparkPagedKvInitialize(cache,configuration,
		&spark_qwen38_paged_kv_geometry,blocks_by_lane,counts_by_lane);
}

void SparkQwen38PagedKvDestroy(SparkQwen38PagedKv *cache)
{
	SparkPagedKvDestroy(cache);
}

void SparkQwen38PagedKvLaneReset(SparkQwen38PagedKv *cache, uint32_t lane)
{
	SparkPagedKvLaneReset(cache,lane);
}

SparkStatus SparkQwen38PagedKvAdmit(SparkQwen38PagedKv *cache,
	uint32_t lane, const uint32_t *tokens, uint32_t token_count,
	SparkQwen38PagedKvMatch *match_out)
{
	return SparkPagedKvAdmit(cache,lane,tokens,token_count,match_out);
}

SparkStatus SparkQwen38PagedKvCover(SparkQwen38PagedKv *cache,
	uint32_t lane, uint64_t end_position, const uint32_t *tokens,
	uint32_t token_count)
{
	return SparkPagedKvCover(cache,lane,end_position,tokens,token_count);
}

uint64_t SparkQwen38PagedKvCommittedTokens(
	const SparkQwen38PagedKv *cache, uint32_t lane)
{
	return SparkPagedKvCommittedTokens(cache,lane);
}

uint32_t SparkQwen38PagedKvFreeBlocks(const SparkQwen38PagedKv *cache)
{
	return SparkPagedKvFreeBlocks(cache);
}

uint32_t SparkQwen38PagedKvCheckpointOffer(SparkQwen38PagedKv *cache,
	uint32_t lane, uint64_t end_position, uint32_t *slot_out)
{
	return SparkPagedKvCheckpointOffer(cache,lane,end_position,slot_out);
}

void SparkQwen38PagedKvCheckpointCommit(SparkQwen38PagedKv *cache,
	uint32_t lane, uint32_t slot, uint64_t end_position)
{
	SparkPagedKvCheckpointCommit(cache,lane,slot,end_position);
}

void SparkQwen38PagedKvCheckpointAbort(SparkQwen38PagedKv *cache,
	uint32_t lane, uint32_t slot)
{
	SparkPagedKvCheckpointAbort(cache,lane,slot);
}
