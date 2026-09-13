#include "spark_glm52_paged_kv.h"








static const SparkPagedKvGeometryCallbacks spark_glm52_paged_kv_geometry =
{
	  SPARK_GLM52_PAGED_KV_SEQUENCE_BASE,
	  SPARK_GLM52_PAGED_KV_MAX_BLOCKS_PER_LANE,
	  0
};

SparkStatus SparkGlm52PagedKvInitialize(SparkGlm52PagedKv *cache,
	const SparkGlm52PagedKvConfiguration *configuration,
	uint32_t *blocks_by_lane, uint32_t *counts_by_lane)
{
	return SparkPagedKvInitialize(cache,configuration,
		&spark_glm52_paged_kv_geometry,blocks_by_lane,counts_by_lane);
}

void SparkGlm52PagedKvDestroy(SparkGlm52PagedKv *cache)
{
	SparkPagedKvDestroy(cache);
}

void SparkGlm52PagedKvLaneReset(SparkGlm52PagedKv *cache, uint32_t lane)
{
	SparkPagedKvLaneReset(cache,lane);
}

SparkStatus SparkGlm52PagedKvAdmit(SparkGlm52PagedKv *cache,
	uint32_t lane, const uint32_t *tokens, uint32_t token_count,
	SparkGlm52PagedKvMatch *match_out)
{
	return SparkPagedKvAdmit(cache,lane,tokens,token_count,match_out);
}

SparkStatus SparkGlm52PagedKvCover(SparkGlm52PagedKv *cache,
	uint32_t lane, uint64_t end_position, const uint32_t *tokens,
	uint32_t token_count)
{
	return SparkPagedKvCover(cache,lane,end_position,tokens,token_count);
}

uint64_t SparkGlm52PagedKvCommittedTokens(
	const SparkGlm52PagedKv *cache, uint32_t lane)
{
	return SparkPagedKvCommittedTokens(cache,lane);
}

uint32_t SparkGlm52PagedKvFreeBlocks(const SparkGlm52PagedKv *cache)
{
	return SparkPagedKvFreeBlocks(cache);
}

uint32_t SparkGlm52PagedKvCheckpointOffer(SparkGlm52PagedKv *cache,
	uint32_t lane, uint64_t end_position, uint32_t *slot_out)
{
	return SparkPagedKvCheckpointOffer(cache,lane,end_position,slot_out);
}

void SparkGlm52PagedKvCheckpointCommit(SparkGlm52PagedKv *cache,
	uint32_t lane, uint32_t slot, uint64_t end_position)
{
	SparkPagedKvCheckpointCommit(cache,lane,slot,end_position);
}

void SparkGlm52PagedKvCheckpointAbort(SparkGlm52PagedKv *cache,
	uint32_t lane, uint32_t slot)
{
	SparkPagedKvCheckpointAbort(cache,lane,slot);
}
