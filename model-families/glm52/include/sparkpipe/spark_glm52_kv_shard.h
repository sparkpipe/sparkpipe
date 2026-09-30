#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_kv_shard.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#if defined(__CUDACC__)
#define SPARK_GLM52_KV_SHARD_FN static inline __host__ __device__
#else
#define SPARK_GLM52_KV_SHARD_FN static inline
#endif

#define SPARK_GLM52_KV_SHARD_PAGE_SLOTS 64u
#define SPARK_GLM52_KV_SHARD_UNIT SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_KV_SHARD_QUERY_WIDTH (SPARK_GLM52_MODEL_LATENT_DIMENSION + SPARK_GLM52_MODEL_ROPE_DIMENSION)
#define SPARK_GLM52_KV_SHARD_RECORD_FLOATS (SPARK_GLM52_MODEL_LATENT_DIMENSION + 2u)
#define SPARK_GLM52_KV_SHARD_LATENT_TOKEN_BYTES \
	((uint64_t)SPARK_GLM52_MODEL_LAYER_COUNT * SPARK_GLM52_KV_SHARD_QUERY_WIDTH * 2u)
#define SPARK_GLM52_KV_SHARD_INDEX_FIRST_LAYERS 3u
#define SPARK_GLM52_KV_SHARD_INDEX_GROUP_LAYER 6u

typedef struct SparkGlm52KvShardLayerRounds
{
	uint32_t query_rounds;
	uint32_t partial_rounds;
	uint64_t query_wire_bytes;
	uint64_t partial_wire_bytes;
}
SparkGlm52KvShardLayerRounds;

SPARK_GLM52_KV_SHARD_FN SparkKvShard SparkGlm52KvShardLatent(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = 1u;
	return(shard);
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardHeads(uint32_t degree)
{
	return(degree == 0u || SPARK_GLM52_MODEL_HEAD_COUNT % degree != 0u ? 0u : SPARK_GLM52_MODEL_HEAD_COUNT / degree);
}

SPARK_GLM52_KV_SHARD_FN uint64_t SparkGlm52KvShardQueryElements(uint32_t rows,uint32_t degree)
{
	return((uint64_t)rows * SparkGlm52KvShardHeads(degree) * SPARK_GLM52_KV_SHARD_QUERY_WIDTH);
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardQuerySequences(uint32_t rows,uint32_t degree)
{
	return((uint32_t)((SparkGlm52KvShardQueryElements(rows,degree) + SPARK_GLM52_KV_SHARD_UNIT - 1u) / SPARK_GLM52_KV_SHARD_UNIT));
}

SPARK_GLM52_KV_SHARD_FN uint64_t SparkGlm52KvShardPartialElements(uint32_t rows,uint32_t degree)
{
	return((uint64_t)rows * SparkGlm52KvShardHeads(degree) * SPARK_GLM52_KV_SHARD_RECORD_FLOATS * 2u);
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardPartialSequences(uint32_t rows,uint32_t degree)
{
	return((uint32_t)((SparkGlm52KvShardPartialElements(rows,degree) + SPARK_GLM52_KV_SHARD_UNIT - 1u) / SPARK_GLM52_KV_SHARD_UNIT));
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardFits(uint32_t degree,uint32_t rows)
{
	if ( degree < 2u || degree > SPARK_KV_SHARD_MAX_DEGREE || rows == 0u || SparkGlm52KvShardHeads(degree) == 0u ||
		SparkKvShardValid(SparkGlm52KvShardLatent(0u,degree),SPARK_GLM52_KV_SHARD_PAGE_SLOTS) == 0u )
		return(0u);
	return(SparkGlm52KvShardQuerySequences(rows,degree) <= rows && SparkGlm52KvShardPartialSequences(rows,degree) <= rows ? 1u : 0u);
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardIndexLayers(void)
{
	return(SPARK_GLM52_KV_SHARD_INDEX_FIRST_LAYERS + (SPARK_GLM52_MODEL_LAYER_COUNT - SPARK_GLM52_KV_SHARD_INDEX_GROUP_LAYER - 1u) / SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT + 1u);
}

SPARK_GLM52_KV_SHARD_FN uint64_t SparkGlm52KvShardIndexTokenBytes(void)
{
	return((uint64_t)SparkGlm52KvShardIndexLayers() * SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * 2u);
}

SPARK_GLM52_KV_SHARD_FN uint64_t SparkGlm52KvShardTokenBytes(uint32_t degree)
{
	return((degree == 0u ? 0u : SPARK_GLM52_KV_SHARD_LATENT_TOKEN_BYTES / degree) + SparkGlm52KvShardIndexTokenBytes());
}

SPARK_GLM52_KV_SHARD_FN uint32_t SparkGlm52KvShardLayerRoundsBuild(uint32_t rows,uint32_t degree,uint64_t slot_bytes,uint64_t peer_slot_bytes,SparkGlm52KvShardLayerRounds *plan)
{
	uint64_t query,partial,peer_capacity;
	if ( plan == 0 || SparkGlm52KvShardFits(degree,rows) == 0u || slot_bytes <= 16u )
		return(0u);
	query = (uint64_t)SparkGlm52KvShardQuerySequences(rows,degree) * SPARK_GLM52_KV_SHARD_UNIT;
	partial = (uint64_t)SparkGlm52KvShardPartialSequences(rows,degree) * SPARK_GLM52_KV_SHARD_UNIT;
	plan->query_rounds = (uint32_t)SparkTpMeshDirectChunks(query * degree,degree,SPARK_TP_MESH_OPERATION_ALL_GATHER,slot_bytes);
	if ( peer_slot_bytes != 0u )
	{
		peer_capacity = SparkTpMeshDirectCapacity(peer_slot_bytes,SPARK_TP_MESH_OPERATION_ALL_GATHER);
		plan->partial_rounds = peer_capacity == 0u ? 0u : (uint32_t)((partial + peer_capacity - 1u) / peer_capacity);
	}
	else
		plan->partial_rounds = (uint32_t)SparkTpMeshAllToAllChunks(partial,degree,slot_bytes);
	plan->query_wire_bytes = (uint64_t)(degree - 1u) * query * 2u;
	plan->partial_wire_bytes = (uint64_t)(degree - 1u) * partial * 2u;
	return(plan->partial_rounds != 0u ? 1u : 0u);
}
