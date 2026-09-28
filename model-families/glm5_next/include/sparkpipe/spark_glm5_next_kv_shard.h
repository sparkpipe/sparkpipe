#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_kv_shard.h"

#if defined(__CUDACC__)
#define SPARK_GLM5_NEXT_KV_SHARD_FN static inline __host__ __device__
#else
#define SPARK_GLM5_NEXT_KV_SHARD_FN static inline
#endif

#define SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_UNIT \
	(SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT)
#define SPARK_GLM5_NEXT_KV_SHARD_RECORD_FLOATS \
	(SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION + 2u)

SPARK_GLM5_NEXT_KV_SHARD_FN SparkKvShard SparkGlm5NextKvShardLatent(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = 1u;
	return(shard);
}

SPARK_GLM5_NEXT_KV_SHARD_FN SparkKvShard SparkGlm5NextKvShardIndex(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL;
	return(shard);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardHeads(uint32_t degree)
{
	return(degree == 0u ? 0u : SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT / degree);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardQuerySequences(uint32_t rows,uint32_t degree)
{
	uint64_t elements = (uint64_t)rows * SparkGlm5NextKvShardHeads(degree) * SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION;
	return((uint32_t)((elements + SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardPartialSequences(uint32_t rows,uint32_t degree)
{
	uint64_t elements = (uint64_t)rows * SparkGlm5NextKvShardHeads(degree) * SPARK_GLM5_NEXT_KV_SHARD_RECORD_FLOATS * 2u;
	return((uint32_t)((elements + SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_UNIT - 1u) / SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_UNIT));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardQueryStride(uint32_t rows,uint32_t degree)
{
	return((uint64_t)SparkGlm5NextKvShardQuerySequences(rows,degree) * SPARK_GLM5_NEXT_KV_SHARD_QUERY_UNIT);
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint64_t SparkGlm5NextKvShardPartialStride(uint32_t rows,uint32_t degree)
{
	return((uint64_t)SparkGlm5NextKvShardPartialSequences(rows,degree) * (SPARK_GLM5_NEXT_KV_SHARD_PARTIAL_UNIT / 2u));
}

SPARK_GLM5_NEXT_KV_SHARD_FN uint32_t SparkGlm5NextKvShardFits(uint32_t degree,uint32_t rows)
{
	if ( degree < 2u || degree > SPARK_KV_SHARD_MAX_DEGREE || rows == 0u || SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT % degree != 0u ||
		SparkKvShardValid(SparkGlm5NextKvShardLatent(0u,degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS) == 0u ||
		SparkKvShardValid(SparkGlm5NextKvShardIndex(0u,degree),SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS) == 0u )
		return(0u);
	return(SparkGlm5NextKvShardQuerySequences(rows,degree) <= rows && SparkGlm5NextKvShardPartialSequences(rows,degree) <= rows ? 1u : 0u);
}
