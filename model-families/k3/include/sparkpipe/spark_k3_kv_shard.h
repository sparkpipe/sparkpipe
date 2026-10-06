#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_llm_defines.h"
#include "sparkpipe/spark_kv_shard.h"

#if defined(__CUDACC__)
#define SPARK_K3_KV_SHARD_FN static inline __host__ __device__
#else
#define SPARK_K3_KV_SHARD_FN static inline
#endif

#define SPARK_K3_KV_SHARD_UNIT SPARK_K3_MODEL_HIDDEN_DIMENSION
#define SPARK_K3_KV_SHARD_QUERY_WIDTH \
	(SPARK_K3_MODEL_MLA_LATENT_DIMENSION + SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION)
#define SPARK_K3_KV_SHARD_RECORD_FLOATS (SPARK_K3_MODEL_MLA_LATENT_DIMENSION + 2u)

SPARK_K3_KV_SHARD_FN SparkKvShard SparkK3KvShardContext(uint32_t rank,uint32_t degree)
{
	SparkKvShard shard;
	shard.degree = degree;
	shard.rank = rank;
	shard.grain = 1u;
	return(shard);
}

SPARK_K3_KV_SHARD_FN uint32_t SparkK3KvShardHeads(uint32_t degree)
{
	return(degree == 0u ? 0u : SPARK_K3_MODEL_MLA_HEAD_COUNT / degree);
}

SPARK_K3_KV_SHARD_FN uint32_t SparkK3KvShardQuerySequences(uint32_t rows,uint32_t degree)
{
	uint64_t elements = (uint64_t)rows * SparkK3KvShardHeads(degree) * SPARK_K3_KV_SHARD_QUERY_WIDTH;
	return((uint32_t)((elements + SPARK_K3_KV_SHARD_UNIT - 1u) / SPARK_K3_KV_SHARD_UNIT));
}

SPARK_K3_KV_SHARD_FN uint64_t SparkK3KvShardQueryStride(uint32_t rows,uint32_t degree)
{
	return((uint64_t)SparkK3KvShardQuerySequences(rows,degree) * SPARK_K3_KV_SHARD_UNIT);
}

SPARK_K3_KV_SHARD_FN uint32_t SparkK3KvShardPartialSequences(uint32_t rows,uint32_t degree)
{
	uint64_t elements = (uint64_t)rows * SparkK3KvShardHeads(degree) * SPARK_K3_KV_SHARD_RECORD_FLOATS * 2u;
	return((uint32_t)((elements + SPARK_K3_KV_SHARD_UNIT - 1u) / SPARK_K3_KV_SHARD_UNIT));
}

SPARK_K3_KV_SHARD_FN uint64_t SparkK3KvShardPartialStride(uint32_t rows,uint32_t degree)
{
	return((uint64_t)SparkK3KvShardPartialSequences(rows,degree) * (SPARK_K3_KV_SHARD_UNIT / 2u));
}

SPARK_K3_KV_SHARD_FN uint32_t SparkK3KvShardFits(uint32_t degree,uint32_t rows)
{
	if ( degree < 2u || degree > SPARK_KV_SHARD_MAX_DEGREE || rows == 0u ||
		SPARK_K3_MODEL_MLA_HEAD_COUNT % degree != 0u ||
		SparkKvShardValid(SparkK3KvShardContext(0u,degree),SPARK_K3_KV_PAGE_SLOTS) == 0u )
		return(0u);
	return(SparkK3KvShardQuerySequences(rows,degree) <= rows && SparkK3KvShardPartialSequences(rows,degree) <= rows ? 1u : 0u);
}

SPARK_K3_KV_SHARD_FN uint64_t SparkK3KvShardScratchBytes(uint32_t rows,uint32_t degree)
{
	return((uint64_t)degree * SparkK3KvShardQueryStride(rows,degree) * sizeof(uint16_t) +
		2u * (uint64_t)degree * SparkK3KvShardPartialStride(rows,degree) * sizeof(float));
}
