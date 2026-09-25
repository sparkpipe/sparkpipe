#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"

#if defined(__CUDACC__)
#define SPARK_GLM5_NEXT_INDEX_CP_FN static inline __host__ __device__
#else
#define SPARK_GLM5_NEXT_INDEX_CP_FN static inline
#endif

#define SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE \
	(SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL)
#define SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS \
	(SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION / 2u)

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpPools(uint32_t context)
{
	return(context / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpLocalStride(uint32_t pools,uint32_t degree)
{
	uint32_t pages;
	pages = (pools + SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE - 1u) / SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE;
	return(degree <= 1u ? pools : ((pages + degree - 1u) / degree) * SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpGlobalPool(uint32_t local,uint32_t rank,uint32_t degree)
{
	return(degree <= 1u ? local : ((local / SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE) * degree + rank) * SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE + local % SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpOwner(uint32_t pool,uint32_t degree)
{
	return(degree <= 1u ? 0u : (pool / SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE) % degree);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpLocalPool(uint32_t pool,uint32_t degree)
{
	return(degree <= 1u ? pool : ((pool / SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE) / degree) * SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE + pool % SPARK_GLM5_NEXT_INDEX_CP_POOLS_PER_PAGE);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpGatherSequences(uint32_t rows,uint32_t local_stride)
{
	return((uint32_t)(((uint64_t)rows * local_stride + SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS - 1u) / SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS));
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpActive(uint32_t context,uint32_t degree)
{
	return(degree > 1u && context > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K ? 1u : 0u);
}

SPARK_GLM5_NEXT_INDEX_CP_FN uint32_t SparkGlm5NextIndexCpFits(uint32_t max_context,uint32_t degree,uint32_t rows)
{
	return(SparkGlm5NextIndexCpGatherSequences(rows,SparkGlm5NextIndexCpLocalStride(SparkGlm5NextIndexCpPools(max_context),degree)) <= rows ? 1u : 0u);
}
