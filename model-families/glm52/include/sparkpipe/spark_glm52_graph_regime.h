#pragma once

#include <stdint.h>

#include "inference/kernels/attn_split.h"
#include "sparkpipe/spark_glm52_model.h"

#define SPARK_GLM52_GRAPH_REGIME_UNSPLIT 0u
#define SPARK_GLM52_GRAPH_REGIME_SPLIT 1u
#define SPARK_GLM52_GRAPH_REGIME_COUNT 2u

static inline uint32_t SparkGlm52GraphReplayable(uint32_t context)
{
	return(context != 0u && context <= SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT ? 1u : 0u);
}

static inline uint32_t SparkGlm52GraphRegime(uint32_t context,uint32_t split_threshold,uint32_t max_positions,uint32_t *bound)
{
	uint32_t limit;
	limit = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT < max_positions ? SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT : max_positions;
	if ( LmLatentAttentionContextSplits(context,split_threshold) != 0u )
	{
		*bound = limit;
		return(SPARK_GLM52_GRAPH_REGIME_SPLIT);
	}
	*bound = split_threshold != 0u && split_threshold - 1u < limit ? split_threshold - 1u : limit;
	return(SPARK_GLM52_GRAPH_REGIME_UNSPLIT);
}
