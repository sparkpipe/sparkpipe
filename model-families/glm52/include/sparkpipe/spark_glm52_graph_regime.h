#pragma once

#include <stdint.h>

#include "inference/kernels/attn_split.h"
#include "sparkpipe/spark_glm52_model.h"

#define SPARK_GLM52_GRAPH_REGIME_UNSPLIT 0u
#define SPARK_GLM52_GRAPH_REGIME_SPLIT 1u
#define SPARK_GLM52_GRAPH_REGIME_SELECTED 2u
#define SPARK_GLM52_GRAPH_CONTEXT_STEP 4096u

static inline uint32_t SparkGlm52GraphRegimeCount(uint32_t max_positions)
{
	if ( max_positions <= SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
		return(SPARK_GLM52_GRAPH_REGIME_SELECTED);
	return(SPARK_GLM52_GRAPH_REGIME_SELECTED + (max_positions + SPARK_GLM52_GRAPH_CONTEXT_STEP - 1u) / SPARK_GLM52_GRAPH_CONTEXT_STEP);
}

static inline uint32_t SparkGlm52GraphReplayable(uint32_t context,uint32_t max_positions)
{
	return(context != 0u && context <= max_positions ? 1u : 0u);
}

static inline uint32_t SparkGlm52GraphRegime(uint32_t context,uint32_t split_threshold,uint32_t max_positions,uint32_t *bound)
{
	uint32_t limit,steps;
	if ( context > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
	{
		steps = (context + SPARK_GLM52_GRAPH_CONTEXT_STEP - 1u) / SPARK_GLM52_GRAPH_CONTEXT_STEP;
		*bound = steps * SPARK_GLM52_GRAPH_CONTEXT_STEP < max_positions ? steps * SPARK_GLM52_GRAPH_CONTEXT_STEP : max_positions;
		return(SPARK_GLM52_GRAPH_REGIME_SELECTED + steps - 1u);
	}
	limit = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT < max_positions ? SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT : max_positions;
	if ( LmLatentAttentionContextSplits(context,split_threshold) != 0u )
	{
		*bound = limit;
		return(SPARK_GLM52_GRAPH_REGIME_SPLIT);
	}
	*bound = split_threshold != 0u && split_threshold - 1u < limit ? split_threshold - 1u : limit;
	return(SPARK_GLM52_GRAPH_REGIME_UNSPLIT);
}

static inline uint32_t SparkGlm52GraphRowClass(uint32_t tokens,uint32_t split_threshold,uint32_t max_positions)
{
	uint32_t bound;
	if ( tokens > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
		return(SPARK_GLM52_GRAPH_REGIME_SELECTED);
	return(SparkGlm52GraphRegime(tokens,split_threshold,max_positions,&bound));
}
