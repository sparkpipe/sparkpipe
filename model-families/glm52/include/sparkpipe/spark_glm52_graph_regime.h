#pragma once

#include <stdint.h>

#include "inference/kernels/attn_split.h"
#include "sparkpipe/spark_glm52_model.h"

#define SPARK_GLM52_GRAPH_REGIME_UNSPLIT 0u
#define SPARK_GLM52_GRAPH_REGIME_SPLIT 1u
#define SPARK_GLM52_GRAPH_REGIME_SELECTED 2u
#define SPARK_GLM52_GRAPH_CONTEXT_STEP 4096u
#define SPARK_GLM52_GRAPH_LINEAR_BUCKETS 4u
#define SPARK_GLM52_GRAPH_OCTAVE_BUCKETS 4u

static inline uint32_t SparkGlm52GraphContextBucket(uint32_t context,uint32_t *bound)
{
	const uint32_t linear = SPARK_GLM52_GRAPH_CONTEXT_STEP * SPARK_GLM52_GRAPH_LINEAR_BUCKETS;
	uint32_t octave,step,steps;
	if ( context <= linear )
	{
		steps = (context + SPARK_GLM52_GRAPH_CONTEXT_STEP - 1u) / SPARK_GLM52_GRAPH_CONTEXT_STEP;
		*bound = steps * SPARK_GLM52_GRAPH_CONTEXT_STEP;
		return(steps - 1u);
	}
	octave = 31u - (uint32_t)__builtin_clz(context - 1u);
	step = (1u << octave) / SPARK_GLM52_GRAPH_OCTAVE_BUCKETS;
	steps = (context + step - 1u) / step;
	*bound = steps * step;
	return(SPARK_GLM52_GRAPH_LINEAR_BUCKETS + (octave - (uint32_t)__builtin_ctz(linear)) * SPARK_GLM52_GRAPH_OCTAVE_BUCKETS + steps - SPARK_GLM52_GRAPH_OCTAVE_BUCKETS - 1u);
}

static inline uint32_t SparkGlm52GraphRegimeCount(uint32_t max_positions)
{
	uint32_t bound;
	if ( max_positions <= SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
		return(SPARK_GLM52_GRAPH_REGIME_SELECTED);
	return(SPARK_GLM52_GRAPH_REGIME_SELECTED + SparkGlm52GraphContextBucket(max_positions,&bound) + 1u);
}

static inline uint32_t SparkGlm52GraphReplayable(uint32_t context,uint32_t max_positions)
{
	return(context != 0u && context <= max_positions ? 1u : 0u);
}

static inline uint32_t SparkGlm52GraphRegime(uint32_t context,uint32_t split_threshold,uint32_t max_positions,uint32_t *bound)
{
	uint32_t limit,bucket;
	if ( context > SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT )
	{
		bucket = SparkGlm52GraphContextBucket(context,bound);
		if ( *bound > max_positions )
			*bound = max_positions;
		return(SPARK_GLM52_GRAPH_REGIME_SELECTED + bucket);
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
