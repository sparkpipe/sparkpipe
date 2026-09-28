#pragma once

#include <stdint.h>

#include "inference/kernels/attn_split.h"
#include "sparkpipe/spark_glm5_next_model.h"

#define SPARK_GLM5_NEXT_GRAPH_REGIME_UNSPLIT 0u
#define SPARK_GLM5_NEXT_GRAPH_REGIME_SPLIT 1u
#define SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED 2u
#define SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT 3u
#define SPARK_GLM5_NEXT_GRAPH_HEAD_GREEDY 0u
#define SPARK_GLM5_NEXT_GRAPH_HEAD_SAMPLED 1u
#define SPARK_GLM5_NEXT_GRAPH_HEAD_COUNT 2u
#define SPARK_GLM5_NEXT_GRAPH_KEY_COUNT (SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT * SPARK_GLM5_NEXT_GRAPH_HEAD_COUNT)
#define SPARK_GLM5_NEXT_GRAPH_CONTEXT_MARGIN 256u

static inline uint32_t SparkGlm5NextAttentionPositionBound(uint32_t context)
{
	return(context > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K ? SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K : context);
}

static inline uint32_t SparkGlm5NextAttentionSplit(uint32_t context,uint32_t split_threshold)
{
	return(LmLatentAttentionContextSplits(SparkGlm5NextAttentionPositionBound(context),split_threshold));
}

static inline uint32_t SparkGlm5NextGraphRegime(uint32_t context,uint32_t split_threshold)
{
	if ( context > SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K )
		return(SPARK_GLM5_NEXT_GRAPH_REGIME_SELECTED);
	return(SparkGlm5NextAttentionSplit(context,split_threshold) != 0u ? SPARK_GLM5_NEXT_GRAPH_REGIME_SPLIT : SPARK_GLM5_NEXT_GRAPH_REGIME_UNSPLIT);
}

static inline uint32_t SparkGlm5NextGraphKey(uint32_t context,uint32_t split_threshold,uint32_t sampled)
{
	return(SparkGlm5NextGraphRegime(context,split_threshold) + (sampled != 0u ? SPARK_GLM5_NEXT_GRAPH_HEAD_SAMPLED : SPARK_GLM5_NEXT_GRAPH_HEAD_GREEDY) * SPARK_GLM5_NEXT_GRAPH_REGIME_COUNT);
}

static inline uint32_t SparkGlm5NextGraphBound(uint32_t context,uint32_t split_threshold,uint32_t max_positions)
{
	uint32_t regime,bound;
	regime = SparkGlm5NextGraphRegime(context,split_threshold);
	bound = context + SPARK_GLM5_NEXT_GRAPH_CONTEXT_MARGIN;
	if ( regime == SPARK_GLM5_NEXT_GRAPH_REGIME_UNSPLIT )
		bound = split_threshold != 0u && split_threshold <= SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K ? split_threshold - 1u : SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K;
	if ( regime == SPARK_GLM5_NEXT_GRAPH_REGIME_SPLIT )
		bound = SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K;
	return(bound < max_positions ? bound : max_positions);
}
