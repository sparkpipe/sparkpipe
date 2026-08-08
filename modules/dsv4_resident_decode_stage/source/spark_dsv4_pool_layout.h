#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv4_model.h"

static inline uint64_t SparkDsv4PoolCacheLaneElements(uint32_t max_sequence_positions,uint32_t kind)
{
	uint64_t slots = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		slots += max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		slots += max_sequence_positions / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
	return(slots * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
}

static inline uint64_t SparkDsv4PoolCompressStateLaneElements(uint32_t kind)
{
	uint64_t overlap,ratio;
	overlap = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR : 1u;
	ratio = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
	return(overlap * ratio * overlap * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
}

static inline uint64_t SparkDsv4PoolLaneOffset(uint64_t layer_offset,uint64_t lane_stride,uint32_t lane)
{
	return(layer_offset + ((uint64_t)lane * lane_stride));
}

static inline uint64_t SparkDsv4PoolIndexCacheLaneElements(uint32_t max_sequence_positions)
{
	return((uint64_t)(max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO) * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
}

static inline uint64_t SparkDsv4PoolIndexStateLaneElements(void)
{
	return((uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO * SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
}

static inline uint64_t SparkDsv4PoolResidentStateBytes(uint64_t cache_elements,uint64_t state_elements,uint32_t csa_layer_count,uint32_t resident_sequence_capacity,uint32_t max_sequence_positions)
{
	uint64_t index_cache,index_state;
	index_cache = SparkDsv4PoolIndexCacheLaneElements(max_sequence_positions) * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	index_state = 2u * SparkDsv4PoolIndexStateLaneElements() * sizeof(float);
	return(cache_elements * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES + 2u * state_elements * sizeof(float) + (uint64_t)csa_layer_count * resident_sequence_capacity * (index_cache + index_state));
}

static inline int32_t SparkDsv4PoolBuildLayout(uint32_t first_layer_index,uint32_t layer_count,uint32_t resident_sequence_capacity,uint32_t max_sequence_positions,uint64_t *cache_offsets,uint64_t *cache_strides,uint64_t *state_offsets,uint64_t *state_strides,uint64_t *cache_elements,uint64_t *state_elements)
{
	uint64_t cache_cursor = 0u,state_cursor = 0u,stride;
	uint32_t kind,layer;
	if ( layer_count == 0u || first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT || layer_count > SPARK_DSV4_MODEL_LAYER_COUNT - first_layer_index || resident_sequence_capacity == 0u || max_sequence_positions < SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO || max_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS || cache_offsets == 0 || cache_strides == 0 || state_offsets == 0 || state_strides == 0 || cache_elements == 0 || state_elements == 0 )
		return(-1);
	for (layer=0u; layer<SPARK_DSV4_MODEL_LAYER_COUNT; layer++)
	{
		cache_offsets[layer] = 0u;
		cache_strides[layer] = 0u;
		state_offsets[layer] = 0u;
		state_strides[layer] = 0u;
	}
	for (layer=first_layer_index; layer<first_layer_index+layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_INVALID )
			return(-2);
		stride = SparkDsv4PoolCacheLaneElements(max_sequence_positions,kind);
		cache_offsets[layer] = cache_cursor;
		cache_strides[layer] = stride;
		cache_cursor += ((uint64_t)resident_sequence_capacity * stride);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
			continue;
		stride = SparkDsv4PoolCompressStateLaneElements(kind);
		state_offsets[layer] = state_cursor;
		state_strides[layer] = stride;
		state_cursor += ((uint64_t)resident_sequence_capacity * stride);
	}
	*cache_elements = cache_cursor;
	*state_elements = state_cursor;
	return(0);
}
