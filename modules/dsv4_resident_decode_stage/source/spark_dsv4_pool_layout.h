#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

static inline uint64_t SparkDsv4PoolCompressStateLaneElements(uint32_t kind);
static inline uint64_t SparkDsv4PoolIndexStateLaneElements(void);

#define SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS \
	SPARK_DSV4_RESIDENT_DECODE_STAGE_CACHE_BLOCK_TOKENS
#define SPARK_DSV4_PAGED_POOL_ALIGNMENT_BYTES 4096u

typedef struct SparkDsv4PagedLayerLayout
{
	uint32_t layer_kind;
	uint32_t attention_entry_capacity;
	uint32_t index_entry_capacity;
	uint32_t reserved0;
	uint64_t attention_offset_bytes;
	uint64_t compressor_kv_offset_bytes;
	uint64_t compressor_score_offset_bytes;
	uint64_t index_cache_offset_bytes;
	uint64_t index_kv_offset_bytes;
	uint64_t index_score_offset_bytes;
}
SparkDsv4PagedLayerLayout;

typedef struct SparkDsv4PagedPoolLayout
{
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t block_token_count;
	uint32_t reserved0;
	uint64_t page_stride_bytes;
	SparkDsv4PagedLayerLayout layers[SPARK_DSV4_MODEL_LAYER_COUNT];
}
SparkDsv4PagedPoolLayout;

typedef struct SparkDsv4PagedScoreSpan
{
	uint64_t offset_words;
	uint64_t element_count;
}
SparkDsv4PagedScoreSpan;

static inline uint64_t SparkDsv4PagedPoolAlignBytes(uint64_t value)
{
	return((value + SPARK_DSV4_PAGED_POOL_ALIGNMENT_BYTES - 1u) &
		~((uint64_t)SPARK_DSV4_PAGED_POOL_ALIGNMENT_BYTES - 1u));
}

static inline int32_t SparkDsv4PagedPoolAppend(
	uint64_t bytes,
	uint64_t *cursor,
	uint64_t *offset)
{
	uint64_t aligned;
	if ( cursor == 0 || offset == 0 )
		return(-1);
	aligned = SparkDsv4PagedPoolAlignBytes(*cursor);
	if ( aligned < *cursor || bytes > UINT64_MAX - aligned )
		return(-2);
	*offset = aligned;
	*cursor = aligned + bytes;
	return(0);
}

static inline uint32_t SparkDsv4PagedPoolCompressedEntries(uint32_t kind)
{
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		return(SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS /
			SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA )
		return(SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS /
			SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO);
	return(0u);
}

static inline int32_t SparkDsv4PagedPoolAppendLayer(
	uint32_t layer,
	SparkDsv4PagedPoolLayout *layout,
	uint64_t *cursor)
{
	SparkDsv4PagedLayerLayout *layer_layout;
	uint64_t bytes,state_elements;
	uint32_t compressed,kind;
	kind = SparkDsv4ModelLayerKind(layer);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_INVALID )
		return(-1);
	layer_layout = &layout->layers[layer];
	layer_layout->layer_kind = kind;
	compressed = SparkDsv4PagedPoolCompressedEntries(kind);
	layer_layout->attention_entry_capacity =
		SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS + compressed;
	bytes = (uint64_t)layer_layout->attention_entry_capacity *
		SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->attention_offset_bytes) != 0 )
		return(-2);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		return(0);
	state_elements = SparkDsv4PoolCompressStateLaneElements(kind);
	bytes = state_elements * sizeof(float);
	if ( SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->compressor_kv_offset_bytes) != 0 ||
		SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->compressor_score_offset_bytes) != 0 )
		return(-3);
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		return(0);
	layer_layout->index_entry_capacity = compressed;
	bytes = (uint64_t)compressed * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION *
		SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->index_cache_offset_bytes) != 0 )
		return(-4);
	bytes = SparkDsv4PoolIndexStateLaneElements() * sizeof(float);
	if ( SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->index_kv_offset_bytes) != 0 ||
		SparkDsv4PagedPoolAppend(bytes,cursor,
		&layer_layout->index_score_offset_bytes) != 0 )
		return(-5);
	return(0);
}

static inline int32_t SparkDsv4PagedPoolBuildLayout(
	uint32_t first_layer_index,
	uint32_t layer_count,
	SparkDsv4PagedPoolLayout *layout)
{
	uint64_t cursor;
	uint32_t layer;
	if ( layout == 0 || layer_count == 0u ||
		first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT ||
		layer_count > SPARK_DSV4_MODEL_LAYER_COUNT - first_layer_index )
		return(-1);
	memset(layout,0,sizeof(*layout));
	layout->first_layer_index = first_layer_index;
	layout->layer_count = layer_count;
	layout->block_token_count = SPARK_DSV4_PAGED_POOL_BLOCK_TOKENS;
	cursor = 0u;
	for (layer=first_layer_index; layer<first_layer_index+layer_count; layer++)
		if ( SparkDsv4PagedPoolAppendLayer(layer,layout,&cursor) != 0 )
			return(-2);
	layout->page_stride_bytes = SparkDsv4PagedPoolAlignBytes(cursor);
	return(layout->page_stride_bytes >= cursor ? 0 : -3);
}

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
