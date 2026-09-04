
#ifndef SPARK_LM_GROUP_GEMM_WORKSPACE_H
#define SPARK_LM_GROUP_GEMM_WORKSPACE_H

#include <stdint.h>

#define LM_WS_ALIGNMENT 256u
#define LM_WS_NVFP4_GROUP 16u
#define LM_WS_REGION_COUNT 9u
#define LM_WS_NVFP4_TILE_K 256u
#define SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL 4u
#define LM_WS_SHARED_LIMIT 131072u

#define LM_WS_REGION_PACKED_HIDDEN 0u
#define LM_WS_REGION_PACKED_HIDDEN_SCALE 1u
#define LM_WS_REGION_ROUTE_ROWS 2u
#define LM_WS_REGION_ROUTE_INDPTR 3u
#define LM_WS_REGION_GROUP_TILE_PREFIX 4u
#define LM_WS_REGION_GATE_UP_BF16 5u
#define LM_WS_REGION_INTERMEDIATE 6u
#define LM_WS_REGION_INTERMEDIATE_SCALE 7u
#define LM_WS_REGION_ROUTE_OUTPUT_BF16 8u

#define LM_WS_OK 0
#define LM_WS_ERR_NULL (-31)
#define LM_WS_ERR_SHAPE (-32)
#define LM_WS_ERR_GROUP (-33)
#define LM_WS_ERR_OVERFLOW (-34)
#define LM_WS_ERR_SHARED (-35)

#define LM_WS_UNFUSED_REGION_MASK ( \
	(1u << LM_WS_REGION_PACKED_HIDDEN) | \
	(1u << LM_WS_REGION_PACKED_HIDDEN_SCALE) | \
	(1u << LM_WS_REGION_GATE_UP_BF16) | \
	(1u << LM_WS_REGION_INTERMEDIATE) | \
	(1u << LM_WS_REGION_INTERMEDIATE_SCALE) | \
	(1u << LM_WS_REGION_ROUTE_OUTPUT_BF16))

typedef struct LmWorkspaceshape
{
	uint32_t tokens,top_k,expert_count,hidden_dimension,intermediate_dimension,tile_m,tile_n;
}
LmWorkspaceshape_t;

typedef struct LmWorkspacelayout
{
	uint64_t offset[LM_WS_REGION_COUNT];
	uint64_t bytes[LM_WS_REGION_COUNT];
	uint64_t total_bytes;
	uint64_t packed_rows;
	uint64_t total_tiles;
	uint64_t shared_bytes;
	uint32_t tile_m;
	uint32_t stages;
	uint32_t ctas_per_sm;
}
LmWorkspacelayout_t;

static uint64_t LmWorkspacealign_up(uint64_t value)
{
	return((value + (uint64_t)LM_WS_ALIGNMENT - 1u)
		& ~((uint64_t)LM_WS_ALIGNMENT - 1u));
}

static uint64_t LmWorkspacepacked_rows(const LmWorkspaceshape_t *shape)
{
	return((uint64_t)shape->tokens * (uint64_t)shape->top_k);
}

static uint64_t LmWorkspacebytes_for_max_batch(uint32_t max_tokens, uint32_t top_k, uint32_t expert_count, uint32_t hidden_dimension, uint32_t intermediate_dimension);

static uint32_t LmWorkspaceselect_tile_m(uint64_t rows_per_expert)
{
	if ( rows_per_expert <= 16u )
		return(16u);
	if ( rows_per_expert <= 32u )
		return(32u);
	return(64u);
}

static uint64_t LmWorkspaceshared_bytes(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t stages, uint32_t element_bits)
{
	uint64_t per_stage;
	per_stage = ((uint64_t)tile_m + (uint64_t)tile_n) * (uint64_t)tile_k
		* (uint64_t)element_bits / 8u;
	return((uint64_t)stages * (per_stage + 16u));
}

static uint32_t LmWorkspaceselect_stages(uint32_t tile_m, uint32_t tile_n, uint32_t tile_k, uint32_t element_bits, uint64_t shared_limit)
{
	if ( LmWorkspaceshared_bytes(tile_m,tile_n,tile_k,2u,element_bits) <= shared_limit )
		return(2u);
	return(0u);
}

static uint32_t LmWorkspacectas_per_sm(uint64_t shared_bytes, uint64_t shared_limit)
{
	if ( shared_bytes == 0u )
		return(0u);
	return((uint32_t)(shared_limit / shared_bytes));
}

static uint64_t LmWorkspacepeak_rows_per_expert(const LmWorkspaceshape_t *shape)
{
	uint64_t mean_rows;
	mean_rows = (LmWorkspacepacked_rows(shape) + (uint64_t)shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	return(mean_rows * 2u);
}

static uint64_t LmWorkspacetotal_tiles_for_tile_m(const LmWorkspaceshape_t *shape, uint32_t tile_m, uint32_t output_dimension)
{
	uint64_t rows_per_expert,m_tiles,n_tiles;
	rows_per_expert = (LmWorkspacepacked_rows(shape) + shape->expert_count - 1u)
		/ (uint64_t)shape->expert_count;
	m_tiles = (rows_per_expert + (uint64_t)tile_m - 1u) / (uint64_t)tile_m;
	if ( m_tiles == 0u )
		m_tiles = 1u;
	n_tiles = ((uint64_t)output_dimension + (uint64_t)shape->tile_n - 1u) / (uint64_t)shape->tile_n;
	return((uint64_t)shape->expert_count * m_tiles * n_tiles);
}

static int32_t LmWorkspacelayout_build(const LmWorkspaceshape_t *shape, LmWorkspacelayout_t *layout)
{
	uint64_t rows,hidden_bytes,hidden_scales,intermediate_bytes,intermediate_scales,cursor;
	uint32_t region,effective;
	if ( shape == 0 || layout == 0 )
		return(LM_WS_ERR_NULL);
	if ( shape->tokens == 0 || shape->top_k == 0 || shape->expert_count == 0
		|| shape->hidden_dimension == 0 || shape->intermediate_dimension == 0
		|| shape->tile_n == 0 )
		return(LM_WS_ERR_SHAPE);
	if ( (shape->hidden_dimension % LM_WS_NVFP4_GROUP) != 0u
		|| (shape->intermediate_dimension % LM_WS_NVFP4_GROUP) != 0u )
		return(LM_WS_ERR_GROUP);
	effective = shape->tile_m != 0u
		? shape->tile_m
		: LmWorkspaceselect_tile_m(LmWorkspacepeak_rows_per_expert(shape));
	layout->tile_m = effective;
	layout->stages = LmWorkspaceselect_stages(effective,shape->tile_n,
		LM_WS_NVFP4_TILE_K,SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL,
		LM_WS_SHARED_LIMIT);
	if ( layout->stages == 0u )
		return(LM_WS_ERR_SHARED);
	layout->shared_bytes = LmWorkspaceshared_bytes(effective,shape->tile_n,
		LM_WS_NVFP4_TILE_K,layout->stages,
		SPARK_LM_TENSOR_MAP_BITS_NVFP4_LOCAL);
	layout->ctas_per_sm = LmWorkspacectas_per_sm(layout->shared_bytes,
		LM_WS_SHARED_LIMIT);
	rows = LmWorkspacepacked_rows(shape);
	if ( rows == 0u || rows > 0xffffffffu )
		return(LM_WS_ERR_OVERFLOW);
	hidden_bytes = rows * ((uint64_t)shape->hidden_dimension / 2u);
	hidden_scales = rows * ((uint64_t)shape->hidden_dimension / LM_WS_NVFP4_GROUP);
	intermediate_bytes = rows * ((uint64_t)shape->intermediate_dimension / 2u);
	intermediate_scales = rows * ((uint64_t)shape->intermediate_dimension / LM_WS_NVFP4_GROUP);
	layout->bytes[LM_WS_REGION_PACKED_HIDDEN] = hidden_bytes;
	layout->bytes[LM_WS_REGION_PACKED_HIDDEN_SCALE] = hidden_scales;
	layout->bytes[LM_WS_REGION_ROUTE_ROWS] = rows * sizeof(uint32_t);
	layout->bytes[LM_WS_REGION_ROUTE_INDPTR] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	layout->bytes[LM_WS_REGION_GROUP_TILE_PREFIX] = ((uint64_t)shape->expert_count + 1u) * sizeof(uint32_t);
	layout->bytes[LM_WS_REGION_GATE_UP_BF16] = rows * (uint64_t)shape->intermediate_dimension * 2u * 2u;
	layout->bytes[LM_WS_REGION_INTERMEDIATE] = intermediate_bytes;
	layout->bytes[LM_WS_REGION_INTERMEDIATE_SCALE] = intermediate_scales;
	layout->bytes[LM_WS_REGION_ROUTE_OUTPUT_BF16] = rows * (uint64_t)shape->hidden_dimension * 2u;
	cursor = 0u;
	for (region = 0u; region < LM_WS_REGION_COUNT; ++region)
	{
		layout->offset[region] = cursor;
		cursor = LmWorkspacealign_up(cursor + layout->bytes[region]);
	}
	layout->total_bytes = cursor;
	layout->packed_rows = rows;
	layout->total_tiles = LmWorkspacetotal_tiles_for_tile_m(shape,effective,
		shape->intermediate_dimension * 2u);
	return(LM_WS_OK);
}

static uint64_t LmWorkspacebytes_for_max_batch(uint32_t max_tokens, uint32_t top_k, uint32_t expert_count, uint32_t hidden_dimension, uint32_t intermediate_dimension)
{
	LmWorkspaceshape_t shape;
	LmWorkspacelayout_t layout;
	uint32_t index;
	for (index = 0u; index < sizeof(shape); ++index)
		((uint8_t *)&shape)[index] = 0u;
	shape.tokens = max_tokens;
	shape.top_k = top_k;
	shape.expert_count = expert_count;
	shape.hidden_dimension = hidden_dimension;
	shape.intermediate_dimension = intermediate_dimension;
	shape.tile_n = 128u;
	if ( LmWorkspacelayout_build(&shape,&layout) != LM_WS_OK )
		return(0u);
	return(layout.total_bytes);
}

#endif
