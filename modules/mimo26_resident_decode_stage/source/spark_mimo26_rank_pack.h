#pragma once

#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_mimo26_model.h"
#include "sparkpipe/spark_status.h"

#include "spark_mimo26_stagepack_format.h"

#define SPARK_MIMO26_RANK_PACK_ABSENT UINT32_MAX
#define SPARK_MIMO26_RANK_PACK_GLOBAL_COUNT 3u
#define SPARK_MIMO26_RANK_PACK_FP8_BLOCK 128u

typedef struct SparkMimo26RankPackLayout
{
	uint32_t tp_degree;
	uint32_t global_entry[SPARK_MIMO26_RANK_PACK_GLOBAL_COUNT];
	uint32_t layer_entry[SPARK_MIMO26_MODEL_LAYER_COUNT][SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT];
} SparkMimo26RankPackLayout;

typedef struct SparkMimo26RankPackShape
{
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint64_t payload_bytes;
	uint64_t scale_bytes;
} SparkMimo26RankPackShape;

static inline uint64_t SparkMimo26RankPackFp8ScaleBytes(uint32_t rows, uint32_t columns)
{
	return((uint64_t)((rows + SPARK_MIMO26_RANK_PACK_FP8_BLOCK - 1u) / SPARK_MIMO26_RANK_PACK_FP8_BLOCK) * (columns / SPARK_MIMO26_RANK_PACK_FP8_BLOCK) * sizeof(float));
}

static inline uint32_t SparkMimo26RankPackShapeSet(SparkMimo26RankPackShape *shape, uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	shape->weight_format = weight_format;
	shape->rows = rows;
	shape->columns = columns;
	shape->scale_bytes = 0u;
	if ( weight_format == SPARK_MIMO26_STAGEPACK_WEIGHT_BF16 )
		shape->payload_bytes = (uint64_t)rows * columns * sizeof(uint16_t);
	else if ( weight_format == SPARK_MIMO26_STAGEPACK_WEIGHT_F32 )
		shape->payload_bytes = (uint64_t)rows * columns * sizeof(float);
	else if ( weight_format == SPARK_MIMO26_STAGEPACK_WEIGHT_FP8_E4M3_F32B128 )
	{
		shape->payload_bytes = (uint64_t)rows * columns;
		shape->scale_bytes = SparkMimo26RankPackFp8ScaleBytes(rows,columns);
	}
	else
	{
		shape->payload_bytes = (uint64_t)rows * columns / 2u;
		shape->scale_bytes = (uint64_t)rows * columns / SPARK_MIMO26_MODEL_EXPERT_MXFP4_SCALE_BLOCK;
	}
	return(1u);
}

static inline uint32_t SparkMimo26RankPackQkvRows(uint32_t layer, uint32_t tp_degree)
{
	uint32_t kv = SparkMimo26ModelLayerKind(layer) == SPARK_MIMO26_MODEL_LAYER_KIND_SWA ? SPARK_MIMO26_MODEL_SWA_KV_HEAD_COUNT : SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT;
	return(SparkMimo26StagePackQkvRankRows(SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT,kv,SPARK_MIMO26_MODEL_ATTN_HEAD_DIMENSION,SPARK_MIMO26_MODEL_ATTN_VALUE_DIMENSION,tp_degree));
}

static inline uint32_t SparkMimo26RankPackExpectedShape(uint32_t kind, uint32_t layer, uint32_t tp_degree, SparkMimo26RankPackShape *shape)
{
	const uint32_t hidden = SPARK_MIMO26_MODEL_HIDDEN_DIMENSION;
	const uint32_t local_experts = SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT / tp_degree;
	uint32_t moe, swa;
	if ( layer == SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER )
	{
		if ( kind == SPARK_MIMO26_STAGEPACK_TENSOR_EMBEDDING || kind == SPARK_MIMO26_STAGEPACK_TENSOR_LM_HEAD )
			return(SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,SPARK_MIMO26_MODEL_VOCAB_COUNT / tp_degree,hidden));
		if ( kind == SPARK_MIMO26_STAGEPACK_TENSOR_FINAL_NORM )
			return(SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,1u,hidden));
		return(0u);
	}
	if ( layer >= SPARK_MIMO26_MODEL_LAYER_COUNT )
		return(0u);
	moe = SparkMimo26ModelLayerIsMoe(layer);
	swa = SparkMimo26ModelLayerKind(layer) == SPARK_MIMO26_MODEL_LAYER_KIND_SWA ? 1u : 0u;
	switch ( kind )
	{
		case SPARK_MIMO26_STAGEPACK_TENSOR_QKV:
			return(SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_FP8_E4M3_F32B128,SparkMimo26RankPackQkvRows(layer,tp_degree),hidden));
		case SPARK_MIMO26_STAGEPACK_TENSOR_O_PROJ:
			return(SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,hidden,SPARK_MIMO26_MODEL_O_INPUT_DIMENSION / tp_degree));
		case SPARK_MIMO26_STAGEPACK_TENSOR_ATTENTION_NORM:
		case SPARK_MIMO26_STAGEPACK_TENSOR_MLP_NORM:
			return(SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,1u,hidden));
		case SPARK_MIMO26_STAGEPACK_TENSOR_SINK_BIAS:
			return(swa != 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,1u,SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT / tp_degree) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE:
			return(moe != 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_BF16,SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT,hidden) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_MOE_GATE_BIAS:
			return(moe != 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_F32,1u,SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE:
		case SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_UP:
			return(moe != 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_MXFP4_E2M1_E8M0G32,local_experts * SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION,hidden) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN:
			return(moe != 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_MXFP4_E2M1_E8M0G32,local_experts * hidden,SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_GATE:
		case SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_UP:
			return(moe == 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_FP8_E4M3_F32B128,SPARK_MIMO26_MODEL_DENSE_INTERMEDIATE_DIMENSION / tp_degree,hidden) : 0u);
		case SPARK_MIMO26_STAGEPACK_TENSOR_DENSE_MLP_DOWN:
			return(moe == 0u ? SparkMimo26RankPackShapeSet(shape,SPARK_MIMO26_STAGEPACK_WEIGHT_FP8_E4M3_F32B128,hidden,SPARK_MIMO26_MODEL_DENSE_INTERMEDIATE_DIMENSION / tp_degree) : 0u);
		default:
			return(0u);
	}
}

static inline uint32_t SparkMimo26RankPackExpectedCount(uint32_t tp_degree)
{
	SparkMimo26RankPackShape shape;
	uint32_t layer, kind, count = SPARK_MIMO26_RANK_PACK_GLOBAL_COUNT;
	for (layer = 0u; layer < SPARK_MIMO26_MODEL_LAYER_COUNT; layer++)
		for (kind = 0u; kind < SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT; kind++)
			count += SparkMimo26RankPackExpectedShape(kind,layer,tp_degree,&shape);
	return(count);
}

static inline SparkStatus SparkMimo26RankPackHeaderCheck(const SparkMimo26StagePackHeader *header, uint64_t file_bytes, uint32_t tp_degree)
{
	if ( header == 0 || tp_degree == 0u || SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT % tp_degree != 0u || SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT % tp_degree != 0u || SPARK_MIMO26_MODEL_VOCAB_COUNT % tp_degree != 0u || SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT % tp_degree != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_MIMO26_STAGEPACK_MAGIC || header->format_version != SPARK_MIMO26_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_MIMO26_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_MIMO26_STAGEPACK_ENTRY_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( header->hidden_dimension != SPARK_MIMO26_MODEL_HIDDEN_DIMENSION || header->layer_count != SPARK_MIMO26_MODEL_LAYER_COUNT || header->first_layer_index != 0u || header->total_layer_count != SPARK_MIMO26_MODEL_LAYER_COUNT || header->attn_query_head_count != SPARK_MIMO26_MODEL_ATTN_HEAD_COUNT || header->attn_kv_head_count != SPARK_MIMO26_MODEL_FULL_KV_HEAD_COUNT || header->attn_head_dimension != SPARK_MIMO26_MODEL_ATTN_HEAD_DIMENSION || header->attn_rope_dimension != SPARK_MIMO26_MODEL_ATTN_ROPE_DIMENSION || header->routed_expert_count != SPARK_MIMO26_MODEL_ROUTED_EXPERT_COUNT || header->experts_per_token != SPARK_MIMO26_MODEL_EXPERTS_PER_TOKEN || header->expert_intermediate_dimension != SPARK_MIMO26_MODEL_EXPERT_INTERMEDIATE_DIMENSION || header->output_vocab_count != SPARK_MIMO26_MODEL_VOCAB_COUNT || header->mxfp4_group_size != SPARK_MIMO26_MODEL_EXPERT_MXFP4_SCALE_BLOCK || header->mtp_layer_count != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( header->tensor_count != SparkMimo26RankPackExpectedCount(tp_degree) || header->file_bytes != file_bytes || header->directory_offset != SPARK_MIMO26_STAGEPACK_HEADER_BYTES || (uint64_t)header->tensor_count * SPARK_MIMO26_STAGEPACK_ENTRY_BYTES > file_bytes - header->directory_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkMimo26RankPackEntryCheck(const SparkMimo26StagePackEntry *entry, uint64_t data_start, uint64_t file_bytes, uint32_t tp_degree)
{
	SparkMimo26RankPackShape shape;
	if ( entry->tensor_kind >= SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT || SparkMimo26RankPackExpectedShape(entry->tensor_kind,entry->layer_index,tp_degree,&shape) == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( (entry->layer_index == SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER) != (SparkMimo26StagePackIsGlobal(entry->tensor_kind) != 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format != shape.weight_format || entry->rows != shape.rows || entry->columns != shape.columns || entry->scale_group_size != 0u || entry->payload_bytes != shape.payload_bytes || entry->scale_bytes != shape.scale_bytes )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset % SPARK_MIMO26_STAGEPACK_PAYLOAD_ALIGNMENT != 0u || entry->payload_offset < data_start || entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.scale_bytes == 0u )
		return(entry->scale_offset == 0u ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_offset % sizeof(float) != 0u || entry->scale_offset < data_start || entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkMimo26RankPackBind(const SparkMimo26StagePackHeader *header, const SparkMimo26StagePackEntry *entries, uint32_t entry_count, uint64_t file_bytes, uint32_t tp_degree, SparkMimo26RankPackLayout *layout)
{
	SparkMimo26RankPackShape shape;
	uint64_t data_start, previous_end = 0u, begin, end;
	uint32_t index, layer, kind, *slot;
	SparkStatus status;
	if ( entries == 0 || layout == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkMimo26RankPackHeaderCheck(header,file_bytes,tp_degree);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry_count != header->tensor_count )
		return(SPARK_STATUS_VALIDATION_FAILED);
	memset(layout,0xff,sizeof(*layout));
	layout->tp_degree = tp_degree;
	data_start = header->directory_offset + (uint64_t)entry_count * SPARK_MIMO26_STAGEPACK_ENTRY_BYTES;
	for (index = 0u; index < entry_count; index++)
	{
		status = SparkMimo26RankPackEntryCheck(&entries[index],data_start,file_bytes,tp_degree);
		if ( status != SPARK_STATUS_OK )
			return(status);
		begin = entries[index].payload_offset;
		end = entries[index].scale_bytes != 0u ? entries[index].scale_offset + entries[index].scale_bytes : begin + entries[index].payload_bytes;
		if ( begin < previous_end || (entries[index].scale_bytes != 0u && entries[index].scale_offset < begin + entries[index].payload_bytes) )
			return(SPARK_STATUS_VALIDATION_FAILED);
		previous_end = end;
		slot = entries[index].layer_index == SPARK_MIMO26_STAGEPACK_GLOBAL_LAYER ? &layout->global_entry[entries[index].tensor_kind] : &layout->layer_entry[entries[index].layer_index][entries[index].tensor_kind];
		if ( *slot != SPARK_MIMO26_RANK_PACK_ABSENT )
			return(SPARK_STATUS_DUPLICATE);
		*slot = index;
	}
	if ( previous_end != file_bytes )
		return(SPARK_STATUS_VALIDATION_FAILED);
	for (kind = 0u; kind < SPARK_MIMO26_RANK_PACK_GLOBAL_COUNT; kind++)
		if ( layout->global_entry[kind] == SPARK_MIMO26_RANK_PACK_ABSENT )
			return(SPARK_STATUS_NOT_FOUND);
	for (layer = 0u; layer < SPARK_MIMO26_MODEL_LAYER_COUNT; layer++)
		for (kind = 0u; kind < SPARK_MIMO26_STAGEPACK_TENSOR_KIND_COUNT; kind++)
			if ( (SparkMimo26RankPackExpectedShape(kind,layer,tp_degree,&shape) != 0u) != (layout->layer_entry[layer][kind] != SPARK_MIMO26_RANK_PACK_ABSENT) )
				return(SPARK_STATUS_NOT_FOUND);
	return(SPARK_STATUS_OK);
}
