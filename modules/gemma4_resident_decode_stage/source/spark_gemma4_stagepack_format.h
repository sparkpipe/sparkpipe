#pragma once

#include <stdint.h>
#include <stddef.h>

#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stagepack_format.h"
#include "sparkpipe/spark_status.h"


#define SPARK_GEMMA4_STAGEPACK_MAGIC 0x50534734u
#define SPARK_GEMMA4_STAGEPACK_FORMAT_VERSION 1u
#define SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER UINT32_MAX
#define SPARK_GEMMA4_STAGEPACK_MTP_LAYER (UINT32_MAX - 1u)
#define SPARK_GEMMA4_STAGEPACK_PAYLOAD_ALIGNMENT 256u

typedef enum SparkGemma4StagePackTensorKind
{
	SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING = 0,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM = 1,
	SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_INPUT_NORM = 2,
	SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_ATTENTION_NORM = 3,
	SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM = 4,
	SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM = 5,
	SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY = 6,
	SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED = 7,
	SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT = 8,
	SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY_NORM = 9,
	SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KEY_NORM = 10,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY = 11,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY = 12,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT = 13,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY_NORM = 14,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY_NORM = 15,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP = 16,
	SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN = 17,
	SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_ROPE_TABLE = 18,
	SPARK_GEMMA4_STAGEPACK_TENSOR_ROUTER_PROJ = 19,
	SPARK_GEMMA4_STAGEPACK_TENSOR_PER_EXPERT_SCALE = 20,
	SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP = 21,
	SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN = 22,
	SPARK_GEMMA4_STAGEPACK_TENSOR_KIND_COUNT = 23
} SparkGemma4StagePackTensorKind;

#define SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FC 100u
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_EMBED_NORM 101u
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_HIDDEN_NORM 102u
#define SPARK_GEMMA4_STAGEPACK_TENSOR_MTP_FINAL_NORM 103u

#define SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL
#define SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER
#define SPARK_GEMMA4_STAGEPACK_CLASS_SLIDING_LAYER 2u
#define SPARK_GEMMA4_STAGEPACK_CLASS_FULL_LAYER 3u

typedef struct SparkGemma4StagePackHeader
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	uint32_t directory_entry_bytes;
	uint32_t tensor_count;
	uint32_t hidden_dimension;
	uint32_t layer_count;
	uint32_t first_layer_index;
	uint32_t total_layer_count;
	uint32_t full_layer_period;
	uint32_t full_layer_phase;
	uint32_t sliding_kv_head_count;
	uint32_t dense_intermediate_dimension;
	uint32_t sliding_head_dimension;
	uint32_t full_head_dimension;
	uint32_t sliding_window_tokens;
	uint32_t query_head_count;
	uint32_t full_kv_head_count;
	uint32_t sliding_rope_dimension;
	uint32_t full_rope_dimension;
	uint32_t routed_expert_count;
	uint32_t experts_per_token;
	uint32_t expert_intermediate_dimension;
	uint32_t output_vocab_count;
	uint32_t weight_element_bytes;
	uint32_t reserved_zero;
	uint64_t directory_offset;
	uint64_t file_bytes;
} SparkGemma4StagePackHeader;

typedef struct SparkGemma4StagePackEntry
{
	uint32_t tensor_kind;
	uint32_t layer_index;
	uint32_t weight_format;
	uint32_t rows;
	uint32_t columns;
	uint32_t scale_group_size;
	uint64_t payload_offset;
	uint64_t payload_bytes;
	uint64_t scale_offset;
	uint64_t scale_bytes;
} SparkGemma4StagePackEntry;

#define SPARK_GEMMA4_STAGEPACK_HEADER_BYTES 120u
#define SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES 56u
_Static_assert(sizeof(SparkGemma4StagePackHeader) == SPARK_GEMMA4_STAGEPACK_HEADER_BYTES,"gemma4 stage pack header must be 120 wire bytes");
_Static_assert(sizeof(SparkGemma4StagePackEntry) == SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES,"gemma4 stage pack directory entry must be 56 wire bytes");

#define SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(gemma_field,common_field) \
	_Static_assert(offsetof(SparkGemma4StagePackHeader,gemma_field) == \
		offsetof(SparkStagePackHeaderCommon,common_field), \
		"gemma4 stage pack header layout drift")
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(magic,magic);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(format_version,format_version);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(header_bytes,header_bytes);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(directory_entry_bytes,directory_entry_bytes);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(tensor_count,tensor_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(hidden_dimension,hidden_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(layer_count,layer_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(first_layer_index,first_layer_index);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(total_layer_count,total_layer_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(full_layer_period,attention_period);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(full_layer_phase,full_attention_phase);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(sliding_kv_head_count,gdn_key_head_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(dense_intermediate_dimension,gdn_value_head_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(sliding_head_dimension,gdn_head_key_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(full_head_dimension,gdn_head_value_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(sliding_window_tokens,gdn_conv_kernel);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(query_head_count,attn_query_head_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(full_kv_head_count,attn_kv_head_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(sliding_rope_dimension,attn_head_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(full_rope_dimension,attn_rope_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(routed_expert_count,routed_expert_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(experts_per_token,experts_per_token);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(expert_intermediate_dimension,expert_intermediate_dimension);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(output_vocab_count,output_vocab_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(weight_element_bytes,mxfp4_group_size);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(reserved_zero,mtp_layer_count);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(directory_offset,directory_offset);
SPARK_GEMMA4_STAGEPACK_FIELD_PROOF(file_bytes,file_bytes);

_Static_assert(SPARK_GEMMA4_MODEL_SLIDING_LAYER_COUNT + SPARK_GEMMA4_MODEL_FULL_LAYER_COUNT == SPARK_GEMMA4_MODEL_LAYER_COUNT,"gemma4 layer split must cover the stack");
_Static_assert((SPARK_GEMMA4_MODEL_LAYER_COUNT % SPARK_GEMMA4_MODEL_FULL_LAYER_PERIOD) == 0u,"gemma4 layer count must be whole periods");
_Static_assert((SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT % SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT) == 0u,"gemma4 sliding query heads must group evenly onto kv heads");
_Static_assert((SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT % SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT) == 0u,"gemma4 full query heads must group evenly onto kv heads");
_Static_assert(SPARK_GEMMA4_MODEL_FULL_ATTENTION_K_EQ_V == 1u,"gemma4 full layers carry no v projection (attention_k_eq_v)");
_Static_assert(SPARK_GEMMA4_MODEL_SLIDING_ROPE_DIMENSION == SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,"gemma4 sliding rope covers the whole head");
_Static_assert(SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS == 256u,"gemma4 proportional rope table is 256 fp32 entries (128 frequencies then zeros)");

static inline uint32_t SparkGemma4StagePackKvHeadsPerRank(uint32_t global_kv_head_count, uint32_t tp_degree)
{
	return(global_kv_head_count >= tp_degree ? global_kv_head_count / tp_degree : 1u);
}

static inline uint32_t SparkGemma4StagePackExpectedTensorCount(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t tensors = layer_count * (SPARK_GEMMA4_MODEL_MOE_BLOCK ? 10u : 6u);
	tensors += layer_count * 5u;
	tensors += 1u;
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_GEMMA4_MODEL_LAYER_COUNT )
		tensors += 1u;
	return(tensors);
}

static inline void SparkGemma4StagePackExpectedGeometry(SparkGemma4StagePackHeader *header, uint32_t first_layer_index, uint32_t layer_count)
{
	header->magic = SPARK_GEMMA4_STAGEPACK_MAGIC;
	header->format_version = SPARK_GEMMA4_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_GEMMA4_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_GEMMA4_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkGemma4StagePackExpectedTensorCount(first_layer_index,layer_count);
	header->hidden_dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = SPARK_GEMMA4_MODEL_LAYER_COUNT;
	header->full_layer_period = SPARK_GEMMA4_MODEL_FULL_LAYER_PERIOD;
	header->full_layer_phase = SPARK_GEMMA4_MODEL_FULL_LAYER_PHASE;
	header->sliding_kv_head_count = SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT;
	header->dense_intermediate_dimension = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	header->sliding_head_dimension = SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	header->full_head_dimension = SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
	header->sliding_window_tokens = SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS;
	header->query_head_count = SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT;
	header->full_kv_head_count = SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT;
	header->sliding_rope_dimension = SPARK_GEMMA4_MODEL_SLIDING_ROPE_DIMENSION;
	header->full_rope_dimension = SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION;
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	header->routed_expert_count = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT;
	header->experts_per_token = SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN;
	header->expert_intermediate_dimension = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
#else
	header->routed_expert_count = 0u;
	header->experts_per_token = 0u;
	header->expert_intermediate_dimension = 0u;
#endif
	header->output_vocab_count = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT;
	header->weight_element_bytes = SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	header->reserved_zero = 0u;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

static inline int32_t SparkGemma4StagePackHeaderMatches(const SparkGemma4StagePackHeader *file_header, const SparkGemma4StagePackHeader *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}

typedef SparkStagePackTensorShape SparkGemma4StagePackTensorShape;

static inline int32_t SparkGemma4StagePackShapeGlobal(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_ROPE_TABLE:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_F32;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeLayerNorm(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_INPUT_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_ATTENTION_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeSliding(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_SLIDING_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY:
		shape->rows = SPARK_GEMMA4_MODEL_SLIDING_QUERY_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED:
		shape->rows = 2u * SPARK_GEMMA4_MODEL_SLIDING_KV_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_SLIDING_QUERY_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeFull(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_FULL_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY:
		shape->rows = SPARK_GEMMA4_MODEL_FULL_QUERY_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY:
		shape->rows = SPARK_GEMMA4_MODEL_FULL_KV_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_FULL_QUERY_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY_NORM:
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY_NORM:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackShapeEveryLayerMatmul(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	shape->layer_class = SPARK_GEMMA4_STAGEPACK_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP:
		shape->rows = 2u * SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN:
		shape->rows = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION;
		return(0);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ROUTER_PROJ:
		shape->rows = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PER_EXPERT_SCALE:
		shape->rows = 1u;
		shape->columns = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_F32;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP:
		shape->rows = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT * SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * 2u;
		shape->columns = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		return(0);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN:
		shape->rows = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		shape->columns = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION;
		return(0);
#endif
	default:
		return(-1);
	}
}

static inline int32_t SparkGemma4StagePackTensorShapeOf(uint32_t tensor_kind, SparkGemma4StagePackTensorShape *shape)
{
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeGlobal(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeLayerNorm(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeSliding(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeFull(tensor_kind,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkGemma4StagePackShapeEveryLayerMatmul(tensor_kind,shape) == 0 )
		return(0);
	return(-1);
}

static inline void SparkGemma4StagePackNarrowShape(SparkGemma4StagePackTensorShape *shape, uint32_t tensor_kind, uint32_t tp_degree, uint32_t tp_rank)
{
	uint32_t kv_per_rank;
	(void)tp_rank;
	if ( tp_degree <= 1u )
		return;
	switch ( tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING:
		shape->rows = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY:
		shape->rows = (SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT / tp_degree) * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED:
		kv_per_rank = SparkGemma4StagePackKvHeadsPerRank(SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT,tp_degree);
		shape->rows = 2u * kv_per_rank * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT:
		shape->columns = SPARK_GEMMA4_MODEL_SLIDING_QUERY_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY:
		shape->rows = (SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT / tp_degree) * SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY:
		kv_per_rank = SparkGemma4StagePackKvHeadsPerRank(SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT,tp_degree);
		shape->rows = kv_per_rank * SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT:
		shape->columns = SPARK_GEMMA4_MODEL_FULL_QUERY_DIMENSION / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP:
		shape->rows = (2u * SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION) / tp_degree;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN:
		shape->columns = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION / tp_degree;
		break;
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP:
		shape->rows = (SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT / tp_degree) * SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * 2u;
		break;
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN:
		shape->rows = (SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT / tp_degree) * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
		break;
#endif
	default:
		break;
	}
}

static inline int32_t SparkGemma4StagePackResolvedShape(uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkGemma4StagePackTensorShape *shape)
{
	if ( SparkGemma4StagePackTensorShapeOf(tensor_kind,shape) < 0 )
		return(-1);
	if ( (shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= SPARK_GEMMA4_MODEL_LAYER_COUNT )
		return(-3);
	if ( shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_SLIDING_LAYER && SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer_index) != 0u )
		return(-4);
	if ( shape->layer_class == SPARK_GEMMA4_STAGEPACK_CLASS_FULL_LAYER && SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer_index) == 0u )
		return(-5);
	return(0);
}

_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 == SPARK_STAGEPACK_FORMAT_WEIGHT_BF16,"gemma4 bf16 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 == SPARK_STAGEPACK_FORMAT_WEIGHT_F32,"gemma4 f32 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 == SPARK_STAGEPACK_FORMAT_WEIGHT_I64,"gemma4 i64 weight code must match the shared format");
_Static_assert(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED == SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED,"gemma4 nvfp4 weight code must match the shared format");

static inline uint64_t SparkGemma4StagePackPayloadBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED )
		return(elements / 2u);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_F32 || weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_U32 )
		return(elements * 4u);
	if ( weight_format == SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_I64 )
		return(elements * 8u);
	return(elements * (uint64_t)SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES);
}

static inline uint64_t SparkGemma4StagePackScaleBytes(uint32_t weight_format, uint32_t rows, uint32_t columns)
{
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	uint64_t plane,per_expert,experts;
	if ( weight_format != SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_NVFP4_PACKED )
		return(0u);
	plane = (uint64_t)rows * ((uint64_t)columns / 16u);
	if ( columns == SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION )
		per_expert = SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * 2u;
	else if ( columns == SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION )
		per_expert = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	else
		return(0u);
	if ( per_expert == 0u || (rows % per_expert) != 0u )
		return(0u);
	experts = (uint64_t)rows / per_expert;
	return(plane + experts * 8u);
#else
	(void)weight_format;
	(void)rows;
	(void)columns;
	return(0u);
#endif
}
