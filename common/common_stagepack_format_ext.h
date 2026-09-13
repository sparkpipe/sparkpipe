#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stagepack_format.h"

#define SPARK_STAGEPACK_FORMAT_WEIGHT_U32 2u
#define SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1 3u
#define SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_E8M0B128 6u

#define SPARK_STAGEPACK_LAYER_GLOBAL UINT32_MAX
#define SPARK_STAGEPACK_LAYER_MTP (UINT32_MAX - 1u)
#define SPARK_STAGEPACK_PAYLOAD_ALIGNMENT 256u
#define SPARK_STAGEPACK_NVFP4_SCALE_GROUP 16u
#define SPARK_STAGEPACK_NVFP4_GLOBAL_SCALE_BYTES 8u

typedef struct SparkStagePackEntry
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
} SparkStagePackEntry;

#define SPARK_STAGEPACK_ENTRY_BYTES 56u

_Static_assert(sizeof(SparkStagePackEntry) == SPARK_STAGEPACK_ENTRY_BYTES,"stage pack directory entry must be 56 wire bytes");

typedef enum SparkStagePackTensorRole
{
	SPARK_STAGEPACK_ROLE_NONE = 0,
	SPARK_STAGEPACK_ROLE_EMBEDDING,
	SPARK_STAGEPACK_ROLE_FINAL_NORM,
	SPARK_STAGEPACK_ROLE_LM_HEAD,
	SPARK_STAGEPACK_ROLE_MTP_FC,
	SPARK_STAGEPACK_ROLE_MTP_EMBED_NORM,
	SPARK_STAGEPACK_ROLE_MTP_HIDDEN_NORM,
	SPARK_STAGEPACK_ROLE_MTP_FINAL_NORM,
	SPARK_STAGEPACK_ROLE_MIXER_DOWN,
	SPARK_STAGEPACK_ROLE_MIXER_UP,
	SPARK_STAGEPACK_ROLE_HC_DOWN,
	SPARK_STAGEPACK_ROLE_HC_UP,
	SPARK_STAGEPACK_ROLE_HC_INJECT,
	SPARK_STAGEPACK_ROLE_ATTN_QUERY,
	SPARK_STAGEPACK_ROLE_ATTN_KEY,
	SPARK_STAGEPACK_ROLE_ATTN_VALUE,
	SPARK_STAGEPACK_ROLE_ATTN_OUTPUT,
	SPARK_STAGEPACK_ROLE_ATTN_QUERY_NORM,
	SPARK_STAGEPACK_ROLE_ATTN_KEY_NORM,
	SPARK_STAGEPACK_ROLE_INDEXER_QK,
	SPARK_STAGEPACK_ROLE_INDEXER_Q_NORM,
	SPARK_STAGEPACK_ROLE_INDEXER_K_NORM,
	SPARK_STAGEPACK_ROLE_PLE_KEY,
	SPARK_STAGEPACK_ROLE_PLE_VALUE,
	SPARK_STAGEPACK_ROLE_PLE_NORM_KEY,
	SPARK_STAGEPACK_ROLE_PLE_NORM_QUERY,
	SPARK_STAGEPACK_ROLE_PLE_NORM_CONV,
	SPARK_STAGEPACK_ROLE_PLE_CONV,
	SPARK_STAGEPACK_ROLE_PLE_MULTIPLIERS,
	SPARK_STAGEPACK_ROLE_PLE_HEAD_VOCABS,
	SPARK_STAGEPACK_ROLE_PLE_HEAD_OFFSETS,
	SPARK_STAGEPACK_ROLE_PLE_NGRAM,
	SPARK_STAGEPACK_ROLE_COUNT
} SparkStagePackTensorRole;

typedef struct SparkStagePackFamilySpec
{
	uint32_t magic;
	uint32_t format_version;
	uint32_t hidden_dimension;
	uint32_t output_vocab_count;
	uint32_t layer_count;
	uint32_t attention_period;
	uint32_t full_attention_phase;
	uint32_t gdn_key_head_count;
	uint32_t gdn_value_head_count;
	uint32_t gdn_head_key_dimension;
	uint32_t gdn_head_value_dimension;
	uint32_t gdn_conv_kernel;
	uint32_t attn_query_head_count;
	uint32_t attn_kv_head_count;
	uint32_t attn_head_dimension;
	uint32_t attn_rope_dimension;
	uint32_t routed_expert_count;
	uint32_t experts_per_token;
	uint32_t expert_intermediate_dimension;
	uint32_t mxfp4_group_size;
	uint32_t fp8_block;
	uint32_t bf16_element_bytes;
	uint32_t mtp_layer_count;
	uint32_t hc_stream_count;
	uint32_t hc_lowrank_dimension;
	uint32_t indexer_head_count;
	uint32_t indexer_kv_head_count;
	uint32_t indexer_head_dimension;
	uint32_t ple_layer_index;
	uint32_t ple_layer_class;
	uint32_t ple_ngram_size;
	uint32_t ple_heads_per_ngram;
	uint32_t ple_embed_dimension;
	uint32_t ple_conv_kernel;
	uint32_t ple_ngram_row_count;
	uint32_t layer_tensor_count;
	uint32_t gdn_layer_tensor_count;
	uint32_t attn_layer_tensor_count;
	uint32_t tail_tensor_count_no_mtp;
	uint32_t tail_tensor_count_mtp;
	uint32_t first_stage_tensor_count;
	uint32_t last_stage_tensor_count;
} SparkStagePackFamilySpec;

typedef struct SparkStagePackKindRow
{
	uint32_t kind;
	uint32_t role;
} SparkStagePackKindRow;

typedef struct SparkStagePackKindTable
{
	const SparkStagePackKindRow *rows;
	uint32_t row_count;
} SparkStagePackKindTable;

static inline uint32_t SparkStagePackFamilyGdnValueHeadsPerKeyHead(const SparkStagePackFamilySpec *spec)
{
	return(spec->gdn_value_head_count / spec->gdn_key_head_count);
}

static inline uint32_t SparkStagePackFamilyGdnQkDimension(const SparkStagePackFamilySpec *spec)
{
	return(spec->gdn_key_head_count * spec->gdn_head_key_dimension);
}

static inline uint32_t SparkStagePackFamilyGdnValueDimension(const SparkStagePackFamilySpec *spec)
{
	return(spec->gdn_value_head_count * spec->gdn_head_value_dimension);
}

static inline uint32_t SparkStagePackFamilyGdnConvChannels(const SparkStagePackFamilySpec *spec)
{
	return((2u * SparkStagePackFamilyGdnQkDimension(spec)) + SparkStagePackFamilyGdnValueDimension(spec));
}

static inline uint32_t SparkStagePackFamilyAttnQueryDimension(const SparkStagePackFamilySpec *spec)
{
	return(spec->attn_query_head_count * spec->attn_head_dimension);
}

static inline uint32_t SparkStagePackFamilyAttnKvDimension(const SparkStagePackFamilySpec *spec)
{
	return(spec->attn_kv_head_count * spec->attn_head_dimension);
}

static inline uint32_t SparkStagePackFamilyHcStreamWidth(const SparkStagePackFamilySpec *spec)
{
	return(spec->hc_stream_count * spec->hidden_dimension);
}

static inline uint32_t SparkStagePackFamilyPleNgramHeadCount(const SparkStagePackFamilySpec *spec)
{
	return(spec->ple_heads_per_ngram * (spec->ple_ngram_size - 1u));
}

static inline uint32_t SparkStagePackFamilyPleNgramHeadDimension(const SparkStagePackFamilySpec *spec)
{
	return(spec->ple_embed_dimension / SparkStagePackFamilyPleNgramHeadCount(spec));
}

static inline uint32_t SparkStagePackFamilyFullAttentionLayersBelow(const SparkStagePackFamilySpec *spec, uint32_t layer_count)
{
	return(layer_count / spec->attention_period);
}

static inline SparkStagePackGeometryTable SparkStagePackFamilyGeometryTable(const SparkStagePackFamilySpec *spec)
{
	SparkStagePackGeometryTable geometry;
	geometry.norm_width = SparkStagePackFamilyHcStreamWidth(spec);
	geometry.hidden_dimension = spec->hidden_dimension;
	geometry.routed_expert_count = spec->routed_expert_count;
	geometry.expert_intermediate_dimension = spec->expert_intermediate_dimension;
	geometry.gdn_conv_channels = SparkStagePackFamilyGdnConvChannels(spec);
	geometry.gdn_value_dimension = SparkStagePackFamilyGdnValueDimension(spec);
	geometry.gdn_value_head_count = spec->gdn_value_head_count;
	geometry.gdn_head_value_dimension = spec->gdn_head_value_dimension;
	geometry.gdn_conv_kernel = spec->gdn_conv_kernel;
	return(geometry);
}

static inline uint32_t SparkStagePackFamilyRoleOf(const SparkStagePackKindTable *table, uint32_t kind)
{
	uint32_t index;
	for (index = 0u; index < table->row_count; index++)
		if ( table->rows[index].kind == kind )
			return(table->rows[index].role);
	return(SPARK_STAGEPACK_ROLE_NONE);
}

static inline int32_t SparkStagePackFamilyShapeGlobal(const SparkStagePackFamilySpec *spec, uint32_t role, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL;
	switch ( role )
	{
	case SPARK_STAGEPACK_ROLE_EMBEDDING:
	case SPARK_STAGEPACK_ROLE_LM_HEAD:
		shape->rows = spec->output_vocab_count;
		shape->columns = spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
		return(0);
	case SPARK_STAGEPACK_ROLE_MTP_FC:
		shape->rows = spec->hidden_dimension;
		shape->columns = 2u * spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_MIXER_DOWN:
		shape->rows = spec->hc_lowrank_dimension;
		shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
		return(0);
	case SPARK_STAGEPACK_ROLE_MIXER_UP:
		shape->rows = SparkStagePackFamilyHcStreamWidth(spec);
		shape->columns = spec->hc_lowrank_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_MTP_EMBED_NORM:
		shape->rows = 1u;
		shape->columns = spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_MTP_HIDDEN_NORM:
	case SPARK_STAGEPACK_ROLE_MTP_FINAL_NORM:
		shape->rows = 1u;
		shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkStagePackFamilyShapeEveryLayer(const SparkStagePackFamilySpec *spec, uint32_t role, SparkStagePackTensorShape *shape)
{
	if ( role != SPARK_STAGEPACK_ROLE_HC_DOWN && role != SPARK_STAGEPACK_ROLE_HC_UP && role != SPARK_STAGEPACK_ROLE_HC_INJECT )
		return(-1);
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	if ( role == SPARK_STAGEPACK_ROLE_HC_DOWN )
	{
		shape->rows = spec->hc_lowrank_dimension;
		shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
		return(0);
	}
	if ( role == SPARK_STAGEPACK_ROLE_HC_UP )
	{
		shape->rows = SparkStagePackFamilyHcStreamWidth(spec);
		shape->columns = spec->hc_lowrank_dimension;
		return(0);
	}
	shape->rows = spec->hc_stream_count;
	shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
	return(0);
}

static inline int32_t SparkStagePackFamilyShapeAttn(const SparkStagePackFamilySpec *spec, uint32_t role, SparkStagePackTensorShape *shape)
{
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER;
	switch ( role )
	{
	case SPARK_STAGEPACK_ROLE_ATTN_QUERY:
		shape->rows = 2u * SparkStagePackFamilyAttnQueryDimension(spec);
		shape->columns = spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_ATTN_KEY:
	case SPARK_STAGEPACK_ROLE_ATTN_VALUE:
		shape->rows = SparkStagePackFamilyAttnKvDimension(spec);
		shape->columns = spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_ATTN_OUTPUT:
		shape->rows = spec->hidden_dimension;
		shape->columns = SparkStagePackFamilyAttnQueryDimension(spec);
		return(0);
	case SPARK_STAGEPACK_ROLE_ATTN_QUERY_NORM:
	case SPARK_STAGEPACK_ROLE_ATTN_KEY_NORM:
		shape->rows = 1u;
		shape->columns = spec->attn_head_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_INDEXER_QK:
		shape->rows = (spec->indexer_head_count + spec->indexer_kv_head_count) * spec->indexer_head_dimension;
		shape->columns = spec->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_INDEXER_Q_NORM:
	case SPARK_STAGEPACK_ROLE_INDEXER_K_NORM:
		shape->rows = 1u;
		shape->columns = spec->indexer_head_dimension;
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkStagePackFamilyShapePle(const SparkStagePackFamilySpec *spec, uint32_t role, SparkStagePackTensorShape *shape)
{
	shape->layer_class = spec->ple_layer_class;
	switch ( role )
	{
	case SPARK_STAGEPACK_ROLE_PLE_KEY:
		shape->rows = SparkStagePackFamilyHcStreamWidth(spec);
		shape->columns = spec->ple_embed_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_VALUE:
		shape->rows = spec->hidden_dimension;
		shape->columns = spec->ple_embed_dimension;
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_NORM_KEY:
	case SPARK_STAGEPACK_ROLE_PLE_NORM_QUERY:
	case SPARK_STAGEPACK_ROLE_PLE_NORM_CONV:
		shape->rows = 1u;
		shape->columns = SparkStagePackFamilyHcStreamWidth(spec);
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_CONV:
		shape->rows = SparkStagePackFamilyHcStreamWidth(spec);
		shape->columns = spec->ple_conv_kernel;
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_MULTIPLIERS:
		shape->rows = 1u;
		shape->columns = spec->ple_ngram_size;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_I64;
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_HEAD_VOCABS:
	case SPARK_STAGEPACK_ROLE_PLE_HEAD_OFFSETS:
		shape->rows = 1u;
		shape->columns = SparkStagePackFamilyPleNgramHeadCount(spec);
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_I64;
		return(0);
	case SPARK_STAGEPACK_ROLE_PLE_NGRAM:
		shape->rows = spec->ple_ngram_row_count;
		shape->columns = SparkStagePackFamilyPleNgramHeadDimension(spec);
		return(0);
	default:
		return(-1);
	}
}

static inline int32_t SparkStagePackFamilyTensorShapeOf(const SparkStagePackFamilySpec *spec, const SparkStagePackKindTable *table, uint32_t tensor_kind, SparkStagePackTensorShape *shape)
{
	uint32_t role = SparkStagePackFamilyRoleOf(table,tensor_kind);
	SparkStagePackGeometryTable geometry;
	SparkStagePackShapeInit(shape);
	if ( role != SPARK_STAGEPACK_ROLE_NONE && SparkStagePackFamilyShapeGlobal(spec,role,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( role != SPARK_STAGEPACK_ROLE_NONE && SparkStagePackFamilyShapeEveryLayer(spec,role,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	geometry = SparkStagePackFamilyGeometryTable(spec);
	if ( SparkStagePackShapeEveryLayerCommon(tensor_kind,&geometry,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( SparkStagePackShapeGdnCommon(tensor_kind,&geometry,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( role != SPARK_STAGEPACK_ROLE_NONE && SparkStagePackFamilyShapeAttn(spec,role,shape) == 0 )
		return(0);
	SparkStagePackShapeInit(shape);
	if ( role != SPARK_STAGEPACK_ROLE_NONE && SparkStagePackFamilyShapePle(spec,role,shape) == 0 )
		return(0);
	return(-1);
}

static inline void SparkStagePackFamilyNarrowShape(const SparkStagePackFamilySpec *spec, const SparkStagePackKindTable *table, uint32_t tensor_kind, SparkStagePackTensorShape *shape, uint32_t tp_degree, uint32_t tp_rank)
{
	uint32_t role = SparkStagePackFamilyRoleOf(table,tensor_kind);
	uint32_t key_heads,value_heads,experts;
	(void)tp_rank;
	if ( tp_degree <= 1u )
		return;
	key_heads = spec->gdn_key_head_count / tp_degree;
	value_heads = spec->gdn_value_head_count / tp_degree;
	experts = spec->routed_expert_count / tp_degree;
	switch ( role )
	{
	case SPARK_STAGEPACK_ROLE_ATTN_QUERY:
		shape->rows = (spec->attn_query_head_count / tp_degree) * 2u * spec->attn_head_dimension;
		break;
	case SPARK_STAGEPACK_ROLE_ATTN_KEY:
	case SPARK_STAGEPACK_ROLE_ATTN_VALUE:
		if ( (spec->attn_kv_head_count % tp_degree) != 0u )
			break;
		shape->rows = (spec->attn_kv_head_count / tp_degree) * spec->attn_head_dimension;
		break;
	case SPARK_STAGEPACK_ROLE_ATTN_OUTPUT:
		shape->columns = SparkStagePackFamilyAttnQueryDimension(spec) / tp_degree;
		break;
	case SPARK_STAGEPACK_ROLE_EMBEDDING:
	case SPARK_STAGEPACK_ROLE_LM_HEAD:
		shape->rows = spec->output_vocab_count / tp_degree;
		break;
	case SPARK_STAGEPACK_ROLE_PLE_NGRAM:
		shape->rows = spec->ple_ngram_row_count / tp_degree;
		break;
	default:
		break;
	}
	switch ( tensor_kind )
	{
	case SPARK_STAGEPACK_TENSOR_GDN_QKV:
	case SPARK_STAGEPACK_TENSOR_GDN_CONV_WEIGHT:
		shape->rows = (2u * (key_heads * spec->gdn_head_key_dimension)) + (value_heads * spec->gdn_head_value_dimension);
		break;
	case SPARK_STAGEPACK_TENSOR_GDN_GATE:
		shape->rows = SparkStagePackFamilyGdnValueDimension(spec) / tp_degree;
		break;
	case SPARK_STAGEPACK_TENSOR_GDN_BETA:
	case SPARK_STAGEPACK_TENSOR_GDN_DECAY:
		shape->rows = value_heads;
		break;
	case SPARK_STAGEPACK_TENSOR_GDN_A_LOG:
	case SPARK_STAGEPACK_TENSOR_GDN_DT_BIAS:
		shape->columns = value_heads;
		break;
	case SPARK_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->columns = SparkStagePackFamilyGdnValueDimension(spec) / tp_degree;
		break;
	case SPARK_STAGEPACK_TENSOR_MOE_GATE:
		shape->rows = experts;
		break;
	case SPARK_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_STAGEPACK_TENSOR_MOE_W3:
		shape->rows = experts * spec->expert_intermediate_dimension;
		break;
	case SPARK_STAGEPACK_TENSOR_MOE_DOWN:
		shape->rows = experts * spec->hidden_dimension;
		break;
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = spec->expert_intermediate_dimension / tp_degree;
		break;
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->columns = spec->expert_intermediate_dimension / tp_degree;
		break;
	default:
		break;
	}
}

static inline int32_t SparkStagePackFamilyResolvedShape(const SparkStagePackFamilySpec *spec, const SparkStagePackKindTable *table, uint32_t tensor_kind, uint32_t layer_index, uint32_t is_global, SparkStagePackTensorShape *shape)
{
	if ( SparkStagePackFamilyTensorShapeOf(spec,table,tensor_kind,shape) < 0 )
		return(-1);
	if ( layer_index == SPARK_STAGEPACK_LAYER_MTP )
		return((is_global == 0u && (shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER || shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER)) ? 0 : -6);
	if ( (shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GLOBAL) != (is_global != 0u) )
		return(-2);
	if ( is_global != 0u )
		return(0);
	if ( layer_index >= spec->layer_count )
		return(-3);
	if ( shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER && ((layer_index % spec->attention_period) == spec->full_attention_phase) )
		return(-4);
	if ( shape->layer_class == SPARK_STAGEPACK_FORMAT_LAYER_CLASS_ATTN_LAYER && ((layer_index % spec->attention_period) != spec->full_attention_phase) )
		return(-5);
	if ( shape->layer_class == spec->ple_layer_class && layer_index != spec->ple_layer_index )
		return(-7);
	return(0);
}

static inline uint32_t SparkStagePackFamilyExpectedTensorCount(const SparkStagePackFamilySpec *spec, uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	uint32_t full = SparkStagePackFamilyFullAttentionLayersBelow(spec,first_layer_index + layer_count) - SparkStagePackFamilyFullAttentionLayersBelow(spec,first_layer_index);
	uint32_t gdn = layer_count - full;
	uint32_t tensors = (layer_count * spec->layer_tensor_count) + (gdn * spec->gdn_layer_tensor_count) + (full * spec->attn_layer_tensor_count);
	if ( first_layer_index == 0u )
		tensors += spec->first_stage_tensor_count;
	if ( first_layer_index + layer_count == spec->layer_count )
	{
		tensors += (spec->mtp_layer_count == 0u) ? spec->tail_tensor_count_no_mtp : spec->tail_tensor_count_mtp;
		if ( first_layer_index != 0u )
			tensors += spec->last_stage_tensor_count;
	}
	if ( include_ple != 0u && first_layer_index <= spec->ple_layer_index && first_layer_index + layer_count > spec->ple_layer_index )
		tensors += 10u;
	return(tensors);
}

static inline void SparkStagePackFamilyExpectedGeometry(const SparkStagePackFamilySpec *spec, SparkStagePackHeaderCommon *header, uint32_t first_layer_index, uint32_t layer_count, uint32_t include_ple)
{
	header->magic = spec->magic;
	header->format_version = spec->format_version;
	header->header_bytes = SPARK_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkStagePackFamilyExpectedTensorCount(spec,first_layer_index,layer_count,include_ple);
	header->hidden_dimension = spec->hidden_dimension;
	header->layer_count = layer_count;
	header->first_layer_index = first_layer_index;
	header->total_layer_count = spec->layer_count;
	header->attention_period = spec->attention_period;
	header->full_attention_phase = spec->full_attention_phase;
	header->gdn_key_head_count = spec->gdn_key_head_count;
	header->gdn_value_head_count = spec->gdn_value_head_count;
	header->gdn_head_key_dimension = spec->gdn_head_key_dimension;
	header->gdn_head_value_dimension = spec->gdn_head_value_dimension;
	header->gdn_conv_kernel = spec->gdn_conv_kernel;
	header->attn_query_head_count = spec->attn_query_head_count;
	header->attn_kv_head_count = spec->attn_kv_head_count;
	header->attn_head_dimension = spec->attn_head_dimension;
	header->attn_rope_dimension = spec->attn_rope_dimension;
	header->routed_expert_count = spec->routed_expert_count;
	header->experts_per_token = spec->experts_per_token;
	header->expert_intermediate_dimension = spec->expert_intermediate_dimension;
	header->output_vocab_count = spec->output_vocab_count;
	header->mxfp4_group_size = spec->mxfp4_group_size;
	header->mtp_layer_count = spec->mtp_layer_count;
	header->directory_offset = 0u;
	header->file_bytes = 0u;
}

static inline uint64_t SparkStagePackFamilyPayloadBytes(const SparkStagePackFamilySpec *spec, uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t elements = (uint64_t)rows * (uint64_t)columns;
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1 || weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED )
		return(elements / 2u);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128 || weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_E8M0B128 )
		return(elements);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_F32 || weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_U32 )
		return(elements * 4u);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_I64 )
		return(elements * 8u);
	return(elements * (uint64_t)spec->bf16_element_bytes);
}

static inline uint64_t SparkStagePackFamilyScaleBytes(const SparkStagePackFamilySpec *spec, uint32_t weight_format, uint32_t rows, uint32_t columns)
{
	uint64_t plane,per_expert,experts;
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_MXFP4_E2M1 )
		return(((uint64_t)rows * (uint64_t)columns) / spec->mxfp4_group_size);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_NVFP4_PACKED )
	{
		plane = (uint64_t)rows * ((uint64_t)columns / SPARK_STAGEPACK_NVFP4_SCALE_GROUP);
		if ( columns == spec->hidden_dimension )
			per_expert = spec->expert_intermediate_dimension;
		else if ( columns == spec->expert_intermediate_dimension )
			per_expert = spec->hidden_dimension;
		else
			return(0u);
		if ( per_expert == 0u || (rows % per_expert) != 0u )
			return(0u);
		experts = (uint64_t)rows / per_expert;
		return(plane + (experts * SPARK_STAGEPACK_NVFP4_GLOBAL_SCALE_BYTES));
	}
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128 )
		return(((uint64_t)rows / spec->fp8_block) * ((uint64_t)columns / spec->fp8_block) * 4u);
	if ( weight_format == SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_E8M0B128 )
		return((uint64_t)rows * ((uint64_t)columns / spec->fp8_block));
	return(0u);
}

static inline int32_t SparkStagePackFamilySpecCheck(const SparkStagePackFamilySpec *spec)
{
	if ( spec->hidden_dimension == 0u || spec->layer_count == 0u || spec->attention_period == 0u )
		return(-1);
	if ( (spec->layer_count % spec->attention_period) != 0u )
		return(-2);
	if ( spec->full_attention_phase >= spec->attention_period )
		return(-3);
	if ( SparkStagePackFamilyFullAttentionLayersBelow(spec,spec->layer_count) != spec->layer_count / spec->attention_period )
		return(-4);
	if ( SparkStagePackFamilyGdnValueHeadsPerKeyHead(spec) == 0u || (spec->gdn_value_head_count % spec->gdn_key_head_count) != 0u )
		return(-5);
	if ( (spec->attn_query_head_count % spec->attn_kv_head_count) != 0u )
		return(-6);
	if ( (spec->attn_rope_dimension % 2u) != 0u )
		return(-7);
	if ( (spec->expert_intermediate_dimension % spec->mxfp4_group_size) != 0u || (spec->hidden_dimension % spec->mxfp4_group_size) != 0u )
		return(-8);
	if ( spec->fp8_block == 0u || (spec->hidden_dimension % spec->fp8_block) != 0u )
		return(-9);
	if ( SparkStagePackFamilyPleNgramHeadCount(spec) == 0u || (spec->ple_embed_dimension % SparkStagePackFamilyPleNgramHeadCount(spec)) != 0u )
		return(-10);
	return(0);
}
