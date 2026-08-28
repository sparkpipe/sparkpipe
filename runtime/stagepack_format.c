/*
 * Stagepack format library implementation. See
 * include/sparkpipe/spark_stagepack_format.h for the contract: the shape
 * algebra and the header comparison stated once, the family geometry as
 * data, and the layout proof as compile-time teeth.
 */

#include <string.h>

#include "sparkpipe/spark_stagepack_format.h"

void SparkStagePackShapeInit(SparkStagePackTensorShape *shape)
{
	shape->rows = 0u;
	shape->columns = 0u;
	shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_BF16;
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
}

static int32_t SparkStagePackShapeMoE(uint32_t tensor_kind,
	const SparkStagePackGeometryTable *geometry,
	SparkStagePackTensorShape *shape)
{
	switch ( tensor_kind )
	{
	case SPARK_STAGEPACK_TENSOR_MOE_GATE:
		shape->rows = geometry->routed_expert_count;
		shape->columns = geometry->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_MOE_W1:
	case SPARK_STAGEPACK_TENSOR_MOE_W3:
		/* The routed projections narrow along the expert-shard axis. */
		shape->rows = geometry->routed_expert_count * geometry->expert_intermediate_dimension;
		shape->columns = geometry->hidden_dimension;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128;
		return(0);
	case SPARK_STAGEPACK_TENSOR_MOE_DOWN:
		shape->rows = geometry->routed_expert_count * geometry->hidden_dimension;
		shape->columns = geometry->expert_intermediate_dimension;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_FP8_E4M3_F32B128;
		return(0);
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE:
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_UP:
		shape->rows = geometry->expert_intermediate_dimension;
		shape->columns = geometry->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		shape->rows = geometry->hidden_dimension;
		shape->columns = geometry->expert_intermediate_dimension;
		return(0);
	default:
		return(-1);
	}
}

int32_t SparkStagePackShapeEveryLayerCommon(uint32_t tensor_kind,
	const SparkStagePackGeometryTable *geometry,
	SparkStagePackTensorShape *shape)
{
	if ( geometry == 0 || shape == 0 )
		return(-1);
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_EVERY_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_STAGEPACK_TENSOR_ATTENTION_NORM:
	case SPARK_STAGEPACK_TENSOR_MLP_NORM:
		shape->rows = 1u;
		shape->columns = geometry->norm_width;
		return(0);
	case SPARK_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT:
		shape->rows = 1u;
		shape->columns = geometry->hidden_dimension;
		return(0);
	default:
		return(SparkStagePackShapeMoE(tensor_kind,geometry,shape));
	}
}

int32_t SparkStagePackShapeGdnCommon(uint32_t tensor_kind,
	const SparkStagePackGeometryTable *geometry,
	SparkStagePackTensorShape *shape)
{
	if ( geometry == 0 || shape == 0 )
		return(-1);
	shape->layer_class = SPARK_STAGEPACK_FORMAT_LAYER_CLASS_GDN_LAYER;
	switch ( tensor_kind )
	{
	case SPARK_STAGEPACK_TENSOR_GDN_QKV:
		shape->rows = geometry->gdn_conv_channels;
		shape->columns = geometry->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_GATE:
		shape->rows = geometry->gdn_value_dimension;
		shape->columns = geometry->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_BETA:
	case SPARK_STAGEPACK_TENSOR_GDN_DECAY:
		shape->rows = geometry->gdn_value_head_count;
		shape->columns = geometry->hidden_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_OUTPUT:
		shape->rows = geometry->hidden_dimension;
		shape->columns = geometry->gdn_value_dimension;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_CONV_WEIGHT:
		shape->rows = geometry->gdn_conv_channels;
		shape->columns = geometry->gdn_conv_kernel;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_A_LOG:
	case SPARK_STAGEPACK_TENSOR_GDN_DT_BIAS:
		shape->rows = 1u;
		shape->columns = geometry->gdn_value_head_count;
		shape->natural_format = SPARK_STAGEPACK_FORMAT_WEIGHT_F32;
		return(0);
	case SPARK_STAGEPACK_TENSOR_GDN_NORM:
		shape->rows = 1u;
		shape->columns = geometry->gdn_head_value_dimension;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkStagePackHeaderMatchesIdentity(
	const SparkStagePackHeaderCommon *f,
	const SparkStagePackHeaderCommon *e)
{
	if ( f->magic != e->magic || f->format_version != e->format_version || f->header_bytes != e->header_bytes || f->directory_entry_bytes != e->directory_entry_bytes )
		return(-1);
	return(0);
}

static int32_t SparkStagePackHeaderMatchesCounts(
	const SparkStagePackHeaderCommon *f,
	const SparkStagePackHeaderCommon *e)
{
	if ( f->tensor_count != e->tensor_count || f->hidden_dimension != e->hidden_dimension || f->layer_count != e->layer_count || f->first_layer_index != e->first_layer_index || f->total_layer_count != e->total_layer_count )
		return(-2);
	return(0);
}

static int32_t SparkStagePackHeaderMatchesGdnGeometry(
	const SparkStagePackHeaderCommon *f,
	const SparkStagePackHeaderCommon *e)
{
	if ( f->attention_period != e->attention_period || f->full_attention_phase != e->full_attention_phase || f->gdn_key_head_count != e->gdn_key_head_count || f->gdn_value_head_count != e->gdn_value_head_count || f->gdn_head_key_dimension != e->gdn_head_key_dimension || f->gdn_head_value_dimension != e->gdn_head_value_dimension || f->gdn_conv_kernel != e->gdn_conv_kernel )
		return(-3);
	return(0);
}

static int32_t SparkStagePackHeaderMatchesAttnGeometry(
	const SparkStagePackHeaderCommon *f,
	const SparkStagePackHeaderCommon *e)
{
	if ( f->attn_query_head_count != e->attn_query_head_count || f->attn_kv_head_count != e->attn_kv_head_count || f->attn_head_dimension != e->attn_head_dimension || f->attn_rope_dimension != e->attn_rope_dimension )
		return(-4);
	return(0);
}

static int32_t SparkStagePackHeaderMatchesMoEGeometry(
	const SparkStagePackHeaderCommon *f,
	const SparkStagePackHeaderCommon *e)
{
	if ( f->routed_expert_count != e->routed_expert_count || f->experts_per_token != e->experts_per_token || f->expert_intermediate_dimension != e->expert_intermediate_dimension || f->output_vocab_count != e->output_vocab_count || f->mxfp4_group_size != e->mxfp4_group_size || f->mtp_layer_count != e->mtp_layer_count )
		return(-5);
	return(0);
}

/* Field-by-field comparison; 0 on match, negative group id on drift. */
int32_t SparkStagePackHeaderMatches(const SparkStagePackHeaderCommon *file_header,
	const SparkStagePackHeaderCommon *expected)
{
	int32_t status;
	status = SparkStagePackHeaderMatchesIdentity(file_header,expected);
	if ( status == 0 )
		status = SparkStagePackHeaderMatchesCounts(file_header,expected);
	if ( status == 0 )
		status = SparkStagePackHeaderMatchesGdnGeometry(file_header,expected);
	if ( status == 0 )
		status = SparkStagePackHeaderMatchesAttnGeometry(file_header,expected);
	if ( status == 0 )
		status = SparkStagePackHeaderMatchesMoEGeometry(file_header,expected);
	return(status);
}
