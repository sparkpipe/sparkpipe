#pragma once

static int32_t SPARK_FAMILY(SynthesizeAppendLayer)(SPARK_FAMILY(SynthesizeContext) *context, uint32_t layer_index)
{
	uint32_t kind;
	for (kind = SPARK_FAMILY_CONST(STAGEPACK_TENSOR_ATTN_NORM);
	     kind < SPARK_FAMILY_CONST(STAGEPACK_TENSOR_KIND_COUNT); kind++)
	{
		int32_t appended = SPARK_FAMILY(SynthesizeAppend)(context,kind,layer_index);
		if ( appended == -1 || appended == -3 )
			return(-(int32_t)kind);
	}
	return(0);
}

static int32_t SPARK_FAMILY(SynthesizeBuild)(SPARK_FAMILY(SynthesizeContext) *context)
{
	uint32_t layer;
	uint32_t last = context->first_layer_index + context->layer_count;
	for (layer = context->first_layer_index; layer < last; layer++)
		if ( SPARK_FAMILY(SynthesizeAppendLayer)(context,layer) < 0 )
			return(-1);
	if ( context->include_mtp != 0u && last == SPARK_FAMILY_CONST(MODEL_MTP_LAYER_INDEX) )
		if ( SPARK_FAMILY(SynthesizeAppendLayer)(context,SPARK_FAMILY_CONST(MODEL_MTP_LAYER_INDEX)) < 0 )
			return(-2);
	if ( context->owns_embedding != 0u )
		if ( SPARK_FAMILY(SynthesizeAppend)(context,SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EMBEDDING),SPARK_FAMILY_CONST(STAGEPACK_GLOBAL_LAYER)) < 0 )
			return(-3);
	if ( context->owns_head != 0u )
	{
		if ( SPARK_FAMILY(SynthesizeAppend)(context,SPARK_FAMILY_CONST(STAGEPACK_TENSOR_FINAL_NORM),SPARK_FAMILY_CONST(STAGEPACK_GLOBAL_LAYER)) < 0 )
			return(-4);
		if ( SPARK_FAMILY(SynthesizeAppend)(context,SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD),SPARK_FAMILY_CONST(STAGEPACK_GLOBAL_LAYER)) < 0 )
			return(-5);
	}
	return(0);
}
