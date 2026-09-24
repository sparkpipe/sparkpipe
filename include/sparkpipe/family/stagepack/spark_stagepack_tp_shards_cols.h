#pragma once

static inline uint32_t SPARK_FAMILY(StagePackTpShardsCols)(uint32_t tensor_kind)
{
	switch ( tensor_kind )
	{
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_ATTN_OUTPUT):
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_DENSE_DOWN):
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EXPERT_DOWN):
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_SHARED_DOWN):
		return(1u);
	default:
		return(0u);
	}
}
