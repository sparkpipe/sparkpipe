#pragma once

static SparkStatus SPARK_FAMILY(ModuleBindGlobal)(SPARK_FAMILY(ModuleState) *state, const SPARK_FAMILY(StagePackEntry) *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EMBEDDING):
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_FINAL_NORM):
		if ( state->owns_final_head == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD):
		if ( state->owns_final_head == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_MTP_EMBED_NORM): state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_MTP_HIDDEN_NORM): state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_MTP_FINAL_NORM): state->mtp.final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
}
