#pragma once

static SparkStatus SPARK_FAMILY(ModuleBindMtp)(SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(StagePackEntry) *entry,void *payload,void *scale)
{
	(void)state;
	(void)entry;
	(void)payload;
	(void)scale;
	SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SPARK_FAMILY(ModuleBindGlobal)(SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(StagePackEntry) *entry,void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EMBEDDING): state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_FINAL_NORM): state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD): state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
}
