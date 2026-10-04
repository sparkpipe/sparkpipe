#pragma once

static uint64_t SPARK_FAMILY(ModuleExpectedGlobalBits)(const SPARK_FAMILY(ModuleState) *state)
{
	uint64_t bits;
	bits = 0u;
	if ( state->owns_embedding != 0u )
		bits |= UINT64_C(1) << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EMBEDDING);
	if ( state->owns_final_head != 0u )
		bits |= (UINT64_C(1) << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_FINAL_NORM)) | (UINT64_C(1) << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD));
	return(bits);
}
