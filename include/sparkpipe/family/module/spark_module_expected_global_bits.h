#pragma once

static uint32_t SPARK_FAMILY(ModuleExpectedGlobalBits)(const SPARK_FAMILY(ModuleState) *state)
{
	uint32_t bits = 0u;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		bits |= 1u << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_EMBEDDING);
	if ( state->owns_final_head != 0u )
		bits |= (1u << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_FINAL_NORM)) | (1u << SPARK_FAMILY_CONST(STAGEPACK_TENSOR_LM_HEAD));
	return(bits);
}
