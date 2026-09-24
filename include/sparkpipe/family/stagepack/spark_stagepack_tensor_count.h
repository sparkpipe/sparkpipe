#pragma once

static inline uint32_t SPARK_FAMILY(StagePackExpectedTensorCount)(uint32_t first_layer_index, uint32_t layer_count)
{
	uint32_t tensors = layer_count * 8u;
	if ( first_layer_index == 0u )
		tensors += 1u;
	if ( first_layer_index + layer_count == SPARK_FAMILY_CONST(MODEL_LAYER_COUNT) )
		tensors += 2u;
	return(tensors);
}
