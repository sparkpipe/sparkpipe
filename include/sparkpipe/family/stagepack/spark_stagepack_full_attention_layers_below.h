#pragma once

static inline uint32_t SPARK_FAMILY(StagePackFullAttentionLayersBelow)(uint32_t layer_count)
{
	return(layer_count / SPARK_FAMILY_CONST(MODEL_ATTENTION_PERIOD));
}
