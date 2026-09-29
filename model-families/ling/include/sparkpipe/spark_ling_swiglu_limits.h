#pragma once

#define SPARK_LING_MODEL_MOE_ROUTED_SWIGLU_LIMIT(layer_index) \
	((layer_index) >= 35u ? 4.0f : 0.0f)
#define SPARK_LING_MODEL_MOE_SHARED_SWIGLU_LIMIT(layer_index) \
	((layer_index) >= 40u ? 7.0f : ((layer_index) >= 34u ? 5.0f : 0.0f))
