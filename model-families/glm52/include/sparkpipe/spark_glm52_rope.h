#pragma once

#include <math.h>
#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"

#if SPARK_GLM52_MODEL_ROPE_INTERLEAVE != 1u || \
    SPARK_GLM52_MODEL_DSA_ROPE_INTERLEAVE != 1u
#error "GLM-5.2 MLA and DSA require interleaved adjacent-pair RoPE"
#endif

static inline void SparkGlm52BuildRopeTables(
	float *cosine_table,
	float *sine_table,
	uint32_t position_count)
{
	uint64_t offset;
	uint32_t pair_count;
	uint32_t pair_index;
	uint32_t position;
	double angle;
	double inverse_frequency;
	pair_count = SPARK_GLM52_MODEL_ROPE_DIMENSION / 2u;
	for (position = 0u; position < position_count; ++position)
	{
		for (pair_index = 0u; pair_index < pair_count; ++pair_index)
		{
			inverse_frequency = pow(
				SPARK_GLM52_MODEL_ROPE_THETA,
				-((double)(pair_index * 2u) /
				 (double)SPARK_GLM52_MODEL_ROPE_DIMENSION));
			angle = (double)position * inverse_frequency;
			offset = ((uint64_t)position * pair_count) + pair_index;
			cosine_table[offset] = (float)cos(angle);
			sine_table[offset] = (float)sin(angle);
		}
	}
}
