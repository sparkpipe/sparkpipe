#pragma once

#include "inference/kernels/project.cuh"
#include <stdint.h>
#include "sparkpipe/spark_gpu.h"

#define LM_PROJECT_CHAIN_THREADS 256u
#define LM_PROJECT_CHAIN_OUTPUTS 8u
#define LM_PROJECT_CHAIN_ROWS 8u

template<uint32_t THREADS, uint32_t IN_DIM, uint32_t OUT_DIM, uint32_t INPUT_HEAD_DIM = IN_DIM, uint32_t INPUT_OFFSET = 0u, uint32_t OUT_HEAD_STRIDE = OUT_DIM>
static inline SparkGpuError LmPerHeadProjectChainLaunch(const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t heads, uint32_t rows, SparkGpuStream stream,
	uint32_t row_exact = 0u, uint32_t force_mma = 0u)
{
	(void)stream;
	(void)row_exact;
	(void)force_mma;
	for (uint32_t row = 0u; row < rows; ++row)
		for (uint32_t head = 0u; head < heads; ++head)
		{
			const uint16_t *in_row = input_bf16 + ((uint64_t)row * heads + head) * INPUT_HEAD_DIM + INPUT_OFFSET;
			uint16_t *out_row = output_bf16 + ((uint64_t)row * heads + head) * OUT_HEAD_STRIDE;
			for (uint32_t o = 0u; o < OUT_DIM; ++o)
			{
				const uint16_t *w = weight_bf16 + ((uint64_t)head * OUT_DIM + o) * IN_DIM;
				float sum = 0.0f;
				for (uint32_t k = 0u; k < IN_DIM; ++k)
					sum += LmBf16ToFloat(in_row[k]) * LmBf16ToFloat(w[k]);
				out_row[o] = LmFloatToBf16(sum);
			}
		}
	return SPARK_GPU_SUCCESS;
}
