#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/hc.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "inference/kernels/norm.cuh"

template<uint32_t THREADS, uint32_t STREAMS, class Lane>
static __device__ __forceinline__ void LmHcCollapseNormRow(const uint16_t *input, const uint16_t *__restrict__ streams, const float *__restrict__ pre,
	const uint16_t *__restrict__ weight, uint16_t *output, uint16_t *__restrict__ quantized, uint32_t row, uint32_t dimension, uint32_t part,
	uint32_t parts, float epsilon, float *shared)
{
	float *values = shared;
	float *reduction = shared + dimension;
	const uint64_t base = (uint64_t)row * dimension;
	const uint32_t span = dimension / parts, begin = part * span;
	uint32_t index;
	float total = 0.0f, scale;
	for ( index = threadIdx.x; index < dimension; index += THREADS )
	{
		float value = LmBf16ToFloat(streams != 0 ? LmHcCollapseElement<STREAMS>(streams, pre, row, dimension, index) : input[base + index]);
		values[index] = value;
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total, reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for ( index = begin + threadIdx.x; index < begin + span; index += THREADS )
	{
		const uint16_t normed = LmFloatToBf16(values[index] * scale * LmScalarToFloat(weight[index]));
		if ( output != 0 )
			output[base + index] = normed;
		quantized[base + index] = Lane::Apply(LmBf16ToFloat(normed));
	}
}
