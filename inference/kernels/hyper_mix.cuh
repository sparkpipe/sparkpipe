#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"

#ifdef __CUDACC__
static __device__ __forceinline__ float LmHyperRound(float value)
{
	return LmBf16ToFloat(LmFloatToBf16(value));
}

static __device__ __forceinline__ float LmHyperSigmoid(float value)
{
	return 1.0f / (1.0f + expf(-value));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmGroupRmsNormKernel(const uint16_t *__restrict__ input_bf16, const float *__restrict__ weight, uint16_t *__restrict__ output_bf16,
	uint32_t groups, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t group = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * groups + group) * dimension;
	const float *group_weight = weight + (uint64_t)group * dimension;
	float total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS )
	{
		const float value = LmBf16ToFloat(input_bf16[base + index]);
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total, reduction);
	const float scale = rsqrtf(total / (float)dimension + epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS )
		output_bf16[base + index] = LmFloatToBf16(LmBf16ToFloat(input_bf16[base + index]) * scale * group_weight[index]);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperExpandKernel(const uint16_t *__restrict__ hidden_bf16, uint16_t *__restrict__ streams_bf16, uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < streams * width; index += gridDim.x * THREADS )
		streams_bf16[(uint64_t)row * streams * width + index] = hidden_bf16[(uint64_t)row * width + index % width];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperGateKernel(uint16_t *__restrict__ values_bf16, uint64_t elements, float divisor)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * THREADS + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * THREADS )
	{
		const float value = LmHyperRound(LmBf16ToFloat(values_bf16[index]) / divisor);
		values_bf16[index] = LmFloatToBf16(value * LmHyperSigmoid(value));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperMixKernel(const uint16_t *__restrict__ up_bf16, const uint16_t *__restrict__ normed_bf16, uint16_t *__restrict__ mixed_bf16,
	uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const uint64_t base = (uint64_t)row * streams * width;
	for ( uint32_t element = blockIdx.x * THREADS + threadIdx.x; element < width; element += gridDim.x * THREADS )
	{
		float total = 0.0f;
		for ( uint32_t stream = 0u; stream < streams; ++stream )
		{
			const uint64_t at = base + (uint64_t)stream * width + element;
			total += LmHyperRound(LmHyperRound(LmHyperSigmoid(LmBf16ToFloat(up_bf16[at]))) * LmBf16ToFloat(normed_bf16[at]));
		}
		mixed_bf16[(uint64_t)row * width + element] = LmFloatToBf16(total / (float)streams);
	}
}

template<uint32_t THREADS, uint32_t STREAMS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperInjectWeightsKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ weight_bf16, float *__restrict__ inject,
	uint32_t width)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x, total_width = STREAMS * width;
	const uint16_t *source = normed_bf16 + (uint64_t)row * total_width;
	float sums[STREAMS];
	for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
		sums[stream] = 0.0f;
	for ( uint32_t index = threadIdx.x; index < total_width; index += THREADS )
	{
		const float value = LmBf16ToFloat(source[index]);
		for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
			sums[stream] += value * LmBf16ToFloat(weight_bf16[(uint64_t)stream * total_width + index]);
	}
	for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
	{
		const float logit = LmHyperRound(LmBlockSum<THREADS>(sums[stream], reduction));
		if ( threadIdx.x == 0u )
			inject[(uint64_t)row * STREAMS + stream] = LmHyperRound(2.0f * LmHyperRound(LmHyperSigmoid(LmHyperRound(logit / (float)STREAMS))));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperKeyGateKernel(const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ value_bf16,
	uint16_t *__restrict__ gated_bf16, uint32_t streams, uint32_t width)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t stream = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * streams + stream) * width;
	float total = 0.0f;
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
		total += LmHyperRound(LmBf16ToFloat(key_bf16[base + index]) * LmBf16ToFloat(query_bf16[base + index]));
	total = LmBlockSum<THREADS>(total, reduction);
	const float score = LmHyperRound(LmHyperRound(total) / sqrtf((float)width));
	const float magnitude = LmHyperRound(sqrtf(fmaxf(fabsf(score), 1e-6f)));
	const float gate = LmHyperRound(LmHyperSigmoid(score < 0.0f ? -magnitude : (score > 0.0f ? magnitude : 0.0f)));
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
		gated_bf16[base + index] = LmFloatToBf16(gate * LmBf16ToFloat(value_bf16[(uint64_t)row * width + index]));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperAddKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ gated_bf16, const uint16_t *__restrict__ conv_bf16,
	uint64_t elements)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * THREADS + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * THREADS )
	{
		const float added = LmHyperRound(LmBf16ToFloat(gated_bf16[index]) + LmBf16ToFloat(conv_bf16[index]));
		streams_bf16[index] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[index]) + added);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperInjectKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ output_bf16, const float *__restrict__ inject,
	uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	for ( uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < streams * width; index += gridDim.x * THREADS )
	{
		const uint64_t at = (uint64_t)row * streams * width + index;
		const float injected = LmHyperRound(LmBf16ToFloat(output_bf16[(uint64_t)row * width + index % width]) * inject[(uint64_t)row * streams + index / width]);
		streams_bf16[at] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[at]) + injected);
	}
}
#endif
