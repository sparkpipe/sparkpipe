#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"

#ifdef __CUDACC__
#include <cooperative_groups.h>

#define LM_HYPER_INJECT_CLUSTER_ROWS 8u
#define LM_HYPER_INJECT_BATCH 16u

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
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		float staged_input[LM_RMS_NORM_STAGED], staged_weight[LM_RMS_NORM_STAGED];
#pragma unroll
		for ( uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot )
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			staged_input[slot] = index < dimension ? LmBf16ToFloat(input_bf16[base + index]) : 0.0f;
			staged_weight[slot] = index < dimension ? group_weight[index] : 0.0f;
		}
#pragma unroll
		for ( uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot )
			if ( threadIdx.x + slot * THREADS < dimension )
				total += staged_input[slot] * staged_input[slot];
		total = LmBlockSum<THREADS>(total, reduction);
		const float scale = rsqrtf(total / (float)dimension + epsilon);
#pragma unroll
		for ( uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot )
			if ( threadIdx.x + slot * THREADS < dimension )
				output_bf16[base + threadIdx.x + slot * THREADS] = LmFloatToBf16(staged_input[slot] * scale * staged_weight[slot]);
		return;
	}
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

template<uint32_t THREADS, uint32_t STREAMS>
__global__ __launch_bounds__(LM_WARP_LANES, 1)
void LmHyperInjectWeightsClusterKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ weight_bf16, float *__restrict__ inject,
	uint32_t width)
{
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES;
	__shared__ float warp_sums[STREAMS];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = cluster.block_rank(), row = blockIdx.x / WARPS, total_width = STREAMS * width;
	const uint32_t thread = part * LM_WARP_LANES + threadIdx.x;
	const uint16_t *source = normed_bf16 + (uint64_t)row * total_width;
	float sums[STREAMS];
	static_assert(THREADS % LM_WARP_LANES == 0u && WARPS <= 8u, "one portable cluster block per warp of the reference block");
	for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
		sums[stream] = 0.0f;
	for ( uint32_t first = thread; first < total_width; first += THREADS * LM_HYPER_INJECT_BATCH )
	{
		uint16_t values[LM_HYPER_INJECT_BATCH], weights[STREAMS][LM_HYPER_INJECT_BATCH];
#pragma unroll
		for ( uint32_t step = 0u; step < LM_HYPER_INJECT_BATCH; ++step )
		{
			const uint32_t index = first + step * THREADS;
			values[step] = index < total_width ? source[index] : (uint16_t)0u;
#pragma unroll
			for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
				weights[stream][step] = index < total_width ? weight_bf16[(uint64_t)stream * total_width + index] : (uint16_t)0u;
		}
#pragma unroll
		for ( uint32_t step = 0u; step < LM_HYPER_INJECT_BATCH; ++step )
		{
			if ( first + step * THREADS >= total_width )
				break;
			const float value = LmBf16ToFloat(values[step]);
#pragma unroll
			for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
				sums[stream] += value * LmBf16ToFloat(weights[stream][step]);
		}
	}
	for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
	{
		float value = sums[stream];
		for ( uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u )
			value += __shfl_down_sync(0xffffffffu, value, offset);
		if ( threadIdx.x == 0u )
			warp_sums[stream] = value;
	}
	LmClusterBarrier();
	if ( part == 0u )
		for ( uint32_t stream = 0u; stream < STREAMS; ++stream )
		{
			float value = threadIdx.x < WARPS ? cluster.map_shared_rank(warp_sums, threadIdx.x)[stream] : 0.0f;
			for ( uint32_t offset = WARPS / 2u; offset > 0u; offset >>= 1u )
				value += __shfl_down_sync(0xffffffffu, value, offset);
			const float logit = LmHyperRound(value);
			if ( threadIdx.x == 0u )
				inject[(uint64_t)row * STREAMS + stream] = LmHyperRound(2.0f * LmHyperRound(LmHyperSigmoid(LmHyperRound(logit / (float)STREAMS))));
		}
	LmClusterBarrier();
}

template<uint32_t THREADS, uint32_t STREAMS>
static inline cudaError_t LmHyperInjectWeightsLaunch(uint32_t rows, cudaStream_t stream, const uint16_t *normed_bf16, const uint16_t *weight_bf16, float *inject,
	uint32_t width)
{
	if ( rows != 0u && rows <= LM_HYPER_INJECT_CLUSTER_ROWS )
		return LmLaunchClusterKernel((LmHyperInjectWeightsClusterKernel<THREADS, STREAMS>), dim3(rows * (THREADS / LM_WARP_LANES)), dim3(LM_WARP_LANES), 0,
			stream, THREADS / LM_WARP_LANES, 0u, normed_bf16, weight_bf16, inject, width);
	LmHyperInjectWeightsKernel<THREADS, STREAMS><<<rows, THREADS, 0, stream>>>(normed_bf16, weight_bf16, inject, width);
	return cudaPeekAtLastError();
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
