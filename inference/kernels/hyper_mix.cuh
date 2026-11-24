#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#ifdef __CUDACC__
#include <cooperative_groups.h>

#define LM_HYPER_INJECT_CLUSTER_ROWS 8u
#define LM_HYPER_INJECT_BATCH 16u

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmGroupRmsNormKernel(const uint16_t *__restrict__ input_bf16, const float *__restrict__ weight, uint16_t *__restrict__ output_bf16,
	uint32_t groups, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float sq_wave[THREADS * 8u];
	const uint32_t group = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * groups + group) * dimension;
	const float *group_weight = weight + (uint64_t)group * dimension;
	float total = 0.0f;
	const bool vec_ok = ((dimension & 7u) == 0u) && LmIsAligned16(input_bf16) && LmIsAligned16(weight) && LmIsAligned16(output_bf16);
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged_input[vec_slots];
			#pragma unroll
			for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
			{
				const uint32_t wave_base = slot * THREADS * 8u;
				if ( wave_base >= dimension )
				{
					staged_input[slot] = make_uint4(0u, 0u, 0u, 0u);
					continue;
				}
				const uint32_t offset = wave_base + threadIdx.x * 8u;
				if ( offset < dimension )
				{
					uint16_t bits[8];
					float sq[8];
					staged_input[slot] = *(const uint4 *)(input_bf16 + base + offset);
					LmUnpackUint4Bf16(staged_input[slot], bits);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
					{
						const float value = LmBf16ToFloat(bits[k]);
						sq[k] = value * value;
					}
					LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
				}
				else
					staged_input[slot] = make_uint4(0u, 0u, 0u, 0u);
				__syncthreads();
				#pragma unroll
				for ( uint32_t s = 0u; s < 8u; ++s )
				{
					const uint32_t idx = wave_base + s * THREADS + threadIdx.x;
					if ( idx < dimension )
						total += sq_wave[s * THREADS + threadIdx.x];
				}
				if ( wave_base + THREADS * 8u < dimension )
					__syncthreads();
			}
			total = LmBlockSum<THREADS>(total, reduction);
			const float scale = SparkMathRmsScale(total, (float)dimension, epsilon);
			#pragma unroll
			for ( uint32_t slot = 0u; slot < vec_slots; ++slot )
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				uint16_t bits[8];
				float w[8], out_f[8];
				LmUnpackUint4Bf16(staged_input[slot], bits);
				LmLoadWeight8(group_weight, offset, w);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					out_f[k] = LmBf16ToFloat(bits[k]) * scale * w[k];
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		for ( uint32_t wave_base = 0u; wave_base < dimension; wave_base += THREADS * 8u )
		{
			const uint32_t offset = wave_base + threadIdx.x * 8u;
			if ( offset < dimension )
			{
				uint16_t bits[8];
				float sq[8];
				LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + offset), bits);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
				{
					const float value = LmBf16ToFloat(bits[k]);
					sq[k] = value * value;
				}
				LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
			}
			__syncthreads();
			#pragma unroll
			for ( uint32_t s = 0u; s < 8u; ++s )
			{
				const uint32_t idx = wave_base + s * THREADS + threadIdx.x;
				if ( idx < dimension )
					total += sq_wave[s * THREADS + threadIdx.x];
			}
			if ( wave_base + THREADS * 8u < dimension )
				__syncthreads();
		}
		total = LmBlockSum<THREADS>(total, reduction);
		const float scale = SparkMathRmsScale(total, (float)dimension, epsilon);
		for ( uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u )
		{
			uint16_t bits[8];
			float w[8], out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + offset), bits);
			LmLoadWeight8(group_weight, offset, w);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out_f[k] = LmBf16ToFloat(bits[k]) * scale * w[k];
			*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
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
		const float scale = SparkMathRmsScale(total, (float)dimension, epsilon);
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
	const float scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for ( uint32_t index = threadIdx.x; index < dimension; index += THREADS )
		output_bf16[base + index] = LmFloatToBf16(LmBf16ToFloat(input_bf16[base + index]) * scale * group_weight[index]);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperExpandKernel(const uint16_t *__restrict__ hidden_bf16, uint16_t *__restrict__ streams_bf16, uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	if ( ((width & 7u) == 0u) && LmIsAligned16(hidden_bf16) && LmIsAligned16(streams_bf16) )
	{
		const uint32_t width_vecs = width >> 3u;
		const uint32_t total_vecs = streams * width_vecs;
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= (uint64_t)streams * width )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t v = blockIdx.x * (THREADS >> 3u) + threadIdx.x;
				if ( v < total_vecs )
					*(uint4 *)(streams_bf16 + (uint64_t)row * streams * width + ((uint64_t)v << 3u)) =
						*(const uint4 *)(hidden_bf16 + (uint64_t)row * width + ((uint64_t)(v % width_vecs) << 3u));
				return;
			}
		}
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < total_vecs; v += gridDim.x * THREADS )
			*(uint4 *)(streams_bf16 + (uint64_t)row * streams * width + ((uint64_t)v << 3u)) =
				*(const uint4 *)(hidden_bf16 + (uint64_t)row * width + ((uint64_t)(v % width_vecs) << 3u));
		return;
	}
	for ( uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < streams * width; index += gridDim.x * THREADS )
		streams_bf16[(uint64_t)row * streams * width + index] = hidden_bf16[(uint64_t)row * width + index % width];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperGateKernel(uint16_t *__restrict__ values_bf16, uint64_t elements, float divisor)
{
	if ( ((elements & 7u) == 0u) && LmIsAligned16(values_bf16) )
	{
		const uint64_t total_vecs = elements >> 3u;
		for ( uint64_t v = (uint64_t)blockIdx.x * THREADS + threadIdx.x; v < total_vecs; v += (uint64_t)gridDim.x * THREADS )
		{
			const uint64_t offset = v << 3u;
			uint16_t bits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(values_bf16 + offset), bits);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out_f[k] = SparkMathHyperGate(LmBf16ToFloat(bits[k]), divisor);
			*(uint4 *)(values_bf16 + offset) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * THREADS + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * THREADS )
	{
		values_bf16[index] = LmFloatToBf16(SparkMathHyperGate(LmBf16ToFloat(values_bf16[index]),divisor));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperMixKernel(const uint16_t *__restrict__ up_bf16, const uint16_t *__restrict__ normed_bf16, uint16_t *__restrict__ mixed_bf16,
	uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	const uint64_t base = (uint64_t)row * streams * width;
	if ( ((width & 7u) == 0u) && LmIsAligned16(up_bf16) && LmIsAligned16(normed_bf16) && LmIsAligned16(mixed_bf16) )
	{
		const uint32_t width_vecs = width >> 3u;
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= width )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t element = (blockIdx.x * THREADS) + (threadIdx.x << 3u);
				if ( element >= width )
					return;
				float total[8];
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					total[k] = 0.0f;
				for ( uint32_t stream = 0u; stream < streams; ++stream )
				{
					const uint64_t at = base + (uint64_t)stream * width + element;
					uint16_t ubits[8], nbits[8];
					LmUnpackUint4Bf16(*(const uint4 *)(up_bf16 + at), ubits);
					LmUnpackUint4Bf16(*(const uint4 *)(normed_bf16 + at), nbits);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
					{
						const float gate = SparkMathHyperMixGate(LmBf16ToFloat(ubits[k]));
						total[k] += SparkMathHyperMixTerm(gate, LmBf16ToFloat(nbits[k]));
					}
				}
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
					total[k] /= (float)streams;
				*(uint4 *)(mixed_bf16 + (uint64_t)row * width + element) = LmPackUint4FloatToBf16(total);
				return;
			}
		}
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < width_vecs; v += gridDim.x * THREADS )
		{
			const uint32_t element = v << 3u;
			float total[8];
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				total[k] = 0.0f;
			for ( uint32_t stream = 0u; stream < streams; ++stream )
			{
				const uint64_t at = base + (uint64_t)stream * width + element;
				uint16_t ubits[8], nbits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(up_bf16 + at), ubits);
				LmUnpackUint4Bf16(*(const uint4 *)(normed_bf16 + at), nbits);
				#pragma unroll
				for ( uint32_t k = 0u; k < 8u; ++k )
				{
					const float gate = SparkMathHyperMixGate(LmBf16ToFloat(ubits[k]));
					total[k] += SparkMathHyperMixTerm(gate, LmBf16ToFloat(nbits[k]));
				}
			}
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				total[k] /= (float)streams;
			*(uint4 *)(mixed_bf16 + (uint64_t)row * width + element) = LmPackUint4FloatToBf16(total);
		}
		return;
	}
	for ( uint32_t element = blockIdx.x * THREADS + threadIdx.x; element < width; element += gridDim.x * THREADS )
	{
		float total = 0.0f;
		for ( uint32_t stream = 0u; stream < streams; ++stream )
		{
			const uint64_t at = base + (uint64_t)stream * width + element;
			const float gate = SparkMathHyperMixGate(LmBf16ToFloat(up_bf16[at]));
			total += SparkMathHyperMixTerm(gate,LmBf16ToFloat(normed_bf16[at]));
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
		const float logit = SparkMathRoundBf16(LmBlockSum<THREADS>(sums[stream], reduction));
		if ( threadIdx.x == 0u )
			inject[(uint64_t)row * STREAMS + stream] = SparkMathHyperInjectWeight(logit,(float)STREAMS);
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
			const float logit = SparkMathRoundBf16(value);
			if ( threadIdx.x == 0u )
				inject[(uint64_t)row * STREAMS + stream] = SparkMathHyperInjectWeight(logit,(float)STREAMS);
		}
	LmClusterBarrier();
}

template<uint32_t THREADS, uint32_t STREAMS>
static inline SparkGpuError LmHyperInjectWeightsLaunch(uint32_t rows, SparkGpuStream stream, const uint16_t *normed_bf16, const uint16_t *weight_bf16, float *inject,
	uint32_t width)
{
	if ( rows != 0u && rows <= LM_HYPER_INJECT_CLUSTER_ROWS )
		return LmLaunchClusterKernel((LmHyperInjectWeightsClusterKernel<THREADS, STREAMS>), dim3(rows * (THREADS / LM_WARP_LANES)), dim3(LM_WARP_LANES), 0,
			stream, THREADS / LM_WARP_LANES, 0u, normed_bf16, weight_bf16, inject, width);
	LM_LAUNCH((LmHyperInjectWeightsKernel<THREADS, STREAMS>), rows, THREADS, 0, stream, normed_bf16, weight_bf16, inject, width);
	return SparkGpuPeekAtLastError();
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
		total += SparkMathRoundBf16(LmBf16ToFloat(key_bf16[base + index]) * LmBf16ToFloat(query_bf16[base + index]));
	total = LmBlockSum<THREADS>(total, reduction);
	const float gate = SparkMathHyperKeyGate(total,width);
	for ( uint32_t index = threadIdx.x; index < width; index += THREADS )
		gated_bf16[base + index] = LmFloatToBf16(gate * LmBf16ToFloat(value_bf16[(uint64_t)row * width + index]));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperAddKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ gated_bf16, const uint16_t *__restrict__ conv_bf16,
	uint64_t elements)
{
	if ( ((elements & 7u) == 0u) && LmIsAligned16(streams_bf16) && LmIsAligned16(gated_bf16) && LmIsAligned16(conv_bf16) )
	{
		const uint64_t total_vecs = elements >> 3u;
		for ( uint64_t v = (uint64_t)blockIdx.x * THREADS + threadIdx.x; v < total_vecs; v += (uint64_t)gridDim.x * THREADS )
		{
			const uint64_t offset = v << 3u;
			uint16_t sbits[8], gbits[8], cbits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(streams_bf16 + offset), sbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gated_bf16 + offset), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(conv_bf16 + offset), cbits);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
			{
				const float added = SparkMathHyperAddend(LmBf16ToFloat(gbits[k]), LmBf16ToFloat(cbits[k]));
				out_f[k] = LmBf16ToFloat(sbits[k]) + added;
			}
			*(uint4 *)(streams_bf16 + offset) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for ( uint64_t index = (uint64_t)blockIdx.x * THREADS + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * THREADS )
	{
		const float added = SparkMathHyperAddend(LmBf16ToFloat(gated_bf16[index]),LmBf16ToFloat(conv_bf16[index]));
		streams_bf16[index] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[index]) + added);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperInjectKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ output_bf16, const float *__restrict__ inject,
	uint32_t streams, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	if ( ((width & 7u) == 0u) && LmIsAligned16(streams_bf16) && LmIsAligned16(output_bf16) )
	{
		const uint32_t width_vecs = width >> 3u;
		const uint32_t total_vecs = streams * width_vecs;
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= (uint64_t)streams * width )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t v = blockIdx.x * (THREADS >> 3u) + threadIdx.x;
				if ( v < total_vecs )
				{
					const uint32_t stream = v / width_vecs;
					const uint32_t element = (v % width_vecs) << 3u;
					const uint64_t at = (uint64_t)row * streams * width + ((uint64_t)v << 3u);
					const float weight = inject[(uint64_t)row * streams + stream];
					uint16_t sbits[8], obits[8];
					float out_f[8];
					LmUnpackUint4Bf16(*(const uint4 *)(streams_bf16 + at), sbits);
					LmUnpackUint4Bf16(*(const uint4 *)(output_bf16 + (uint64_t)row * width + element), obits);
					#pragma unroll
					for ( uint32_t k = 0u; k < 8u; ++k )
						out_f[k] = LmBf16ToFloat(sbits[k]) + SparkMathHyperInjected(LmBf16ToFloat(obits[k]), weight);
					*(uint4 *)(streams_bf16 + at) = LmPackUint4FloatToBf16(out_f);
				}
				return;
			}
		}
		for ( uint32_t v = blockIdx.x * THREADS + threadIdx.x; v < total_vecs; v += gridDim.x * THREADS )
		{
			const uint32_t stream = v / width_vecs;
			const uint32_t element = (v % width_vecs) << 3u;
			const uint64_t at = (uint64_t)row * streams * width + ((uint64_t)v << 3u);
			const float weight = inject[(uint64_t)row * streams + stream];
			uint16_t sbits[8], obits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(streams_bf16 + at), sbits);
			LmUnpackUint4Bf16(*(const uint4 *)(output_bf16 + (uint64_t)row * width + element), obits);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				out_f[k] = LmBf16ToFloat(sbits[k]) + SparkMathHyperInjected(LmBf16ToFloat(obits[k]), weight);
			*(uint4 *)(streams_bf16 + at) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for ( uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < streams * width; index += gridDim.x * THREADS )
	{
		const uint64_t at = (uint64_t)row * streams * width + index;
		const float injected = SparkMathHyperInjected(LmBf16ToFloat(output_bf16[(uint64_t)row * width + index % width]),inject[(uint64_t)row * streams + index / width]);
		streams_bf16[at] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[at]) + injected);
	}
}
#endif
