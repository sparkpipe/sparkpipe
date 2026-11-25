#pragma once

#include <stdint.h>
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#ifdef __CUDACC__
#include <cooperative_groups.h>

#define LM_HYPER_INJECT_CLUSTER_ROWS 8u
#define LM_HYPER_INJECT_BATCH 16u

struct LmHyperGateOp { float divisor; __device__ __forceinline__ float operator()(float x) const { return(SparkMathHyperGate(x, divisor)); } };

template<uint32_t THREADS>
static __device__ __forceinline__ void LmGroupRmsWaveSq(const uint16_t *row_ptr, float *sq_wave, uint32_t wave_base, uint32_t dimension, uint4 *staged_slot, float *total)
{
	const uint32_t offset = wave_base + threadIdx.x * 8u;
	if ( offset < dimension )
	{
		const uint4 vec = __ldg((const uint4 *)(row_ptr + offset));
		if ( staged_slot != 0 ) *staged_slot = vec;
		float vals[8], sq[8];
		LmLoadBf16x8AsFloat(vec, vals);
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k) sq[k] = vals[k] * vals[k];
		LmStoreFloat8(sq_wave + threadIdx.x * 8u, sq);
	}
	else if ( staged_slot != 0 ) *staged_slot = make_uint4(0u, 0u, 0u, 0u);
	__syncthreads();
	#pragma unroll
	for (uint32_t s = 0u; s < 8u; ++s)
		if ( wave_base + s * THREADS + threadIdx.x < dimension ) *total += sq_wave[s * THREADS + threadIdx.x];
	if ( wave_base + THREADS * 8u < dimension ) __syncthreads();
}

static __device__ __forceinline__ void LmGroupRmsStore8(uint4 vec, const float *group_weight, uint32_t offset, float scale, uint16_t *output_row)
{
	float vals[8], w[8];
	LmLoadBf16x8AsFloat(vec, vals);
	LmLoadWeight8(group_weight, offset, w);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) vals[k] = vals[k] * scale * w[k];
	*(uint4 *)(output_row + offset) = LmPackUint4FloatToBf16(vals);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmGroupRmsNormKernel(const uint16_t *__restrict__ input_bf16, const float *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t groups, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()], sq_wave[THREADS * 8u];
	LmDependentWait();
	const uint64_t base = ((uint64_t)blockIdx.y * groups + blockIdx.x) * dimension;
	const uint16_t *in_row = input_bf16 + base;
	uint16_t *out_row = output_bf16 + base;
	const float *group_weight = weight + (uint64_t)blockIdx.x * dimension;
	float total = 0.0f;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(input_bf16) && LmIsAligned16(weight) && LmIsAligned16(output_bf16) )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged_input[vec_slots];
			#pragma unroll
			for (uint32_t slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t wave_base = slot * THREADS * 8u;
				if ( wave_base >= dimension ) staged_input[slot] = make_uint4(0u, 0u, 0u, 0u);
				else LmGroupRmsWaveSq<THREADS>(in_row, sq_wave, wave_base, dimension, &staged_input[slot], &total);
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
			#pragma unroll
			for (uint32_t slot = 0u; slot < vec_slots; ++slot)
				if ( (slot * THREADS + threadIdx.x) * 8u < dimension )
					LmGroupRmsStore8(staged_input[slot], group_weight, (slot * THREADS + threadIdx.x) * 8u, scale, out_row);
			LmDependentRelease();
			return;
		}
		for (uint32_t wave_base = 0u; wave_base < dimension; wave_base += THREADS * 8u)
			LmGroupRmsWaveSq<THREADS>(in_row, sq_wave, wave_base, dimension, (uint4 *)0, &total);
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
		for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			LmGroupRmsStore8(__ldg((const uint4 *)(in_row + offset)), group_weight, offset, scale, out_row);
		LmDependentRelease();
		return;
	}
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		float staged_input[LM_RMS_NORM_STAGED], staged_weight[LM_RMS_NORM_STAGED];
		#pragma unroll
		for (uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			staged_input[slot] = index < dimension ? LmBf16ToFloat(__ldg(in_row + index)) : 0.0f;
			staged_weight[slot] = index < dimension ? __ldg(group_weight + index) : 0.0f;
			if ( index < dimension ) total += staged_input[slot] * staged_input[slot];
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
		#pragma unroll
		for (uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
			if ( threadIdx.x + slot * THREADS < dimension )
				out_row[threadIdx.x + slot * THREADS] = LmFloatToBf16(staged_input[slot] * scale * staged_weight[slot]);
		LmDependentRelease();
		return;
	}
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) { const float v = LmBf16ToFloat(__ldg(in_row + index)); total += v * v; }
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		out_row[index] = LmFloatToBf16(LmBf16ToFloat(__ldg(in_row + index)) * scale * __ldg(group_weight + index));
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperExpandKernel(const uint16_t *__restrict__ hidden_bf16, uint16_t *__restrict__ streams_bf16, uint32_t streams, uint32_t width)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y, total = streams * width;
	if ( ((width & 7u) == 0u) && LmIsAligned16(hidden_bf16) && LmIsAligned16(streams_bf16) )
	{
		uint32_t elem, stride;
		if ( !LmVecElement<THREADS>(total, &elem, &stride) )
		{
			LmDependentRelease();
			return;
		}
		const uint32_t width_vecs = width >> 3u, total_vecs = total >> 3u, v_stride = stride >> 3u;
		for (uint32_t v = elem >> 3u; v < total_vecs; v += v_stride)
			*(uint4 *)(streams_bf16 + (uint64_t)row * total + ((uint64_t)v << 3u)) = __ldg((const uint4 *)(hidden_bf16 + (uint64_t)row * width + ((uint64_t)(v % width_vecs) << 3u)));
		LmDependentRelease();
		return;
	}
	for (uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < total; index += gridDim.x * THREADS)
		streams_bf16[(uint64_t)row * total + index] = __ldg(hidden_bf16 + (uint64_t)row * width + index % width);
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperGateKernel(uint16_t *__restrict__ values_bf16, uint64_t elements, float divisor)
{
	LmDependentWait();
	LmUnaryBf16GridBody<THREADS>(values_bf16, elements, LmHyperGateOp{divisor});
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperMixKernel(const uint16_t *__restrict__ up_bf16, const uint16_t *__restrict__ normed_bf16, uint16_t *__restrict__ mixed_bf16, uint32_t streams, uint32_t width)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y;
	const uint64_t base = (uint64_t)row * streams * width;
	if ( ((width & 7u) == 0u) && LmIsAligned16(up_bf16) && LmIsAligned16(normed_bf16) && LmIsAligned16(mixed_bf16) )
	{
		uint32_t elem, stride;
		if ( !LmVecElement<THREADS>(width, &elem, &stride) )
		{
			LmDependentRelease();
			return;
		}
		const uint32_t width_vecs = width >> 3u, v_stride = stride >> 3u;
		for (uint32_t v = elem >> 3u; v < width_vecs; v += v_stride)
		{
			const uint32_t element = v << 3u;
			float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			for (uint32_t stream = 0u; stream < streams; ++stream)
			{
				const uint64_t at = base + (uint64_t)stream * width + element;
				float u[8], n[8];
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(up_bf16 + at)), u);
				LmLoadBf16x8AsFloat(__ldg((const uint4 *)(normed_bf16 + at)), n);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k) total[k] += SparkMathHyperMixTerm(SparkMathHyperMixGate(u[k]), n[k]);
			}
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) total[k] /= (float)streams;
			*(uint4 *)(mixed_bf16 + (uint64_t)row * width + element) = LmPackUint4FloatToBf16(total);
		}
		LmDependentRelease();
		return;
	}
	for (uint32_t element = blockIdx.x * THREADS + threadIdx.x; element < width; element += gridDim.x * THREADS)
	{
		float total = 0.0f;
		for (uint32_t stream = 0u; stream < streams; ++stream)
		{
			const uint64_t at = base + (uint64_t)stream * width + element;
			total += SparkMathHyperMixTerm(SparkMathHyperMixGate(LmBf16ToFloat(__ldg(up_bf16 + at))), LmBf16ToFloat(__ldg(normed_bf16 + at)));
		}
		mixed_bf16[(uint64_t)row * width + element] = LmFloatToBf16(total / (float)streams);
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t STREAMS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperInjectWeightsKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ weight_bf16, float *__restrict__ inject, uint32_t width)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	LmDependentWait();
	const uint32_t row = blockIdx.x, total_width = STREAMS * width;
	const uint16_t *source = normed_bf16 + (uint64_t)row * total_width;
	float sums[STREAMS];
	#pragma unroll
	for (uint32_t stream = 0u; stream < STREAMS; ++stream) sums[stream] = 0.0f;
	for (uint32_t index = threadIdx.x; index < total_width; index += THREADS)
	{
		const float value = LmBf16ToFloat(__ldg(source + index));
		#pragma unroll
		for (uint32_t stream = 0u; stream < STREAMS; ++stream) sums[stream] += value * LmBf16ToFloat(__ldg(weight_bf16 + (uint64_t)stream * total_width + index));
	}
	#pragma unroll
	for (uint32_t stream = 0u; stream < STREAMS; ++stream)
	{
		const float logit = SparkMathRoundBf16(LmBlockSum<THREADS>(sums[stream], reduction));
		if ( threadIdx.x == 0u ) inject[(uint64_t)row * STREAMS + stream] = SparkMathHyperInjectWeight(logit, (float)STREAMS);
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t STREAMS>
__global__ __launch_bounds__(LM_WARP_LANES, 1)
void LmHyperInjectWeightsClusterKernel(const uint16_t *__restrict__ normed_bf16, const uint16_t *__restrict__ weight_bf16, float *__restrict__ inject, uint32_t width)
{
	constexpr uint32_t WARPS = THREADS / LM_WARP_LANES;
	__shared__ float warp_sums[STREAMS];
	LmDependentWait();
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = cluster.block_rank(), row = blockIdx.x / WARPS, total_width = STREAMS * width, thread = part * LM_WARP_LANES + threadIdx.x;
	const uint16_t *source = normed_bf16 + (uint64_t)row * total_width;
	float sums[STREAMS];
	static_assert(THREADS % LM_WARP_LANES == 0u && WARPS <= 8u, "one portable cluster block per warp of the reference block");
	#pragma unroll
	for (uint32_t stream = 0u; stream < STREAMS; ++stream) sums[stream] = 0.0f;
	for (uint32_t first = thread; first < total_width; first += THREADS * LM_HYPER_INJECT_BATCH)
	{
		uint16_t values[LM_HYPER_INJECT_BATCH], weights[STREAMS][LM_HYPER_INJECT_BATCH];
		#pragma unroll
		for (uint32_t step = 0u; step < LM_HYPER_INJECT_BATCH; ++step)
		{
			const uint32_t index = first + step * THREADS;
			values[step] = index < total_width ? __ldg(source + index) : (uint16_t)0u;
			#pragma unroll
			for (uint32_t stream = 0u; stream < STREAMS; ++stream)
				weights[stream][step] = index < total_width ? __ldg(weight_bf16 + (uint64_t)stream * total_width + index) : (uint16_t)0u;
		}
		#pragma unroll
		for (uint32_t step = 0u; step < LM_HYPER_INJECT_BATCH; ++step)
		{
			if ( first + step * THREADS >= total_width ) break;
			const float value = LmBf16ToFloat(values[step]);
			#pragma unroll
			for (uint32_t stream = 0u; stream < STREAMS; ++stream) sums[stream] += value * LmBf16ToFloat(weights[stream][step]);
		}
	}
	#pragma unroll
	for (uint32_t stream = 0u; stream < STREAMS; ++stream)
	{
		float value = sums[stream];
		#pragma unroll
		for (uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u) value += __shfl_down_sync(0xffffffffu, value, offset);
		if ( threadIdx.x == 0u ) warp_sums[stream] = value;
	}
	LmClusterBarrier();
	if ( part == 0u )
		#pragma unroll
		for (uint32_t stream = 0u; stream < STREAMS; ++stream)
		{
			float value = threadIdx.x < WARPS ? cluster.map_shared_rank(warp_sums, threadIdx.x)[stream] : 0.0f;
			#pragma unroll
			for (uint32_t offset = WARPS / 2u; offset > 0u; offset >>= 1u) value += __shfl_down_sync(0xffffffffu, value, offset);
			const float logit = SparkMathRoundBf16(value);
			if ( threadIdx.x == 0u ) inject[(uint64_t)row * STREAMS + stream] = SparkMathHyperInjectWeight(logit, (float)STREAMS);
		}
	LmClusterBarrier();
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t STREAMS>
static inline SparkGpuError LmHyperInjectWeightsLaunch(uint32_t rows, SparkGpuStream stream, const uint16_t *normed_bf16, const uint16_t *weight_bf16, float *inject, uint32_t width)
{
	if ( rows != 0u && rows <= LM_HYPER_INJECT_CLUSTER_ROWS )
		return LmLaunchClusterKernel((LmHyperInjectWeightsClusterKernel<THREADS, STREAMS>), dim3(rows * (THREADS / LM_WARP_LANES)), dim3(LM_WARP_LANES), 0, stream, THREADS / LM_WARP_LANES, 0u, normed_bf16, weight_bf16, inject, width);
	LM_LAUNCH_DEPENDENT((LmHyperInjectWeightsKernel<THREADS, STREAMS>), rows, THREADS, 0, stream, normed_bf16, weight_bf16, inject, width);
	return SparkGpuPeekAtLastError();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperKeyGateKernel(const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ query_bf16, const uint16_t *__restrict__ value_bf16, uint16_t *__restrict__ gated_bf16, uint32_t streams, uint32_t width)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	LmDependentWait();
	const uint32_t stream = blockIdx.x, row = blockIdx.y;
	const uint64_t base = ((uint64_t)row * streams + stream) * width;
	float total = 0.0f;
	for (uint32_t index = threadIdx.x; index < width; index += THREADS)
		total += SparkMathRoundBf16(LmBf16ToFloat(__ldg(key_bf16 + base + index)) * LmBf16ToFloat(__ldg(query_bf16 + base + index)));
	const float gate = SparkMathHyperKeyGate(LmBlockSumOnce<THREADS>(total, reduction), width);
	if ( ((width & 7u) == 0u) && LmIsAligned16(value_bf16) && LmIsAligned16(gated_bf16) )
	{
		const uint16_t *val_row = value_bf16 + (uint64_t)row * width;
		uint16_t *out_row = gated_bf16 + base;
		for (uint32_t off = threadIdx.x * 8u; off < width; off += THREADS * 8u)
		{
			float v[8];
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(val_row + off)), v);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) v[k] = gate * v[k];
			*(uint4 *)(out_row + off) = LmPackUint4FloatToBf16(v);
		}
		LmDependentRelease();
		return;
	}
	for (uint32_t index = threadIdx.x; index < width; index += THREADS)
		gated_bf16[base + index] = LmFloatToBf16(gate * LmBf16ToFloat(__ldg(value_bf16 + (uint64_t)row * width + index)));
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperAddKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ gated_bf16, const uint16_t *__restrict__ conv_bf16, uint64_t elements)
{
	LmDependentWait();
	if ( ((elements & 7u) == 0u) && LmIsAligned16(streams_bf16) && LmIsAligned16(gated_bf16) && LmIsAligned16(conv_bf16) )
	{
		for (uint64_t v = (uint64_t)blockIdx.x * THREADS + threadIdx.x; v < (elements >> 3u); v += (uint64_t)gridDim.x * THREADS)
		{
			const uint64_t offset = v << 3u;
			float s[8], g[8], c[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(streams_bf16 + offset), s);
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(gated_bf16 + offset)), g);
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(conv_bf16 + offset)), c);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) s[k] += SparkMathHyperAddend(g[k], c[k]);
			*(uint4 *)(streams_bf16 + offset) = LmPackUint4FloatToBf16(s);
		}
		LmDependentRelease();
		return;
	}
	for (uint64_t index = (uint64_t)blockIdx.x * THREADS + threadIdx.x; index < elements; index += (uint64_t)gridDim.x * THREADS)
		streams_bf16[index] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[index]) + SparkMathHyperAddend(LmBf16ToFloat(__ldg(gated_bf16 + index)), LmBf16ToFloat(__ldg(conv_bf16 + index))));
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHyperInjectKernel(uint16_t *__restrict__ streams_bf16, const uint16_t *__restrict__ output_bf16, const float *__restrict__ inject, uint32_t streams, uint32_t width)
{
	LmDependentWait();
	const uint32_t row = blockIdx.y, total = streams * width;
	if ( ((width & 7u) == 0u) && LmIsAligned16(streams_bf16) && LmIsAligned16(output_bf16) )
	{
		uint32_t elem, stride;
		if ( !LmVecElement<THREADS>(total, &elem, &stride) )
		{
			LmDependentRelease();
			return;
		}
		const uint32_t width_vecs = width >> 3u, total_vecs = total >> 3u, v_stride = stride >> 3u;
		for (uint32_t v = elem >> 3u; v < total_vecs; v += v_stride)
		{
			const uint64_t at = (uint64_t)row * total + ((uint64_t)v << 3u);
			const float weight = __ldg(inject + (uint64_t)row * streams + (v / width_vecs));
			float s[8], o[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(streams_bf16 + at), s);
			LmLoadBf16x8AsFloat(__ldg((const uint4 *)(output_bf16 + (uint64_t)row * width + ((v % width_vecs) << 3u))), o);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) s[k] += SparkMathHyperInjected(o[k], weight);
			*(uint4 *)(streams_bf16 + at) = LmPackUint4FloatToBf16(s);
		}
		LmDependentRelease();
		return;
	}
	for (uint32_t index = blockIdx.x * THREADS + threadIdx.x; index < total; index += gridDim.x * THREADS)
	{
		const uint64_t at = (uint64_t)row * total + index;
		streams_bf16[at] = LmFloatToBf16(LmBf16ToFloat(streams_bf16[at]) + SparkMathHyperInjected(LmBf16ToFloat(__ldg(output_bf16 + (uint64_t)row * width + index % width)), __ldg(inject + (uint64_t)row * streams + index / width)));
	}
	LmDependentRelease();
}
#endif
