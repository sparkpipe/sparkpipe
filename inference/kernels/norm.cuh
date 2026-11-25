#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/vec.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include <stdint.h>

#define LM_RMS_NORM_STAGED 32u
#define LM_GATE_SIGMOID 0u
#define LM_GATE_SOFTPLUS 1u
#define LM_GATE_SILU 2u
#define LM_MOE_FINALIZE_ADD_ELEMENTS 8u

template<bool ROUND_FIRST> struct LmRmsGainOp { __device__ __forceinline__ float operator()(float v, float s, float w) const { return(ROUND_FIRST ? SparkMathNormGainRounded(v * s, w) : v * s * w); } };
struct LmRmsCenteredOp { __device__ __forceinline__ float operator()(float v, float s, float w) const { return(v * s * (1.0f + w)); } };
template<bool ROUND_FIRST> struct LmHeadRmsOp { float head_multiply; __device__ __forceinline__ float operator()(float v, float s, float w, float, bool has_w) const { float x = v * s; if ( has_w ) x = ROUND_FIRST ? SparkMathNormGainRounded(x, w) : x * w; return(SparkMathRoundBf16(x) * head_multiply); } };
struct LmRmsSigmoidGateOp { __device__ __forceinline__ float operator()(float v, float s, float w, float g, bool) const { return(v * s * w * SparkMathSigmoidFast(g)); } };
struct LmSiluOp { __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathSiluFast(g) * u); } };
struct LmSiluLimitOp { float limit; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathClampedGlu(SparkMathSiluFast(g), u, limit)); } };
struct LmSiluRoundedLimitOp { float limit; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathClampedGlu(SparkMathRoundBf16(SparkMathSilu(g)), u, limit)); } };
struct LmSiluGateLimitOp { float limit; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathGateLimitedGlu(g, u, limit)); } };
struct LmClampedUpGateOp { float limit; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathSiluFast(g > limit ? limit : g) * (u > limit ? limit : (u < -limit ? -limit : u))); } };
struct LmSituOp { float beta, linear_beta; __device__ __forceinline__ float operator()(float g, float u) const { return(SparkMathSitu(g, u, beta, linear_beta)); } };
template<bool ROUND_GATE> struct LmSigmoidGateOp { __device__ __forceinline__ float operator()(float g, float v) const { return(v * (ROUND_GATE ? SparkMathRoundBf16(SparkMathSigmoid(g)) : SparkMathSigmoidFast(g))); } };
template<bool ROUND_ACT> struct LmStridedGateUpOp { __device__ __forceinline__ float operator()(float g, float u) const { return((ROUND_ACT ? SparkMathRoundBf16(SparkMathSilu(g)) : SparkMathSiluFast(g)) * u); } };
struct LmAddOp { __device__ __forceinline__ float operator()(float a, float b) const { return(a + b); } };
struct LmScaleOp { float scale; __device__ __forceinline__ float operator()(float x) const { return(x * scale); } };

template<bool HAS_RESIDUAL>
static __device__ __forceinline__ uint4 LmRmsLoadAdd8(const uint16_t *__restrict__ in_ptr, const uint16_t *__restrict__ res_ptr, float vals[8])
{
	const uint4 in_vec = *(const uint4 *)in_ptr;
	LmLoadBf16x8AsFloat(in_vec, vals);
	if constexpr ( HAS_RESIDUAL )
		if ( res_ptr != 0 )
		{
			float res[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)res_ptr, res);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) vals[k] += res[k];
			const uint4 packed = LmPackUint4FloatToBf16(vals);
			LmLoadBf16x8AsFloat(packed, vals);
			return(packed);
		}
	return(in_vec);
}

template<class Weight, class OutOp>
static __device__ __forceinline__ void LmRmsStore8(const float vals[8], const Weight *__restrict__ weight, uint32_t offset, float scale, uint16_t *__restrict__ out_ptr, OutOp op)
{
	float w[8], out_f[8];
	LmLoadWeight8(weight, offset, w);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) out_f[k] = op(vals[k], scale, w[k]);
	*(uint4 *)out_ptr = LmPackUint4FloatToBf16(out_f);
}

template<uint32_t THREADS, uint32_t VEC_SLOTS, class Weight, bool EXACT_ORDER, bool HAS_RESIDUAL, class OutOp>
static __device__ __forceinline__ void LmRmsNormRowStagedVec(float *__restrict__ row, float *__restrict__ reduction, const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint64_t base, uint32_t dimension, float epsilon, OutOp op)
{
	uint4 staged[VEC_SLOTS];
	float total = 0.0f;
	#pragma unroll
	for (uint32_t slot = 0u; slot < VEC_SLOTS; ++slot)
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset < dimension )
		{
			float vals[8];
			staged[slot] = LmRmsLoadAdd8<HAS_RESIDUAL>(input_bf16 + base + offset, HAS_RESIDUAL && residual_bf16 != 0 ? residual_bf16 + base + offset : (const uint16_t *)0, vals);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) { if constexpr ( EXACT_ORDER ) vals[k] *= vals[k]; else total += vals[k] * vals[k]; }
			if constexpr ( EXACT_ORDER ) LmStoreFloat8(row + offset, vals);
		}
		else staged[slot] = make_uint4(0u, 0u, 0u, 0u);
	}
	if constexpr ( EXACT_ORDER )
	{
		__syncthreads();
		#pragma unroll
		for (uint32_t slot = 0u; slot < VEC_SLOTS * 8u; ++slot)
			if ( threadIdx.x + slot * THREADS < dimension ) total += row[threadIdx.x + slot * THREADS];
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
	#pragma unroll
	for (uint32_t slot = 0u; slot < VEC_SLOTS; ++slot)
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset >= dimension ) continue;
		if constexpr ( HAS_RESIDUAL ) if ( residual_out_bf16 != 0 ) *(uint4 *)(residual_out_bf16 + base + offset) = staged[slot];
		float vals[8];
		LmLoadBf16x8AsFloat(staged[slot], vals);
		LmRmsStore8(vals, weight, offset, scale, output_bf16 + base + offset, op);
	}
}

template<uint32_t THREADS, class Weight, bool EXACT_ORDER, bool HAS_RESIDUAL, class OutOp>
static __device__ __forceinline__ void LmRmsNormRowBody(float *__restrict__ shared, const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon, OutOp op)
{
	float *row = shared, *reduction = shared + dimension, total = 0.0f;
	const uint64_t base = (uint64_t)blockIdx.x * row_stride;
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(output_bf16) && LmIsAligned16(weight) &&
		(!HAS_RESIDUAL || ((residual_bf16 == 0 || LmIsAligned16(residual_bf16)) && (residual_out_bf16 == 0 || LmIsAligned16(residual_out_bf16))));
	if ( vec_ok )
	{
		if ( dimension <= THREADS * 16u && (!EXACT_ORDER || LmIsAligned16(row)) )
		{
			LmRmsNormRowStagedVec<THREADS, 2u, Weight, EXACT_ORDER, HAS_RESIDUAL>(row, reduction, input_bf16, residual_bf16, weight, residual_out_bf16, output_bf16, base, dimension, epsilon, op);
			return;
		}
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED && (!EXACT_ORDER || LmIsAligned16(row)) )
		{
			LmRmsNormRowStagedVec<THREADS, LM_RMS_NORM_STAGED / 8u, Weight, EXACT_ORDER, HAS_RESIDUAL>(row, reduction, input_bf16, residual_bf16, weight, residual_out_bf16, output_bf16, base, dimension, epsilon, op);
			return;
		}
		if ( LmIsAligned16(row) )
		{
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				float vals[8];
				LmRmsLoadAdd8<HAS_RESIDUAL>(input_bf16 + base + offset, HAS_RESIDUAL && residual_bf16 != 0 ? residual_bf16 + base + offset : (const uint16_t *)0, vals);
				if constexpr ( !EXACT_ORDER )
				{
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k) total += vals[k] * vals[k];
				}
				LmStoreFloat8(row + offset, vals);
			}
			if constexpr ( EXACT_ORDER )
			{
				__syncthreads();
				for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) total += row[index] * row[index];
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				float vals[8];
				LmLoadFloat8(row + offset, vals);
				if constexpr ( HAS_RESIDUAL ) if ( residual_out_bf16 != 0 ) *(uint4 *)(residual_out_bf16 + base + offset) = LmPackUint4FloatToBf16(vals);
				LmRmsStore8(vals, weight, offset, scale, output_bf16 + base + offset, op);
			}
			return;
		}
	}
	if ( dimension <= THREADS * 16u )
	{
		uint16_t staged[16u];
		#pragma unroll
		for (uint32_t slot = 0u; slot < 16u; ++slot)
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			if ( index < dimension )
			{
				uint16_t bits = input_bf16[base + index];
				float value = LmBf16ToFloat(bits);
				if constexpr ( HAS_RESIDUAL ) if ( residual_bf16 != 0 ) { bits = LmFloatToBf16(value + LmBf16ToFloat(residual_bf16[base + index])); value = LmBf16ToFloat(bits); }
				staged[slot] = bits;
				total += value * value;
			}
			else staged[slot] = 0u;
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
		#pragma unroll
		for (uint32_t slot = 0u; slot < 16u; ++slot)
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			if ( index >= dimension ) continue;
			if constexpr ( HAS_RESIDUAL ) if ( residual_out_bf16 != 0 ) residual_out_bf16[base + index] = staged[slot];
			output_bf16[base + index] = LmFloatToBf16(op(LmBf16ToFloat(staged[slot]), scale, LmScalarToFloat(weight[index])));
		}
		return;
	}
	#pragma unroll 8
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		if constexpr ( HAS_RESIDUAL ) if ( residual_bf16 != 0 ) value = SparkMathResidualSum(value, LmBf16ToFloat(residual_bf16[base + index]));
		row[index] = value;
		total += value * value;
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
	#pragma unroll 8
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
	{
		if constexpr ( HAS_RESIDUAL ) if ( residual_out_bf16 != 0 ) residual_out_bf16[base + index] = LmFloatToBf16(row[index]);
		output_bf16[base + index] = LmFloatToBf16(op(row[index], scale, LmScalarToFloat(weight[index])));
	}
}

template<class Weight, class OutOp>
static __device__ __forceinline__ void LmRmsSmallStore8(uint4 vec, const uint16_t *gate, const Weight *weight, uint32_t offset, float scale, uint16_t *output, OutOp op)
{
	float x[8], g[8], w[8], out[8];
	LmLoadBf16x8AsFloat(vec, x);
	if ( gate != 0 ) LmLoadBf16x8AsFloat(*(const uint4 *)(gate + offset), g);
	if ( weight != 0 ) LmLoadWeight8(weight, offset, w);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) out[k] = op(x[k], scale, weight != 0 ? w[k] : 1.0f, gate != 0 ? g[k] : 0.0f, weight != 0);
	*(uint4 *)(output + offset) = LmPackUint4FloatToBf16(out);
}

template<uint32_t THREADS, uint32_t VEC_SLOTS, class Weight, class OutOp>
static __device__ __forceinline__ void LmRmsNormSmallBody(const uint16_t *input, const uint16_t *gate, const Weight *weight, uint16_t *output, uint32_t dimension, float epsilon, OutOp op)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	float total = 0.0f;
	const bool vec_ok = ((dimension & 7u) == 0u) && LmIsAligned16(input) && LmIsAligned16(output) &&
		(weight == 0 || LmIsAligned16(weight)) && (gate == 0 || LmIsAligned16(gate));
	if ( vec_ok )
	{
		if ( dimension <= THREADS * VEC_SLOTS * 8u )
		{
			uint4 staged[VEC_SLOTS];
			#pragma unroll
			for (uint32_t slot = 0u; slot < VEC_SLOTS; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				staged[slot] = offset < dimension ? *(const uint4 *)(input + offset) : make_uint4(0u, 0u, 0u, 0u);
				if ( offset < dimension ) total = LmSumSqBf16x8(staged[slot], total);
			}
			const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
			#pragma unroll
			for (uint32_t slot = 0u; slot < VEC_SLOTS; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension ) LmRmsSmallStore8(staged[slot], gate, weight, offset, scale, output, op);
			}
			return;
		}
		for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u) total = LmSumSqBf16x8(*(const uint4 *)(input + index), total);
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
		for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
			LmRmsSmallStore8(*(const uint4 *)(input + index), gate, weight, index, scale, output, op);
		return;
	}
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) { const float v = LmBf16ToFloat(input[index]); total += v * v; }
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)dimension, epsilon);
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		output[index] = LmFloatToBf16(op(LmBf16ToFloat(input[index]), scale, weight != 0 ? LmScalarToFloat(weight[index]) : 1.0f, gate != 0 ? LmBf16ToFloat(gate[index]) : 0.0f, weight != 0));
}

template<uint32_t THREADS, class Weight, bool ROUND_FIRST = false>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedResidualRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	LmDependentWait();
	LmDependentRelease();
	LmRmsNormRowBody<THREADS, Weight, true, true>(lm_norm_shared, input_bf16, residual_bf16, weight, residual_out_bf16, output_bf16, dimension, row_stride, epsilon, LmRmsGainOp<ROUND_FIRST>{});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmBf16RmsNormKernel(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	LmDependentWait();
	LmDependentRelease();
	LmRmsNormRowBody<THREADS, uint16_t, true, false>(lm_norm_shared, input, (const uint16_t *)0, weight, (uint16_t *)0, output, dimension, row_stride, epsilon, LmRmsGainOp<true>{});
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmLayerNormStats(float total, float total_squared, float *pair_reduction, uint32_t dimension, float epsilon, float *mean, float *inverse)
{
	LmBlockSum2Once<THREADS>(&total, &total_squared, pair_reduction);
	*mean = total / (float)dimension;
	*inverse = rsqrtf(fmaxf((total_squared / (float)dimension) - ((*mean) * (*mean)), 0.0f) + epsilon);
}

template<class Weight>
static __device__ __forceinline__ void LmLayerNormStore8(const float v[8], const Weight *weight, const Weight *bias, uint32_t offset, float mean, float inverse, uint16_t *out_ptr)
{
	float w[8], b[8], out_f[8];
	LmLoadWeight8(weight, offset, w);
	LmLoadWeight8(bias, offset, b);
	#pragma unroll
	for (uint32_t k = 0u; k < 8u; ++k) out_f[k] = ((v[k] - mean) * inverse * w[k]) + b[k];
	*(uint4 *)out_ptr = LmPackUint4FloatToBf16(out_f);
}

template<uint32_t THREADS, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmLayerNormKernel(const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, const Weight *__restrict__ bias, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	__shared__ float pair_reduction[2u * LmBlockWarps<THREADS>()];
	LmDependentWait();
	LmDependentRelease();
	float *row = lm_norm_shared, total = 0.0f, total_squared = 0.0f, mean, inverse;
	const uint64_t base = (uint64_t)blockIdx.x * row_stride;
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(weight) && LmIsAligned16(bias) && LmIsAligned16(output_bf16);
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged[vec_slots];
			#pragma unroll
			for (uint32_t slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				staged[slot] = offset < dimension ? *(const uint4 *)(input_bf16 + base + offset) : make_uint4(0u, 0u, 0u, 0u);
				if ( offset < dimension )
				{
					float v[8];
					LmLoadBf16x8AsFloat(staged[slot], v);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k) { total += v[k]; total_squared += v[k] * v[k]; }
				}
			}
			LmLayerNormStats<THREADS>(total, total_squared, pair_reduction, dimension, epsilon, &mean, &inverse);
			#pragma unroll
			for (uint32_t slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension ) { float v[8]; LmLoadBf16x8AsFloat(staged[slot], v); LmLayerNormStore8(v, weight, bias, offset, mean, inverse, output_bf16 + base + offset); }
			}
			return;
		}
		if ( LmIsAligned16(row) )
		{
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				float v[8];
				LmLoadBf16x8AsFloat(*(const uint4 *)(input_bf16 + base + offset), v);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k) { total += v[k]; total_squared += v[k] * v[k]; }
				LmStoreFloat8(row + offset, v);
			}
			LmLayerNormStats<THREADS>(total, total_squared, pair_reduction, dimension, epsilon, &mean, &inverse);
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				float v[8];
				LmLoadFloat8(row + offset, v);
				LmLayerNormStore8(v, weight, bias, offset, mean, inverse, output_bf16 + base + offset);
			}
			return;
		}
	}
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		uint16_t staged[LM_RMS_NORM_STAGED];
		#pragma unroll
		for (uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
		{
			const uint32_t index = threadIdx.x + slot * THREADS;
			staged[slot] = index < dimension ? input_bf16[base + index] : (uint16_t)0u;
			if ( index < dimension ) { const float v = LmBf16ToFloat(staged[slot]); total += v; total_squared += v * v; }
		}
		LmLayerNormStats<THREADS>(total, total_squared, pair_reduction, dimension, epsilon, &mean, &inverse);
		#pragma unroll
		for (uint32_t slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
			if ( threadIdx.x + slot * THREADS < dimension )
				output_bf16[base + threadIdx.x + slot * THREADS] = LmFloatToBf16(((LmBf16ToFloat(staged[slot]) - mean) * inverse * LmScalarToFloat(weight[threadIdx.x + slot * THREADS])) + LmScalarToFloat(bias[threadIdx.x + slot * THREADS]));
		return;
	}
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
	{
		const float v = LmBf16ToFloat(input_bf16[base + index]);
		row[index] = v; total += v; total_squared += v * v;
	}
	LmLayerNormStats<THREADS>(total, total_squared, pair_reduction, dimension, epsilon, &mean, &inverse);
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] = LmFloatToBf16(((row[index] - mean) * inverse * LmScalarToFloat(weight[index])) + LmScalarToFloat(bias[index]));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base + (gate_first ? 0u : dimension), gate_up_bf16 + base + (gate_first ? dimension : 0u), output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmSiluOp{});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base + (gate_first ? 0u : dimension), gate_up_bf16 + base + (gate_first ? dimension : 0u), output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmSiluLimitOp{limit});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluRoundedMulLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base + (gate_first ? 0u : dimension), gate_up_bf16 + base + (gate_first ? dimension : 0u), output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmSiluRoundedLimitOp{limit});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulGateLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base, gate_up_bf16 + base + dimension, output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmSiluGateLimitOp{limit});
}

template<uint32_t THREADS, class Weight = uint16_t, bool ROUND_FIRST = false>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormKernel(const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon, float head_multiply)
{
	LmDependentWait();
	LmDependentRelease();
	if ( blockIdx.y >= rows || blockIdx.x >= heads ) return;
	const uint64_t base = ((uint64_t)blockIdx.y * heads + blockIdx.x) * head_dimension;
	LmRmsNormSmallBody<THREADS, 1u>(input_bf16 + base, (const uint16_t *)0, weight_bf16, output_bf16 + base, head_dimension, epsilon, LmHeadRmsOp<ROUND_FIRST>{head_multiply});
}

template<uint32_t THREADS, uint32_t GATE, bool ONCE = false>
static __device__ __forceinline__ void LmHeadRmsNormGateBody(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	LmDependentWait();
	LmDependentRelease();
	if ( blockIdx.y >= rows || blockIdx.x >= heads ) return;
	const uint64_t base = ((uint64_t)blockIdx.y * heads + blockIdx.x) * head_dimension;
	const uint16_t *in_row = input_bf16 + base;
	const uint16_t *gate_row = gate_bf16 + base;
	uint16_t *out_row = output_bf16 + base;
	if ( head_dimension <= THREADS )
	{
		const float in = threadIdx.x < head_dimension ? LmBf16ToFloat(in_row[threadIdx.x]) : 0.0f;
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(in * in, reduction), (float)head_dimension, epsilon);
		if ( threadIdx.x < head_dimension )
		{
			const float w = LmBf16ToFloat(__ldg(weight_bf16 + threadIdx.x));
			const float g = LmBf16ToFloat(gate_row[threadIdx.x]);
			out_row[threadIdx.x] = LmFloatToBf16(ONCE ? SparkMathRmsNormGateOnce(in, scale, w, SparkMathSigmoidFast(g))
				: SparkMathRoundBf16(SparkMathRoundBf16(in * scale) * w) * (GATE == LM_GATE_SIGMOID ? SparkMathSigmoidFast(g) : SparkMathSiluFast(g)));
		}
		return;
	}
	if ( head_dimension <= THREADS * 4u )
	{
		float staged[4], total = 0.0f;
		#pragma unroll
		for (uint32_t s = 0u; s < 4u; ++s)
		{
			const uint32_t index = threadIdx.x + s * THREADS;
			staged[s] = index < head_dimension ? LmBf16ToFloat(in_row[index]) : 0.0f;
			total += staged[s] * staged[s];
		}
		const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)head_dimension, epsilon);
		#pragma unroll
		for (uint32_t s = 0u; s < 4u; ++s)
		{
			const uint32_t index = threadIdx.x + s * THREADS;
			if ( index < head_dimension )
			{
				const float w = LmBf16ToFloat(__ldg(weight_bf16 + index));
				const float g = LmBf16ToFloat(gate_row[index]);
				out_row[index] = LmFloatToBf16(ONCE ? SparkMathRmsNormGateOnce(staged[s], scale, w, SparkMathSigmoidFast(g))
					: SparkMathRoundBf16(SparkMathRoundBf16(staged[s] * scale) * w) * (GATE == LM_GATE_SIGMOID ? SparkMathSigmoidFast(g) : SparkMathSiluFast(g)));
			}
		}
		return;
	}
	float total = 0.0f;
	for (uint32_t index = threadIdx.x; index < head_dimension; index += THREADS) { const float v = LmBf16ToFloat(in_row[index]); total += v * v; }
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)head_dimension, epsilon);
	for (uint32_t index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		const float in = LmBf16ToFloat(in_row[index]), w = LmBf16ToFloat(__ldg(weight_bf16 + index)), g = LmBf16ToFloat(gate_row[index]);
		out_row[index] = LmFloatToBf16(ONCE ? SparkMathRmsNormGateOnce(in, scale, w, SparkMathSigmoidFast(g))
			: SparkMathRoundBf16(SparkMathRoundBf16(in * scale) * w) * (GATE == LM_GATE_SIGMOID ? SparkMathSigmoidFast(g) : SparkMathSiluFast(g)));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSigmoidGateOnceKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormGateBody<THREADS, LM_GATE_SIGMOID, true>(input_bf16, gate_bf16, weight_bf16, output_bf16, rows, heads, head_dimension, epsilon);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSiluGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormGateBody<THREADS, LM_GATE_SILU, false>(input_bf16, gate_bf16, weight_bf16, output_bf16, rows, heads, head_dimension, epsilon);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSigmoidGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormGateBody<THREADS, LM_GATE_SIGMOID, false>(input_bf16, gate_bf16, weight_bf16, output_bf16, rows, heads, head_dimension, epsilon);
}

template<uint32_t THREADS, uint32_t ACTIVATION, bool ROUND_GATE = false>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadGateBroadcastKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension, uint32_t gate_stride = 0u)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.x, head = blockIdx.y, stride = gate_stride != 0u ? gate_stride : heads;
	uint16_t *base = heads_bf16 + ((((uint64_t)row * heads) + head) * head_dimension);
	float gate = LmBf16ToFloat(gate_bf16[((uint64_t)row * stride) + head]);
	if constexpr ( ACTIVATION == LM_GATE_SOFTPLUS ) gate = ROUND_GATE ? SparkMathSoftplusGate(gate) : (gate > 20.0f ? gate : log1pf(__expf(gate)));
	if constexpr ( ACTIVATION == LM_GATE_SIGMOID ) gate = ROUND_GATE ? SparkMathRoundBf16(SparkMathSigmoid(gate)) : SparkMathSigmoidFast(gate);
	if ( ((head_dimension & 7u) == 0u) && LmIsAligned16(base) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u)
		{
			float v[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(base + index), v);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) v[k] *= gate;
			*(uint4 *)(base + index) = LmPackUint4FloatToBf16(v);
		}
		return;
	}
	for (uint32_t index = threadIdx.x; index < head_dimension; index += THREADS) base[index] = LmFloatToBf16(LmBf16ToFloat(base[index]) * gate);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRmsNormSigmoidGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const float *__restrict__ weight, uint16_t *__restrict__ output_bf16, uint32_t dimension, float epsilon)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmDependentWait();
	LmDependentRelease();
	LmRmsNormSmallBody<THREADS, LM_RMS_NORM_STAGED / 8u>(input_bf16 + base, gate_bf16 + base, weight, output_bf16 + base, dimension, epsilon, LmRmsSigmoidGateOp{});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmClampedUpGateKernel(const uint16_t *__restrict__ up_gate_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float limit)
{
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	LmDependentWait();
	LmDependentRelease();
	LmGateUpRowBody<THREADS>(up_gate_bf16 + base + dimension, up_gate_bf16 + base, output_bf16 + (uint64_t)blockIdx.x * dimension, dimension, LmClampedUpGateOp{limit});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSituMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint64_t output_pitch, float beta, float linear_beta, uint32_t row_groups)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	const uint64_t out_base = (uint64_t)(blockIdx.x / row_groups) * output_pitch + (uint64_t)(blockIdx.x % row_groups) * dimension;
	LmGateUpRowBody<THREADS>(gate_up_bf16 + base, gate_up_bf16 + base + dimension, output_bf16 + out_base, dimension, LmSituOp{beta, linear_beta});
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16, float *__restrict__ scores, uint32_t width)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *src = logits_bf16 + base;
	float *dst = scores + base;
	if ( ((width & 7u) == 0u) && LmIsAligned16(src) && LmIsAligned16(dst) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < width; index += THREADS * 8u)
		{
			float v[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(src + index), v);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) v[k] = SparkMathSigmoidFast(v[k]);
			LmStoreFloat8(dst + index, v);
		}
		return;
	}
	for (uint32_t index = threadIdx.x; index < width; index += THREADS) dst[index] = SparkMathSigmoidFast(LmBf16ToFloat(src[index]));
}

template<uint32_t THREADS, bool DOUBLED = false>
__global__ __launch_bounds__(THREADS, 1)
void LmBf16SigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16, float *__restrict__ scores, uint32_t width)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t base = (uint64_t)blockIdx.x * width;
	const uint16_t *src = logits_bf16 + base;
	float *dst = scores + base;
	if ( ((width & 7u) == 0u) && LmIsAligned16(src) && LmIsAligned16(dst) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < width; index += THREADS * 8u)
		{
			float v[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(src + index), v);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				const float r = SparkMathRoundBf16(SparkMathSigmoidFast(v[k]));
				v[k] = DOUBLED ? 2.0f * r : r;
			}
			LmStoreFloat8(dst + index, v);
		}
		return;
	}
	for (uint32_t index = threadIdx.x; index < width; index += THREADS)
	{
		const float r = SparkMathRoundBf16(SparkMathSigmoidFast(LmBf16ToFloat(src[index])));
		dst[index] = DOUBLED ? 2.0f * r : r;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSigmoidGateKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension, uint32_t gate_head_stride, uint32_t gate_offset)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t pair = ((uint64_t)blockIdx.x * heads) + blockIdx.y;
	uint16_t *row = heads_bf16 + pair * head_dimension;
	LmGateUpRowBody<THREADS>(gate_bf16 + (pair * gate_head_stride) + gate_offset, row, row, head_dimension, LmSigmoidGateOp<false>{});
}

template<uint32_t THREADS, uint32_t RANK>
__global__ __launch_bounds__(THREADS)
void LmLowRankGateDownKernel(const uint16_t *__restrict__ rows_bf16, const uint16_t *__restrict__ down_bf16, float *__restrict__ bottleneck, uint32_t dimension)
{
	__shared__ float partial[LmBlockWarps<THREADS>()];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t rank = blockIdx.x, row = blockIdx.y;
	const uint64_t base = (uint64_t)row * dimension;
	float total = 0.0f;
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		total += LmBf16ToFloat(rows_bf16[base + index]) * LmBf16ToFloat(__ldg(down_bf16 + ((uint64_t)rank * dimension) + index));
	total = LmBlockSumOnce<THREADS>(total, partial);
	if ( threadIdx.x == 0u ) bottleneck[(uint64_t)row * RANK + rank] = SparkMathRoundBf16(SparkMathSiluFast(SparkMathRoundBf16(total)));
}

template<uint32_t THREADS, uint32_t RANK>
__global__ __launch_bounds__(THREADS)
void LmLowRankGateApplyKernel(uint16_t *__restrict__ rows_bf16, const float *__restrict__ bottleneck, const uint16_t *__restrict__ up_bf16, uint32_t dimension)
{
	__shared__ float factors[RANK];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t index = blockIdx.x * THREADS + threadIdx.x, row = blockIdx.y;
	for (uint32_t r = threadIdx.x; r < RANK; r += THREADS)
		factors[r] = bottleneck[(uint64_t)row * RANK + r];
	__syncthreads();
	if ( index >= dimension ) return;
	float gate = 0.0f;
	#pragma unroll
	for (uint32_t rank = 0u; rank < RANK; ++rank) gate += factors[rank] * LmBf16ToFloat(__ldg(up_bf16 + ((uint64_t)index * RANK) + rank));
	rows_bf16[(uint64_t)row * dimension + index] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[(uint64_t)row * dimension + index]) * SparkMathSigmoidFast(SparkMathRoundBf16(gate)));
}

template<uint32_t THREADS, bool ROUND_GATE = false>
__global__ __launch_bounds__(THREADS, 1)
void LmOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t base = (uint64_t)blockIdx.x * dimension;
	LmGateUpRowBody<THREADS>(gate_bf16 + base, output_bf16 + base, output_bf16 + base, dimension, LmSigmoidGateOp<ROUND_GATE>{});
}

template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmQuantiseRowsKernel(const uint16_t *__restrict__ input_bf16, const uint32_t *__restrict__ source_row_map, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t row_count, uint32_t dimension)
{
	static_assert(Format::kScaleGroup > 0u && (Format::kScaleGroup % 8u) == 0u, "quantization groups must be positive multiples of eight");
	extern __shared__ float lm_quant_shared[];
	LmDependentWait();
	LmDependentRelease();
	float *row = lm_quant_shared, *reduction = lm_quant_shared + Format::kScaleGroup, absmax = 0.0f;
	const uint32_t groups = dimension / Format::kScaleGroup, destination = blockIdx.x, group = blockIdx.y;
	if ( destination >= row_count || group >= groups ) return;
	const uint64_t source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination) * (uint64_t)dimension + (group * Format::kScaleGroup);
	const uint16_t *src = input_bf16 + source;
	if ( LmIsAligned16(src) && LmIsAligned16(row) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < Format::kScaleGroup; index += THREADS * 8u)
		{
			float v[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(src + index), v);
			LmStoreFloat8(row + index, v);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) absmax = fmaxf(absmax, fabsf(v[k]));
		}
	}
	else
	{
		for (uint32_t index = threadIdx.x; index < Format::kScaleGroup; index += THREADS) { row[index] = LmBf16ToFloat(src[index]); absmax = fmaxf(absmax, fabsf(row[index])); }
	}
	uint8_t scale_code = LmFloatToUe4m3(LmBlockMax<THREADS>(absmax, reduction) / Format::kMax);
	if ( scale_code == 0u ) scale_code = 1u;
	const float inverse = 1.0f / LmUe4m3ToFloat(scale_code);
	if ( threadIdx.x == 0u ) output_scales[((uint64_t)destination * groups) + group] = scale_code;
	__syncthreads();
	for (uint32_t index = threadIdx.x * 8u; index < Format::kScaleGroup; index += THREADS * 8u)
	{
		float values[8];
		const uint64_t bit = (((uint64_t)destination * dimension) + (group * Format::kScaleGroup) + index) * Format::kStoredBits;
		for (uint32_t v = 0u; v < 8u; ++v) values[v] = row[index + v] * inverse;
		LmStoreCodeOctet<Format>(output_codes, bit, values);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(packed_bf16) && LmIsAligned16(output_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		for (uint32_t token = blockIdx.y; token < tokens; token += gridDim.y)
		{
			float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			for (uint32_t route = 0u; route < top_k; ++route)
			{
				const uint64_t slot = ((uint64_t)token * top_k) + route;
				LmFmaBf16x8(total, *(const uint4 *)(packed_bf16 + ((uint64_t)__ldg(packed_row_of_token_route + slot) * dimension) + vec_element), __ldg(route_weight + slot));
			}
			*(uint4 *)(output_bf16 + ((uint64_t)token * dimension) + vec_element) = LmPackUint4FloatToBf16(total);
		}
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	for (uint32_t token = blockIdx.y; token < tokens; token += gridDim.y)
	{
		float total = 0.0f;
		for (uint32_t route = 0u; route < top_k; ++route)
		{
			const uint64_t slot = ((uint64_t)token * top_k) + route;
			total += LmBf16ToFloat(packed_bf16[((uint64_t)packed_row_of_token_route[slot] * dimension) + element]) * route_weight[slot];
		}
		output_bf16[((uint64_t)token * dimension) + element] = LmFloatToBf16(total);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeAddKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, const uint16_t *__restrict__ addend_bf16, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t element = ((blockIdx.x * THREADS) + threadIdx.x) * LM_MOE_FINALIZE_ADD_ELEMENTS;
	if ( element >= dimension ) return;
	for (uint32_t token = blockIdx.y; token < tokens; token += gridDim.y)
	{
		float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, add[8];
		for (uint32_t route = 0u; route < top_k; ++route)
		{
			const uint64_t slot = ((uint64_t)token * top_k) + route;
			LmFmaBf16x8(total, *(const uint4 *)(packed_bf16 + ((uint64_t)__ldg(packed_row_of_token_route + slot) * dimension) + element), __ldg(route_weight + slot));
		}
		LmLoadBf16x8AsFloat(*(const uint4 *)(addend_bf16 + ((uint64_t)token * dimension) + element), add);
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k) total[k] = SparkMathRoundBf16(total[k]) + add[k];
		*(uint4 *)(output_bf16 + ((uint64_t)token * dimension) + element) = LmPackUint4FloatToBf16(total);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsKernel(const uint16_t *__restrict__ a_bf16, const uint16_t *__restrict__ b_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(a_bf16) && LmIsAligned16(b_bf16) && LmIsAligned16(out_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
		{
			const uint64_t index = ((uint64_t)row * dimension) + vec_element;
			float a[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(a_bf16 + index), a);
			LmAddBf16x8(a, *(const uint4 *)(b_bf16 + index));
			*(uint4 *)(out_bf16 + index) = LmPackUint4FloatToBf16(a);
		}
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
	{
		const uint64_t index = ((uint64_t)row * dimension) + element;
		out_bf16[index] = LmFloatToBf16(LmBf16ToFloat(a_bf16[index]) + LmBf16ToFloat(b_bf16[index]));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCopyRowsKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(source_bf16) && LmIsAligned16(destination_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
			*(uint4 *)(destination_bf16 + ((uint64_t)row * dimension) + vec_element) = *(const uint4 *)(source_bf16 + ((uint64_t)row * dimension) + vec_element);
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
		destination_bf16[((uint64_t)row * dimension) + element] = source_bf16[((uint64_t)row * dimension) + element];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRankMajorRowsKernel(const uint16_t *__restrict__ gathered_bf16, uint16_t *__restrict__ rows_bf16, uint32_t rows, uint32_t width, uint32_t ranks, uint32_t first_rank = 0u)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t total = width * ranks;
	if ( ((width & 7u) == 0u) && LmIsAligned16(gathered_bf16) && LmIsAligned16(rows_bf16) )
	{
		const uint32_t width_v = width >> 3u, total_v = width_v * ranks;
		const uint4 *g4 = (const uint4 *)gathered_bf16;
		uint4 *r4 = (uint4 *)rows_bf16;
		for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
			for (uint32_t v = (blockIdx.x * THREADS) + threadIdx.x; v < total_v; v += gridDim.x * THREADS)
				r4[(uint64_t)row * total_v + v] = g4[(((uint64_t)(first_rank + (v / width_v)) * rows) + row) * width_v + (v % width_v)];
		return;
	}
	for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
		for (uint32_t index = (blockIdx.x * THREADS) + threadIdx.x; index < total; index += gridDim.x * THREADS)
			rows_bf16[((uint64_t)row * total) + index] = gathered_bf16[((((uint64_t)(first_rank + (index / width)) * rows) + row) * width) + (index % width)];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSliceColumnsKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ target_bf16, uint32_t source_width, uint32_t offset, uint32_t width, uint32_t target_stride = 0u)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y, dst_stride = target_stride != 0u ? target_stride : width;
	const uint16_t *src = source_bf16 + ((uint64_t)row * source_width) + offset;
	uint16_t *dst = target_bf16 + ((uint64_t)row * dst_stride);
	if ( ((width & 7u) == 0u) && LmIsAligned16(src) && LmIsAligned16(dst) )
	{
		for (uint32_t v = (blockIdx.x * THREADS) + threadIdx.x; v < (width >> 3u); v += gridDim.x * THREADS) ((uint4 *)dst)[v] = ((const uint4 *)src)[v];
		return;
	}
	for (uint32_t index = (blockIdx.x * THREADS) + threadIdx.x; index < width; index += gridDim.x * THREADS) dst[index] = src[index];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCopyRowsStridedKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension, uint64_t destination_stride)
{
	LmDependentWait();
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && ((destination_stride & 7u) == 0u) && LmIsAligned16(source_bf16) && LmIsAligned16(destination_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
			*(uint4 *)(destination_bf16 + ((uint64_t)row * destination_stride) + vec_element) = *(const uint4 *)(source_bf16 + ((uint64_t)row * dimension) + vec_element);
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
		destination_bf16[((uint64_t)row * destination_stride) + element] = source_bf16[((uint64_t)row * dimension) + element];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsStridedKernel(const uint16_t *__restrict__ source_bf16, const uint16_t *__restrict__ addend_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension, uint64_t destination_stride)
{
	LmDependentWait();
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && ((destination_stride & 7u) == 0u) && LmIsAligned16(source_bf16) && LmIsAligned16(addend_bf16) && LmIsAligned16(destination_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
		{
			const uint64_t src_index = ((uint64_t)row * dimension) + vec_element;
			float a[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(source_bf16 + src_index), a);
			LmAddBf16x8(a, *(const uint4 *)(addend_bf16 + src_index));
			*(uint4 *)(destination_bf16 + ((uint64_t)row * destination_stride) + vec_element) = LmPackUint4FloatToBf16(a);
		}
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	for (uint32_t row = blockIdx.y; row < rows; row += gridDim.y)
	{
		const uint64_t source = ((uint64_t)row * dimension) + element;
		destination_bf16[((uint64_t)row * destination_stride) + element] = LmFloatToBf16(LmBf16ToFloat(source_bf16[source]) + LmBf16ToFloat(addend_bf16[source]));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmStreamMeanRowsKernel(const uint16_t *__restrict__ streams_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t streams, uint32_t dimension, uint64_t destination_stride)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	if ( row >= rows ) return;
	if ( ((dimension & 7u) == 0u) && ((destination_stride & 7u) == 0u) && LmIsAligned16(streams_bf16) && LmIsAligned16(destination_bf16) )
	{
		uint32_t vec_element;
		if ( !LmVecElement<THREADS>(dimension, &vec_element) ) return;
		const uint16_t *source = streams_bf16 + ((uint64_t)row * streams * dimension) + vec_element;
		float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		for (uint32_t stream = 0u; stream < streams; ++stream)
			LmAddBf16x8(total, *(const uint4 *)(source + (uint64_t)stream * dimension));
		const float inv = (float)streams;
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k) total[k] /= inv;
		*(uint4 *)(destination_bf16 + ((uint64_t)row * destination_stride) + vec_element) = LmPackUint4FloatToBf16(total);
		return;
	}
	const uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension ) return;
	const uint16_t *source = streams_bf16 + ((uint64_t)row * streams * dimension) + element;
	float total = 0.0f;
	for (uint32_t stream = 0u; stream < streams; ++stream) total += LmBf16ToFloat(source[(uint64_t)stream * dimension]);
	destination_bf16[((uint64_t)row * destination_stride) + element] = LmFloatToBf16(total / (float)streams);
}

static __device__ __forceinline__ const uint16_t *LmAttnResSourceRow(const uint16_t *bank_bf16, const uint16_t *partial_bf16, uint32_t source, uint32_t sources, uint32_t rows, uint32_t row, uint32_t dimension)
{
	return(source + 1u == sources ? partial_bf16 + ((uint64_t)row * dimension) : bank_bf16 + ((((uint64_t)source * rows) + row) * dimension));
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmAttnResScoreValue(const uint16_t *values, const uint16_t *score_weight_bf16, uint32_t dimension, float epsilon, float *reduction, bool vec_ok)
{
	float square = 0.0f, dot = 0.0f;
	if ( vec_ok )
	{
		if ( dimension <= THREADS * 32u )
		{
			uint4 staged[4];
			#pragma unroll
			for (uint32_t chunk = 0u; chunk < 4u; ++chunk)
			{
				const uint32_t index = (chunk * THREADS + threadIdx.x) * 8u;
				if ( index < dimension )
				{
					staged[chunk] = *(const uint4 *)(values + index);
					square = LmSumSqBf16x8(staged[chunk], square);
				}
			}
			const float inverse = SparkMathRmsScale(LmBlockSum<THREADS>(square, reduction), (float)dimension, epsilon);
			#pragma unroll
			for (uint32_t chunk = 0u; chunk < 4u; ++chunk)
			{
				const uint32_t index = (chunk * THREADS + threadIdx.x) * 8u;
				if ( index < dimension )
				{
					float v[8], w[8];
					LmLoadBf16x8AsFloat(staged[chunk], v);
					LmLoadBf16x8AsFloat(*(const uint4 *)(score_weight_bf16 + index), w);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k) dot += v[k] * inverse * w[k];
				}
			}
			return(LmBlockSum<THREADS>(dot, reduction));
		}
		for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u) square = LmSumSqBf16x8(*(const uint4 *)(values + index), square);
		const float inverse = SparkMathRmsScale(LmBlockSum<THREADS>(square, reduction), (float)dimension, epsilon);
		for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			float v[8], w[8];
			LmLoadBf16x8AsFloat(*(const uint4 *)(values + index), v);
			LmLoadBf16x8AsFloat(*(const uint4 *)(score_weight_bf16 + index), w);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k) dot += v[k] * inverse * w[k];
		}
		return(LmBlockSum<THREADS>(dot, reduction));
	}
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) { const float v = LmBf16ToFloat(values[index]); square += v * v; }
	const float inverse = SparkMathRmsScale(LmBlockSum<THREADS>(square, reduction), (float)dimension, epsilon);
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) dot += LmBf16ToFloat(values[index]) * inverse * LmBf16ToFloat(score_weight_bf16[index]);
	return(LmBlockSum<THREADS>(dot, reduction));
}

static __device__ __forceinline__ uint4 LmAttnResMix8(const uint16_t *bank_bf16, const uint16_t *partial_bf16, const float *weight, uint32_t sources, uint32_t rows, uint32_t row, uint32_t dimension, uint32_t offset)
{
	float total[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (uint32_t s = 0u; s < sources; ++s) LmFmaBf16x8(total, *(const uint4 *)(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, dimension) + offset), weight[s]);
	return(LmPackUint4FloatToBf16(total));
}

static __device__ __forceinline__ uint16_t LmAttnResMix1(const uint16_t *bank_bf16, const uint16_t *partial_bf16, const float *weight, uint32_t sources, uint32_t rows, uint32_t row, uint32_t dimension, uint32_t index)
{
	float total = 0.0f;
	for (uint32_t s = 0u; s < sources; ++s) total += weight[s] * LmBf16ToFloat(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, dimension)[index]);
	return(LmFloatToBf16(total));
}

template<uint32_t THREADS, uint32_t MAX_SOURCES>
__global__ __launch_bounds__(THREADS)
void LmAttnResKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const uint16_t *__restrict__ score_weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t sources, uint32_t rows, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()], score[MAX_SOURCES], weight[MAX_SOURCES];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.x;
	const bool vec_ok = ((dimension & 7u) == 0u) && LmIsAligned16(bank_bf16) && LmIsAligned16(partial_bf16) && LmIsAligned16(output_bf16);
	for (uint32_t s = 0u; s < sources; ++s)
	{
		const float dot = LmAttnResScoreValue<THREADS>(LmAttnResSourceRow(bank_bf16, partial_bf16, s, sources, rows, row, dimension), score_weight_bf16, dimension, epsilon, reduction, vec_ok && LmIsAligned16(score_weight_bf16));
		if ( threadIdx.x == 0u ) score[s] = dot;
		__syncthreads();
	}
	LmWarpSoftmax<THREADS>(score, weight, sources);
	if ( vec_ok ) { for (uint32_t index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u) *(uint4 *)(output_bf16 + ((uint64_t)row * dimension) + index) = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, dimension, index); return; }
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS) output_bf16[((uint64_t)row * dimension) + index] = LmAttnResMix1(bank_bf16, partial_bf16, weight, sources, rows, row, dimension, index);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS)
void LmAttnResScoreKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const uint16_t *__restrict__ score_weight_bf16, float *__restrict__ score_f32, uint32_t sources, uint32_t rows, uint32_t dimension, uint32_t score_stride, float epsilon)
{
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.x, source = blockIdx.y;
	const uint16_t *values = LmAttnResSourceRow(bank_bf16, partial_bf16, source, sources, rows, row, dimension);
	const float dot = LmAttnResScoreValue<THREADS>(values, score_weight_bf16, dimension, epsilon, reduction, ((dimension & 7u) == 0u) && LmIsAligned16(values) && LmIsAligned16(score_weight_bf16));
	if ( threadIdx.x == 0u ) score_f32[((uint64_t)row * score_stride) + source] = dot;
}

template<uint32_t THREADS, uint32_t MAX_SOURCES>
__global__ __launch_bounds__(THREADS)
void LmAttnResMixKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const float *__restrict__ score_f32, uint16_t *__restrict__ output_bf16, uint32_t sources, uint32_t rows, uint32_t dimension, uint32_t score_stride)
{
	__shared__ float weight[MAX_SOURCES];
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y, index = (blockIdx.x * THREADS) + threadIdx.x;
	LmWarpSoftmax<THREADS>(score_f32 + ((uint64_t)row * score_stride), weight, sources);
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(bank_bf16) && LmIsAligned16(partial_bf16) && LmIsAligned16(output_bf16) )
	{
		if ( index * 8u < dimension ) *(uint4 *)(output_bf16 + ((uint64_t)row * dimension) + index * 8u) = LmAttnResMix8(bank_bf16, partial_bf16, weight, sources, rows, row, dimension, index * 8u);
		return;
	}
	if ( index < dimension ) output_bf16[((uint64_t)row * dimension) + index] = LmAttnResMix1(bank_bf16, partial_bf16, weight, sources, rows, row, dimension, index);
}

template<uint32_t THREADS, bool SCATTER>
static __device__ __forceinline__ void LmIndexedRowCopyBody(const uint16_t *source_bf16, const uint32_t *row_map, uint16_t *destination_bf16, uint32_t rows, uint32_t dimension)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y;
	if ( row >= rows ) return;
	const uint32_t mapped = row_map != 0 ? row_map[row] : row;
	const uint16_t *src = source_bf16 + ((uint64_t)(SCATTER ? row : mapped) * dimension);
	uint16_t *dst = destination_bf16 + ((uint64_t)(SCATTER ? mapped : row) * dimension);
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(src) && LmIsAligned16(dst) )
	{
		for (uint32_t v = (blockIdx.x * THREADS) + threadIdx.x; v < (dimension >> 3u); v += gridDim.x * THREADS) ((uint4 *)dst)[v] = ((const uint4 *)src)[v];
		return;
	}
	for (uint32_t index = (blockIdx.x * THREADS) + threadIdx.x; index < dimension; index += gridDim.x * THREADS) dst[index] = src[index];
}

template<uint32_t THREADS>
__global__ void LmGatherRowsKernel(const uint16_t *__restrict__ source_bf16, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	LmIndexedRowCopyBody<THREADS, false>(source_bf16, source_row_map, destination_bf16, rows, dimension);
}

template<uint32_t THREADS>
__global__ void LmScatterRowsKernel(const uint16_t *__restrict__ source_bf16, const uint32_t *__restrict__ dest_row_map, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	LmIndexedRowCopyBody<THREADS, true>(source_bf16, dest_row_map, destination_bf16, rows, dimension);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCenteredRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	LmDependentWait();
	LmDependentRelease();
	LmRmsNormRowBody<THREADS, uint16_t, false, false>(lm_norm_shared, input_bf16, (const uint16_t *)0, weight_bf16, (uint16_t *)0, output_bf16, dimension, row_stride, epsilon, LmRmsCenteredOp{});
}

static __global__ void LmAddElementsKernel(uint16_t *__restrict__ destination_bf16, const uint16_t *__restrict__ addend_bf16, uint64_t elements)
{
	LmDependentWait();
	LmDependentRelease();
	LmBinaryBf16GridBody(destination_bf16, addend_bf16, destination_bf16, elements, LmAddOp{});
}

template<class T = uint16_t>
__global__ void LmScaleElementsKernel(T *__restrict__ values, uint64_t elements, float scale)
{
	LmDependentWait();
	LmDependentRelease();
	if constexpr ( sizeof(T) == 2u ) { LmUnaryBf16GridBody(values, elements, LmScaleOp{scale}); return; }
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
	if ( LmIsAligned16(values) )
	{
		const uint64_t vecs = elements >> 2u;
		float4 *v4 = (float4 *)values;
		for (uint64_t v = tid; v < vecs; v += stride) { float4 x = v4[v]; v4[v] = {x.x * scale, x.y * scale, x.z * scale, x.w * scale}; }
		for (uint64_t index = (vecs << 2u) + tid; index < elements; index += stride) values[index] *= scale;
		return;
	}
	for (uint64_t index = tid; index < elements; index += stride) values[index] *= scale;
}

static __global__ void LmConcatRowsKernel(const uint16_t *__restrict__ left_bf16, const uint16_t *__restrict__ right_bf16, uint16_t *__restrict__ output_bf16, uint32_t left_width, uint32_t right_width, uint32_t right_stride = 0u, uint32_t right_offset = 0u)
{
	LmDependentWait();
	LmDependentRelease();
	const uint32_t row = blockIdx.y, total = left_width + right_width, r_stride = right_stride != 0u ? right_stride : right_width;
	const uint16_t *left_row = left_bf16 + (uint64_t)row * left_width, *right_row = right_bf16 + (uint64_t)row * r_stride + right_offset;
	uint16_t *out_row = output_bf16 + (uint64_t)row * total;
	if ( ((left_width & 7u) == 0u) && ((right_width & 7u) == 0u) && LmIsAligned16(left_row) && LmIsAligned16(right_row) && LmIsAligned16(out_row) )
	{
		const uint32_t left_v = left_width >> 3u, total_v = total >> 3u;
		for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < total_v; v += gridDim.x * blockDim.x)
			((uint4 *)out_row)[v] = v < left_v ? ((const uint4 *)left_row)[v] : ((const uint4 *)right_row)[v - left_v];
		return;
	}
	for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += gridDim.x * blockDim.x)
		out_row[index] = index < left_width ? left_row[index] : right_row[index - left_width];
}

static __global__ void LmWidenBf16ToF32Kernel(const uint16_t *__restrict__ source_bf16, float *__restrict__ target_f32, uint64_t elements)
{
	LmDependentWait();
	LmDependentRelease();
	const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (uint64_t)gridDim.x * blockDim.x;
	if ( LmIsAligned16(source_bf16) && LmIsAligned16(target_f32) )
	{
		const uint64_t vecs = elements >> 3u;
		const uint4 *s4 = (const uint4 *)source_bf16;
		for (uint64_t v = tid; v < vecs; v += stride) { float out[8]; LmLoadBf16x8AsFloat(s4[v], out); LmStoreFloat8(target_f32 + (v << 3u), out); }
		for (uint64_t index = (vecs << 3u) + tid; index < elements; index += stride) target_f32[index] = LmBf16ToFloat(source_bf16[index]);
		return;
	}
	for (uint64_t index = tid; index < elements; index += stride) target_f32[index] = LmBf16ToFloat(source_bf16[index]);
}

template<uint32_t THREADS, bool ROUND_ACT = false>
__global__ __launch_bounds__(THREADS, 1)
void LmGateUpStridedKernel(const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ up_bf16, uint32_t source_stride, uint16_t *__restrict__ output_bf16, uint32_t width)
{
	LmDependentWait();
	LmDependentRelease();
	LmBinaryBf16GridBody<THREADS>(gate_bf16 + (uint64_t)blockIdx.y * source_stride, up_bf16 + (uint64_t)blockIdx.y * source_stride, output_bf16 + (uint64_t)blockIdx.y * width, width, LmStridedGateUpOp<ROUND_ACT>{});
}
